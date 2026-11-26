# Independent review: K1 patch (Pascal K-quant MMVQ), p104-exp a96dce0 (v8)

Reviewer: independent, host z4-p104, 2026-11-23 03:40-03:55 CET. I did not use the GPU: no llama binaries, no benchmarks, no compile. I read the code with `git diff 5bacb7c a96dce0` and the surrounding mmvq.cu/ggml-cuda.cu. Everything else comes from the raw receipts in ~/p104-bench/runs (*.out, *.meta.json, *.trace.csv, chain logs and scripts). I modified nothing except this file.

## Verdict: APPROVE-WITH-CONDITIONS

- **The kernel math is correct.** I checked the new vec_dot line by line against upstream `vec_dot_q4_K_q8_1` and `vec_dot_q5_K_q8_1`. The integer math, the scale/min unpacking, the exact min-term sum and the Q5_K high-bit handling are all equivalent. The only difference is the order of the fp32 additions.
- **The speedup is real.** The raw files reproduce every headline number. The ABBA runs are clean, not contaminated, and thermally even.
- **Not yet production-ready, for these reasons:**
  - The v8 change (wide vec_dot for 2 and 3 columns) has no model-level numerics check. `llama-server -np 2` runs exactly this path when both slots decode.
  - The ub-1 KLD has one unexplained 0.395 outlier token, and the sample is too small (2.5k tokens) to rule out a real problem.
  - There are no out-of-bounds checks for the new 4-rows-per-block tail reads.
  - The second local Q4_K_M model (Qwen3-8B) was never measured.

None of these is a known bug. All can be closed in about an hour of GPU time (tests listed below).

## Findings

| # | Severity | Finding | Reference |
|---|---|---|---|
| 1 | Info (pass) | **Q4_K math matches upstream.** With iqs = 8j, the thread reads `qs[32j..32j+31]`, which equals upstream's `16*bq8_offset` with bq8_offset = 2j. It pairs quant int l with q8 int l of q8 blocks 2j and 2j+1, which equals upstream's u[2i], u[2i+1] pairing over ints t and t+4. The scale unpack matches upstream for both j<2 and j>=2: for j>=2, `s4=scales[j+2]`, `s0=scales[j-2]`, `s2=scales[j]`. The `scales[jm+4]` index is at most 5, inside the 6 uint16 values. The min term uses the exact dp4a(0x01010101) sum and the same d8 multipliers as `vec_dot_q4_K_q8_1_impl_vmmq`. So the result is the same integer math; only the per-thread fp32 accumulation order changes. | mmvq.cu:99-158 |
| 2 | Info (pass) | **Q5_K math matches upstream.** `(vh << (4-i)) & 0x10101010` equals upstream `((vh >> i) << 4) & 0x10101010` on every masked bit (i is 0 or 1). `qh >> 2j` equals upstream `>> bq8_offset`. Both use the same signed arithmetic shift. | mmvq.cu:161-217 |
| 3 | Info (pass) | **Index bounds are fine.** K always divides into whole 256-element blocks, because `ncols_x % blck_size` is asserted at mmvq.cu:1383. The K loop `kbx < blocks_per_row_x` handles a partial last iteration: with 64 threads at vdr 8, one iteration covers 16 super-blocks. The column count is a template parameter. 2 or 3 columns use the generic nwarps 4 and 2 rows per block. 4 to 8 columns use upstream vdr 2. Multi-token MoE (`ids` with ncols>1) goes to `mul_mat_vec_q_moe`, which this patch does not touch. Shared memory is `[nwarps-1][ncols][rpb][32]` floats: 512 B for Q4_K, 1.5 KB for Q6_K, so nothing is near a limit. The bias prefetch and the write-back both use `threadIdx.x < rpb` and `threadIdx.x == i`, so they stay consistent at 4 rows per block. | mmvq.cu:1000-1041, 1088 |
| 4 | Medium | **Overrun past the last row grows from 0 to 3 rows.** Rows past the matrix end are not guarded when reading. With 1 column, Q4_K, Q5_K and Q6_K now use 4 rows per block instead of 1. If nrows % 4 != 0, the kernel reads up to 3 rows past the end of the tensor. For Q6_K at k=4096 that is about 10 KB. The values are discarded at the write guard. The CUDA allocator pads only up to MATRIX_ROW_PADDING columns, never extra rows. Upstream already accepts a 1-row overrun at 2..8 columns, so this is the same class of risk, just three times larger and now on the common 1-column path. All Qwen3.5-9B row counts are multiples of 4, as are Qwen3-8B's (151936 included). An odd-vocab output tensor that is last in its buffer could fault. `test-backend-ops` m=1023 n=1 k=4096 passed, but that does not detect out-of-bounds reads that hit allocated memory. **Condition:** run compute-sanitizer (test T4 below). Suggested hardening: clamp the row index to nrows-1 when reading. | mmvq.cu:998, 1032 |
| 5 | Low | **16-byte alignment is assumed, not checked.** Both block sizes are multiples of 16 (Q4_K is 144 B with qs at offset 16; Q5_K is 176 B with qh at 16 and qs at 48). So every block-granular offset keeps the base pointer's 16-byte alignment. That covers row and expert slices (MUL_MAT_ID `stride_channel_x`), split-mode row slices, and column views. Device buffers are 128-byte aligned (ggml-cuda.cu:1298), and K-quants cannot be KV-cache types. So I found no reachable misaligned case on a discrete GPU. The patch does tighten the requirement from 4 to 16 bytes, unchecked. Upstream has `offset_a` view tests at 1/2/4/16 bytes (test-backend-ops.cpp:12217, currently F8 only). A host-pointer buffer from a GGUF with alignment < 16 would fault. Suggested: a host-side check `((uintptr_t)vx \| (uintptr_t)gate) & 15` that falls back to vdr 2. | mmvq.cu:105, 167-168 |
| 6 | Info (pass) | **Arch gating is clean.** Device side: the Pascal table is selected only for `610 <= __CUDA_ARCH__ < 700`. HIP and MUSA define `__CUDA_ARCH__` as 1300 (vendors/hip.h:184, musa.h:11), so they are excluded. Host side: `GGML_CUDA_CC_IS_NVIDIA && highest_compiled_arch in [610,700)` mirrors the existing Turing/Orin logic, so host and device pick the same table, including when 61 PTX is JIT-compiled on a newer GPU. All types other than Q4_K/Q5_K/Q6_K fall through to GENERIC values, so their kernels are unchanged. P100 (6.0) is unaffected. **Caveat:** the table also applies to every GP102/GP106/GP107 card and to TX2 (6.2), and it was tuned only on P104. That matters for upstreaming, not for this host. | mmvq.cu:276-277, 308-310, 790-796, 815-827 |
| 7 | Info | **The fused path cannot run on this GPU.** `ggml_cuda_should_fuse_mul_mat_vec_q` returns false for cc < Volta. So MMVQ gate/bias/GLU fusion never runs on the P104: that kernel code is compiled but unreachable here. This matches the dispatch receipt (`gate=0` everywhere). The new fused perf case in test-backend-ops never ran either: its filter matched no lines in k1-op-*.out. MUL_MAT_FUSION correctness tests pass but go through the unfused path. Likewise, MUL_MAT_ID 1-column tests use only k=256, which takes the small_k path. So the 4-rows-per-block path with `ids`, at k >= 4096, has no correctness test. That matters only for MoE models. | ggml-cuda.cu:2350; runs/k1-test-v8-K.out |
| 8 | Low (perf) | **The small_k threshold moved.** The host now uses vdr 8 in `blocks_per_iter_1warp` for Pascal Q4_K/Q5_K. So small_k (2 rows per block, 2 warps) now triggers for K < 4096; upstream triggered for K < 2048. For K = 2048 (for example MoE experts or a 2048-hidden model), half the threads in each block are idle. The same happens for 2 and 3 columns at K = 4096, where one iteration covers 32 blocks and only 16 exist: measured q4_K 4096x4096 n=2 gives only +0.9%. These shapes do not occur in Qwen3.5-9B or Qwen3-8B and were not measured, so this is not a correctness issue. | mmvq.cu:1401-1411 |
| 9 | Info (pass) | **The debug trace costs nothing when unset.** It is a function-local static bool, so there is one guarded load per host launch, and only on the 1-column path. CUDA graph replay skips host code entirely. It adds `<cstdio>`/`<cstdlib>`. The `P104_*` macros and the dead `P104_EXACT_SUM=0` branch should be removed before upstreaming. | mmvq.cu:1368-1372, 1491-1495 |
| 10 | Medium (evidence) | **The 2..3-column change (v8) has no model-level numerics check.** The ub-1 KLD exercises only the 1-column path, which is why v2/v6/v7/v8 give bit-identical KLD. The v8 change to 2 and 3 columns is covered only by test-backend-ops NMSE against CPU, and that threshold is loose. Yet 2 columns is what `-np 2` runs when both slots decode. **Condition:** KLD at -ub 2 and -ub 3 (T1). | LEDGER row k1-opn-v8 |
| 11 | Medium (evidence) | **The ub-1 KLD outlier and sample size need resolving.** Counts come from k1-kldub1-v8.out against the in-fixture floor k1-kldub1-floor-fa0.out (unpatched, fa off). Mean KLD 0.000604 +- 0.000155 against 0.000435 +- 0.000029; median 0.000302 against 0.000286; 99.9% 0.0084 against 0.0078; RMS dp 0.591% against 0.542%; same-top 98.43% against 98.71%. With about 2560 scored tokens, the single max token (0.395) adds 0.395/2560 = 0.000154 to the mean, which equals the whole standard error. Without it the mean is about 0.00045, the same as the floor. So the bulk of the distribution sits at the noise floor. But one token at 0.395 is 6x the floor's maximum (0.067), and one sample cannot show whether it is chance. A plausible cause is that Qwen3.5's recurrent DeltaNet state carries tiny reorder differences forward in ub-1 mode. The "0.00075 q3a floor" in the LEDGER is a different fixture (ub 512, 20 chunks) and is not the right comparator. **Condition:** 20-chunk ub-1 floor and v8 runs, plus a non-recurrent model (T2, T3). | runs/k1-kldub1-*.out |
| 12 | Low (evidence) | **Determinism of the baseline was never measured.** There is no base-vs-base ub-1 KLD, and no server output-equality check: fit_k1.py does not save completions. The kernel itself is deterministic (fixed reduction order, no atomics). | tools/fit_k1.py |
| 13 | Low (evidence) | **Receipt gaps on provenance and shape coverage.** The server-arm binary is chosen through the `LLAMA_BIN` env var, which is not recorded in meta.json; provenance comes only from k1srv*.sh and the chain logs, which are consistent. There is no patched dispatch trace showing table=9/vdr=8/nwarps=2 in a real model run, only the operator deltas. The n=2..8 "no regression" claim holds for the measured shapes (all v8/base ratios are 0.998-1.24). But the actual Qwen3.5 Q5_K shapes (k=4096, m=4096/8192) and the 12288/248320 shapes were never measured at n=2/3. | runs/k1-opn-*.out |
| 14 | Info | **Doc mismatches.** The task brief says vecdotq.cuh was changed; it was not (mmvq.cu and tests only, as the README says). FINAL-REPORT.md contains no K1 results, so the K1 numbers live only in LEDGER.md rows 19-50. The patch file is byte-identical to `git diff 5bacb7c a96dce0`. The worktree is clean, mmvq.cu (03:04) predates mmvq.cu.o (03:05), and the sha256 of v8-bin libggml-cuda equals both build/bin and ~/llama-k1-20261123 (8643fc59...). So the measured binary is the committed source. | - |

## Evidence cross-check (claim vs raw value)

| Claim (LEDGER / README) | Raw value (my extraction) | OK? |
|---|---|---|
| ABBA v8 tg d0 29.83/38.56/38.55/29.82 (+29.3%) | A1 29.83+-0.03, B1 38.56+-0.01, B2 38.55+-0.01, A2 29.82+-0.04 → +29.27% | yes |
| ABBA v8 tg d8192 +28.2% | 28.44 / 36.46 / 36.48 / 28.44 → +28.23% | yes |
| pp unchanged | pp512 d0 536.8-537.3 in all arms; d8192 A 425.3/424.8, B 419.4+-13.7/424.6 | yes (B1 d8192 is noisy, within its stddev) |
| v2/v6/v7 ABBA | v2 +24.1%, v6 +29.1%, v7 +29.2%; A arms stable 29.82-29.86 across all four sessions | yes |
| Arms | A = ~/llama-prod (5bacb7c, same CMake: 61-real, Release, graphs on, gcc-12). B = ~/p104-exp/vN-bin | yes: like-for-like build config |
| Thermal "start 44-45C, max 55-56C, no trip" | meta start 44/45/45/45, max 55/55/55/56, trip None, rc 0. All 72 k1 traces: no hw_thermal / sw_thermal / hw_slowdown flag. Max over all k1 runs 61 C | yes. `sw_power_cap` is Active in every arm (the 140 W cap) |
| SM avg A 1690/1685, B 1767/1764 | identical (samples with util>=50, whole run) | yes |
| tok/s per MHz | **During the tg phases only:** A at about 1652 MHz, B at about 1803 MHz (+9%). Power is equal (about 137-139 W, at the cap). Power-cap samples: A 19-25 of about 40, B 8-11 of about 31. Mem clock is 5005 in all arms. tok/s per GHz: d0 A 18.06, B 21.40 (+18.5%); d8192 A 17.10, B 20.22 (+18.2%) | see note |
| Op GB/s Q4_K +44..+57%, Q5_K +27..+49%, Q6_K +13..+18% | recomputed from us/run against mean(base-2, base-3): Q4_K 1.442-1.567, Q5_K 1.269-1.491, Q6_K 1.128-1.179. Base repeats agree within 1%. E.g. Q4_K 12288x4096 167.1 us → 109.45 us (170 → 259 GB/s) | yes |
| n=4..8 equal to base within 0.5%; n=2/3 up to +24% | ratios at n=4..8 are 0.998-1.004; n=2 1.003-1.237; n=3 0.999-1.225 | yes, measured shapes only (finding 13) |
| Server single stream +28.7% | tg 29.25 / 37.68 / 37.61 / 29.26 → +28.7% | yes |
| Server 2 streams +2.6% | 25.98, 25.98 / 26.67, 26.67 / 26.67, 26.67 / 26.00, 26.00 → +2.6%. The wall-time claim was withdrawn: polling quantises it to 2 s | yes |
| KLD ub1 0.0006 at noise level | 0.000604 +- 0.000155 against in-fixture floor 0.000435 +- 0.000029. Max 0.395 against floor 0.067. Same-top 98.43 +- 0.25 against 98.71 +- 0.22 | mostly (finding 11) |
| KLD ub512 0.000000 | weak gate: ub 512 goes through MMQ, as the builder notes | n/a |
| Test pass 562/562 (v8), 2724/2724 all-type MUL_MAT (v6) | "562/562 tests passed" in k1-test-v8-K.out; 2724 is a v6 run. v7/v8 changed only Q4_K/Q5_K selection, so this is acceptable | yes |
| INVALID-overlap receipts excluded | Neither INVALID-* receipt is cited by any claim. The INVALID kldub1 .out is 0 bytes and has no meta. Clean reruns: k1-op-v4a (02:06) and k1-kldub1-floor-fa0 (01:38Z). Interval check over all 151 guarded meta.json files found 0 overlaps. The v8 chain waits for DONEV7 | yes |

**Note on the clock confound.** The higher B clock is endogenous, not an external confound. Both arms sit at the same 140 W power cap with the same temperatures and the same memory clock. The baseline kernel just spends more power per clock (more load and unpack instructions), so the governor clocks it lower. The +29% is therefore the real deployable gain on this card at this cap. The clock-normalised lower bound is about +18%, which would apply if decode were SM-clock-bound. The truth lies between the two, because MMVQ on GP104 is latency- and memory-bound. Changing the cap or the cooling could move the gain inside that range. The operator gains (+44-57% Q4_K) are far larger than the 6-7% clock gap seen in the operator runs, so the kernel improvement is genuine.

## Conditions: tests the GPU owner should run (in this order, about 60-75 min GPU)

Environment for every test:
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-uuid>
```
- Base binaries: `B0=~/p104-exp/base-bin`.
- Patched binaries: `B8=~/p104-exp/v8-bin`.
- Shortcuts: `M=~/models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf`, `W=~/p104-bench/data/wiki.test.raw`, `G="python3 -I ~/p104-bench/gguard.py"`.
- Each test needs `LD_LIBRARY_PATH` set to the binary's own directory.

**T1. Numerics of the 2..3-column path (v8 change).** For ub in 2, 3 and 4 (4 is a control on the unchanged path):
```
for U in 2 3 4; do
  LD_LIBRARY_PATH=$B0 $G k1-kldub$U-base -- $B0/llama-perplexity -m $M -f $W -ngl 99 -c 1024 -b 1024 -ub $U --chunks 5 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-ub$U-base.logits
  LD_LIBRARY_PATH=$B8 $G k1-kldub$U-v8 -- $B8/llama-perplexity -m $M -f $W -ngl 99 -c 1024 -b 1024 -ub $U --chunks 5 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-ub$U-base.logits --kl-divergence
done
```
- Pass: ub2 and ub3 mean KLD within about 1.5x the ub4 control and the ub1 floor (about 0.00045), with no token above 0.1.
- Each logits file is about 1.3 GB; delete them afterwards.
- `-ub` controls tokens per ubatch, so this hits MMVQ with 2 and 3 columns. Confirm with `GGML_P104_MMVQ_LOG=1`; note the log prints only 1-column launches.

**T2. Resolve the ub-1 outlier with a larger sample (20 chunks).** Generate a 20-chunk base, then run floor (base with fa off) and v8 against it:
```
LD_LIBRARY_PATH=$B0 $G k1-kldub1x20-base -- $B0/llama-perplexity -m $M -f $W -ngl 99 -c 1024 -b 1024 -ub 1 --chunks 20 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-ub1x20-base.logits
LD_LIBRARY_PATH=$B0 $G k1-kldub1x20-floor -- $B0/llama-perplexity -m $M -f $W -ngl 99 -c 1024 -b 1024 -ub 1 --chunks 20 -t 4 -fa off -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-ub1x20-base.logits --kl-divergence
LD_LIBRARY_PATH=$B8 $G k1-kldub1x20-v8 -- $B8/llama-perplexity -m $M -f $W -ngl 99 -c 1024 -b 1024 -ub 1 --chunks 20 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-ub1x20-base.logits --kl-divergence
```
- Also add a base-vs-base determinism run: rerun the first base command with `--kl-divergence`. Expect KLD exactly 0.
- Pass:
  - v8 mean KLD minus floor < 2 combined standard errors.
  - v8 99.9% KLD and maximum KLD are comparable to the floor's.
  - same-top within 0.5 pp.
- Cost: about 4x 3 min per run.

**T3. Second production candidate, Qwen3-8B Q4_K_M (non-recurrent, so the cleanest numerics signal).**
```
Q=~/models/Qwen3-8B-Q4_K_M.gguf
LD_LIBRARY_PATH=$B0 $G k1-q3-kldub1-base -- $B0/llama-perplexity -m $Q -f $W -ngl 99 -c 1024 -b 1024 -ub 1 --chunks 5 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-q3-ub1-base.logits
LD_LIBRARY_PATH=$B8 $G k1-q3-kldub1-v8 -- $B8/llama-perplexity -m $Q -f $W -ngl 99 -c 1024 -b 1024 -ub 1 --chunks 5 -t 4 -fa on -ctk f16 -ctv f16 --kl-divergence-base ~/p104-bench/data/k1-q3-ub1-base.logits --kl-divergence
cd ~/p104-bench && sed -e "s#qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf#../Qwen3-8B-Q4_K_M.gguf#" k1abba.sh > /tmp/k1abba-q3.sh   # or pass the model path if k1abba.sh supports it
bash /tmp/k1abba-q3.sh k1-abba-q3-v8 $B8
```
- Pass: KLD at its own fa-toggle floor; tg gain > 0 at d0 and d8192; pp unchanged.
- Also run a GGML_P104_MMVQ_LOG=1 dispatch on Qwen3-8B with v8. It should show table=9 vdr=8 nwarps=2 for q4_K/q5_K.

**T4. Out-of-bounds and odd rows under compute-sanitizer (small shapes, a few minutes).**
```
LD_LIBRARY_PATH=$B8 compute-sanitizer --tool memcheck $B8/test-backend-ops test -b CUDA0 -o MUL_MAT -p "type_a=q[456]_K.*m=(1023|1000|4100|3000)" 2>&1 | tee ~/p104-bench/runs/k1-sanitize-v8.out
LD_LIBRARY_PATH=$B8 compute-sanitizer --tool memcheck $B8/test-backend-ops test -b CUDA0 -o MUL_MAT_ID -p "q[45]_K" 2>&1 | tee -a ~/p104-bench/runs/k1-sanitize-v8.out
```
- Pass: "ERROR SUMMARY: 0 errors".
- memcheck only flags reads outside allocations, so this is necessary but not sufficient. The real fix for finding 4 is the row clamp.

**T5. Gemma q4_0 is untouched (sanity, about 4 min).** Run the ABBA script with `-m ~/models/experimental/gemma4-12b-qat/gemma-4-12b-it-qat-q4_0.gguf -d 0` for A=llama-prod against B=v8. Expect tg and pp equal within 0.5%, since the q4_0 kernels are unchanged.

**T6. Production-like soak and determinism (30 min).**
- Run llama-server from ~/llama-k1-20261123 with the production flags (`-np 2 -c 24576 -fa on -ctk f16 -ctv f16`).
- Run fit_k1.py in a loop (FITSTREAMS=2, then 1) for 30 minutes.
- Change fit_k1.py to also save each completion's `content` (copy the script; do not edit the receipt tool in place).
- Pass:
  - no CUDA errors in server.log;
  - temperature max < 75 C;
  - identical temp-0 completions across repeats for the same slot layout;
  - per-stream tg stable within 1%.

**T7 (optional, perf coverage).**
```
$B0/test-backend-ops perf -o MUL_MAT -p "type_a=q[456]_K,type_b=f32,m=(12288|4096|8192|1024|248320),n=[1-3],k=(4096|12288),"
$B8/test-backend-ops perf -o MUL_MAT -p "type_a=q[456]_K,type_b=f32,m=(12288|4096|8192|1024|248320),n=[1-3],k=(4096|12288),"
```
- Same commands with k=(2048|2560|3072) to cover finding 8.
- This covers the actual Qwen3.5 shapes at 2 and 3 columns.

**Before upstreaming (not needed for local production):**
- the row clamp (finding 4);
- the alignment fallback (finding 5);
- remove the trace hook and the `P104_*` macros;
- benchmark on GP102/GP106/GP107 (finding 6).

Per llama.cpp AGENTS.md, a human must own and be able to defend the change.
