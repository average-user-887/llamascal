# P104-100 optimisation: final report (GPU worker, 2026-11-22/23)
Host z4-p104, 140 W cap (unchanged), build 5bacb7c pinned (~/llama-prod), experimental build (same commit, only used for llama-perplexity and test-backend-ops). Ledger: ~/p104-bench/LEDGER.md. Receipts: ~/p104-bench/runs/ (tag.out, tag.trace.csv, tag.meta.json). No thermal trip occurred in any run (max 66 C, no hw_thermal/sw_thermal/hw_slowdown flag; sw_power_cap Active is the expected 140 W cap).

## 1. Verdict in one paragraph
Serving-side tuning is worth doing and is already mostly harvested: switching the model to Qwen3.5-9B Q4_K_M and the KV cache to f16 gives +13% decode at 8k depth (single stream), +18% aggregate with two streams at ~7k each, +55% prompt processing at 8k depth, with no quality loss (f16 is the reference). Every other "pure speed" knob (-ub, -b, -t, CUDA graphs) is flat. Short-context single-stream decode is ~10% lower than the Qwen3-8B baseline because the model is 13% larger. Further kernel work has a real but bounded ceiling: the Q4_K GEMV runs at ~174 GB/s versus ~290 GB/s practical peak (f16 mat-vec 292, ADD 286), so decode is ~60% of the memory roofline. A better Q4_K GEMV (q4_1-like efficiency is 239 GB/s on the same card) could plausibly deliver +15-25% tg; theoretical ceiling ~+35-40% (about 40 tok/s on Qwen3.5-9B at depth 0). That is worth a bounded Opus card (K1) but not an open-ended programme; K3 (attention) is solved by config, not kernels (see 5); launch/graph overhead (K5) is not a lever.

## 2. q1 results (Task 1) - llama-bench, fa1, q8_0 KV, ub512, t4, r3, 140 W
CPU-contended by a concurrent nice-19 compile (small effect; later reruns q7 agree within 1%).
| model | depth | pp512 tok/s | tg128 tok/s | start/end/max C | SM MHz avg/min | W avg/max |
|---|---|---|---|---|---|---|
| Qwen3-8B Q4_K_M | 0 | 468.8 +-19.8 (warm-up; 532 in q7) | 32.83 +-0.03 | 29/48/50 (whole run) | 1726/1594 | 116/176 |
| | 4096 | 360.3 +-0.3 | 28.65 +-0.02 | | | |
| | 8192 | 272.3 +-0.4 | 25.22 +-0.01 | | | |
| Qwen3.5-9B Q4_K_M | 0 | 426.0 +-38.7 (warm-up; 523-538 later) | 29.30 +-0.02 | 45/53/55 | 1692/1607 | 121/147 |
| | 4096 | 463.4 +-1.0 | 27.83 +-0.02 | | | |
| | 8192 | 421.2 +-0.9 | 26.94 +-0.02 | | | |
| Gemma-4-12B QAT Q4_0 | 0 | 406.3 +-1.0 | 22.85 +-0.05 | 45/52/54 | 1743/1607 | 114/166 |
| | 4096 | 291.2 +-1.8 | 20.49 +-0.02 | | | |
No OOM or load failures (Gemma only tested to depth 4096 by the q1 script; 6.5 GiB weights leave little KV room).
Judgement: Gemma-4-12B fails the >=25 tok/s gate at depth 0. Qwen3.5-9B is the best candidate: a newer, larger model whose decode barely degrades with context (hybrid layers: 29.3 -> 26.9 at 8k, vs 32.8 -> 25.2 for Qwen3-8B) and whose pp is far higher at depth. Qwen3-8B is the fastest at depth 0 only.

## 3. Flag sweep (Task 2), Qwen3.5-9B unless noted (tg tok/s, d0 / d8192)
| knob | result |
|---|---|
| -ub 128/256/512/1024 | tg 29.3-29.5 / 26.9 all; pp512 480/525/529/528 (ub128 -9%); keep 512 (pp2048: 520/524/521/514 for ub256/512/1024/2048) |
| -b 512/1024/2048/4096 | tg flat; pp512 is lower for b<2048 (431/454 vs 515-525), keep default 2048 |
| -t 2/4/6/8 | tg 29.47/29.40/29.32/29.24 (decode is GPU bound; 2 threads suffice) |
| CUDA graphs on/off, ABBA | 29.65/29.61/29.61/29.63 d0, 27.00/27.01/27.03/27.01 d8k: no effect |
| -fa 0 vs 1 (f16 KV) | d0 29.51 vs 29.75; d8192 22.97 vs 28.37: always fa1 (Qwen3-8B: 13.1 vs 28.3) |
| KV type f16/f16 | 29.50 / 28.07-28.37 (best) |
| KV q8_0/q8_0 | 29.35 / 26.81 |
| KV q4_0/q4_0 | 29.18 / 25.26; mixed types 25.5-26.7 at d8k |
| -ngl | model fully resident at 99; not varied (any offload would only slow decode over PCIe 1.0 x4) |
Qwen3-8B shows the same pattern, stronger: f16/f16 d8192 tg 28.5 vs q8/q8 25.0 vs mixed 19.4-22.3. But Qwen3-8B f16 KV does not fit -c 24576 -np 2 (only -c 16384 np2 fits, 6991 MiB peak).

## 4. Quality (Task 3): KLD vs f16-KV reference, wikitext-2 raw test split (downloaded from huggingface.co/datasets/ggml-org/ci wikitext-2-raw-v1.zip, sha256 of wiki.test.raw 173c87a5...), -c 1024, 20 chunks (~20k tokens), fa on; llama-perplexity from the experimental build.
| model | config | mean KLD | PPL ratio | same-top-p |
|---|---|---|---|---|
| Qwen3.5-9B | fa off, f16 (noise floor) | 0.00075 | 1.0010 | 98.97% |
| | q8_0/q8_0 | 0.00075 | 1.0014 | 98.77% |
| | k q8_0 / v q4_0 | 0.00238 | 1.0034 | 98.11% |
| | q4_0/q4_0 | 0.00490 | 1.0035 | 97.31% |
| Qwen3-8B | fa off, f16 | 0.00060 | 1.0079 | 98.94% |
| | q8_0/q8_0 | 0.00072 | 1.0077 | 98.88% |
| | k q8_0 / v q4_0 | 0.00388 | 1.0093 | 97.51% |
| | q4_0/q4_0 | 0.03195 | 1.0143 | 93.12% |
q8_0 KV is indistinguishable from the noise floor; q4_0 KV is not acceptable for Qwen3-8B. f16 KV (the deployed choice) is the reference itself. Caveat: short 1024-token contexts exercise KV quantisation less than 8k+ contexts; the file has wikitext tokenisation artefacts. Ranking is still reliable. Logits file (5 GB, Qwen3.5) kept in ~/p104-bench/data.

## 5. Profile attribution and K3 (Task 5)
Tooling: nsys 2022.4 recorded no CUDA events with the 580 driver (empty reports; also its importer is mis-installed, workaround QdstrmImporter run manually), and ncu fails with ERR_NVGPUCTRPERM (needs the NVreg_RestrictProfilingToAdminUsers=0 module option and root; human item). So no per-kernel nsys/ncu table and no measured launches/token. I used test-backend-ops perf (same commit, no kernel changes) instead:
| op (n=1 decode shape) | time | achieved GB/s |
|---|---|---|
| f16 mat-vec 4096x14336 | 402 us | 292 (practical peak, ADD 4096x512 gives 286) |
| Q4_K 4096x14336 / 14336x4096 / 4096x4096 | 190 / 195 / 59 us | 174 / 169 / 160 |
| Q6_K 4096x14336 | 268 us | 180 |
| q8_0 / q4_0 / q4_1 4096x14336 | 284 / 156 / 154 us | 220 / 212 / 239 |
| FLASH_ATTN_EXT f16, 8 KV heads GQA4, kv=8192 / 65536 | 154 / 1160 us | 217 / 231 |
End-to-end cross-check: Qwen3-8B decode reads 5.03 GB/token at 32.9 tok/s = 165 GB/s; Qwen3.5-9B 5.67 GB at 29.7 = 168 GB/s, consistent with the Q4_K figure. Real peak is ~290 GB/s, not 320.
K3 result: the handbook's "KV read at ~70 GB/s" is true only for the q8_0 (and q4) KV flash-attention path. With f16 KV the flash-attention decode kernel reaches ~220 GB/s, and the predicted attention time (36 layers x 154 us = 5.5 ms at 8k) matches the measured f16 slowdown (33.8 -> 28.5 tok/s = +5.6 ms). So the long-context loss was dequantisation cost in the quantised-KV kernel, and is fixed by f16 KV where memory allows. Remaining attention headroom with f16 is small (<= ~25% of 5.5 ms = ~4% tg at 8k). A faster q8_0-KV decode kernel would only matter for models that cannot afford f16 KV (Qwen3-8B at 2x24k).
Launch/gap (K5): graphs on/off ABBA is within 0.1% at d0 and d8k, so launch overhead and inter-kernel gaps are fully hidden; zero upside. Exact launches/token not measured (see tooling).

## 6. Speculative decoding (Task 4), Qwen3.5-9B, server, -c 8192 -np 1, temp 0, 256 tokens, ms per committed token wall-clock
| arm | copy-heavy | chat1/2/3 | code |
|---|---|---|---|
| serial (ABBA bookends) | 40.7-41.8 | 35.0-35.7 | 35.0-35.4 |
| ngram-simple | 13.8 (acc 240/240) | 35.0-37.0 (no drafts) | 37.4-37.6 (acc 29/160): +7% slower |
| ngram-map-k | 13.8 (240/240) | 34.9-35.9 | 37.8 (14/106): +7% slower |
| ngram-mod | 12.3-15.3 (229-251/..) | 35 on first pass; much faster on repeated prompts (it learns across requests) | 19-35 |
simple/map-k outputs were byte-identical to serial in all cases. ngram-mod outputs diverged from serial at near-tie tokens in 3 of 4 prompts (coherent, but not bit-exact). Gain is ~3x only for copy-heavy work (quoting, rewriting, editing documents); for normal chat/code it is zero to slightly negative. Recommendation: do not enable by default; enable ngram-map-k only if the workload is mostly copy/edit. No draft-model gguf is available locally, so none tested (a 0.6-4B draft needs ~0.5-2.5 GB; Qwen3.5-9B f16 at 24k leaves >2 GiB headroom, so it would fit).

## 7. Best deployable config and gains
Model ~/models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf. Verified to load and serve:
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-uuid> \
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 24576 -np 2 -fa on -ctk f16 -ctv f16 -ub 512 -t 4 --host ... --port ...
```
(-c 24576 = 12288 per slot; load takes ~31 s.) Memory: 5975 MiB peak with two ~8.4k-token prompts in flight (of 8109); it also fits -c 49152 -np 2 (24k/slot) at 6755 MiB, which is the recommended production setting if longer context is wanted. Per-slot context is the real constraint: the llama-server layout gives each slot c/np.
Counterbalanced comparison vs the q1 Qwen3-8B baseline (q8_0 KV), llama-bench, ABBA (A1 Z1 Z2 A2, repeat agreement <=0.3%):
| depth | pp baseline -> final | tg baseline -> final |
|---|---|---|
| 0 | 532 -> 530 (-0%) | 32.9 -> 29.8 (-9%) |
| 4096 | 358 -> 465 (+30%) | 28.6 -> 29.0 (+1.5%) |
| 8192 | 270 -> 420 (+56%) | 25.15 -> 28.4 (+13%) |
Server, two concurrent streams (ABBA, np2 -c 24576): short context 29.0 vs 25.8 tok/s per stream (-11%); ~7k context each 20.8 vs 24.6 per stream (+18%, aggregate 41.6 -> 49.2). Honest caveat: with two streams active at depth, the final config is just under the 25 tok/s per-stream target (24.6) and the baseline well under (20.8); single stream meets it everywhere (>=28).
Thermal at 140 W (card cap held): 11-min soak (q9-soak, 60 reps, d4096, tg256) start 45 C -> max 66 C, SM 1582-1735 MHz (avg 1642), 133 W avg, no throttle flags other than sw_power_cap (the cap itself). Every other run peaked at 50-66 C. Margin to the 85 C trip: ~19 C. (Earlier tests ran with a concurrent compile; thermal results are unaffected.)
Quality evidence: f16 KV is the reference; q8_0 KV (if memory forced it) costs ~0.00075 KLD = noise floor. The model change itself (Qwen3-8B -> Qwen3.5-9B) is a capability choice not measured here: I did not run task benchmarks (BFCL/IFEval); recommend the the lab orchestrator host harnesses before promoting.

## 8. Open issues / human items
1. ncu needs GPU counter permission (root: `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d, reboot/reload); nsys 2022.4 is too old for driver 580 (install a newer nsight-systems if per-kernel timelines are wanted).
2. Decide the production trade-off: Qwen3.5-9B is slower at short context than Qwen3-8B (-9% to -11%) but faster where long prompts matter; and its quality has not been measured on lab task suites.
3. The experimental build needed `-DCMAKE_CXX_STANDARD_LIBRARIES="-lstdc++ -lm"` because the tree was configured with gcc-12 as the C++ compiler (the earlier link errors were that, not code). Build dir: ~/p104-exp/wt/build (not production; ~/llama-prod untouched).
4. The q1 run overlapped a nice-19 compile and the q3/q5 runs overlapped brief compiles/imports; all key conclusions were re-measured in clean ABBA runs (q7).
5. Next kernel card if approved: Q4_K (and Q6_K) MMVQ decode GEMV efficiency (K1) with ceiling ~+15-25% realistic, +35-40% theoretical; gate on KLD and counterbalanced whole-model tg.
