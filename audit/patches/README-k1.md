# K1 patch: Pascal (sm_61) K-quant MMVQ decode GEMV

Base: llama.cpp 5bacb7c. Branch `p104-exp` in `~/p104-exp/wt`. Final commit: a96dce0 (v8).

Files:
- `k1-mmvq-pascal-kquant.patch` - single diff 5bacb7c..final (apply with `git apply` on a 5bacb7c checkout).
- `k1-series/` - the per-variant commit series 5bacb7c..a96dce0 (git format-patch), including the rejected Q6_K
  variants (commit 0004 adds them, 0005 removes them again; kept only as a record). `k1-series-old/` (up to v6) and
  `k1-series-v7/` are superseded exports.

## What it changes (ggml/src/ggml-cuda/mmvq.cu only, plus test cases)

1. New MMVQ parameter table `MMVQ_PARAMETERS_PASCAL`, selected for NVIDIA cc 6.1-6.x (dp4a, pre-Volta), both in
   the device-side constexpr selector and the host-side selector. All types not listed below fall back to the GENERIC
   table, so behaviour for them is unchanged.
2. Q4_K and Q5_K, Pascal only: a "wide" vec_dot (`vec_dot_q4_K_q8_1_wide`, `vec_dot_q5_K_q8_1_wide`, vdr 8). Each
   thread processes one whole sub-block pair of a super-block: 32 contiguous quant bytes loaded with two 16-byte
   loads (LDG.128) instead of two 4-byte loads 16 bytes apart (Q5_K additionally loads the 32 qh bytes with two
   16-byte loads). Scale/min unpacking is done once per 64 weights instead of once per 32. The min term still uses an
   exact dp4a sum of the q8_1 quants (the inexact `ds.y` shortcut was measured and gave no speed, so it is off;
   macro `P104_EXACT_SUM`).
3. The wide vec_dot is used for ncols_dst <= 3 (`P104_WIDE_MAX_NCOLS`). With 4..8 columns it regressed some shapes
   by up to 28% (v6), so upstream vdr 2 is kept there; measured n = 4..8 timings equal baseline within 0.5%, n = 2, 3
   are equal or faster (up to +24%) (runs/k1-opn-*). n = 2 is what llama-server -np 2 runs when both slots decode.
4. Pascal launch shape for ncols_dst == 1: Q4_K/Q5_K nwarps 2 and 4 rows per block for ncols_dst == 1 (64-thread blocks); Q6_K keeps the
   upstream vec_dot (vdr 2) with nwarps 4 and 4 rows per block. More rows per block = more independent weight loads
   in flight per thread, which is what the latency-bound sm_61 kernel needed.
5. Env-gated dispatch trace (`GGML_P104_MMVQ_LOG=1` prints one line per mat-vec launch: type, k, m, table, vdr,
   nwarps, grid). Off by default; cost is one cached getenv. Can be dropped for upstreaming.
6. tests/test-backend-ops.cpp: perf cases for the Qwen3.5-9B decode shapes (Q4_K/Q5_K/Q6_K, n = 1) and a fused
   gate/up Q4_K case. Test-only.

Not changed: MMQ (prompt processing), MoE multi-token kernel, any FP16 path, ncols_dst > 3, any other architecture.
Numerics: per-row float summation order changes for Q4_K/Q5_K (same integer dp4a math, exact min-term sum), so
logits are not bit-identical; decode-path KLD vs unpatched is 0.0006 (noise level, see LEDGER k1-kldub1-*).

## Build (experimental build dir, same config as the measured one)

```
cd ~/p104-exp/wt && git checkout p104-exp
cmake -S . -B build-k1 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=gcc-12 -DCMAKE_CUDA_HOST_COMPILER=gcc-12 \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-lstdc++ -lm" -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=ON
nice -n 19 cmake --build build-k1 -j3 --target llama-server llama-bench llama-perplexity test-backend-ops
```
(`build/` in the worktree is already configured exactly like this and was used for every measurement:
`cd ~/p104-exp/wt/build && nice -n 19 make -j3 llama-server llama-bench`. The fresh `build-k1` line above mirrors its
CMakeCache but was not itself run, a full CUDA build takes long on 4 cores.)
The measured binaries are kept in `~/p104-exp/v8-bin/` (libggml-cuda.so sha256 in the final report).

## Deploy (human / orchestrator action, not done by K1)

Copy `build-k1/bin/*` (llama-server + all lib*.so*) into a new dated directory, e.g. `~/llama-k1-YYYYMMDD/`, record
sha256 sums, and point the service at it with `LD_LIBRARY_PATH` set to that directory, in a stop window. Do not
rebuild in `~/llama.cpp` or overwrite `~/llama-prod`. Rollback = point the unit back at `~/llama-prod`.

## Update 2026-11-23 05:20: v10 (6768e5b) supersedes v8 for deployment

`k1-v10-mmvq-pascal-kquant.patch` = 5bacb7c..6768e5b (series in `k1-series-v10/`). Changes over v8 (review REVIEW-K1.md):
- No read past the last matrix row: with several rows per block on Pascal, the last block is shifted back to end at
  the last row (rows it shares with the previous block are recomputed bit-identically and written twice with the same
  value). A first attempt (v9: per-row clamp + in-kernel alignment fallback) made 2..3-column launches up to 2x slower
  and was rejected (LEDGER k2).
- 16-byte alignment of the quantized weights is asserted on the host for the wide path (unreachable on device buffers).
- small_k threshold back to upstream (K < 2048); dispatch trace compiled out unless GGML_P104_MMVQ_TRACE is defined;
  explicit __ldg loads; extra perf cases (n = 1..3, K = 2048..3072).
- Numerics bit-identical to v8 (KLD at ub 1/2/3 identical), llama-bench tg d0 v10 vs v8 -0.23% (ABBA).
Build: `~/p104-exp/wt/build-v9` (relocated copy of build/, same CMakeCache), binaries in `~/p104-exp/v10-bin/`.
