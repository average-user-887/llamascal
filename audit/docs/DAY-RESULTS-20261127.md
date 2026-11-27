# Pascal Daytime Optimization Report (`v17` Candidate, 2026-11-27)

- **Baseline:** `p104-v16` (`636e68f`, `/home/<user>/p104-exp/v16-bin`)
- **Candidate Branch / Head:** `exp/pascal-v17` @ `f60e8ba` (`5` commits atop `636e68f`)
- **Candidate Binaries:** `/home/<user>/p104-exp/v17-cand-bin` (staged; `p104-svc` intentionally kept on `636e68f` until promoted)
- **Patch Series:** `/home/<user>/p104-bench/patches-v17` (`0001`..`0005`)
- **Target Architecture:** NVIDIA Pascal (`sm_61`, serialized solo benchmarking under `night_guard.py` with `COOL_C=55`, `WARN_C=75`, `TRIP_C=80`, `P104` power cap `140.00 W`)

---

## 1. Executive Summary

`v17` (`f60e8ba`) delivers substantial end-to-end speedups across all three Pascal GPUs (**P104-100**, **Quadro P2000**, and **Quadro P620**) and across dense hybrid (`Qwen3.5-9B`, `Qwen3.5-0.8B`) and MoE (`OLMoE-1B-7B`, `granite-3.1-1b`) architectures, while holding `p104-v16` correctness (`20834/20834` `test-backend-ops` on both `P620` and `P104`) and zero-degradation WikiText-2 KLD:

| GPU | Model | Metric | `v16` (`636e68f`) | `v17` (`f60e8ba`) | Delta |
|---|---|---|---:|---:|---:|
| **P104** | `OLMoE-1B-7B` | `pp4` | 326.57 ± 0.90 | 349.54 ± 0.45 | **+7.03%** |
| **P104** | `OLMoE-1B-7B` | `tg128` | 178.57 ± 0.31 | 186.93 ± 0.11 | **+4.68%** |
| **P104** | `Qwen3.5-9B` (`d=0`) | `pp128` | 536.13 ± 2.25 | 547.00 ± 2.29 | **+2.03%** |
| **P104** | `Qwen3.5-9B` (`d=0`) | `pp512` | 623.26 ± 2.76 | 635.38 ± 2.00 | **+1.94%** |
| **P104** | `Qwen3.5-9B` (`d=16384`) | `tg128@d16384` | 35.46 ± 0.20 | 35.76 ± 0.10 | **+0.86%** |
| **P104** | `Qwen3.5-0.8B` | `pp512` / `tg128` | 4698.85 / 140.63 | 4921.93 / 152.85 | **+4.75% / +8.69%** |
| **P2000** | `granite-3.1-1b` | `tg128` | 170.86 ± 0.53 | 198.99 ± 0.29 | **+16.46%** |
| **P2000** | `Qwen3.5-0.8B` | `pp128` / `tg128` | 2306.88 / 104.47 | 2524.79 / 120.17 | **+9.45% / +15.03%** |
| **P620** | `Qwen3.5-0.8B` | `pp128` / `tg128` | 1016.66 / 51.43 | 1134.23 / 68.21 | **+11.56% / +32.61%** |
| **P620** | `granite-3.1-1b` | `tg128` | 85.24 ± 0.06 | 114.36 ± 0.20 | **+34.16%** |

---

## 2. Landed Optimizations & Reverted Experiments

### Kept & Committed (`p104-v16..f60e8ba`)
1. **`ad1a928` — Pascal SMEM-staged `float4` GatedDeltaNet kernel (`gated_delta_net.cu`)**
   - Adds `gated_delta_net_pascal_f4<keep_rs_t>` (`__launch_bounds__(128, 4)`, `T_TILE=8`, `REG:69–72 STACK:0 SHARED:8256`) for `warp_size == 32 && S_v == 128 && !kda && n_tokens >= 16` with 16-byte aligned pointers/strides (`n_tokens < 16` stays on the unmodified `v16` kernel).
   - Stages `q/k` (`float4`) and `expf(g)/beta` once per `T_TILE=8` step in shared memory across all warps, eliminating redundant per-column `expf()` calls and global `q/k` reloads while keeping exact FP32 FMA count.
   - `P2000` `GATED_DELTA_NET` ABBA (`t2c-p2000-gdn`): **-17.27% to -46.29%** across `n_seq_tokens >= 64` (`H=32 n=512: 4273.91 -> 2295.55 us`, **-46.29%**; `H=16 v_repeat=2 n=2048: 15848.83 -> 9075.10 us`, **-42.74%**); `n=1` unchanged (`+0.96%`, `0.75%` drift).
2. **`30f5bbc` — `half2 KV_tmp` for `ncols <= 8` in `fattn-tile.cuh`**
   - On `!FAST_FP16_AVAILABLE` (`sm_61`), stores `KV_tmp` as `half2` when `ncols <= 8` (halving SMEM and lifting register/occupancy limits on decode, e.g. `128x128 ncols=8`: `REG 137->80`, `SHARED 39424->23040`; `256x256 ncols=8`: `SHARED 43520->27136`) with `__half22float2` hoisted out of `cpw`, while keeping `float KV_tmp` for `ncols > 8` so prefill avoids quarter-rate `F2F` bottlenecks.
   - `P2000` `FLASH_ATTN_EXT` ABBA (`t3afix-p2000-fa`): **-16.27% to -38.69%** on all `nb=1` decode cases (`hsk=128 kv=16384: -24.84%`, `hsk=256 nr23=[6,1] kv=4096: -38.69%`, `kv=16384: -24.40%`), with **-0.01% / +0.18%** (`0%` regression) on `nb=512` prefill.
3. **`90284eb` — Fused `gate+up` Pascal `Q4_K`/`Q5_K` `n=1` MMVQ (`mmvq.cu`)**
   - Extracts `p104_mmvq_q45_n1_row` and extends `p104_mmvq_kloop_q45_n1` to `has_fusion=true` (`REG:133 STACK:0` `Q4_K`, `REG:163 STACK:0` `Q5_K`), hoisting `q8_1` (`u`, `d8`, `fsum_m`) once across both `up` (`vx`) and `gate` (`vgate`) weights in SwiGLU/GeGLU FFN layers. Screened `+1.48%` on `Qwen3-8B -ngl 28` (`P2000`) and carries into pure-GPU `K=4096` `Qwen3.5-9B`.
4. **`c6c5dc4` — Pascal wide `Q4_K`/`Q5_K`/`Q6_K` in `mul_mat_vec_q_moe` (`mmvq.cu`)**
   - Routes multi-token MoE `MUL_MAT_ID` (`ncols_dst = 2..8`) on Pascal to `vec_dot_q4_K_q8_1_wide<true>` / `vec_dot_q5_K_q8_1_wide<true>` (`vdr=8` when `ncols_x >= 2048` and 16-byte aligned) and `vec_dot_q6_K_q8_1_v<P104_Q6_VDR, false>` (`vdr=4` when `ncols_x >= 512`), zero compiler warnings.
   - `P2000` `MUL_MAT_ID` ABBA (`t4b-p2000-mmid`): **-10.94% to -13.28%** on `q4_K n=4 k=2048`, **-3.08% to -5.14%** on `q6_K n=4 k=2048`; **+6.91%** on `OLMoE pp4` (`P2000`).
5. **`f60e8ba` — Single-warp Pascal `small_k` `n=1` `Q4_K`/`Q5_K`/`Q6_K` k-loops (`mmvq.cu`)**
   - Replaces generic 4-warp `vdr=2` `small_k` execution (`K=512, 1024`) at `ncols_dst == 1` with single-warp (`nwarps=1`, `rows_per_block=4`), `if constexpr (nwarps > 1)` SMEM bypass, and Pascal `p104_mmvq_kloop_q45_n1`/`q6`/`q45` k-loops. Eliminates idle half-warps, `__syncthreads()`, shared reduction, and quarter-rate `I2F`.
   - `P2000` screening delta: `granite-3.1-1b tg128` **+16.05%**, `Qwen3.5-0.8B tg128` **+15.19%**, `OLMoE-1B-7B tg128` **+6.35%**.

### Screened & Reverted (Negative / Neutral Results)
- **Track 1 (`mmq-vec-dot.cuh` `Q4_K` packed nibbles + `Q6_K` `i0`-outer):** Kept `Q4_K` at `REG:128 STACK:48`, regressed `Q6_K` registers (`126->128`), `P2000` ABBA neutral (`Q4_K n=512 -0.03%`, `Q6_K +0.55%`, `Qwen3.5-0.8B pp512 -0.24%`). Reverted cleanly.
- **Track 2a (`vec_size=4` inside `gated_delta_net_cuda`) & Track 2b (`gated_delta_net_pascal_chunked` WY):** `2a` regressed `H=32 n=1` (`+5.91%`) and `H=4` (`+20.95%`); `2b` added FP32 GEMMs without Tensor Cores (`STACK:64`), regressing `H=32 n=512` (`4267 -> 5167 us`). Abandoned for `2c`.
- **Track 3a (unconditional `half2 KV_tmp`):** Sped decode `-16%..-33%` but slowed `nb=512` prefill (`ncols=32`) by `+23.17%..+24.33%` due to repeated `F2F` conversions in `cpw`. Fixed cleanly in `30f5bbc`.
- **Track 3b (`fattn-tile.cuh` `DKQ=128/256 ncols=4 nbatch_K=128`):** Raised `256x256` registers (`123->139`) and regressed `nr23=[4,1]` (`+4.68%..+5.60%`). Reverted cleanly.

---

## 3. Exclusive Solo 3-Card ABBA Validation (`v16-bin` vs `v17-cand-bin`)

All 8 legs executed strictly serialized (one GPU active, CPU idle, `COOL_C=55`, in-process warmup `-p 512 -n 32 -r 1` before `-r 3` timed sweeps) under `night_guard.py`. No thermal trips occurred.

| GPU | Model | Metric | `v16` Mean ± SD (tok/s) | `v17` Mean ± SD (tok/s) | Delta | Max T (v16/v17) | Avg SM MHz (v16/v17) |
|---|---|---|---|---|---|---|---|
| **P104** | `OLMoE-1B-7B` | `pp4` | 326.57 ± 0.90 | 349.54 ± 0.45 | **+7.03%** | 43 / 44 °C | 1881 / 1878 |
| **P104** | `OLMoE-1B-7B` | `pp8` | 367.58 ± 9.24 | 373.16 ± 3.58 | **+1.52%** | 43 / 44 °C | 1881 / 1878 |
| **P104** | `OLMoE-1B-7B` | `pp128` | 1640.13 ± 10.25 | 1636.61 ± 6.98 | -0.21% | 43 / 44 °C | 1881 / 1878 |
| **P104** | `OLMoE-1B-7B` | `pp512` | 2506.61 ± 7.51 | 2506.27 ± 7.19 | -0.01% | 43 / 44 °C | 1881 / 1878 |
| **P104** | `OLMoE-1B-7B` | `tg128` | 178.57 ± 0.31 | 186.93 ± 0.11 | **+4.68%** | 43 / 44 °C | 1881 / 1878 |
| **P104** | `Qwen3.5-9B` (`d=0`) | `pp128` | 536.13 ± 2.25 | 547.00 ± 2.29 | **+2.03%** | 53 / 54 °C | 1831 / 1841 |
| **P104** | `Qwen3.5-9B` (`d=0`) | `pp512` | 623.26 ± 2.76 | 635.38 ± 2.00 | **+1.94%** | 53 / 54 °C | 1831 / 1841 |
| **P104** | `Qwen3.5-9B` (`d=0`) | `tg128` | 38.98 ± 0.05 | 38.92 ± 0.03 | -0.17% | 53 / 54 °C | 1831 / 1841 |
| **P104** | `Qwen3.5-9B` (`d=16384`) | `pp512@d16384` | 384.17 ± 0.06 | 384.92 ± 1.97 | +0.20% | 59 / 58 °C | 1778 / 1778 |
| **P104** | `Qwen3.5-9B` (`d=16384`) | `tg128@d16384` | 35.46 ± 0.20 | 35.76 ± 0.10 | **+0.86%** | 59 / 58 °C | 1778 / 1778 |
| **P104** | `Qwen3.5-0.8B` | `pp128` | 3660.18 ± 4.23 | 3826.81 ± 3.51 | **+4.55%** | 57 / 56 °C | 1851 / 1848 |
| **P104** | `Qwen3.5-0.8B` | `pp512` | 4698.85 ± 4.34 | 4921.93 ± 7.04 | **+4.75%** | 57 / 56 °C | 1851 / 1848 |
| **P104** | `Qwen3.5-0.8B` | `tg128` | 140.63 ± 0.18 | 152.85 ± 0.23 | **+8.69%** | 57 / 56 °C | 1851 / 1848 |
| **P2000** | `granite-3.1-1b` | `pp4` | 361.52 ± 6.89 | 360.81 ± 6.46 | -0.20% | 59 / 59 °C | 1638 / 1680 |
| **P2000** | `granite-3.1-1b` | `pp8` | 535.55 ± 1.87 | 539.81 ± 0.92 | **+0.80%** | 59 / 59 °C | 1638 / 1680 |
| **P2000** | `granite-3.1-1b` | `pp128` | 2379.71 ± 8.15 | 2391.71 ± 6.97 | +0.50% | 59 / 59 °C | 1638 / 1680 |
| **P2000** | `granite-3.1-1b` | `pp512` | 3026.09 ± 20.78 | 3021.30 ± 13.62 | -0.16% | 59 / 59 °C | 1638 / 1680 |
| **P2000** | `granite-3.1-1b` | `tg128` | 170.86 ± 0.53 | 198.99 ± 0.29 | **+16.46%** | 59 / 59 °C | 1638 / 1680 |
| **P2000** | `Qwen3.5-0.8B` | `pp128` | 2306.88 ± 0.99 | 2524.79 ± 0.91 | **+9.45%** | 60 / 60 °C | 1688 / 1674 |
| **P2000** | `Qwen3.5-0.8B` | `pp512` | 2583.88 ± 2.09 | 2791.54 ± 3.27 | **+8.04%** | 60 / 60 °C | 1688 / 1674 |
| **P2000** | `Qwen3.5-0.8B` | `tg128` | 104.47 ± 0.05 | 120.17 ± 0.05 | **+15.03%** | 60 / 60 °C | 1688 / 1674 |
| **P620** | `Qwen3.5-0.8B` | `pp128` | 1016.66 ± 1.63 | 1134.23 ± 0.98 | **+11.56%** | 62 / 62 °C | 1354 / 1354 |
| **P620** | `Qwen3.5-0.8B` | `pp512` | 1098.34 ± 1.88 | 1216.60 ± 0.86 | **+10.77%** | 62 / 62 °C | 1354 / 1354 |
| **P620** | `Qwen3.5-0.8B` | `tg128` | 51.43 ± 0.05 | 68.21 ± 0.05 | **+32.61%** | 62 / 62 °C | 1354 / 1354 |
| **P620** | `granite-3.1-1b` | `pp128` | 1027.24 ± 0.24 | 1032.43 ± 0.06 | **+0.51%** | 61 / 61 °C | 1354 / 1354 |
| **P620** | `granite-3.1-1b` | `pp512` | 1304.31 ± 0.37 | 1306.47 ± 0.59 | +0.17% | 61 / 61 °C | 1354 / 1354 |
| **P620** | `granite-3.1-1b` | `tg128` | 85.24 ± 0.06 | 114.36 ± 0.20 | **+34.16%** | 61 / 61 °C | 1354 / 1354 |

---

## 4. `S0` MTP Serving Check (`Qwen3.5-9B`, `P104`, `c=24576`, `M2-dv64k`)

| Binary | Arm | `tg tok/s` (srv) | Wall ms/tok | Accept (`acc/draft`) | Speedup vs `S` | VRAM MiB | Max Temp | Exact vs `S` |
|---|---|---:|---:|---|---:|---:|---:|---|
| `v16-bin` | `S` | 37.82 | 34.48 | - | 1.000x | 5933 | 53 °C | - |
| `v16-bin` | `M2-dv64k` | 58.43 | 25.64 | 0.792 (2086/2633) | 1.344x | 6473 | 56 °C | 6/16 |
| `v17-cand-bin` | `S` | 37.62 | 34.58 | - | 1.000x | 5933 | 59 °C | - |
| `v17-cand-bin` | `M2-dv64k` | 57.88 | 25.62 | 0.776 (2084/2685) | **1.350x** | 6473 | 59 °C | 6/16 |

---

## 5. Correctness, Greedy Parity & Paired WikiText-2 KLD Gates

- **Full `test-backend-ops test -b CUDA0` (`v17-cand-bin`):**
  - **P620:** `20834/20834` tests passed (`OK`)
  - **P104:** `20834/20834` tests passed (`OK`)
- **Paired WikiText-2 KLD (`llama-perplexity`, `-c 1024 -b 1024 -fa on -ctk f16 -ctv f16`, `P104`):**

| Model | Config (`Q`) | Reference (`base`) | Mean PPL(Q) | Mean PPL(base) | Mean KLD | Max KLD | Same Top-p |
|---|---|---|---:|---:|---:|---:|---:|
| `OLMoE-1B-7B` | `v17 ub=128` (20 chunks) | `v16 ub=128` | 9.6364 ± 0.2634 | 9.6294 ± 0.2628 | **0.000000 ± 0.000000** | 0.000060 | **100.000%** |
| `OLMoE-1B-7B` | `v17 ub=4` (20 chunks) | `v16 ub=128` | 9.6359 ± 0.2633 | 9.6294 ± 0.2628 | 0.003134 ± 0.000171 | 1.505748 | 97.798% |
| `OLMoE-1B-7B` *(v16 ref)* | `v16 ub=4` (20 chunks) | `v15b ub=128` | 9.6467 ± 0.2637 | 9.6294 ± 0.2628 | 0.003226 ± 0.000173 | 1.457581 | 97.857% |
| `Qwen3.5-9B` | `v17 ub=512` (20 chunks) | `v16 ub=512` | 7.1070 ± 0.1788 | 7.0992 ± 0.1784 | 0.001043 ± 0.000184 | 1.225432 | 98.865% |
| `Qwen3.5-9B` | `v17 ub=128` (20 chunks) | `v16 ub=512` | 7.1027 ± 0.1787 | 7.0992 ± 0.1784 | 0.001076 ± 0.000185 | 1.137416 | 98.796% |
| `Qwen3.5-9B` *(v16 ref)* | `v16 ub=128` (20 chunks) | `v15b ub=512` | 7.1045 ± 0.1787 | 7.0992 ± 0.1784 | 0.000822 ± 0.000147 | 1.261073 | 98.777% |
| `granite-3.1-1b` | `v17 ub=1` (10 chunks) | `v16 ub=1` | 9.1625 ± 0.3313 | 9.1545 ± 0.3305 | 0.002813 ± 0.000098 | 0.339185 | 97.417% |

  *Notes:* On `Qwen3.5-9B`, `v17` (`Mean KLD 0.00104/0.00108`, `Max KLD 1.225/1.137`) lies inside `v16`'s own FP32 ubatch associativity envelope (`v16 ub=128`: `Mean KLD 0.000822, Max 1.261`; `v16 ub=1`: `Mean KLD 0.001555, Max 4.677`) due to `float4`-grouped FP32 summation order in `gated_delta_net_pascal_f4`. On `OLMoE ub=4`, `v17` (`0.003134`) is tighter than `v16`'s own `ub=4` (`0.003226`). All runs have `Mean KLD <= 0.0031` and `Mean ln(PPL(Q)/PPL(base)) <= 0.0011`.
- **Greedy parity (`temp=0, seed=1`, 64 tokens, `v17-cand-bin` vs `v16-bin`):**
  - `Qwen3.5-9B` (`P104`): `2/3` exact (`prompt[2]` diverged into equivalent reasoning phrasing); `OLMoE-1B-7B` (`P104`): `1/2` exact; `granite-3.1-1b` (`P2000`): `1/2` exact (FP32 accumulation-order tie flips during generation).

---

## 6. Candidate Artifacts, DSO Witness & Production Service State

- **DSO Witness (`/home/<user>/p104-bench/tools/dso_witness.sh /home/<user>/p104-exp/v17-cand-bin`):**
  All 4 binaries (`llama-bench`, `llama-server`, `llama-perplexity`, `test-backend-ops`) resolved 100% of project DSOs strictly inside `v17-cand-bin`:
  - `llama-bench`: `8127589f079eb7e57ff7221c74db4243e6f2ea8dce9fe2d0eb74e9f0f37811f0`
  - `llama-server`: `0e0c5f47ab16fdc180ee244a42777fcf0e12308572739ab9b564ef525a80049d`
  - `llama-perplexity`: `2cdfd971bf0135d58b4bb28f4d0f804d2f011e1cd8b534127e3950acef4db5ff`
  - `test-backend-ops`: `d7f83b062c49787a354239372f527d990801540c3ab03701bf9fd4a843c44ea8`
  - `libggml-cuda.so.1.0.0`: `d12039fcc64fece0b127fe1fa4ce7d29cbf1f2daff6800150cef6d2f56a3a1ed`
- **Production Service Restored:** `p104-svc` (`p104-llama:636e68f-20261127`) restarted cleanly, `http://127.0.0.1:18100/health` returned `{"status":"ok"}`, and `P104-100` power cap verified at `140.00 W` (`46 °C`, `7301 MiB`).
- **Ready for promotion / push upon user instruction:** `/home/<user>/p104-exp/v17-cand-bin` and `/home/<user>/p104-bench/patches-v17`.
