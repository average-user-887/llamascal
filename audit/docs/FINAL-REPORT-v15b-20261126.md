# Final Pascal (`sm_61`) Optimization & Benchmark Report — `llamascal` (`v15b`)

**Date:** 2026-11-26  
**Repository:** `/home/<user>/p104-exp/wt` (branch `llamascal`, certified commit `c01d483`, 3 commits over `p104-v14` `585eb84`; Stock upstream `5bacb7c`)  
**Staged Binary Set:** `/home/<user>/p104-exp/v15b-bin` (exported patches in `/home/<user>/p104-bench/patches/llamascal-v15b/`)  
**Hardware Targets:** `P620` (2 GB), `P2000` (5 GB), `P104` (8 GB, `140.00 W` power limit preserved)  
**Production Service State:** `/home/<user>/p104-exp/llamascal-bin -> v14-bin` and Docker container `p104-svc` (`:18100`) left untouched pending promotion approval.

---

## 1. Executive Summary

Following the audit rules ingested from `llamascal/audit/pascal-next-20261126` (`LLAMASCAL_ORCHESTRATOR.md`, `PASCAL_MODELS.json`, `NEXT_EXPERIMENTS.md`), we unified Pascal (`sm_61`) dense and MoE development into one `llamascal` codebase, closed loaded-DSO provenance risks (`PROV-1`), pulled and SHA-256-verified the pinned `COMMON` and `P104-MoE` models, and established the first apples-to-apples exclusive solo baseline across **Stock (`5bacb7c`)**, **`v14` (`585eb84`)**, and **`v15b` (`c01d483`)** on all three GPUs (`P620`, `P2000`, `P104`).

Key outcomes:
1. **Cross-card baseline caught a hidden `v14` regression (`small_k` MMVQ bug):** Running Stock against `v14` on `COMMON` revealed that `v14` regressed `tg64` on `Qwen3.5-0.8B` and `granite-3.1-1b-a400m` by **`-9.8%` to `-24.0%`** across all three Pascal cards. CUDA graph disable testing (`NOGRAPH`) ruled out CUDA graphs as the cause; source inspection localized the bug to `MMVQ_PARAMETERS_PASCAL` in `mmvq.cu` forcing `vdr=8` and `nwarps=2` on `k < 2048` (`50–75%` of warp lanes idle on `k=512..1024`), plus a `MUL_MAT_ID Q4_K` regression from an unguarded `n=1` k-loop.
2. **`v15b` (`c01d483`) fixes the regression and adds net MoE + Dense gains:**
   - Recovers **`+9.37%` to `+30.69%` `tg64`** vs `v14` across all 3 GPUs on `COMMON` (`Qwen3.5-0.8B` and `Granite-1B`), restoring Stock parity or beating Stock (`P620` `Qwen3.5-0.8B` **`+4.76%` `tg64`**, `pp512` **`+8.85%`** vs Stock).
   - Improves `P104` `OLMoE-1B-7B` **`tg128`** to **`179.19 t/s`** (**`+27.71%` vs Stock**, **`+4.97%` vs `v14`**) and `pp512` to **`2497.29 t/s`** (**`+8.69%` vs Stock**).
   - Preserves and extends `P104` dense gains on `Qwen3.5-9B` (`pp512@d0` **`+16.05%`**, `tg128@d0` **`+31.53%`**, `tg128@d16384` **`+31.26%` vs Stock** and **`+2.14%` vs `v14`**).
   - Reduces `hsk=256` `FLASH_ATTN_EXT` tile op latency by **`-5.15%` to `-18.14%`** across all three GPUs via `nbatch_fa=64` in `fattn-tile.cuh`.
3. **Full verification passed with zero errors or thermal trips:** `9857/9857` ops on `P104`, `398/398` `Q4_K`/`Q5_K`/`Q6_K` ops on `P620`, `5/5` KLD quality gates (`Qwen3.5-9B-Q4_K_M`, `ub1..ub512`), max temperatures `60°C–63°C`, `trips=[]`.

---

## 2. Audit Compliance, Provenance (`PROV-1`), and Pinned Models

### 2.1 Loaded-DSO Provenance (`PROV-1`)
Created `/home/<user>/p104-bench/tools/dso_witness.sh` and verified (`exit 0`) that `LD_LIBRARY_PATH=<bin_dir>` cleanly overrides the embedded `DT_RUNPATH` (`/home/<user>/p104-exp/wt/build-v9/bin`) so every project `.so` resolves strictly inside its target `<bin_dir>`:

| Build Label | Commit / Dir | `llama-bench` SHA-256 | `libggml-cuda.so.1.0.0` SHA-256 |
|---|---|---|---|
| **Stock (`base-bin`)** | `5bacb7c` (`archive/base-bin`) | `3dcb1266cc6ad408c5fd7d65c8d404ef9528bc137829e6a9ab58a0cd214b3c94` | `b5b40012264dbb6e50a6b1b53ef3e3df654d47821373f5061295448eb3194830` |
| **Daily Control (`v14-bin`)** | `585eb84` (`v14-bin`) | `643cdf54859a226bfd3ab5157aefa53582b9daa7ae58cc2e3ee384b15f3670ec` | `34d3ecff4b10c1826ed4d48580ba5951acc3b88ad12e321700db6a3b3e8a38bb` |
| **`v15` (pre-small_k fix)** | `e2ccb81` (`v15-bin`) | `bf51d42ed06266a58971b27f03127b8f5fefb465084d9ba1319326ed951d4a1a` | `00640d975fd0bbe24bf35c19b17cb5493db93e2748bfcb66fce8f4e82e645854` |
| **`v15b` (Certified)** | `c01d483` (`v15b-bin`) | `ea9dc89f952168075f0a578fdb9492005e645fe62af62f0052cd435b5ab5f3b7` | `8627f886182aa4fe6995367e97a1e210ead44ea0d34a6c27fc72000738310006` |
| **`MOE-1` Profiler** | `archive/c0-prof-bin` | `e5c3f49aa88aeee2fdd093aacf5b84d580f17aeea33244f914670526f241005`... (`...4110c`) | `48794bd093a2a6712146f23af3e810de70f540286cbd4f8be4eeaae241005bbd7` |

*(Full `llama-bench` hash for `archive/c0-prof-bin`: `e5c3f49aa88aeee2fdd093aacf5b84d580f17aeea33244f914670526f243110c`.)*

### 2.2 Pinned Models & Fit Smoke Checks (`/home/<user>/models/pascal-bench/LOCAL_MANIFEST.json`)
All four audit/ladder models were verified by SHA-256 and smoke-checked for 100% GPU residency (`0` CPU spill):

| Model File | Revision | Size (Bytes) | SHA-256 | Verified Card Fit (100% GPU, 0 Spill) |
|---|---|---|---|---|
| `Qwen3.5-0.8B-Q4_K_M.gguf` | `unsloth/Qwen3.5-0.8B-GGUF` @ `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | `532517120` | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` | `P620`, `P2000`, `P104`: `25/25` layers (`497.40 MiB` model, `3.00 MiB` KV, `123.32 MiB` compute) |
| `granite-3.1-1b-a400m-instruct.Q4_K_M.gguf` | `QuantFactory/granite-3.1-1b-a400m-instruct-GGUF` @ `d1057c7e1300230a3d9d06278a74e01bc5267708` | `850160576` | `2e9e6946fbeb63f7cdc39a302979988daf04a9c9651fdcc0fc1df6f2aa05d459` | `P620`, `P2000`, `P104`: `25/25` layers (`782.12 MiB` model, `12.00 MiB` KV, `26.07 MiB` compute) |
| `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` | `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF` @ `7448e4317d367a3d6838fe367fe996758da6fc6e` | `4213513024` | `3bd9ec48045f526508956ef663c5c11dbb3ef791a06f4e10a6be88379c40e1fe` | `P104`: `17/17` layers (`3961.35 MiB` model, `32.00 MiB` KV, `28.63 MiB` compute) |
| `Qwen3.5-9B-Q4_K_M.gguf` | Existing pinned `P104` ladder model | — | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | `P104`: full GPU offload (`-ngl 99`) |

Observed free VRAM at fit check: `P620` `1861 MiB` (with Xorg), `P2000` `4978 MiB`, `P104` `8023 MiB`.

---

## 3. Exclusive Solo Benchmarks: Stock (`5bacb7c`) vs `v14` (`585eb84`) vs `v15b` (`c01d483`)

All runs executed strictly solo (`-ngl 99 -fa 1 -ctk f16 -ctv f16`) using `dso_witness.sh` provenance checks, max `60°C` (`COMMON`) / `63°C` (`LADDER`), and zero thermal/power trips (`trips=[]`).

### 3.1 `COMMON` Across All 3 Cards (`-b 128 -ub 128 -t 4 -p 128,512 -n 64 -d 0 -r 3`)

| Card | Model | Metric | Stock (`5bacb7c`) | `v14` (`585eb84`) | `v15b` (`c01d483`) | `v15b` vs Stock | `v15b` vs `v14` |
|---|---|---|---|---|---|---|---|
| **`P620`** | `Qwen3.5-0.8B` | `pp128` | `943.14` | `991.48` | **`1016.20`** | **`+7.75%`** | **`+2.49%`** |
| **`P620`** | `Qwen3.5-0.8B` | `pp512` | `924.19` | `980.64` | **`1005.99`** | **`+8.85%`** | **`+2.58%`** |
| **`P620`** | `Qwen3.5-0.8B` | `tg64` | `49.06` | `41.98` | **`51.39`** | **`+4.76%`** | **`+22.42%`** |
| **`P620`** | `Granite-1B` | `pp128` | `990.41` | `1020.67` | **`1027.07`** | **`+3.70%`** | `+0.63%` |
| **`P620`** | `Granite-1B` | `pp512` | `978.52` | `1006.82` | **`1012.44`** | **`+3.47%`** | `+0.56%` |
| **`P620`** | `Granite-1B` | `tg64` | `85.78` | `65.22` | `85.23` | `-0.64%` | **`+30.69%`** |
| **`P2000`** | `Qwen3.5-0.8B` | `pp128` | `1897.10` | `2249.42` | **`2300.26`** | **`+21.25%`** | **`+2.26%`** |
| **`P2000`** | `Qwen3.5-0.8B` | `pp512` | `2183.46` | `2232.32` | **`2278.25`** | **`+4.34%`** | **`+2.06%`** |
| **`P2000`** | `Qwen3.5-0.8B` | `tg64` | `103.80` | `89.60` | **`104.17`** | **`+0.36%`** | **`+16.26%`** |
| **`P2000`** | `Granite-1B` | `pp128` | `2011.54` | `2385.71` | **`2399.52`** | **`+19.29%`** | `+0.58%` |
| **`P2000`** | `Granite-1B` | `pp512` | `2266.04` | `2349.18` | **`2358.17`** | **`+4.07%`** | `+0.38%` |
| **`P2000`** | `Granite-1B` | `tg64` | `172.73` | `139.91` | `171.57` | `-0.67%` | **`+22.63%`** |
| **`P104`** | `Qwen3.5-0.8B` | `pp128` | `3703.07` | `3603.09` | `3660.65` | `-1.15%` | **`+1.60%`** |
| **`P104`** | `Qwen3.5-0.8B` | `pp512` | `3662.75` | `3598.50` | `3654.14` | `-0.24%` | **`+1.55%`** |
| **`P104`** | `Qwen3.5-0.8B` | `tg64` | `144.13` | `129.76` | `141.92` | `-1.53%` | **`+9.37%`** |
| **`P104`** | `Granite-1B` | `pp128` | `4182.79` | `4216.55` | **`4237.37`** | **`+1.30%`** | `+0.49%` |
| **`P104`** | `Granite-1B` | `pp512` | `4036.09` | `4109.84` | **`4125.80`** | **`+2.22%`** | `+0.39%` |
| **`P104`** | `Granite-1B` | `tg64` | `242.90` | `211.44` | `240.51` | `-0.98%` | **`+13.75%`** |

### 3.2 `P104` `LADDER` (`OLMoE-1B-7B` & `Qwen3.5-9B`)

| Model & Flags | Metric | Stock (`5bacb7c`) | `v14` (`585eb84`) | `v15b` (`c01d483`) | `v15b` vs Stock | `v15b` vs `v14` |
|---|---|---|---|---|---|---|
| `OLMoE-1B-7B` (`-b 512 -ub 512 -p 128,512 -n 128`) | `pp128` | `1596.10` | `1524.95` | `1523.67` | `-4.54%` | `-0.08%` |
| `OLMoE-1B-7B` | `pp512` | `2297.70` | `2503.21` | `2497.29` | **`+8.69%`** | `-0.24%` |
| `OLMoE-1B-7B` | `tg128` | `140.32` | `170.71` | **`179.19`** | **`+27.71%`** | **`+4.97%`** |
| `Qwen3.5-9B` (`-ub 512 -p 512,2048 -n 128`) | `pp512@d0` | `532.90` | `618.20` | **`618.43`** | **`+16.05%`** | `+0.04%` |
| `Qwen3.5-9B` | `pp2048@d0` | `521.75` | `603.60` | `603.44` | **`+15.66%`** | `-0.03%` |
| `Qwen3.5-9B` | `tg128@d0` | `29.63` | `39.00` | `38.98` | **`+31.53%`** | `-0.05%` |
| `Qwen3.5-9B` | `pp512@d16384` | `340.82` | `381.06` | `379.28` | **`+11.29%`** | `-0.47%` |
| `Qwen3.5-9B` | `pp2048@d16384` | `344.40` | `381.88` | `379.51` | **`+10.20%`** | `-0.62%` |
| `Qwen3.5-9B` | `tg128@d16384` | `27.10` | `34.82` | **`35.57`** | **`+31.26%`** | **`+2.14%`** |

Additional card check observed during `v15` development (`task-429`): on `P2000`, `Qwen3-8B` (`-ngl 28`) held flat at `283.3` `pp512@d0` and `11.15` `tg128@d0`, while `hsk=256` `FLASH_ATTN_EXT` ops improved `-5.15%..-18.14%` across `P2000` and `P620`.

---

## 4. Profiling (`MOE-1`), Root Causes, and Code Changes in `v15b`

### 4.1 `MOE-1` CUDA-Event Attribution (`archive/c0-prof-bin`, `GGML_P104_OPPROF=1 GGML_CUDA_DISABLE_GRAPHS=1`, `P104`)
- **`OLMoE-1B-7B` (`n8 kv256 nodes534`, `gpu_ms=16.264`):** `MUL_MAT:q4_K:2048x2048` `20.30%`, `MUL_MAT:f32:64x2048` (router) `14.26%`, `MUL_MAT_ID:q4_K` `12.06%` (`1024x2048` `8.55%`), `ADD` `7.61%`, `MUL_MAT_ID:q6_K` `6.12%`, `RMS_NORM` `5.96%`, `FLASH_ATTN_EXT` `5.32%`, `QUANT_Q8_1` `5.09%`.
- **`granite-1b` (`n8 kv256 nodes800`, `gpu_ms=9.449`):** `MUL_MAT_ID:q4_K` `15.67%` (`512x1024` `11.07%`), `MUL_MAT:q4_K` `13.24%`, `FLASH_ATTN_EXT` `9.71%`, `QUANT_Q8_1` `8.98%`, `MUL_MAT:f32` `7.97%`, `MUL_MAT_ID:q6_K` `7.27%`, `MUL_MAT:q6_K` `5.27%`.
- **`Qwen3.5-0.8B` (`n1 kv256 nodes1188`, `gpu_ms=14.141`):** `MUL_MAT:q6_K` `14.68%`, `MUL_MAT:q4_K` `13.00%`, `MUL_MAT:q5_K` `12.45%`, `QUANT_Q8_1` `7.59%`, `RMS_NORM` `7.22%`, `FLASH_ATTN_EXT` `5.94%`.
- **`NOGRAPH` Check (`GGML_CUDA_DISABLE_GRAPHS=1`) on `P104`:** Even without CUDA graphs, `granite` dropped Stock `242.55` -> `v14` `208.47` and `q08b` dropped Stock `144.06` -> `v14` `129.74`, proving the regression was inside CUDA kernels rather than graph capture/launch overhead.

### 4.2 Landed Commits in `llamascal` (`c01d483` over `585eb84`)
1. **`ce70cb5` (`mmq-vec-dot.cuh`, `mmvq.cu`):**
   - `Q5_K` `i0`-outer loop in `mmq-vec-dot.cuh` (`-0.5%..-0.8%` at `n=512`).
   - Hoisted `n=1` k-loops for `Q4_K`/`Q5_K`/`Q6_K` in `mmvq.cu` (`Q4_K` LM-head `-1.22%`, `MUL_MAT_ID Q6_K n=1` `-0.61%..-0.78%` on `P104`).
2. **`69288c5` / `e2ccb81` (`fattn-tile.cuh`):**
   - Set `nbatch_fa=64` for `DKQ=256` `ncols=4` `(128,3,64,64)` and `ncols=8` `(256,2,64,128)`, cutting `hsk=256` `FLASH_ATTN_EXT` op latency `-5.95%..-18.14%` across all three Pascal cards.
3. **`c01d483` (`mmvq.cu` `small_k` & MoE `has_ids` fix):**
   - Fixed the `small_k` (`k < 2048`) bug in `MMVQ_PARAMETERS_PASCAL` (`calc_nwarps`, `get_vdr_mmvq_tbl`, `get_vec_dot_q_cuda_tbl`, k-loops, alignment assert) where `v14` forced `vdr=8` and `nwarps=2`, leaving `50–75%` of warp lanes idle on `k=512..1024`.
   - Guarded `p104_mmvq_kloop_q45_n1` `Q4_K` with `!has_ids`, eliminating the `P620` `+4.5%..+5.9%` `MUL_MAT_ID Q4_K` regression.

### 4.3 Rejected & Reverted Hypotheses (Documented Not to Retry)
- `MMQ I=128, J=64`: `Q4_K` `+1267%..+1400%`, `Q6_K` `+22.6%..+30.2%`.
- `Q4_0`/`Q8_0` MMQ magic-float: `+2.3%..+3.1%`.
- `Q4_K`/`Q5_K` `MMQ occ=1`: `+15.8%..+33.7%`.
- `quantize_q8_1` `__shfl_sync`: `-0.02%..+0.71%` (wash).
- `FA __ldg` KV loads: `~0%` (wash).
- `FA (128,128,8)->(256,2,64,128)`: `+16%..+29%`.
- `FA (128,128,32)->(256,2,32,128)`: `Qwen3-8B` `pp2048@d4096` `-7.8%`.
- `FA (128,128,4)->(128,3,64,64)`: `P2000` `+1.7%..+1.9%`.
- `FA (256,256,16)->(256,2,64,64)`: `nb=1` `+1.1%`.
- `MMVQ Q4_K n=2`: `+16%..+18%`.
- `MMVQ Q6_K n=2,3`: `+1.7%..+9.5%` (124 regs).

---

## 5. Correctness & KLD Quality Gates (`v15b-bin`)

### 5.1 Backend Op Correctness
- **`P104` (`test-backend-ops`):** **`9857/9857` passed** (`MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION,FLASH_ATTN_EXT,GATED_DELTA_NET`, max `42°C`).
- **`P620` (`test-backend-ops`):** **`398/398` passed** (`MUL_MAT,MUL_MAT_ID -p "type_a=(q4_K|q5_K|q6_K)"`, max `43°C`).

### 5.2 KLD Quality (`Qwen3.5-9B-Q4_K_M`, `-ngl 99 -c 1024 -b 1024 -t 4 -ctk f16 -ctv f16 -fa on --chunks 5`, max `63°C`)

| `ubatch` | Mean KLD | Max KLD | `99.9%` Quantile | Same Top-Token Rate |
|---|---|---|---|---|
| `ub1` | `0.000490 ± 0.000023` | — | `0.014951` | `98.982 ± 0.199 %` |
| `ub2` | `0.000439 ± 0.000015` | `0.023422` | `0.007095` | `98.591 ± 0.233 %` |
| `ub3` | `0.000521 ± 0.000057` | `0.127220` | `0.016352` | `98.748 ± 0.220 %` |
| `ub4` | `-0.000000 ± 0.000000` | `0.000047` | `0.000046` | `100.000 ± 0.000 %` |
| `ub512` | `0.000493 ± 0.000043` | — | `0.011691` | `98.630 ± 0.230 %` |

---

## 6. Conclusions & Recommended Next Steps

1. **Promotion Ready:** `v15b-bin` (`c01d483`) strictly dominates `v14-bin` on tiny dense & MoE models (`+9.37%` to `+30.69%` `tg64`), improves `OLMoE-1B-7B` `tg128` by `+4.97%` (`+27.71%` vs Stock), preserves `P104` `Qwen3.5-9B` gains (`+31.53%` `tg128@d0`, `+31.26%` `tg128@d16384` vs Stock), and clears all correctness and KLD checks. You may tag `p104-v15b` and promote `/home/<user>/p104-exp/llamascal-bin -> v15b-bin` when ready.
2. **Next High-Value Optimization Targets (`MOE-1` & `DENSE-1`):**
   - **MoE Router (`MUL_MAT:f32:64x2048`):** Accounts for `14.26%` of `OLMoE-1B-7B` GPU time and `7.97%` of `granite-1b` time (`MUL_MAT_F32` tiny-`n` warp specialization / shared-memory reduction in `mmvf.cu`).
   - **`DENSE-1` Profiling:** Capture `GGML_P104_OPPROF=1` on `Qwen3.5-9B` (`pp512`/`pp2048`/`tg128` at `d0` and `d16384`) to identify remaining `QUANT_Q8_1` (`5%–9%`), `RMS_NORM` (`6%–7%`), and `GATED_DELTA_NET` opportunities.
   - **`OLMoE` `pp128` (`-4.54%` vs Stock):** Profile batch `n=128` `MUL_MAT_ID` / `MMQ` tile dispatch thresholds on `P104`.
