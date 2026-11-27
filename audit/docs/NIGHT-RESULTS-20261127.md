# Overnight Pascal (`sm_61`) Inference Optimization Report — 2026-11-27

## 1. Executive Summary & Production State
- **Baseline (`v15b`, unchanged in production):** Commit `c01d483e142aa758209e350ea9ba717763087dc3` (`c01d483`), `<home>/p104-exp/llamascal-bin -> v15b-bin`. Stock reference: `5bacb7c`.
- **Candidate (`v16-cand`):** Branch `exp/pascal-v16-cand` (`636e68f21fb07bf1f723e1c78b4d0368667dfa08`) in `<home>/p104-exp/wt-v16`, built to `<home>/p104-exp/v16-cand-bin`, patch exported to `<home>/p104-bench/patches/0001-p104-Pascal-MMQ-tail-wave-J-64-guard-F32-MMVF-128-ro.patch`.
- **Headline No-MTP Kernel Results (Exclusive Solo 8-Leg ABBA, `v16-cand-bin` vs `v15b-bin`):**
  - `P104` `OLMoE-1B-7B` `pp128`: **`+7.12%`** (`1529.92 ± 6.75 -> 1638.81 ± 1.53 tok/s`), exceeding stock (`1631.47`) while retaining `v15b` decode gains (`179 tok/s`) and zero `pp512` loss (`+0.10%`).
  - `P104` `Qwen3.5-0.8B` `pp128`: **`+1.54%`** (`3605.00 ± 3.70 -> 3660.53 ± 2.90 tok/s`).
  - `P2000` `granite-3.1-1b` `pp8`: **`+1.96%`** (`522.35 ± 2.50 -> 532.57 ± 1.61 tok/s`), `pp4`: **`+0.96%`** (`364.88 -> 368.40 tok/s`), with `pp128` neutral (`-0.40%`).
  - `P620` (`Qwen3.5-0.8B`, `granite-1b`) and `P104` `Qwen3.5-9B`: zero regressions across all prompt/decode shapes.
- **Headline Separate Serving / MTP Stack Result (`S0` on `P104`, `Qwen3.5-9B-MTP-Q4_K_M.gguf`):**
  - `M2` (`--spec-type draft-mtp --spec-draft-n-max 2`) + `P104_MTP_DRAFT_VOCAB=65536`: **`58.51 tok/s`** server decode on `v16-cand-bin` (`25.6 ms/tok` end-to-end wall, **`1.37x` wall speedup**, `+54.6%` decode tok/s over no-MTP `S` `37.85 tok/s` / `35.1 ms/tok`), `0.792` acceptance (`2086/2633`), `6473 MiB` VRAM, max `56 C`.
- **Quality & Safety Verification:**
  - `test-backend-ops test` (`v16-cand-bin`): **`20834/20834` passed** on `P620` and **`20834/20834` passed** on `P104` (`2/2` backends OK).
  - Greedy parity (`v16-cand-bin` vs `v15b-bin`): `OLMoE` `[True]`, `granite-1b` `[True]`, `Qwen3.5-9B` `[True, True, True]`.
  - Paired WikiText-2 KLD (`v16-cand-bin` vs `v15b-bin`): **`Mean KLD = 0.000000 ± 0.000000`**, `100.000%` top-p match on both `OLMoE-1B-7B` (`ub=128`) and `Qwen3.5-9B` (`ub=512`).
  - **Restored Production State:** `p104-svc` restored and `/health` healthy (`7301 MiB` on `P104`), `P104` power limit `140.00 W`, `<home>/p104-exp/llamascal-bin -> v15b-bin`, `<home>/p104-exp/wt` clean at `c01d483`.

---

## 2. Host Provenance, DSO Witnesses & Concurrency Canary
- **Model SHA-256 Verification:**
  - `granite-3.1-1b-a400m-instruct.Q4_K_M.gguf`: `2e9e6946fbeb63f7cdc39a302979988daf04a9c9651fdcc0fc1df6f2aa05d459`
  - `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`: `3bd9ec48045f526508956ef663c5c11dbb3ef791a06f4e10a6be88379c40e1fe`
  - `Qwen3.5-0.8B-Q4_K_M.gguf`: `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`
  - `Qwen3.5-9B-Q4_K_M.gguf`: `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`
- **DSO Witnesses (`tools/dso_witness.sh`):**
  - `v15b-bin`: `llama-bench` `ea9dc89f952168075f0a578fdb9492005e645fe62af62f0052cd435b5ab5f3b7`, `libggml-cuda.so.1.0.0` `8627f886182aa4fe6995367e97a1e210ead44ea0d34a6c27fc72000738310006`
  - `v16-cand-bin`: `llama-bench` `af34d327b1a19eaf0e60a0c7db455bf6a7ceabb979f762d602da581889ad0a59`, `libggml-cuda.so.1.0.0` `6651270d6a5f3390eeec258a45a17a11b120cad3d5996d6e1f5910b60073d9c9`
- **Concurrency Canary (`Qwen3.5-0.8B` on `v15b-bin`):** Running 2-up/3-up produced up to `33.11%` drift on `P2000 pp128` (`1727.81 -> 2299.95`, exceeding the `3%` threshold). Accordingly, all publishable A/Bs ran strictly serialized solo under `night_guard.py` (`WARN_C=75`, `TRIP_C=80`, `COOL_C=55`, zero thermal trips recorded).

---

## 3. Baseline Profiling (`D0`, `M0`) & Root-Cause Findings
1. **True `n=1` MoE Decode (`OPPROF MAXTOK=8`):** Router `MUL_MAT:f32` is only `2.00%` on `OLMoE` (`P104`) and `1.92%` on `granite-1b` (`P2000`); the earlier `14.26%` observation was an `n=8` `cuBLAS` fallback. At `n=4/8` (`OLMoE`), `mmvf.cu` excluded `F32` when `ne11 > 3`, forcing small `64x2048` / `32x1024` routers through `cuBLAS`.
2. **`OLMoE pp128` Bisect (`Stock -> v13 -> v14 -> v15b`):**
   - `Stock (5bacb7c)`: `pp128 1631.47 ± 40.08`, `pp512 2300.78`, `tg128 140.47`
   - `v13 (67df1fa)`: `pp128 1564.35 ± 18.91`, `pp512 2483.19`, `tg128 170.24`
   - `v14 (585eb84)`: `pp128 1519.89 ± 23.21`, `pp512 2483.18`, `tg128 168.66`
   - `v15b (c01d483)`: `pp128 1518.76 ± 7.09`, `pp512 2491.69`, `tg128 178.96 ± 0.19`
   - Root cause (`GGML_P104_MMQ_LOG=1` + `cuobjdump`): on `2048`-row layers (`OLMoE`), `J=128` launched `(2048/64)*1 = 32` blocks across `20` SMs on `P104` (tail wave: `1 < 32/20 < 2`) while spilling `STACK:48` at `REG:128`. On 8-SM `P2000`, `512`-row GQA layers launch `8 == nsm` blocks (zero tail), so guarding `nblocks_128 > nsm && nblocks_128 < 2*nsm` fixes `P104` tail waves without hurting `P2000` (`512`/`1024` rows) or `Qwen3.5-9B` (`>=4096` rows).
3. **`Dense` `Qwen3.5-9B` (`P104`):** `tg128@d0` is dominated by `Q4_K` (`41.05%`), `Q6_K` (`23.69%`), `Q5_K` (`13.66%`); `pp512@d16384` by `Q4_K` (`45.65%`) and `FLASH_ATTN_EXT` (`24.35%`).

---

## 4. Candidate Changes in `v16-cand` (`exp/pascal-v16-cand`)
Patch: `<home>/p104-bench/patches/0001-p104-Pascal-MMQ-tail-wave-J-64-guard-F32-MMVF-128-ro.patch` (`3 files, +57 lines`):
1. **`ggml/src/ggml-cuda/mmq.cu`:** On Pascal (`sm_61`..`sm_62`), when `!ids && J_best == 128`, compute `nblocks_128 = (ne01 / 64) * ((ne11 + 127) / 128)`. If `nblocks_128 > nsm && nblocks_128 < 2 * nsm`, re-select `J_best = 64` (keeps `<= nsm` single-wave layers such as `512`-row GQA on `P2000` at `J=128`, and fixes `32`-block tail waves on `20`-SM `P104`).
2. **`ggml/src/ggml-cuda/mmvf.cu`:** On Pascal (`cc < GGML_CUDA_CC_TURING`), enable `GGML_TYPE_F32` `MMVF` for thin matrices (`src0_ne[1] <= 128`) up to `ne11 <= (src0_ne[0] <= 1024 ? MMVF_MAX_BATCH_SIZE : 4)`, accelerating small MoE routers (`pp4`/`pp8`) without hurting `cuBLAS` on wider `K=2048` `pp8`.
3. **`ggml/src/ggml-cuda/mmq-vec-dot.cuh`:** For `Q4_K` `J <= 16`, use an `i0`-outer loop (matching `Q5_K`) to cut register usage from `96` (`2` blocks/SM) to `72..76` (`3` blocks/SM, `STACK:0`).

---

## 5. Phase 4 Exclusive Solo ABBA Results (No-MTP, `v15b-bin` vs `v16-cand-bin`)

All `d=0` suites ran 8 solo legs (`A/B/B/A/B/A/A/B`, 4 launches/arm, 3 intra-run repeats after in-process warmup).

| GPU | Model | Metric | `v15b` (mean ± sd) | `v16-cand` (mean ± sd) | Delta | Max T (A/B) |
|---|---|---|---|---|---|---|
| P104 | OLMoE-1B-7B | `pp4` | 324.43 ± 0.58 | 325.94 ± 0.79 | **+0.47%** | 44C / 44C |
| P104 | OLMoE-1B-7B | `pp8` | 366.20 ± 3.24 | 366.16 ± 7.20 | -0.01% | 44C / 44C |
| P104 | OLMoE-1B-7B | **`pp128`** | **1529.92 ± 6.75** | **1638.81 ± 1.53** | **+7.12%** | 44C / 44C |
| P104 | OLMoE-1B-7B | `pp512` | 2503.67 ± 9.31 | 2506.09 ± 5.82 | +0.10% | 44C / 44C |
| P104 | OLMoE-1B-7B | `tg128` | 179.40 ± 0.12 | 178.71 ± 0.27 | -0.38% | 44C / 44C |
| P104 | Qwen3.5-0.8B | **`pp128`** | **3605.00 ± 3.70** | **3660.53 ± 2.90** | **+1.54%** | 56C / 56C |
| P104 | Qwen3.5-0.8B | `pp512` | 4674.36 ± 11.92 | 4704.99 ± 7.94 | +0.66% | 56C / 56C |
| P104 | Qwen3.5-0.8B | `tg128` | 140.70 ± 0.14 | 140.74 ± 0.35 | +0.03% | 56C / 56C |
| P104 | Qwen3.5-9B `d=0` | `pp128` | 535.26 ± 2.63 | 536.10 ± 1.33 | +0.16% | 53C / 54C |
| P104 | Qwen3.5-9B `d=0` | `pp512` | 621.93 ± 2.57 | 622.72 ± 2.04 | +0.13% | 53C / 54C |
| P104 | Qwen3.5-9B `d=0` | `tg128` | 39.03 ± 0.03 | 38.98 ± 0.04 | -0.14% | 53C / 54C |
| P104 | Qwen3.5-9B `d=16k` (init 2/arm) | `pp512@d16384` | 385.46 ± 0.04 | 377.16 ± 10.44* | -2.15%* | 58C / 58C |
| P104 | Qwen3.5-9B `d=16k` (re-check) | `pp512@d16384` | 387.96 `[5]` | 388.37 `[4]` | **+0.11%** | - |
| P104 | Qwen3.5-9B `d=16k` (re-check) | `tg128@d16384` | 35.65 `[5]` | 35.62 `[4]` | -0.08% | - |
| P2000 | granite-3.1-1b | **`pp4`** | 364.88 ± 0.71 | 368.40 ± 1.00 | **+0.96%** | 59C / 59C |
| P2000 | granite-3.1-1b | **`pp8`** | **522.35 ± 2.50** | **532.57 ± 1.61** | **+1.96%** | 59C / 59C |
| P2000 | granite-3.1-1b | `pp128` | 2387.18 ± 6.05 | 2377.69 ± 6.03 | -0.40% | 59C / 59C |
| P2000 | granite-3.1-1b | `pp512` | 3021.65 ± 22.53 | 3016.74 ± 17.92 | -0.16% | 59C / 59C |
| P2000 | granite-3.1-1b | `tg128` | 171.28 ± 0.35 | 170.43 ± 0.54 | -0.50% | 59C / 59C |
| P2000 | Qwen3.5-0.8B | `pp128` | 2304.00 ± 0.35 | 2304.32 ± 1.13 | +0.01% | 60C / 60C |
| P2000 | Qwen3.5-0.8B | `pp512` | 2572.64 ± 0.89 | 2577.56 ± 8.37 | +0.19% | 60C / 60C |
| P2000 | Qwen3.5-0.8B | `tg128` | 104.33 ± 0.03 | 104.34 ± 0.05 | +0.00% | 60C / 60C |
| P620 | Qwen3.5-0.8B | `pp128` | 1016.72 ± 1.48 | 1017.22 ± 0.78 | +0.05% | 62C / 63C |
| P620 | Qwen3.5-0.8B | `pp512` | 1097.88 ± 1.39 | 1099.17 ± 0.86 | +0.12% | 62C / 63C |
| P620 | Qwen3.5-0.8B | `tg128` | 51.47 ± 0.06 | 51.46 ± 0.05 | -0.02% | 62C / 63C |
| P620 | granite-3.1-1b | `pp128` | 1026.97 ± 0.16 | 1027.09 ± 0.20 | +0.01% | 61C / 62C |
| P620 | granite-3.1-1b | `pp512` | 1304.31 ± 0.13 | 1304.13 ± 0.24 | -0.01% | 61C / 62C |
| P620 | granite-3.1-1b | `tg128` | 85.33 ± 0.02 | 85.24 ± 0.11 | -0.10% | 61C / 62C |

\*`q9b-d16k` leg `[1 v16]` had one transient jitter rep (`353.89 / 378.67 / 376.76`); `[2 v16]` (`384.54`) and re-check legs `[4 v16]` (`388.37`) vs `[5 v15b]` (`387.96`) confirmed zero regression (`+0.11%`).

---

## 6. Rejected Experiments (Cleanly Reverted)

| ID | Hypothesis Tested | Observed Result | Disposition |
|---|---|---|---|
| `M1a` | `Q6_K` `small_k` routing + `p104_mmvq_kloop_q45_smallk` | `P2000 gran tg128 -7.46%`, `P620 q08 -6.06%`, `P620 gran -9.85%` | Reverted |
| `M2c` sub | `Q4_K` `J >= 120` `i0`-outer in `mmq-vec-dot.cuh` | Still `REG:128 STACK:48`, regressed `P2000 gran pp128` | Reverted |
| `D1a` | `mmvq.cu` `__launch_bounds__` `6/12` blocks/SM (`<= 80` regs) | `Qwen3.5-9B tg128@d0 -2.38%`, `tg128@d16k -2.26%` | Reverted |
| `D1b` | `mmvq.cu` `__launch_bounds__` `5/10` blocks/SM (`<= 96` regs) | `Qwen3.5-9B tg128@d0 -0.45%` | Reverted |
| `D2a` | `fattn-tile.cuh` `DKQ=256,DV=256` `ncols=8,16,32` `occ=1` (`REG:148`) | `Qwen3.5-9B pp512@d16384 -12.02%` (`381.71 -> 335.84`) | Reverted |
| `D2b` | `fattn-tile.cuh` `DKQ=256,DV=256` `ncols=16,32` `nthreads=512, occ=1` | `Qwen3.5-9B pp512@d16384 -20.36%` (`385.97 -> 307.37`) | Reverted |

---

## 7. Separate `S0` MTP & Serving-Stack Results (`P104`, `16` Workload Requests)

Strictly separated from no-MTP kernel A/Bs (`llama-server -c 24576 -np 1 -fa on -ctk f16 -ctv f16 -ub 512 -t 4 --cache-ram 1024 --jinja --metrics --no-warmup` under `night_guard.py`):

| Binary | Arm | Draft Vocab Env | Srv `tg tok/s` | Wall `ms/tok` | Speedup vs `S` (wall) | Accept (acc/draft) | VRAM MiB | Max Temp C | Parity vs `M2` |
|---|---|---|---|---|---|---|---|---|---|
| `v15b-bin` | `S` (`Qwen3.5-9B`, no MTP) | - | 37.85 | 35.1 | 1.000x | - | 5933 | 58 | - |
| `v15b-bin` | `M1` (`draft-mtp n_max=1`) | - | 50.01 | 28.5 | 1.234x | 0.889 (1622/1824) | 6423 | 60 | 10/16 vs M2 |
| `v15b-bin` | `M2` (`draft-mtp n_max=2`) | - | 53.66 | 27.1 | 1.295x | 0.818 (2139/2615) | 6473 | 61 | 16/16 |
| `v15b-bin` | `M3` (`draft-mtp n_max=3`) | - | 46.86 | 30.2 | 1.165x | 0.723 (2329/3221) | 6523 | 61 | - |
| `v15b-bin` | `M2-dv16k` | `16384` | 54.54 | 26.8 | 1.311x | 0.657 (1953/2971) | 6473 | 61 | **16/16** |
| `v15b-bin` | `M2-dv32k` | `32768` | 57.12 | 25.9 | 1.355x | 0.730 (2041/2797) | 6473 | 61 | **16/16** |
| `v15b-bin` | `M2-dv64k` | `65536` | 57.92 | 25.8 | 1.362x | 0.780 (2097/2687) | 6473 | 61 | **16/16** |
| `v16-cand-bin` | `M2-dv64k` | `65536` | **58.51** | **25.6** | **1.373x** | 0.792 (2086/2633) | 6473 | 56 | 8/16 vs `v15b` |

*(Note: `S` uses `Qwen3.5-9B-Q4_K_M.gguf` and `M1..M3` use `Qwen3.5-9B-MTP-Q4_K_M.gguf`; `M2-dv16k/32k/64k` match `M2` 16/16 identically while lifting decode from `53.66` to `57.92 tok/s`. Profiling `Qwen3.5-9B` verify batches `pp=2/3/4` confirmed `pp3` `429.4 ms` vs `pp4` `557.5 ms` where `n=4` exits wide `vdr=8`, explaining why `M2` outperforms `M3`.)*

---

## 8. Correctness, Greedy Parity & Paired WikiText-2 KLD Gates (`v16-cand-bin`)
- **Full `test-backend-ops test`:**
  - `P620`: **`20834/20834` passed** (`2/2 backends passed OK`).
  - `P104`: **`20834/20834` passed** (`2/2 backends passed OK`).
- **Deterministic Greedy Parity (`temp=0, seed=1` vs `v15b-bin`):**
  - `OLMoE-1B-7B` (`P104`): `[True]` (`1/1`)
  - `granite-3.1-1b` (`P2000`): `[True]` (`1/1`)
  - `Qwen3.5-9B` (`P104`): `[True, True, True]` (`3/3`)
- **Paired WikiText-2 KLD (`20` chunks, `c=1024`, `-ctk f16 -ctv f16 -fa on`, `P104`):**

| Model | Ref (`v15b`) | Test (`v16-cand`) | PPL (ref -> test) | Mean KLD | Max KLD | RMS Δp | Same Top-p |
|---|---|---|---|---|---|---|---|
| `OLMoE-1B-7B` | `ub=128` | `ub=128` | `9.6364 -> 9.6364 ± 0.2634` | **`0.000000 ± 0.000000`** | `0.000060` | `0.001 ± 0.000 %` | **`100.000 ± 0.000 %`** |
| `OLMoE-1B-7B` | `ub=128` | `ub=4` *(cross-ub)* | `9.6364 -> 9.6467 ± 0.2637` | `0.003226 ± 0.000173` | `1.457581` | `1.388 ± 0.039 %` | `97.857 ± 0.143 %` |
| `Qwen3.5-9B` | `ub=512` | `ub=512` | `7.1026 -> 7.1026 ± 0.1787` | **`0.000000 ± 0.000000`** | `0.000059` | `0.001 ± 0.000 %` | **`100.000 ± 0.000 %`** |
| `Qwen3.5-9B` | `ub=512` | `ub=128` *(cross-ub)* | `7.1026 -> 7.1045 ± 0.1787` | `0.000822 ± 0.000147` | `1.261073` | `0.824 ± 0.059 %` | `98.777 ± 0.109 %` |
| `Qwen3.5-9B` | `ub=512` | `ub=4` *(cross-ub)* | `7.1026 -> 7.1063 ± 0.1788` | `0.001142 ± 0.000295` | `2.844869` | `0.951 ± 0.084 %` | `98.836 ± 0.106 %` |
| `Qwen3.5-9B` | `ub=512` | `ub=1` *(cross-ub)* | `7.1026 -> 7.1121 ± 0.1789` | `0.001555 ± 0.000512` | `4.676535` | `1.167 ± 0.209 %` | `98.836 ± 0.106 %` |

---

## 9. Recommended Next Actions
1. **Promote `v16-cand` (`636e68f`)** on review: `+7.12%` `OLMoE pp128`, `+1.54%` `Qwen3.5-0.8B pp128`, `+1.96%` `granite-1b pp8` on `P2000`, `20834/20834` full backend-ops pass on `P620` & `P104`, and `0.000000` matched-`ub` KLD.
2. **Deploy `S0` MTP Serving Tuning** (`--spec-type draft-mtp --spec-draft-n-max 2` + `P104_MTP_DRAFT_VOCAB=65536`) for single-stream `Qwen3.5-9B-MTP` serving (`58.51 tok/s` decode, `25.6 ms/tok` wall, `1.37x` end-to-end speedup over non-MTP baseline).
