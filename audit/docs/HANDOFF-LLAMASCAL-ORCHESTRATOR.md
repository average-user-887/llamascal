# Orchestrator Handoff: `llamascal` (`p104-v13` Clean Base & Multi-Pascal Guide)

**Host**: `z4-p104` (HP Z4 G4)
**Updated**: 2026-11-26
**Live Serving Endpoint**: Docker container `p104-svc` on `:18100` (`p104-llama:67df1fa-20261126`)

---

## 1. Clean Workspace & Build State

| Artifact | Path / Ref | Notes |
|---|---|---|
| Upstream reference repo | `~/llama.cpp` | `master` @ `5bacb7c` (stock baseline, clean) |
| Active development worktree | `~/p104-exp/wt` | Branch **`llamascal`** == tag **`p104-v14`** @ **`585eb84`** (clean over **`p104-v13`** @ `67df1fa`, `git status --short` empty) |
| Previous release tags | **`p104-v13`** @ `67df1fa`, **`p104-v11`** @ `b098f6f` | Previous `mmvq`+`c7a`+`c2` (`v13`) and `k1/k3` (`v11`) baselines |
| Active CMake build dir | `~/p104-exp/wt/build` -> `build-v9` | Configured for `-DCMAKE_CUDA_ARCHITECTURES=61-real` (all `sm_61` GPUs) |
| Current host binaries | `~/p104-exp/llamascal-bin` -> `v14-bin` | Rebuilt from `585eb84`; sha256 checksums in `v14-bin/HASHES.txt` (note: use explicit `LD_LIBRARY_PATH` when comparing bins due to `DT_RUNPATH` pointing at `wt/build-v9/bin`) |
| Previous host binaries | `~/p104-exp/v13-bin`, `v11-bin` | Kept for direct ABBA comparison (live `p104-svc` container still runs `67df1fa-20261126`) |
| Archived scratch bins | `~/p104-exp/archive/` | Old intermediate `k1`/`k3`/`card*` bins & scratch notes |
| Exported patch series | `~/p104-bench/patches/llamascal-v14/` | 20 `git format-patch` patches (`5bacb7c..p104-v14`); `llamascal-v13/` (`0001..0017`) preserved |
| Combined diffs | `patches/v14-p104-bundle.patch`, `patches/v14-from-v13.patch` | Single-file diffs from `5bacb7c` and `p104-v13` |
| Untested WIP archive | `patches/archive-wip/c7b-i128-untested.patch` | Saved `cardC` 6-line `mmq-config-pascal-dp4a.cuh` (`I=128`) experiment |
| Serving config | `~/p104-bench/container/serving.env` | `SRV_FLAGS` with `-bs`, `SRV_ENV="P104_MTP_DRAFT_VOCAB=96256"` |
| Image pins | `~/p104-bench/images/CURRENT` | `p104-llama:67df1fa-20261126` (`PREVIOUS`: `p104-llama:b098f6f-20261123`) |

---

## 2. Included Optimizations & Verified Receipts

Commit range on top of upstream `5bacb7c` (`5bacb7c..p104-v13`, branch `llamascal` @ `67df1fa`):
1. **`k1` + `k3` (`0ad6995..b098f6f`)**: Pascal `MMVQ_PARAMETERS_PASCAL` dispatch, wide (`vdr=8`) `Q4_K`/`Q5_K` matrix-vector kernels for decode/small-batch, templated `Q6_K` `vec_dot_q6_K_q8_1_v`, `P104_Q6_PRESUB_MINCOLS` infrastructure.
2. **`k4` (`8845a20`)**:
   - **Q6_K**: Enabled `P104_Q6_PRESUB_MINCOLS=2` (exact integer 32-offset presubtraction) + `0x4B400000` (`12582912.0f`) magic-number `int->float` conversion in `vec_dot_q6_K_q8_1_v` (avoids quarter-rate `I2F` & `XMAD`). Won all 15 benchmarked shapes (`-2.1%` to `-8.0%`; `248320x4096 n=3`: `4756 -> 4376 us`).
   - **Q5_K**: `p104_mmvq_kloop_q45` hoists exact `q8_1` column `sumi_m` once per column across `rows_per_cuda_block` and uses magic-number float conversion (`-8.7%` at `n=2`, `-14.4%` at `n=3` on `4096x4096`). Restricted to `type == GGML_TYPE_Q5_K` after operator ABBA showed `Q4_K` regressed on `k=12288, n=3` (`168.3 -> 213.0 us`).
3. **`c7a` (`27b632d`, cherry-picked `361199d`)**: Pascal dp4a MMQ prefill K-quant inner loop rewrite (`mmq-vec-dot.cuh`, `mmq.cu`, `mmq.cuh`) eliminating integer scale multiplies and quarter-rate conversions.
4. **`c2` (`d1ea74a`, `67df1fa`, cherry-picked `ed8b209`, `b1d2b1a`)**: Draft head vocab truncation via env `P104_MTP_DRAFT_VOCAB` (`96256` validated: `+2.80%` over `131072`, identical `draft_acc 2130/2637`).
5. **`p104-v14` (`baf7a5d..585eb84`, verified on P2000 GP106 & P104-100 GP104)**:
   - **Avenue A (`baf7a5d`, `mmvq.cu`)**: Column-outer `p104_mmvq_kloop_q4_col` for `Q4_K` at `ncols_dst == 3` (`-1.6%..-2.4%` P104-100, `-1.6%..-12.2%` P2000; `n=1,2` kept on `v13`).
   - **Avenue B (`84f6210`, `c7b` in `mmq-config-pascal-dp4a.cuh`)**: Added `I=64, J=128` Pascal dp4a MMQ tiles (`-6.18%..-16.69%` P104-100 `n=512`, `+6.26%/+6.40%` `pp512/pp2048` on `Qwen3.5-9B`; `+6.90%/+5.86%` on P2000 `Qwen3-8B`).
   - **Avenue C (`585eb84`, `gated_delta_net.cu`)**: Read-only `__ldg` loads + single `expf` in `KDA` (`-0.06%..-2.12%` P104-100 / `-0.65%..-7.14%` P2000 at `n_seq_tokens <= 512`).
   - **Excluded (`v13`)**: `c36`/`c6` (`q8_1` cache + `ADD+RMS_NORM+MUL` fusion, no MTP gain), `P104_GLU_FUSE=1` (`-14.1%` MTP), `c5` (`ac2fb21`, FA GQA `-7.6%`), `c8` (`dbaa2da`, CUDA graphs sm_61 no gain), `c4` (`Q5_K` `output.weight` failed same-top gate).

### Headline Verification Summary (P104-100, 140.00 W power cap enforced)
- **Correctness (`v13-bin`, `cd-v13-test`)**: `5403/5403` `test-backend-ops` (`MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION`) passed, `0 FAIL`.
- **KLD (`Qwen3.5-9B-Q4_K_M`, 5 chunks WikiText-2)**: `ub1 0.000609` (Same top `98.904%`), `ub2 0.000434` (`98.630%`), `ub4 -0.000000` (`100.000%`), `ub512 0.000493` (`98.630%`).
- **`llama-bench` ABBA vs `v11` (`cd-v13-abba`)**: `pp512@d0` `535.98 -> 581.57` (**+8.51%**), `pp2048@d0` `524.23 -> 568.79` (**+8.50%**), `pp512@d8192` `422.04 -> 452.44` (**+7.20%**), `pp2048@d8192` `417.24 -> 444.43` (**+6.52%**), `tg128@d0` `38.86 -> 38.91`.
- **Container Release ABBA (`p104-llama:67df1fa-20261126` vs `b098f6f-20261123`, `rel-abba.sh SOAK=1`)**:
  - 8-arm single-stream MTP: `38.09 -> 44.11 tok/s` (**+15.79%**: `chat +19.3%`, `code +20.0%`, `copy +15.0%`, `agent +4.2%`, `acc 0.813 -> 0.806`, drift `<= 0.66%`).
  - Concurrent `np=2`: `26.595 -> 29.60 tok/s` (**+11.30%**).
  - 16k NIAH + JSON schema: `pp 393.15 -> 418.22` (**+6.38%**), `tg 48.68 -> 57.86` (**+18.86%**), `VRAM 7369 MiB` (`+0 MiB`).
  - 20-minute soak (`rel-soak`): `213` requests over `1204s`, mean `57.05 tok/s`, max GPU temp `65 C`, `trip None`, `0` warn/error lines.

Full chronological receipts live in [LEDGER.md](file://~/p104-bench/LEDGER.md) and [cards/results/SUMMARY.md](file://~/p104-bench/cards/results/SUMMARY.md).

---

## 3. Multi-Pascal Hardware & Transfer Guide (`P104-100` + `P2000` + `P620`)

### Architecture Compatibility
All `llamascal` builds target `-DCMAKE_CUDA_ARCHITECTURES=61-real` (`sm_61` / `GGML_CUDA_CC_DP4A`). Dispatch checks in `mmvq.cu` (`MMVQ_PARAMETERS_PASCAL`) and `mmq.cu` / `mmq-vec-dot.cuh` activate automatically for all **`sm_61` GP10x Pascal GPUs**:
- **NVIDIA P104-100** (`GP104`, 20 SMs, 8 GiB GDDR5X, 140.00 W cap, `UUID=<gpu-uuid>`, PCIe Gen1 x4)
- **Quadro P2000** (`GP106`, 8 SMs, 5 GiB GDDR5, 75.00 W cap, `UUID=<gpu-uuid>`, PCIe Gen3 x16, 0 MiB Xorg)
- **Quadro P620** (`GP107`, 4 SMs, 2 GiB GDDR5, `UUID=<gpu-uuid>`, PCIe Gen3 x4, Xorg/GDM display)

### Slot-Independent Thermal Guards
`gguard.py`, `gguard-fleetreg.py`, and `container/run-experiment.sh` were updated to target GPUs by **UUID** (`GGUARD_GPU` / `CUDA_VISIBLE_DEVICES` / `UUID`) instead of the hardcoded `<pci-bus-id>` PCIe bus address, so moving P620 or adding P2000 will not break monitoring or hit the wrong GPU.

### Verified Cross-Card Transfer Receipt on Quadro P620 (`GP107`, `runs/p620-*`)
Executed `./pascal-transfer-bench.sh <gpu-uuid> p620` without touching P104-100 (`p104-svc` stayed online):
- **`test-backend-ops` (`q[456]_K`, `MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION`)**: **562/562 tests passed** (`max_temp=41 C`).
- **Operator Perf ABBA (`llamascal-bin` vs `v11-bin`, `max_temp <= 65 C`, `trip=null`)**:
  - `Q5_K 4096x4096`: `n=1`: `171.5 -> 168.3 us` (**-1.9%**); `n=2`: `228.6 -> 209.2 us` (**-8.5%**); `n=3`: `370.3 -> 277.6 us` (**-25.0%**)
  - `Q5_K 8192x4096`: `n=3`: `730.0 -> 538.9 us` (**-26.2%**)
  - `Q5_K 12288x4096`: `n=3`: `1090.5 -> 799.6 us` (**-26.7%**)
  - `Q6_K 4096x4096`: `n=1`: `246.1 -> 243.1 us` (**-1.2%**); `n=2`: `275.9 -> 266.3 us` (**-3.5%**); `n=3`: `361.1 -> 322.3 us` (**-10.7%**)
  - `Q6_K 12288x4096`: `n=3`: `1062.2 -> 936.5 us` (**-11.8%**)
  - `Q4_K`: unchanged (`v11` fast path retained).

Note: `test-backend-ops perf` sweeps small batch sizes (`n=1..3` matched by filter); to benchmark `c7a` prefill MMQ transfer on **P2000** / **P620**, use `llama-bench` with `-p 512,2048` on a model that fits VRAM.

### Checklist After Adding P2000 & Moving P620
1. Discover updated PCIe bus IDs and UUIDs:
   ```bash
   nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,pcie.link.gen.current,pcie.link.width.current,memory.total,power.limit --format=csv
   ```
2. Verify P104-100 power cap still reads `140.00 W` after reboot (`systemctl status p104-powercap.service`).
3. Run the cross-card operator test & ABBA on the new P2000:
   ```bash
   cd ~/p104-bench
   ./pascal-transfer-bench.sh <P2000_UUID> p2000
   ```
4. Measure end-to-end prefill + decode transfer on P2000 (5 GiB VRAM) and P620 (2 GiB VRAM) using a smaller model (e.g. Qwen3-4B / 1.5B `Q4_K_M` or `Q5_K_M`, setting `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<UUID> GGUARD_GPU=<UUID>` and `gguard.py`):
   ```bash
   # Compare v11-bin vs llamascal-bin on P2000/P620:
   LD_LIBRARY_PATH=$HOME/p104-exp/llamascal-bin flock ~/.local/state/lab-exp/gpu-bench-p2000.lock \
     python3 -I ~/p104-bench/gguard.py p2000-bench -- \
     $HOME/p104-exp/llamascal-bin/llama-bench -m <SMALL_MODEL_GGUF> -ngl 99 -fa 1 -ctk f16 -ctv f16 -ub 512 -t 4 -p 512,2048 -n 128 -d 0,4096 -r 3 -o jsonl
   ```
5. Check candidate tuning next (if continuing optimization across GP104/GP106/GP107):
   - Test `c7b` (`patches/archive-wip/c7b-i128-untested.patch`) on prefill MMQ (check register count / spills and `pp512`/`pp2048` across P104, P2000, P620).
   - Check multi-GPU tensor split / PCIe link utilization (`P2000` in full-speed slot vs `P620` in x4 slot vs `P104-100`).

---

## 4. Operating Rules & Guardrails
- **Never** increase the P104-100 `140.00 W` power cap or run `sudo`.
- **Never** run CPU builds (`make`, `docker build`) concurrently with GPU benchmarks — hold `~/.local/state/lab-exp/cpu-build.lock`.
- Hold per-GPU lock `~/.local/state/lab-exp/gpu-bench-<tag>.lock` and always run GPU jobs under `gguard.py` (host) or `container/run-experiment.sh` (container).
- Do not touch `p104-svc` (`:18100`) when benchmarking `P2000` or `P620` (pin target card via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<UUID>` and `GGUARD_GPU=<UUID>`).

---

## 5. GitHub Release (`llamascal`) Checklist

Once your GitHub token/remote is configured in `~/p104-exp/wt`:
```bash
cd ~/p104-exp/wt
git remote add llamascal https://github.com/<your-org-or-user>/llamascal.git
git push llamascal llamascal:main
git push llamascal p104-v11 p104-v13 p104-v14
```
(Optionally export `patches/llamascal-v14/*.patch`, `patches/v14-p104-bundle.patch`, and `LEDGER.md` / `cards/results/SUMMARY.md` as release notes.)
