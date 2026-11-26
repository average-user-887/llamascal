# `llamascal` Audit Bundle (Pascal `sm_61`: P104-100 / Quadro P2000 / Quadro P620)

This directory contains the sanitized documentation, exported patch series, thermal/benchmark scripts, binary hash records, raw text benchmark receipts, and SHA-256 manifest for auditing the `llamascal` CUDA optimization work on NVIDIA Pascal (`sm_61` / `GGML_CUDA_CC_DP4A`) GPUs.

## Repository Branches & Tags

| Ref | Commit | Description |
|---|---|---|
| `upstream-base` | `5bacb7c868a2bc98682de15f447fc1ce17c9c49c` | Upstream `ggml-org/llama.cpp` baseline (`5bacb7c`). |
| `p104-v11` | `b098f6f564900f6fa9798d85694b0a6eb4bc2f62` | `k1` + `k3`: Pascal `MMVQ_PARAMETERS_PASCAL` decode kernels (`Q4_K`/`Q5_K`/`Q6_K`) and per-op CUDA event attribution hooks. |
| `p104-v13` | `67df1facf36aca85e0b19d9f75e8b0868ac550aa` | `v11` + `k4` (`Q5_K`/`Q6_K` MMVQ) + `c7a` (Pascal dp4a MMQ prefill) + `c2` (`P104_MTP_DRAFT_VOCAB`). |
| `p104-v14` | `585eb84d0cd95b3126172754364514278e4c25a7` | **Latest promoted release**: `v13` + `baf7a5d` (Avenue A: column-outer `Q4_K` MMVQ at `ncols_dst == 3`) + `84f6210` (Avenue B / `c7b`: `I=64, J=128` Pascal dp4a MMQ tiles) + `585eb84` (Avenue C: `gated_delta_net` `__ldg` loads + single `expf` in KDA). |
| `llamascal` | `e2ccb81536c3dd6ab5cee23938a31fddbd05780c` | Pure code branch: `p104-v14` plus 2 WIP `v15` commits (`ce70cb5`, `e2ccb81`) whose P104-100 validation (`v15-p104-*`) was in progress at snapshot time. |
| `main` | — | `llamascal` code tree plus this `audit/` directory. |

## Hardware Tested

- **NVIDIA P104-100** (`GP104`, `sm_61`, 20 SMs, 8 GiB GDDR5X, 140.00 W power cap enforced)
- **NVIDIA Quadro P2000** (`GP106`, `sm_61`, 8 SMs, 5 GiB GDDR5, 75.00 W power cap)
- **NVIDIA Quadro P620** (`GP107`, `sm_61`, 4 SMs, 2 GiB GDDR5)

## Headline Verified Results

### `p104-v14` (`585eb84`) vs `p104-v13` (`67df1fa`)
- **P104-100 (140.00 W cap)**:
  - Correctness: `test-backend-ops` (`MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION,GATED_DELTA_NET`) 2/2 PASS.
  - KLD (`Qwen3.5-9B-Q4_K_M`, 5 chunks WikiText-2): `ub1 0.000609` (`98.904%` same top), `ub2 0.000434` (`98.630%`), `ub3 0.000521` (`98.748%`), `ub4 -0.000000` (`100.000%`), `ub512 0.000493` (`98.630%`).
  - Prefill / Decode (`Qwen3.5-9B-Q4_K_M`): `pp512@d0` `584.11 -> 620.68 tok/s` (**+6.26%** vs `v13`), `pp2048@d0` `570.79 -> 607.35 tok/s` (**+6.40%** vs `v13`), `tg128@d0` `39.01 -> 39.06 tok/s`.
  - Operators: `Q4_K n=3` `-1.60%..-2.37%` (`248320x4096 -1.85%`); `n=512` (`4096x14336`) `q4_K -6.18%`, `q5_K -6.08%`, `q6_K -16.69%`; `GATED_DELTA_NET` (`n<=512`) `-0.06%..-2.12%`.
- **Quadro P2000**:
  - `Qwen3-8B-Q4_K_M` (`-ngl 33`): `pp512@d0` `282.91 -> 302.43 tok/s` (**+6.90%** vs `v13`), `pp2048@d0` `255.72 -> 270.71 tok/s` (**+5.86%** vs `v13`).
  - Operators: `Q4_K n=3` `-1.60%..-3.23%` (`248320x4096 -12.18%`); `n=512` (`4096x14336`) `-14.30%..-15.98%`.

### `p104-v13` (`67df1fa`) Baseline Gates (P104-100 & Quadro P620)
- **P104-100**:
  - `test-backend-ops`: `5403/5403` passed, `0 FAIL`.
  - Container Release ABBA vs `v11` (`-bs` + `P104_MTP_DRAFT_VOCAB=96256`): 8-arm single-stream MTP `38.09 -> 44.11 tok/s` (**+15.79%**); concurrent `np=2` `26.60 -> 29.60 tok/s` (**+11.30%**); 16k NIAH + JSON schema `pp 393.15 -> 418.22` (`+6.38%`), `tg 48.68 -> 57.86` (`+18.86%`), `VRAM 7369 MiB (+0 MiB)`; 20-minute soak `213` requests, mean `57.05 tok/s`, max `65 C`, `trip None`.
- **Quadro P620 (`p104-v13` transfer)**:
  - `test-backend-ops` (`q[456]_K`): `562/562` passed (`max 41 C`); `Q5_K n=3` `-25.0%..-26.7%`, `Q6_K n=3` `-10.7%..-11.8%`, `n=512` `q[456]_K` `-7.1%..-8.2%` vs `v11`.

See [`docs/LEDGER.md`](docs/LEDGER.md) and [`docs/CARDS-SUMMARY.md`](docs/CARDS-SUMMARY.md) for the complete chronological record.

## Layout

- `docs/` — `LEDGER.md`, `CARDS-SUMMARY.md`, `FINAL-REPORT.md`, `REVIEW-K1.md`, `PLACEMENT-ANALYSIS.md`, `HANDOFF-LLAMASCAL-ORCHESTRATOR.md`, and `cards/` (`CARD-*` specs & result reports).
- `patches/` — Single-file diffs (`k3-v11-mmvq-pascal-kquant.patch`, `v13-*.patch`, `v14-*.patch`) and `git format-patch` series (`llamascal-v13/`, `llamascal-v14/`).
- `scripts/` — `gguard.py` (thermal/slowdown guard), `bench.py`, `pascal-transfer-bench.sh`, and `container/` build/serving scripts.
- `receipts/hashes/` — Binary SHA-256 checksums (`v11-HASHES.txt`, `v13-HASHES.txt`, `v14-HASHES.txt`, `v15-HASHES.txt`).
- `receipts/runs/` — Sanitized raw `.out`, `.meta.json`, `.trace.csv`, `.log`, `.json`, and `.summary.log` run receipts.
- `MANIFEST.sha256` — SHA-256 manifest for all files in `audit/`.

## Integrity Verification & Build Reproduction

Verify all audit files against the manifest:

```bash
cd audit
sha256sum -c MANIFEST.sha256
```

Build from any release tag (for example `p104-v14`) targeting Pascal `sm_61`:

```bash
git checkout p104-v14
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-cli llama-server llama-bench llama-perplexity test-backend-ops
```

## Sanitization & Exclusions

- Local paths (`/home/<user>`), usernames, GPU UUIDs (`<gpu-uuid>`), PCIe bus IDs (`<pci-bus-id>`), internal LAN IPs (`<lan-ip>`), and internal orchestrator references were redacted prior to computing `MANIFEST.sha256`.
- Private message payloads (`private/`), binary Nsight Systems traces (`*.qdstrm`), and external dataset files (`nq-data/wiki.test.raw`) are excluded.
- `INVALID-*` receipts are retained for completeness and are not performance evidence.
