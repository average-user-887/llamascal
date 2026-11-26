# Overnight card verdicts 2026-11-25 (verdict analyst; thresholds predeclared in each card)
| Card | Change | Gates (1 disp / 2 ops / 3 numerics / 4 speed / 5 vram+thermal) | Key number | Verdict | Ships |
|---|---|---|---|---|---|
| 0 | A depth/KV curve + B profiler | B tables n512/n1 + QUANT_Q8_1 ok; unprofiled tg check not run | q8_0 KV tg 0.918x f16 @16k, 0.886x @32k; prefill ~85% MMQ; draft LM head 76% of draft graph | PASS (measurement) | n/a |
| 1 | `-bs` backend sampling | pass / pass (16/16 text) / n.a. / M2 +6.0%, S +4.3% / VRAM +0, 56 C | M2 wall 37.49 -> 39.74 | PASS | serving flag `-bs` (no code) |
| 2 | MTP draft vocab prefix (env P104_MTP_DRAFT_VOCAB) | ok / 16/16 text / ok / N=96256 42.90 vs N=131072 41.73 (+2.80%, acc identical 2130/2637); N=65536 acc -3.4 pp FAIL / VRAM +0 | 38.00 -> 42.90 (with -bs, N=96256) | PASS at 96256; 65536 FAIL | env P104_MTP_DRAFT_VOCAB=96256 |
| 3 | q8_1 shared cache (+GLU fusion opt-in) | ok / 4797 pass, KLD bit-identical / MTP ABBA: c36bs 39.94/39.15 vs 40.00/40.00, c6only 39.72 vs 39.99 (-0.68%); GLU -14.1% FAIL | no MTP speedup | EXCLUDED | no |
| 4 | output.weight Q5_K | KLD mean ok (0.0021) but same-top 96.9/97.6/97.8% < 98.0% FAIL / speed tg +3.4%, M2 +3.5% PASS | tg 38.85 -> 40.15; 8/16 MTP text differs | FAIL (numerics gate) | no (needs human quality OK; model file only) |
| 5 | GQA-packed VEC FA for q8_0/q4_0 KV | 4498 tests pass / KLD <= 0.00046 / q8_0 tg d16k -7.6% (needs +6%), d0 -1.5% / VRAM 8101 of 8192 MiB | q8_0 d16k 31.98 -> 29.55 | FAIL | no |
| 6 | ADD+RMS_NORM+MUL fusion | fired 2079x / pass / KLD identical / c6only MTP ABBA 39.72 vs 39.99 (-0.68%) | -0.68% MTP | EXCLUDED | no |
| 7 | prefill MMQ (c7a `361199d`) | 2724+1077 tests pass / KLD ub512 0.000493 / pp512@d0 +8.5%, pp2048@d0 +8.5%, @d8192 +5.3%..+7.2% / max 60 C | pp2048@d0 524.23 -> 568.79 | PASS | yes (`p104-v13`) |
| 8 | CUDA graphs sm_61 (P104_CUDA_GRAPHS=1) | pass / 5007 pass / KLD identical / tg d0 +0.31%, d8192 +0.37%, M2 -0.12%, np2 -1.09% | tg 38.85 -> 38.97 | FAIL | no |
| k4 | MMVQ Q5_K sumi_m hoist + Q6_K presub (`MINCOLS=2`) & magic float | 5403/5403 tests pass / KLD ub1 0.000609, ub2 0.000434 / op Q5_K n=2 -8.7%, n=3 -14.4%, Q6_K -2.1%..-8.0% | combined v13 rel-ABBA 38.09 -> 44.11 tok/s (+15.79%) | PASS (Q5_K + Q6_K) | yes (`p104-v13`) |

## Promoted `p104-v13` (`67df1fa`, live container `p104-llama:67df1fa-20261126`)
- Merged: `-bs` serving flag, `c2` (`ed8b209`, `b1d2b1a`) with `P104_MTP_DRAFT_VOCAB=96256`, `c7a` (`361199d`) dp4a MMQ, and `k4` MMVQ `Q5_K` + `Q6_K` (`8845a20`; `Q4_K` excluded from `p104_mmvq_kloop_q45` after operator ABBA regression on `k=12288 n=3`).
- Excluded: `c36`/`c6` (regressed MTP), `c5`, `c8`, `GLU` fusion, `c4` GGUF.
- Container release ABBA (`p104-llama:67df1fa-20261126` vs `b098f6f-20261123`, `SOAK=1`): 8-arm MTP `38.09 -> 44.11 tok/s` (`+15.79%`: `chat +19.3%`, `code +20.0%`, `copy +15.0%`, `agent +4.2%`, `acc 0.813->0.806`, greedy prompt diffs `6/16` benign FP-reordering synonyms, drift `<= 0.66%`); `np=2` `26.60 -> 29.60 tok/s` (`+11.30%`); 16k NIAH + JSON schema `pp 393.15 -> 418.22 (+6.38%)`, `tg 48.68 -> 57.86 (+18.86%)`, `VRAM 7369 MiB (+0)`; 20m soak `213` reqs mean `57.05 tok/s`, max `65 C`, `trip None`, 0 warn/error lines. Power cap `140.00 W` intact.
## Suspicious / flags
- ABBA drift > 1%: card 1 M2 A1 vs A2 1.45%; card 2 N=131072 0.98% on both arms (edge), card 4 pp d0 A 1.5%, card 8 np2 B 2.48%. Effects reported above exceed their drift except card 2 N=131072 margin.
- Text mismatches: card 5 mtp 9/16 and card 4 8/16 (expected for numerics change); card 8 np2 7/8 (unexplained, no A-vs-A control).
- VRAM: card 5 q8_0 -c 98304 peaked 8101 of 8192 MiB (both arms). Serving config in mtp arms 7369 MiB.
- Thermal: max 60 C in all receipts, no trips.
- CARD-0A pp2048 d0 reports exactly the same value as pp512 (526.74): likely a duplicated row. Gate 2 of card 0 (unprofiled c0prof tg within 1%) and the card-1 sampled (temp 0.7) arm were not verified.
- I overwrote the builders' CARD-3.md and CARD-6.md drafts by mistake; their key content is re-created in the 'Builder notes' sections.

---

## Multi-Pascal Cross-Card Transfer & `p104-v14` (`585eb84`) Validation (2026-11-26)

### 3-Chip Pascal Transfer Table (`5bacb7c` stock vs `v11` vs `v13` vs `v14`)
- **P104-100 (`GP104`, 20 SMs, GDDR5X, 140 W cap)**:
  - Prefill `Qwen3.5-9B-Q4_K_M`: `v11 535.98` -> `v13 581.57 (+8.51%)` -> **`v14 620.68 tok/s` (`+6.26%` vs `v13`, `+15.8%` vs `v11`)** at `pp512@d0`; `v11 524.23` -> `v13 568.79 (+8.50%)` -> **`v14 607.35 tok/s` (`+6.40%` vs `v13`, `+15.9%` vs `v11`)** at `pp2048@d0`; `tg128@d0` steady (`39.01 -> 39.06`).
  - Ops vs `v13`: `Q4_K n=3` `-1.60%..-2.37%` (`248320x4096` `-1.85%`); `n=512` `c7b` (`4096x14336`) `q4_K -6.18%`, `q5_K -6.08%`, `q6_K -16.69%`; `GATED_DELTA_NET` (`n<=512`) `-0.06%..-2.12%`.
- **Quadro P2000 (`GP106`, 8 SMs, GDDR5, 75 W cap, 5 GiB VRAM)**:
  - Ops (`m=4096, k=14336`, `base -> v11 -> v13 -> v14`): `q4_K n=1/2/3` `372.4/400.1/486.3 -> 269.0/332.3/391.3`, `n=512` `13977.6 -> 13903.5 -> 12470.7 (c7a, -10.3%) -> 11496.4 (c7b, -14.3% vs v13)`; `q5_K n=512` `14609.2 -> 14614.1 -> 13380.1 (-8.4%) -> 12353.0 (-16.0% vs v13)`; `q6_K n=512` `15815.9 -> 15829.9 -> 13763.6 (-13.1%) -> 11391.5 (-15.9% vs v13)`. `p2000-op` `v13` vs `v11`: `Q5_K n=2/3 -2.1%..-2.6% / -13.3%..-14.0%`, `Q6_K n=2/3 -2.1%..-2.4% / -5.8%..-6.3%`; `v14` `Q4_K n=3` `-1.60%..-3.23%` (`248320x4096 -12.18%`).
  - End-to-end `Qwen3-8B-Q4_K_M`: `-ngl 99` (`pp512`) `270.61 (base) -> 271.68 (v11) -> 300.45 (v13, +10.59%)`, `tg128 17.80 -> 23.92 (+34.3%)`; `-ngl 33` `pp512 257.95 -> 258.13 -> 284.92 (v13) -> 302.43 (v14, +6.90% vs v13)`, `pp2048 233.75 -> 233.66 -> 255.60 (v13) -> 270.71 (v14, +5.86% vs v13)`.
- **Quadro P620 (`GP107`, 4 SMs, GDDR5, 2 GiB VRAM, Xorg display)**:
  - Ops (`m=4096, k=14336`, `base -> v11 -> v13`): `q4_K n=1/2/3/512` `786.5/804.3/994.6/32899.7 -> 466.3/673.3/842.4/32772.6 -> 466.1/673.2/842.1/30450.6` (`n=512 -7.1%`); `q5_K` `968.9/963.7/1137.3/35532.1 -> 572.7/733.6/996.5/35494.4 -> 571.0/750.2/989.2/32600.2` (`-8.2%`); `q6_K` `1145.9/1142.2/1325.8/35429.2 -> 853.2/899.0/1004.9/35445.4 -> 852.1/893.8/982.5/32771.7` (`-7.5%`). Plus `p620-op` `Q5_K n=3 -25.0%..-26.7%`, `Q6_K n=3 -10.7%..-11.8%`.

### Promoted Code Tag `p104-v14` (`585eb84`, `llamascal-bin -> v14-bin`, host binaries ready)
- Added 3 validated commits on top of `p104-v13` (`67df1fa`):
  1. `baf7a5d` (Avenue A): `p104_mmvq_kloop_q4_col` column-outer MMVQ for `Q4_K` at `ncols_dst == 3` (`-1.6%..-2.4%` P104-100, `-1.6%..-12.2%` P2000).
  2. `84f6210` (Avenue B / `c7b`): `I=64, J=128` Pascal dp4a MMQ tiles (`mmq-config-pascal-dp4a.cuh`), `+6.26%/+6.40%` `pp512/pp2048` on P104-100, `+6.90%/+5.86%` on P2000.
  3. `585eb84` (Avenue C): `gated_delta_net_cuda` `__ldg` loads + single `expf` in `KDA` (`-0.06%..-2.12%` P104-100 / `-0.65%..-7.14%` P2000 for `n<=512`).
- P104-100 gates: `test-backend-ops` 2/2 PASS (`MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION,GATED_DELTA_NET`); KLD `ub1 0.000609 (98.904%)`, `ub2 0.000434 (98.630%)`, `ub3 0.000521 (98.748%)`, `ub4 -0.000000 (100.000%)`, `ub512 0.000493 (98.630%)`; `p104-svc` restarted healthy (`:18100/health` `{"status":"ok"}`, still running `p104-llama:67df1fa-20261126`), `140.00 W` cap intact.
