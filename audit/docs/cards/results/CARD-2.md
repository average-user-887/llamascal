# CARD-2 results: reduced MTP draft vocabulary (P104_MTP_DRAFT_VOCAB=N)
Branch c2 b1d2b1a (also in c25 57111de = c2 + c5 merge), bin ~/p104-exp/c2-bin; A = v11-bin, B = c2-bin with env, MTP Qwen3.5-9B-MTP Q4_K_M, n_max 2, np1, mtpab ABBA (receipts runs/cd-c2-mtp-summary.txt N=65536, runs/cd-c2v131k-mtp-summary.txt N=131072, queueA-1.log).
## Results (wall tok/s, workload 'all')
| N | A1 | A2 | B1 | B2 | mean A | mean B | delta | draft_acc A | draft_acc B | text |
|---|---|---|---|---|---|---|---|---|---|---|
| 65536 | 37.89 | 37.80 | 40.16 | 40.27 | 37.845 | 40.215 | +6.26% | 2136/2627 = 81.3% | 2098/2694 = 77.9% (-3.4 pp) | 16/16 identical |
| 131072 | 37.99 | 37.62 | 39.49 | 39.88 | 37.805 | 39.685 | +4.97% | 81.3% | 2130/2637 = 80.8% (-0.5 pp) | 16/16 identical |
Per workload N=65536 (mean B/A): agent +2.6%, chat +7.8%, code +8.0%, copy +4.3%; N=131072: agent +1.5%, chat +6.3%, code +5.7%, copy +3.4% (derived from the summaries).
ABBA drift (A1 vs A2 / B1 vs B2): 65536: 0.24% / 0.27%; 131072: 0.98% / 0.98% (at the 1% edge). VRAM max 7369 MiB in A and B (delta 0, gate < +150 MiB). Max temp 57 / 59 C. test-backend-ops step: CPU-only 'OK' (no kernel change; trivial). The n_max 3 arm and the tokenizer-coverage output (cards/bench/c2cov.sh, runs/c2cov-*.ids) were not scored.
## Verdict
Gate 1 dispatch: log line added in b1d2b1a; not independently verified by me (speed and acceptance change prove it is active). Gate 2 text identical: PASS (16/16 both). Gate 4 threshold (mean M2 >= +4% AND acceptance drop <= 3 pp):
- N=65536: speed +6.26% >= +4% OK, acceptance drop 3.4 pp > 3 pp -> FAIL (strict, conjunctive gate) despite higher wall speed.
- N=131072: +4.97% >= +4% OK, drop 0.5 pp <= 3 pp -> PASS. Margin over threshold is only 0.97 pt with drift 0.98% -> pass is real but narrow; recommend re-run before default-on.
Overall: PASS for N=131072 (env switch P104_MTP_DRAFT_VOCAB=131072), FAIL by strict gate for N=65536. Ships: env switch (default off) at 131072; agent workload gains least (+1.5%).
