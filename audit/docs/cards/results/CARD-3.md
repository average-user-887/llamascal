# CARD-3 results: shared q8_1 activation cache (+ opt-in Pascal GLU fusion)
Branch c36 9159353 (cardB), bin ~/p104-exp/c36-bin, receipts runs/cd-c36-driver.log, cd-c36-abba-*, cd-c36g-*, cd-c36-stats-*. Note: c36 default arm contains BOTH card 3 (q8 cache) and card 6 (ADD+RMS_NORM+MUL); the ABBA cannot separate them (P104_NO_Q8CACHE / P104_NOFUSE_ADDNORM arms were not run in ABBA).
## Results
- Dispatch (stats tg32): default arm q8_hits=3960 / misses=4257 over 33 graphs, addnorm_fused=2079; GLU arm mmvq_glu_fused=1056, q8_hits 2904; off arm all zeros. tg32 default 39.44, GLU 38.64, off 38.89 tok/s (single runs).
- test-backend-ops: 4797/4797 and 4326/4326 passed (first line is the card set).
- KLD vs unpatched: ub1 0.000667 +-0.000180, same-top 98.787%; ub2 0.000451 +-0.000016, 98.865%; ub4 0.000000 (base itself) - identical to v11 in A and B arms (bit-identical).
- llama-bench ABBA (default arm, c36 vs v11): tg128 d0 38.835 -> 39.47 (+1.64%), d8192 36.76 -> 37.36 (+1.63%), pp512 d0 +0.36%, pp512 d8192 +0.28%. Drift A 0.18%, B 0.15% (d0 tg).
- GLU fusion arm (P104_GLU_FUSE=1; runs/cd-c36g-*): llama-bench tg d0 38.87 -> 38.59 (-0.72%), d8192 -0.63%; MTP M2 wall (cd-c36g-mtp-summary.txt) 37.86 -> 32.53 tok/s (-14.1%), acceptance identical 2136/2627, text identical 16/16. The c8x arm (graphs+c36+GLU) shows the same -13.3%.
- MTP M2 with the default c36 arm (no GLU): NOT RUN (c36g was GLU only).
## Verdict
Gates 1-3 PASS. Gate 4 (card text: tg gain, mtpab M2): default arm tg +1.64% (S decode target +2-3% not reached; hypothesis was 0.5-0.7 ms/token = 2-3%), MTP M2 NOT MEASURED for the default arm -> INCONCLUSIVE for ship-on-MTP; the q8-cache/addnorm combination is safe numerically. GLU fusion: FAIL (slower: -0.7% tg, -14% MTP); keep P104_GLU_FUSE off, do not enable.
Ships: q8 cache default-on only after an MTP M2 ABBA of the default c36 arm (queue it); it is no-risk numerically (bit-identical KLD). Until then env-switchable (P104_NO_Q8CACHE exists).

## Builder notes (re-created by the analyst; the builder's original CARD-3.md draft was overwritten in error, content below is from what had been read)
Branch c36 in ~/p104-exp/cardB (base c0prof e318c44 on p104-v11 b098f6f); patch script ~/p104-exp/cardB-patches/c3-apply.py. Iteration 1: q8_1 cache in mmvq.cu (per backend-context 2-slot persistent buffer, key = src1 data ptr + ne + nb, cleared at every graph_compute, slot dropped when any executed node writes overlapping memory; no cache on concurrent streams, MUL_MAT_ID, >8 cols; off: P104_NO_Q8CACHE=1; counters P104_Q8C_STATS=1). GLU fusion on cc 6.1 behind P104_GLU_FUSE=1 (P104_GLU_FUSE_MAXN default 3), fused MMVQ for Q4_K/Q5_K/Q6_K at 2..3 columns.
