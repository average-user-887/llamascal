# CARD-6 results: ADD+RMS_NORM+MUL fusion
Same branch/bin as CARD-3 (c36 9159353, ~/p104-exp/c36-bin; env P104_NOFUSE_ADDNORM=1 disables). Receipts as in CARD-3.md. Fusion fired: addnorm_fused=2079 over 33 graphs (default arm).
## Results
Combined c36 default arm vs v11 (ABBA, llama-bench): tg d0 +1.64% (38.835 -> 39.47), tg d8192 +1.63%, pp512 d0 +0.36%, pp512 d8192 +0.28% (A d8192 drift 1.09% from one pp outlier +-8). test-backend-ops 4797/4797, 4326/4326 pass. KLD ub1/2/4 identical to v11. temp-0 text on mtpab: only measured with GLU arm (identical 16/16, acceptance identical). mtpab M2 for the default arm: NOT RUN. No separate ADD-only ABBA exists, so the 1.64% cannot be attributed between card 3 and card 6.
## Verdict (thresholds: tg d0 >= +1.5%, pp >= -1%, mtpab M2 >= +1.5%)
tg d0 +1.64% >= +1.5% : met (margin 0.14 pt, drift 0.18%); pp >= -1% : met; mtpab M2 : NOT RUN. INCONCLUSIVE (two of three speed gates met, third missing; gain not isolated from card 3). Do not claim card 6 alone.

## Builder notes (re-created by the analyst; original builder draft overwritten in error)
Branch c36 (same as CARD-3), patch script ~/p104-exp/cardB-patches/c6-apply.py. Already fused in v11: RMS_NORM+MUL(+ADD), RMS_NORM+MUL+ROPE, SSM_CONV+SILU, SOFTPLUS+MUL, GDN + state CPY. New: residual ADD -> RMS_NORM -> MUL kernel (norm.cu p104_add_rms_norm_mul_f32, off: P104_NOFUSE_ADDNORM=1). The recurrent-state GET_ROWS gather (llama-graph.cpp:4119) was identified as a further lever but is not implemented; profile shows GET_ROWS only 0.66 ms/token (c0b), so it is at most 2.3%.
