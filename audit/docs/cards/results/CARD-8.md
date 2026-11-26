# CARD-8 receipt (builder B) — CUDA graphs on sm_61
Branch c8 (dbaa2da, off c36 9159353, base p104-v11). Patch script: ~/p104-exp/cardB-patches/c8-apply.py. Bin: ~/p104-exp/c8-bin.
## Why graphs are off (code reading)
ggml-cuda.cu ggml_cuda_graph_set_enabled(): when a graph key has no instances yet and cc < GGML_CUDA_CC_VOLTA it sets
graph->disable_due_to_gpu_arch (debug log "disabling CUDA graphs due to GPU architecture"). No feature check is involved:
capture, cudaGraphInstantiate and the update path are driver/runtime features available on sm_61. The cutoff is a
performance choice carried over from the upstream CUDA-graph introduction (validated on Ampere, later relaxed to Volta);
nobody measured Pascal. Other disable reasons (GGML_CUDA_DISABLE_GRAPHS, pool debug, per-graph compatibility checks) unchanged.
## Tried
- P104_CUDA_GRAPHS=1 skips the arch check (default off). P104_Q8C_STATS=1 adds cuda_graph_captures / cuda_graph_replays counters.
- Bench arms (cards/bench/c8bench.sh): c8 = graphs on, c36 paths off (graphs-only vs v11): test, dispatch stats tg64 on/off, kld ub1/2/4, llama-bench ABBA d0/d8192, mtpab np1 and np2; c8x = graphs + c36 + GLU fusion, mtp np1.

## Predeclared gate (orchestrator, written BEFORE any c8 numbers were read; 2026-11-24 ~23:10 CET)
- Dispatch: server log no longer prints "disabling CUDA graphs due to GPU architecture" with P104_CUDA_GRAPHS=1; graph replays counted > 0 (P104_Q8C_STATS).
- Op correctness: test-backend-ops unchanged (same pass counts as v11).
- Numerics: KLD vs unpatched at ub 1/2/4 within the noise floor (<= 0.0008 mean, same-top-p within 0.3 pp of the v11 value); MTP draft/verify and np2 outputs complete without capture errors/crashes.
- Speed (ABBA, llama-bench + mtpab M2): PASS if tg d0 >= +1.5% AND tg d8192 >= +1.0% AND pp512 d0 >= -1% AND MTP np1 wall >= +1.5%; no instability at -np 2 (no capture failures, no throughput drop > 2% vs v11 on the np2 arm). Anything between 0 and the threshold = INCONCLUSIVE (report, do not ship by default).
- Ship rule: a passing card ships as a default-on behaviour in v13 only if np1+MTP AND np2 both pass; otherwise keep it behind the env switch.

## Results (verdict analyst)
Receipts runs/cd-c8-driver.log, cd-c8-abba-*, cd-c8-mtp-summary.txt, cd-c8-mtpnp2-summary.txt, cd-c8x-mtp-summary.txt. Branch c8 dbaa2da, bin ~/p104-exp/c8-bin, P104_CUDA_GRAPHS=1.
- Dispatch: stats on: cuda_graph_captures=2 / replays=61 (tg64), 0 off; server logs 0 "disabling CUDA graphs" lines in B; MTP server counters captures 77, replays 5542 (np1). PASS.
- test-backend-ops 5007/5007 pass. KLD ub1 0.000667, ub2 0.000451, ub4 0 : identical to v11. PASS.
- llama-bench ABBA: tg d0 38.845 -> 38.965 (+0.31%), tg d8192 36.74 -> 36.875 (+0.37%), pp512 d0 535.83 -> 535.39 (-0.08%); drift <= 0.1%. tg64 single-run stats: graphs on 38.33 vs off 38.93 (-1.5%).
- mtpab M2 np1: A 37.84, B 37.795 tok/s (-0.12%), draft_acc identical, text 16/16 identical, drift A 0.90% B 0.08%.
- np2 (mtpnp2): A 26.55, B 26.26 (-1.09%), drift B 2.48% (B1 17.60 agent vs B2 18.47), text identical 7/8 (one mismatch, not investigated; A-vs-A np2 nondeterminism possible), no capture failures seen. 
- c8x (graphs + c36 + GLU) np1 MTP: 37.705 -> 32.675 (-13.3%), caused by GLU fusion (see CARD-3).
## Verdict
Predeclared: tg d0 >= +1.5% (got +0.31%) FAIL; tg d8192 >= +1.0% (got +0.37%) FAIL; pp >= -1% OK; MTP np1 >= +1.5% (got -0.12%) FAIL; np2 no drop > 2%: OK (-1.09%) but noisy. Overall FAIL. Graphs give no measurable gain on Pascal here (launch overhead is already hidden, host gap 1.3 ms is not the bound). Ships: NO (leave P104_CUDA_GRAPHS unset).
