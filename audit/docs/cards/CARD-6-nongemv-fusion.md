# CARD-6 non-GEMV decode bundle: GDN block fusion, add+rms_norm fusion, fewer small launches
Question: ~5 ms/token is not GEMV: GDN 0.69, SSM_CONV 0.22, SOFTPLUS 0.29, CONCAT 0.25, CPY 0.38, ADD 0.67, RMS_NORM 0.90, GET_ROWS 0.75, SIGMOID/SILU/FILL/ROPE/CONT/SET_ROWS ~0.1 each, gaps 1.38 [M k3-g3, graphs off]. Which are launch-bound and fusable on sm_61?
Hypothesis: (a) gated_delta_net decode reads+writes 2 MiB state/layer: bound 96 MiB/token -> 0.33 ms vs 0.69: ~0.3 ms by state-in-registers/vectorised float4 IO and fusing the state copy (ggml_cuda_try_gdn_cache_fusion exists: check that it fires on Qwen3.5 decode, ggml-cuda.cu:3591); (b) softplus+mul(alpha)+exp / sigmoid(beta) / concat feeding GDN fold into the GDN kernel prologue (~0.4 ms); (c) residual ADD + next RMS_NORM(+MUL) as one kernel (both read the same 16 KB; 64 pairs, ~0.3 ms); (d) GET_ROWS 0.75 ms: find which get_rows run per token besides the embedding (inp_out_ids, recurrent state s_copy) and fuse or skip the no-op ones (~0.3 ms). Total ~1-1.3 ms = +4-5% S if all land; expect half: +2-3%.
Files: `ggml/src/ggml-cuda/gated_delta_net.cu`, `ssm-conv.cu`, `norm.cu` (add+rms_norm+mul kernel), `getrows.cu`, `ggml-cuda.cu` fusion matcher (`ggml_cuda_can_fuse` ~4153-4262, graph loop ~4470-4830); read `src/models/qwen35.cpp` 375-500 for the exact op order. Each fusion behind its own env off-switch (`P104_NOFUSE_GDN`, `P104_NOFUSE_ADDNORM`) for A/B in one binary.
Control: v11-bin; plain Qwen3.5-9B f16 KV (llama-bench, KLD); MTP serving config (mtpab).
Gates: (1) dispatch: node/launch count per token drops (profiler from CARD-0 or GGML_CUDA_GRAPH_DEBUG node count), fused kernels visible; (2) test-backend-ops -o GATED_DELTA_NET,SSM_CONV,RMS_NORM,ADD,GET_ROWS plus the fusion tests pass vs CPU; (3) KLD ub1/2/4 within noise of v11; recurrent-state correctness through MTP rollback: temp-0 text identical to v11 on all 4 mtpab workloads; (4) speed: llama-bench tg d0 >= +1.5%, pp512 d0/d8192 >= -1%; mtpab M2 >= +1.5%; (5) thermal/VRAM.
Haiku benchmark (~45 GPU min):
```
cd ~/p104-bench; B=~/p104-exp/c6-bin
OPS=GATED_DELTA_NET,SSM_CONV,RMS_NORM,ADD,GET_ROWS,MUL_MAT_FUSION cards/bench/cardbench.sh c6 $B test,kld,bench,mtp
cat runs/cd-c6-mtp-summary.txt; python3 -I tools/abba.py cd-c6-abba
```
Budget: builder 4 h (do b and c first, then a, then d); build ggml-cuda ~8 min; GPU 45 min/iteration; max 2.
Bundling: same branch and builder as CARD-3 ("decode-fusion" series), benchmarked together as one bin (c36-bin) if both are ready in the same window; record each fusion's on/off via env in one extra mtp arm.
Stop rule: total < +1% after 2 iterations: keep only the sub-fusions that each show >= +0.5% with env on/off, drop the rest.
