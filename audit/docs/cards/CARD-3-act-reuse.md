# CARD-3 activation reuse: quantise each activation to q8_1 once, fuse norm into the quantise, enable gate/up GLU fusion on Pascal
Question: Every MUL_MAT re-quantises its src1 (mmvq.cu:1996-2001 `quantize_row_q8_1_cuda` per call). Consumers of the same tensor: in 24 GDN layers wqkv, wqkv_gate (z), ssm_beta, ssm_alpha (qwen35.cpp:265-394); in 32 layers ffn gate+up; in 8 attention layers wq/wk/wv. Gate/up+GLU MMVQ fusion is turned off for cc < Volta (ggml-cuda.cu:2349 "fusion is not universally faster on Pascal"), but that verdict came from upstream kernels, not ours.
Hypothesis: ~120 redundant quantise launches/token x ~3 us = ~0.36 ms, plus 32 SWIGLU kernels and 32 extra GEMV launches removed by gate/up fusion (~0.15 ms), plus rms_norm(+mul)->q8_1 direct output where the norm output only feeds matmuls (~64 launches, ~0.15 ms). Total ~0.5-0.7 ms/token = +2-3% S decode; the same saving per verify step (3 columns) and per draft. Prefill: quantize_mmq_q8_1 also repeats per matmul (4 MB/call at ub 512); share it too (~1% pp).
Files: `ggml/src/ggml-cuda/ggml-cuda.cu` graph loop (~4470-4830): detect runs of MUL_MAT nodes whose src1 is the same tensor (same data ptr, same ne/nb, no node in between writes to src1's buffer), quantise once into a pool buffer that lives for the run, and pass it to `ggml_cuda_mul_mat_vec_q` / mmq through a new optional arg (mmvq.cu:1990-2010, mmq.cu quantize path). `ggml_cuda_should_fuse_mul_mat_vec_q` (2336-2362): allow cc 6.1 for ncols 1..3 behind env `P104_GLU_FUSE=1`, and check that the fused path dispatches our wide vec_dot (mmvq.cu:765-790) rather than the generic one. norm.cu: optional rms_norm_mul -> q8_1 output when every consumer is MMVQ (phase 2, only if phase 1 passes).
Control: v11-bin; plain Qwen3.5-9B Q4_K_M f16 KV for llama-bench/KLD; MTP GGUF serving config for mtpab.
Gates: (1) dispatch: profiler (CARD-0 bin rebased) or `GGML_SCHED`/node count shows quantise launches/token drop from ~190 to ~70 and fused GLU nodes; CUDA graph still captured (GGML_CUDA_GRAPH_DEBUG); (2) test-backend-ops MUL_MAT, MUL_MAT_FUSION q4_K/q5_K/q6_K pass vs CPU; shared quantise must be bit-identical to per-call quantise (same kernel, same input), so KLD should match v11 exactly at ub1; (3) KLD ub1/2/4 within noise of v11 (ub1 0.000667, ub2 0.000451; |delta| <= 0.0002); (4) speed: llama-bench tg d0/d8192 >= +1.5%, pp512 not slower than -1%; mtpab M2 >= +1.5%; (5) thermal/VRAM.
Haiku benchmark (~45 GPU min):
```
cd ~/p104-bench; B=~/p104-exp/c3-bin
OPS=MUL_MAT,MUL_MAT_FUSION OPP="q[456]_K" BARGS="" cards/bench/cardbench.sh c3 $B test,kld,bench,mtp
# GLU fusion arm (env knob) only for the decode bench:
MTPB_ENV=P104_GLU_FUSE=1 cards/bench/cardbench.sh c3g $B mtp
cat runs/cd-c3-mtp-summary.txt runs/cd-c3g-mtp-summary.txt; python3 -I tools/abba.py cd-c3-abba
```
(llama-bench B arm with the GLU fusion: run `P104_GLU_FUSE=1 cards/bench/cardbench.sh c3gb $B bench`; the env is inherited by gguard.)
Budget: builder 3-4 h; build ggml-cuda (~8 min incremental -j2); GPU 45 min/iteration; max 2.
Bundling: kernel/graph series "decode-fusion" with CARD-6 (same file, same builder, one branch); merge before CARD-5.
Stop rule: < +1% tg after 2 iterations, or any KLD outside noise with bit-identical claim broken: drop the shared-quantise part, keep GLU fusion only if it alone passes.
