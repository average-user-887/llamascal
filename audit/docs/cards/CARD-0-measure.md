# CARD-0 measure: fill the [?] cells of STAGE-MAP before GPU time goes to cards 1-7
Question: How much does a token cost in logits D2H/host sampling, the MTP draft, the q8_1 quantise and with CUDA graphs in the serving path? How does prefill split per op? How does quantised KV scale with depth?
Hypothesis: D2H ~1.2 ms/output (0.99 MB at ~0.8 GB/s, PCIe 1.0 x4); draft ~4.5-5 ms/draft token, of which the LM head is 3.45 ms; quantise ~0.5 ms/token; q8_0 KV tg loses ~8% at 16k and more at 32k (VEC reads K/V 4x); prefill is 65-80% MMQ.
Part A (no build, run first): `cards/bench/card0a.sh` (pp512/2048 d0 x5 re-measures the queue's 457.8 outlier; f16/q8_0/q4_0 KV at d16k (r2) and f16/q8_0 at d32k (r1), tg64 + pp512; CUDA graph debug log under the MTP serving config).
Part B (builder, ~1 h CPU, in builder B's tree, build target ggml-cuda only): extend the event profiler (b098f6f, GGML_P104_OPPROF_BUILD) so that (1) it profiles every graph, not only decode, keyed by the graph's n_tokens (ubatch size), which separates prefill (512), draft (1, ctx_dft) and verify (3); (2) the q8_1 quantise inside mul_mat gets its own event label "QUANT_Q8_1"; (3) it prints wall time between graph_compute calls (host gap) per n_tokens. Output to ~/p104-exp/c0-prof-bin.
Part B run (Haiku, ~15 GPU min): 
```
cd ~/p104-bench; B=~/p104-exp/c0-prof-bin; L=~/.local/state/lab-exp; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-uuid>
flock $L/cpu-build.lock sg docker -c "container/gpu-window.sh stop"
M=~/models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf; G="flock $L/gpu-bench-p104.lock python3 -I gguard.py"
LD_LIBRARY_PATH=$B $G cd-c0b-pp -- env GGML_CUDA_DISABLE_GRAPHS=1 GGML_P104_OPPROF=1 $B/llama-bench -m $M -ngl 99 -fa 1 -ctk f16 -ctv f16 -ub 512 -t 4 -p 512 -n 0 -d 0,8192 -r 1
LD_LIBRARY_PATH=$B $G cd-c0b-tg -- env GGML_CUDA_DISABLE_GRAPHS=1 GGML_P104_OPPROF=1 $B/llama-bench -m $M -ngl 99 -fa 1 -ctk f16 -ctv f16 -ub 512 -t 4 -p 0 -n 64 -d 0 -r 1
python3 -I cards/bench/mtpab.py cd-c0b-mtp $B $B --order A --a-env GGML_CUDA_DISABLE_GRAPHS=1,GGML_P104_OPPROF=1
sg docker -c "container/gpu-window.sh start"; grep -h p104opprof runs/cd-c0b-*.out runs/cd-c0b-mtp-A1-server.out | head -200
```
Control: v11-bin (A); profiler bin = v11 kernels + profiler only (check that mmvq SASS is identical, as for k3-prof-bin).
Gates: (1) the profiler prints separate tables for n_tokens 512/1/3 and a QUANT_Q8_1 row; (2) the profiled tg without OPPROF is within 1% of v11 (no kernel change); (3)-(4) n/a (measurement); (5) thermal receipts.
Output: the builder updates STAGE-MAP rows 3, 5, 7, 9, 10, 13 (marked [M c0]) and re-ranks cards 1-3 if a hypothesis is off by more than 2x (e.g. D2H < 0.3 ms: drop CARD-1; draft LM head < 30% of the draft: shrink CARD-2).
Budget: builder 1 h; GPU 25 (A) + 15 (B) min; 1 iteration. Stop rule: no fixing of the profiler beyond 1.5 h; ship part A alone if B is late.
