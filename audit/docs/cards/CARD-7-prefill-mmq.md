# CARD-7 prefill on sm_61: MMQ dp4a tile shapes for Q4_K/Q5_K/Q6_K, chunked GDN, attention at depth (run only after CARD-0 part B shows the prefill split)
Question: pp512 is ~520 tok/s (d0), ~412 at 8k, ~345 at 16k. dp4a use is ~35-40% of peak [E]. Pascal dp4a MMQ configs (mmq-config-pascal-dp4a.cuh) use I=64 tiles, occupancy 2, no stream-K for K-quants, on 15 SMs (tile-count quantisation). Which part of prefill is the bottleneck, and do bigger tiles or stream-K help?
Hypothesis: if MMQ is >= 65% of prefill: an I=128 x J=64 tile for Q4_K/Q6_K (as the Q4_0 entries have) raises the reuse of the B tile 2x; on 15 SMs a 12288-row matmul at ub 512 has 192 I-tiles x 8 J-tiles = 1536 tiles, so the quantisation loss is small, and gains come from arithmetic intensity: +10-15% MMQ time -> +7-10% pp. If chunked gated-delta-net (gated-delta-net-mma.cu needs tensor cores; check the sm_61 fallback) is >= 15%, a dp4a/fp32 chunked variant is the target instead. The depth drop (-21% at 8k) is FA TILE: phase B of CARD-5 covers the quantised part.
Files: `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` (table entries for Q4_K, Q5_K, Q6_K), `mmq.cuh` (tile loaders, `mmq_get_nwarps`, occupancy), `mmq.cu` (stream-K decision); `gated_delta_net.cu` / `gated-delta-net-mma.cu` (sm_61 path). Restrict the build to the 3 K-quant MMQ instances while iterating (template instances under template-instances/mmq-instance-q{4,5,6}_k.cu) to keep -j2 compile < 15 min.
Control: v11-bin; plain Qwen3.5-9B f16 KV; llama-bench pp512/pp2048 at d0/d8192 (decode must stay flat).
Gates: (1) dispatch: profiler (CARD-0 bin) shows the new tile config in MUL_MAT at n_tokens 512; (2) test-backend-ops -o MUL_MAT -p "q[456]_K" incl. the n=512 cases pass vs CPU; (3) KLD ub512 (default perplexity batch = prefill path): candidate vs v11, |delta Mean KLD| <= 0.0002; (4) speed: pp512 d0 and pp2048 d0 >= +4%, d8192 >= +2%, tg unchanged (+-0.5%); also mtpab prompt_ms (pp_tps field) on the long-prompt workloads; (5) thermal: prefill is the hottest load, so record max C vs v11 (+3 C max).
Haiku benchmark (~35 GPU min):
```
cd ~/p104-bench; B=~/p104-exp/c7-bin
OPS=MUL_MAT OPP="q[456]_K" cards/bench/cardbench.sh c7 $B test,kld,bench
python3 -I tools/abba.py cd-c7-abba
```
(kld step uses ub 1/2/4; the builder adds one ub-512 perplexity pair (A and B, `-ub 512 -b 1024`) to a copy cards/bench/c7kld.sh: base logits for ub512 = v11's own output, generated in the same window.)
Budget: builder 4 h; long compiles; GPU 35 min/iteration; max 2. Lowest priority tonight: start only if a builder is idle by ~01:30.
Bundling: "prefill" series, independent of the decode cards; can merge into v13 if done by 03:30, else next night.
Stop rule: CARD-0 part B shows MMQ < 50% of prefill -> do not start; < +3% pp after 2 iterations -> stop.
