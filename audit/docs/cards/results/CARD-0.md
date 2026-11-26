# CARD-0 results (verdict analyst, 2026-11-25)
## Results
Part A (v11-bin, llama-bench, receipts runs/cd-c0a-*.out, r1-G1-card0a.out; max GPU temp 43-59 C, rc 0 all):
| test | tg64 tok/s | pp512 tok/s |
|---|---|---|
| f16 KV d0 | - | 526.74 +-13.3 (pp2048 row also 526.74, identical to the digit: suspicious duplicate, treat pp2048 d0 as unverified) |
| f16 KV d16k | 34.82 | 342.98 +-10.7 |
| q8_0 KV d16k | 31.97 | 349.38 |
| q4_0 KV d16k | 27.98 | 345.97 |
| f16 KV d32k | 31.52 | 255.10 |
| q8_0 KV d32k | 27.94 | 245.44 |
Ratios: q8_0/f16 tg = 0.918 at d16k, 0.886 at d32k; q4_0/f16 = 0.804 at d16k. pp512 d0 = 526.7: the 457.8 queue outlier did not reproduce (single re-measure, one run).
CUDA graphs: cd-c0a-graph-A1-server.out has "disabling CUDA graphs due to GPU architecture" on every graph key (graphs off on sm_61 in v11, as expected).
Part B (c0-prof-bin e318c44, graphs off, profiler on; runs/cd-c0b-{pp,tg,mtp-A1-server}.out, cd-c0b-driver.log): profiler tables per n/kv/nodes exist.
- Prefill n512 kv1024: MUL_MAT (q4_K 58.2% + q5_K 16.1% + q6_K 10.5% classes) = 84.8% of 951 ms graph; GATED_DELTA_NET 4.9%; FLASH_ATTN 3.5% (21.8% at kv16384); QUANT_Q8_1 (mmq) 1.5%. Prefill at d0 is therefore ~85% MMQ.
- Decode n1 (45-node draft graph, ctx_dft): 4.56 ms gpu, of which MUL_MAT q6_K 248320x4096 (LM head) 3.48 ms = 76%; host gap 1.3-1.6 ms. (The 1582-node graphs are the target n1 graphs: 29.3 ms gpu profiled, QUANT_Q8_1 1.09 ms/token over 249 launches = 3.7%, LM head 3.45 ms = 11.8%, GET_ROWS 0.66 ms.)
- Profiled tg64 32.25 tok/s (profiler overhead, graphs off); pp512 517.6 / d8192 414.5 under profiler.
## Verdict
Gate 1 (separate n=512/1/3 tables + QUANT_Q8_1 row): PASS for 512 and 1 and QUANT_Q8_1 present; the n3 verify table was not confirmed by me (only 'n1' and 'n90' keys seen in the head of the output).
Gate 2 (profiled tg within 1% of v11 without OPPROF): NOT TESTED (no unprofiled c0prof tg run); SASS md5 identical per ledger b-c0prof-build.
Gate 5: thermal max 59 C. 
Measurement verdict: PASS (measurement card). Re-rank: D2H hypothesis not directly measured (no D2H row; card 1 measured end-to-end +6.0% MTP, so keep). Draft head = 76% of the 4.56 ms draft graph (hypothesis 3.45 of 4.5-5 ms confirmed) -> card 2 stays high. Prefill is ~85% MMQ so the card-7 start condition (>= 50%) is MET, but card 7 was not started: NOT RUN.
