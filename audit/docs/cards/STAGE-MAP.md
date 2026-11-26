# P104 stage map (planner, 2026-11-24 21:20 CET), baseline = p104-v11 (b098f6f), Qwen3.5-9B Q4_K_M, f16 KV, fa on, ub 512, t 4

Sources. [M] = measured: per-op event profile on v11 kernels (LEDGER k3-g3-*, 65 tokens, graphs off; the profiled token is 29.4 ms against 25.8 ms real, so the profiler adds ~3.5 ms of gaps and the ms figures are slightly inflated), llama-bench and queue receipts. [E] = planner estimate from bytes and launch counts, not measured. [?] = no data; CARD-0 measures it.
Decode token (S, no MTP) = 25.9 ms real (38.6 tok/s). Serving step (MTP n_max 2, np1) ~ 43 tok/s wall.
Prefill pp512 d0 ~520 tok/s = 1.92 ms/token; d8192 ~412 = 2.43 ms/token. Note: queue depth-sweep pp512 d0 gave 457.8 (FAIL against 520), while prefill-ub-b gave 520 in the same night: an unexplained outlier. CARD-0 re-measures it.

| # | Stage | Decode ms/token (share) | Prefill share | Bound / headroom | Cards |
|---|---|---|---|---|---|
| 1 | tokenise + embed get_rows (Q4_K row) | GET_ROWS 0.75 [M] (2.9%) for all get_rows in the graph; the embedding row is only 2.3 KB, so most of this is launches and other gathers [E] | ~0 | launch-bound; ~0.4 ms recoverable [E] | 6 |
| 2 | per-layer RMS norm (+mul) | 0.90 [M] (3.5%) (~97 norms incl. q/k norms) | small [?] | 16 KB per norm: pure launch cost; fusion with the next op | 3, 6 |
| 3 | q8_1 quantise of activations | inside the MUL_MAT events; ~190 quantise launches/token, ~120 of them redundant (qkv/z/beta/alpha share one input in 24 GDN layers, gate/up in 32 layers, q/k/v in 8 attn layers) [E]: 0.4-0.6 ms (~2%) | per ubatch: quantize_mmq_q8_1 per matmul, 4 MB per call [E] | launch-bound at decode | 3 |
| 4 | weight GEMV (MMVQ) Q4_K / Q5_K / Q6_K | 22.7 profiled [M] (~78%): Q4_K 11.35 (235-259 GB/s), Q6_K 6.58 (206-217), Q5_K 4.37 (232-250), q8_0 0.39 | - | 290 GB/s practical: bound ~17.6 ms, 3-5 ms headroom but v9/v12/Q6_K reworks gave <=1.3%: kernel line exhausted | 4 (fewer bytes), 3 (fusion) |
| 4a | of which output.weight 248320x4096 Q6_K (796 MiB) | 3.45 [M] (12-13%) | 1 row per ubatch (last token only) | at ~237 GB/s; byte reduction is the remaining lever | 4, 2 |
| 5 | weight MMQ (prefill) | - | [?] est 65-80% [E] | ~9.7 of ~26 TOPS dp4a = 35-40% [E] (RESEARCH-ROUND2 s3) | 7 |
| 6 | attention FA (8 layers, f16 KV, TILE gqa_opt) | 0.39 d0 / 1.57 d8192 [M] | rises with depth: pp drops 21% at 8k [M] | d8192 KV read 256 MiB/token -> 0.88 ms at 290 GB/s: 1.8x headroom at depth [E] | 5 |
| 6a | quantised KV (q8_0/q4_0) | VEC kernel, one block per Q head: K/V read 4x (GQA 4) [src fattn-vec.cuh:108-113]; tg d16k 32.2 (q8_0) vs 35.2 (f16) [M] | TILE: whole cache dequantised to an f16 temp buffer per call [src fattn-common.cuh:1087]; at 128k that is ~1 GiB temp, so it does not fit | GQA packing gives K/V read 1x | 5 |
| 7 | GDN / ssm block (gated_delta_net, ssm_conv, softplus, sigmoid, silu, concat, fill, cpy of state) | GDN 0.69 + SSM_CONV 0.22 + SOFTPLUS 0.29 + CONCAT 0.25 + small ops ~0.4 + CPY 0.38 = ~2.2 [M] (~8%) | [?] chunked GDN in prefill | GDN state 2 MiB/layer r+w: 96 MiB/token -> 0.33 ms bound vs 0.69 [E]; the rest is launches | 6 |
| 8 | residual add | 0.67 [M] (2.6%) | small | launch-bound; fuse add+rms_norm | 6 |
| 9 | output norm + logits D2H + sampling | norm in 2; logits 248320 x f32 = 0.99 MB per output row over PCIe 1.0 x4 (~0.8 GB/s) = ~1.2 ms per output [E] (~4.5% S; at MTP verify 3 rows per step ~3.6 ms [E]); host sampling over 248k [?] | 1 row | removable: backend sampling (`-bs`) skips the raw logits copy [src llama-context.cpp:2412 needs_raw_logits] | 1 |
| 10 | MTP draft (1 nextn block + full 248k LM head + backend top-k 10) | per draft token ~4.5-5 ms [E] (LM head 3.45 + block ~0.7 + glue); 2 drafts/step = ~18-20% of step [E] | MTP head also runs over the prompt [?] (upstream notes a prefill slowdown) | LM head over 248k for a top-10 draft is mostly waste | 2, 4 |
| 11 | MTP verify (3-column GEMV) | c(3) ~1.4-1.5x single column [M k3] | - | v12 4-5 col +0.5%: no more kernel work | 3 (shared quantise also helps verify) |
| 12 | KV cache ops (set_rows, f16 cpy) | SET_ROWS+CPY ~0.2 of the CPY total [M] | small | - | 5, 6 |
| 13 | launch gaps / host (graph build, set_inputs, CUDA graph replay) | gaps 1.38 [M, graphs off]; with graphs on [?]; graph reuse on draft/verify shapes [?] | ubatch-level [?] | fusion cuts node count | 0 (measure), 3, 6 |

Uncertainty: rows 3, 9, 10 and 13 are the least certain and carry the cheapest cards (1, 2, 3). CARD-0 measures them first; its receipts re-rank cards 1-3 before builders spend GPU time.
Ceiling check: S decode with every decode card passing: 25.9 -> ~23.0 ms (+12%); MTP serving +15-25% (multiplicative, mostly cards 1 and 2).
