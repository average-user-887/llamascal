
## Measured 2026-11-25 (verdict analyst, CARD-0; receipts runs/cd-c0a-*, cd-c0b-*)
| item | measured |
|---|---|
| pp512 d0 (v11, f16) | 526.7 +-13.3 tok/s (the 457.8 outlier did not reproduce) |
| pp512 d16k / d32k f16 | 343.0 / 255.1 |
| tg64 d16k f16 / q8_0 / q4_0 | 34.82 / 31.97 (0.918x) / 27.98 (0.804x); d32k f16 / q8_0 31.52 / 27.94 (0.886x) |
| prefill n512 kv1024 split | MUL_MAT q4_K 58.2%, q5_K 16.1%, q6_K 10.5% (total ~85%); GDN 4.9%; FA 3.5% (21.8% at kv16384); QUANT_Q8_1 (mmq) 1.5% |
| decode n1 target graph (profiled, 29.3 ms) | QUANT_Q8_1 1.09 ms (249 launches, 3.7%); LM head q6_K 3.45 ms (11.8%); GET_ROWS 0.66 ms |
| MTP draft graph (n1, 45 nodes, 4.56 ms) | LM head 3.48 ms = 76%; host gap 1.3-1.6 ms |
| CUDA graphs | disabled by arch on sm_61 in v11; enabling gives +0.3% (CARD-8 FAIL) |
Row notes: row 3 quantise ~1.1 ms/token (more than the 0.5 ms hypothesis) but the shared-cache + addnorm change only gave +1.6% tg; row 5 prefill MMQ ~85% (CARD-7 start condition met); row 10 draft LM head 76% of draft (CARD-2 confirmed, +5-6% M2).
