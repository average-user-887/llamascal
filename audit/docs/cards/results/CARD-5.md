# CARD-5 quantised-KV flash attention for Pascal, phase A (GQA-packed VEC) - builder A receipt
Builder A, 2026-11-24. Branch c5 (worktree edits in ~/p104-exp/cardA-edit, built in ~/p104-exp/cardA as part of branch c25 = c2 + c5).

## Hypothesis
On cc 6.1, q8_0/q4_0 KV decode runs flash_attn_ext_vec with one block per Q head (gqa 4: every K/V row read 4x), and n=3..4 (MTP verify) goes to TILE with a whole-cache f16 dequant. Packing the 4 Q heads of a KV head into one block reads K/V once. Target: tg q8_0 d16k >= +6% vs v11 q8_0 and >= 0.98x v11 f16.

## Iteration 1 (commit ac2fb21)
- flash_attn_ext_vec gains template ncols2 (Q heads per block); column j -> (token j/ncols2, head head0 + j%ncols2) for Q load, mask row, sinks, dst and dst_meta; grid z = ne12*ne03*(gqa/ncols2) (launch_fattn already supports ncols2).
- Host: D=256 and K=V in {q8_0, q4_0}, gqa % 4 == 0, mask without head dim, max_bias 0, no softcap, no kv_rows, Q->ne[1] <= 4 -> ncols1 1 (decode) or 2 (2..4 tokens), ncols2 4. Only these 4 instances are new (f16 path unchanged).
- Kernel selection (fattn.cu): quant KV with Q->ne[1] 3..4 also goes to VEC when the packing applies (no TILE f16 temp at verify).
- Env P104_FA_VEC_GQA=0 restores the v11 behaviour in the same binary.
- test-backend-ops: new FLASH_ATTN_EXT cases hs 256, 4 KV heads, gqa 4/8, kv 512/4096, nb 1..4, sinks on/off, nr3 1/2, q8_0/q4_0, permuted Q.
- Bench script: cards/bench/c5bench.sh (test, kld at q8_0 KV with fresh v11 base logits, llama-bench ABBA q8_0/f16 at d 0/16384, mtpab at -c 98304 q8_0).

## Results
(pending)

## Results (verdict analyst; receipts runs/c5bench.log, cd-c5t-test, cd-c5-kldbase-ub1, cd-c5-q8_0-abba-*, cd-c5-f16-abba-*, cd-c5-mtp-*)
A = v11-bin, B = c5-bin (P104_FA_VEC_GQA default on). 
- test-backend-ops (cd-c5t-test.out): 4498/4498 passed, 2/2 backends (includes the new FLASH_ATTN_EXT cases).
- KLD at q8_0 KV vs fresh v11 q8_0 base logits: ub1 B 0.000190 +-0.000013 same-top 99.530%; ub2 B 0.000190 +-0.000008 (99.374%); ub4 B 0.000462 +-0.000021 (98.865%); A = 0 (self). Small but non-zero: kernel is not bit-identical (different reduction order).
- llama-bench ABBA q8_0 KV (tg128 / pp512): d0 tg A 38.525 -> B 37.935 (-1.5%); d16384 tg A 31.98 -> B 29.545 (-7.6%); pp512 d0 535.1 -> 534.6, pp512 d16384 347.7 -> 347.1. Drift A 0.03% / 0.00%, B 0.03% / 0.10%. B slower on both depths; A at d16k = 0.917 of f16 (34.82), B = 0.848.
- f16 KV ABBA: tg d0 38.89 -> 38.88 (-0.03%), d16384 34.82 -> 34.86 (+0.1%); pp512 d16384 A 337.85 (+-16 outlier) / 349.91 vs B 349.1 / 349.07.
- mtpab M2 q8_0 KV at -c 98304: A 37.69 -> B 37.375 tok/s (-0.84%), draft_acc A 2124/2583 B 2144/2614; text identical only 9/16 (kernel numerics change greedy output on 7/16 prompts); both ran, no OOM; VRAM peak 8101 MiB in both arms (of 8192: 91 MiB free, near limit); max temp 59 C.
## Verdict
Gate 2 PASS (4498/4498). Gate 3 PASS-ish (KLD <= 0.00046, small). Gate 4: tg q8_0 d16k >= +6% vs v11 q8_0: got -7.6% -> FAIL; >= 0.98x v11 f16: 29.5/34.86 = 0.85 -> FAIL; f16 no regression: PASS; pp d16k not slower: PASS; mtpab runs at 98k: PASS (capability), but -0.84% and 9/16 text identical. Overall FAIL: the GQA-packed VEC kernel (4 heads per block, ncols2 4) is slower than v11's per-head VEC. Ships: NO. Keep P104_FA_VEC_GQA=0 semantics (c5 not merged); the builder needs an iteration 2 (occupancy/register pressure likely) before it is worth a re-test.
