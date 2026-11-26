# CARD-7 results: prefill MMQ (builder C, Opus, branch c7 in ~/p104-exp/cardC, bin ~/p104-exp/c7-bin)
Earlier note: CARD-0B profile (runs/cd-c0b-pp.out): MUL_MAT = ~85% of the n512 prefill graph (q4_K 58.2%, q5_K 16.1%, q6_K 10.5%) -> start condition met.

## Predeclared gates (written 2026-11-25 ~02:40 CET, before any c7 number was read)
1. Dispatch: env GGML_P104_MMQ_LOG=1 log shows Q4_K/Q5_K/Q6_K at ne11=512 dispatched to J=64 (I=64, 256 threads) with the new y-f32-ds layout for Q4_K/Q5_K.
2. Op correctness: test-backend-ops -o MUL_MAT and -o MUL_MAT_ID (CUDA0 vs CPU), all pass, 0 FAIL (as v11).
3. Numerics: KLD at ub 512 and ub 1024 (-b 1024, 5 chunks wiki, base = v11's own logits at the same ub generated in the same window): |Mean KLD(B) - Mean KLD(A)| <= 0.0002 (noise floor 0.0004-0.0008); not expected bit-identical (float reassociation of scale products), expected ~1e-6.
4. Speed (llama-bench Qwen3.5-9B Q4_K_M, f16 KV, fa 1, ub 512, -r 5, ABBA vs v11-bin): PASS iff pp512 d0 >= +3% AND pp2048 d8192 >= +2% AND tg128 within +-1% at both depths.
5. Thermal/VRAM: max GPU temp <= v11 + 3 C, no trip; VRAM no regression.
Verdict: PASS only if 1-5 all hold. Max two iterations.

## Hypothesis and arithmetic (CPU-only, SASS of v11 libggml-cuda, cuobjdump)
Hot inner loop of mul_mat_q<Q4_K, J=64> (one k01 step: 2 rows x 8 cols x 16 dp4a per thread) = 666 SASS instr: IDP 256, XMAD 96, FFMA 80, LDS 66, I2F 36, F2F 36, LOP32I 32, SHR 18, FMUL 16, FADD 16.
sm_61 throughput: IDP/XMAD/FP32 = 128 lanes/clk/SM, but I2F/F2F conversions = 32/clk/SM (quarter rate). Issue-slot cost ~= 256 IDP + 96 XMAD + 112 FP + 50 LOP/SHR + 72 cvt x4 = ~800 -> IDP share ~32%, matching the measured "dp4a ~35-40% of peak". Q5_K loop: IDP 256, XMAD 100, I2F 36, F2F 36, FP 112. Q6_K loop: IDP 256, XMAD 198 (int scale multiplies), I2F 32, FP 48.
Overhead per (row, col, sub-block): int sc*sumi (3 XMAD), I2F(sumi) (4 slots), F2F of q8_1 ds half2 (y side, 2 cvt per col-sb, amortised over only 2 rows).
Changes (iteration 1, "c7a"):
 a) dp4a accumulator starts at 0x4B400000 (float 12582912 = 1.5*2^23); |sumi| < 2^22 for q4_K/q5_K/q6_K blocks, so __int_as_float(acc) - 12582912.0f == (float)sumi exactly: I2F (4 slots) -> FADD (1).
 b) scale products in float, hoisted per row: wd = d*sc, wm = dmin*m (Q4_K/Q5_K), w = d6*sc (Q6_K); per (row,col,sb) FMUL+FFMA instead of 3 XMAD + FFMA; accumulate straight into sum[] (removes the per-call FMUL/FFMA/FADD tail).
 c) Q4_K/Q5_K: q8_1 (d, s) converted to float2 once per block into an extra 2 KB smem area (J*4 float2) during the y-tile copy, instead of 32 F2F per thread per k-step.
 Estimated Q4_K inner loop: ~800 -> ~500 issue slots (-35%); if the inner loop is ~70% of MMQ time, MMQ -20..25%, prefill +15..20% at d0 (less at depth where FA grows). Smem 19.2 -> 21.3 KB/block (occupancy 2 unchanged); registers must stay <= 128 (launch bounds 256x2), no spills (checked with cuobjdump -res-usage before any GPU run).
