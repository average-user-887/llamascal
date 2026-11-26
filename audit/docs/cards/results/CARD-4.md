# CARD-4 output.weight Q6_K -> Q5_K requant (model file change, no code) - builder A receipt
Builder A, 2026-11-24. Production use needs the human's OK (quality change): benchmark only.

## Finding before any run: the card's COPY command does not work
`llama-quantize --allow-requantize --output-tensor-type q5_K IN OUT COPY` would write an unchanged copy:
ftype COPY sets only_copy, and tensor_allows_quantization() (src/llama-quant.cpp:304) returns false for every tensor before the output override (line 822) is read.
Replacement (cards/bench/c4quant.sh): ftype Q4_K_M plus `--tensor-type-file` that pins every 2D tensor to its current type with anchored exact names (`^blk\.0\.attn_q\.weight$=q4_K`, ...; manual overrides take precedence over the mixture logic, llama-quant.cpp:831-850) and `^output\.weight$=q5_K`. A tensor whose target type equals its current type is byte-copied (llama-quant.cpp:1322 `quantize = cur_type != new_type`), so only output.weight is requantised (Q6_K -> f32 -> Q5_K, no imatrix).
Check: cards/bench/c4types.py diff IN OUT hashes every tensor: expected 1 CHANGED (output.weight) and all others identical.
The quantise stages 4.07 GB f32 on a 7.5 GiB host, so c4quant.sh opens the serving window (gpu-window stop) and holds both locks.

## Scripts
- cards/bench/c4quant.sh [q5_K|q4_K]: quantise both GGUFs + diff (runner).
- cards/bench/c4speed.sh [outQ5K]: llama-bench ABBA (same v11-bin, -m per arm, A1 orig B1 new B2 new A2 orig, -p 512 -n 128 -d 0,8192 -r 3) + mtpab M2 (--a-model orig MTP, --b-model new MTP).
- KLD: `MODEL=<outQ5K plain> cards/bench/cardbench.sh c4kldB ~/p104-exp/v11-bin kld` (read the B rows; base logits = unpatched Q6_K reference).

## Results
(pending)

## Results (verdict analyst; receipts runs/queueA-2.log, c4quant-outQ5K-*.log, cd-c4kldB-kld-ub{1,2,4}-{A,B}.out, cd-c4Q5K-abba-*.out, cd-c4Q5K-mtp-*)
Model: Qwen3.5-9B-Q4_K_M-outQ5K.gguf (and the MTP variant) from ~/p104-bench/cards/bench/c4quant.sh; v11-bin for all arms. (The ggufinfo 'only output.weight changed' check and the -130 MiB size line were not looked at by me: gate 1 not verified.)
- KLD vs unpatched Q6_K reference logits (5 chunks; A and B rows are identical because both arms loaded the Q5K file via MODEL): ub1 Mean KLD 0.002087 +-0.000177, same-top 96.947 +-0.340%, max KLD 0.4376, 99.9% 0.01487; ub2 0.001892 +-0.000035, 97.613 +-0.302%; ub4 0.001444 +-0.000027, 97.808 +-0.290%. Reference v11 Q6_K: 0.00067 / 98.8%.
- llama-bench ABBA (v11-bin, -m per arm): tg128 d0 A 38.845 -> B 40.15 (+3.36%); d8192 36.72 -> 37.91 (+3.24%); pp512 d0 A 534.6 (538.61 / 530.65, drift 1.5%) -> B 536.6 (+0.4%, within noise); pp512 d8192 424.2 -> 424.9. Drift tg: A 0.28%, B 0.05%. Max temp 57 C.
- mtpab M2 np1 (orig MTP file vs outQ5K MTP file): A 37.455 (37.63 / 37.28, drift 0.93%), B 38.77 (+3.51%); draft_acc 2136/2627 -> 2133/2635; text identical only 8/16 (expected: different weights).
## Verdict
Gate 4 speed: tg d0 +3.36% >= +1.5% PASS, d8192 PASS, mtpab M2 +3.51% >= +1.5% PASS. Gate 3 numerics (Mean KLD <= 0.004 AND same-top >= 98.0%): mean KLD PASS (0.0021/0.0019/0.0014) but same-top 96.95 / 97.61 / 97.81 % < 98.0% at ub1/ub2/ub4 -> FAIL. Overall FAIL on the predeclared numerics gate; a speed gain of ~3.4% but 8/16 MTP outputs change and same-top drops ~1-2 pp. Ships: no (production use needs the human's quality OK in any case; if accepted, it is a model-file swap, no code).
