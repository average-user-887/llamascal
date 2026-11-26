# CARD-4 LM head bytes: output.weight Q6_K -> Q5_K (and Q4_K) in both GGUFs (model change, no kernel change)
Question: output.weight is 796 MiB of Q6_K read once per output token (3.45 ms, 12-13%) and once per MTP draft. Does a smaller type pass the quality gate?
Hypothesis: Q5_K = 5.5 bpw vs 6.5625: -16% bytes, and Q5_K runs at ~240 GB/s vs Q6_K ~237 on this shape: 3.45 -> ~2.85 ms (-0.6 ms, +2.3% S). Q4_K (4.5 bpw, ~250 GB/s): -> ~2.25 ms (+4.8% S). MTP drafts also use this head (unless CARD-2 shrinks it), so M2 gains ~+3-5%. Quality: requantising from Q6_K (near-lossless) to Q5_K should add KLD ~0.001-0.003 [E]; Q4_K more. The head only shapes the logit tail; the same-top % is the key number.
Files: none in ggml. Tool: `llama-quantize` (build target in builder A's tree, CPU only). Command, all other tensors copied unchanged:
`llama-quantize --allow-requantize --output-tensor-type q5_K <in.gguf> <out.gguf> COPY`. Check that COPY honours the override (mtp/ggufinfo.py must list output.weight as Q5_K and every other tensor with type and size unchanged). If not, write a <=60-line gguf-py rewrite (dequantise Q6_K rows -> quantise via ggml's quantize through llama-quantize's tensor-type override `--tensor-type output=q5_K` with ftype Q4_K_M and `--pure` off: verify the same way). Host RAM: the f32 staging of 248320x4096 is 4.07 GB of 7.5 GiB: run with the serving container stopped and nothing else heavy (inside a cardbench window or a CPU window).
Outputs: ~/models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M-outQ5K.gguf and qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M-outQ5K.gguf (and -outQ4K if Q5K passes, as iteration 2). Disk: ~6 GB each, 111 GB free.
Control: v11-bin with the original GGUFs; B = v11-bin with the new GGUF (same binary).
Gates: (1) dispatch: ggufinfo diff = only output.weight; llama-bench logs the model size -130 MiB (Q5K); (2) op correctness: n/a (existing kernels; test-backend-ops q5_K already passes); (3) numerics: KLD of the new file vs the existing unpatched-ref logits ub1/ub2 (5 chunks): accept Mean KLD <= 0.004 and same-top >= 98.0% (current 0.00067 / 98.8%); also the queued kld-wiki-t2 style larger run if time; (4) speed: llama-bench ABBA tg d0/d8192 >= +1.5%, mtpab M2 >= +1.5%; (5) VRAM -130 MiB (more context headroom), thermal.
Haiku benchmark (~45 GPU min; the quantise runs first under the cpu-build lock, ~10 min each):
```
cd ~/p104-bench; Q=~/p104-exp/cardA/build/bin/llama-quantize; L=~/.local/state/lab-exp; MD=~/models/experimental
flock $L/cpu-build.lock nice -n 19 $Q --allow-requantize --output-tensor-type q5_K $MD/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf $MD/qwen35-9b/Qwen3.5-9B-Q4_K_M-outQ5K.gguf COPY 4
flock $L/cpu-build.lock nice -n 19 $Q --allow-requantize --output-tensor-type q5_K $MD/qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M.gguf $MD/qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M-outQ5K.gguf COPY 4
python3 -I mtp/ggufinfo.py $MD/qwen35-9b/Qwen3.5-9B-Q4_K_M-outQ5K.gguf | grep -i output
A=~/p104-exp/v11-bin
MODEL=$MD/qwen35-9b/Qwen3.5-9B-Q4_K_M-outQ5K.gguf cards/bench/cardbench.sh c4kldB $A kld     # both arms use the new file: read only the B rows, the A rows are a duplicate
```
Then the speed ABBA with different files per arm (same binary): `flock $L/cpu-build.lock` window + `python3 -I cards/bench/mtpab.py cd-c4-mtp $A $A --b-model $MD/qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M-outQ5K.gguf` and llama-bench ABBA with `-m` per arm (k3validate.sh pattern: A1 orig, B1 new, B2 new, A2 orig; `-p 512 -n 128 -d 0,8192 -r 3`). The builder writes these 2 lines into cards/bench/c4speed.sh before the window (needs no model knowledge beyond this card).
Budget: builder 0.5 h (script + COPY check); CPU 25 min; GPU 45 min; max 2 iterations (Q5_K, then Q4_K only if Q5_K passes with KLD margin).
Bundling: ships as a model-file switch in serving.env, independent of code; combine with CARD-2 (both act on the head; measure CARD-2 on the new file in the final ABBA).
Stop rule: KLD > 0.004 or same-top < 98% at Q5_K: stop, keep Q6_K. Production switch needs the human's OK (quality change, also a task eval if available).
