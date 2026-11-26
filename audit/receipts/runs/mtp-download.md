# MTP model download (m1-dl)  2026-11-23
Source: https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF  file Qwen3.5-9B-Q4_K_M.gguf (saved as Qwen3.5-9B-MTP-Q4_K_M.gguf)
Path: ~/models/experimental/qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M.gguf  (own new dir; other models untouched)
Size: 5868826976 bytes (matches tree API size)
sha256: e8dd94817e95d6c0939102049d068418269978377b13616c4726235e232841fe == HF lfs.oid (VERIFIED)
Header check (python3 -I, ~/p104-bench/mtp/ggufinfo.py, own pure-python reader): GGUF v3, 442 tensors, arch qwen35, block_count 33, qwen35.nextn_predict_layers = 1, full_attention_interval 4.
MTP head BUNDLED: blk.32.nextn.{eh_proj.weight (8192x4096 Q8_0), enorm, hnorm, shared_head_norm (F32)} plus a full blk.32 layer (33rd block). Original file: block_count 32, no nextn key, 427 tensors.
Quant types (tensor counts): Q4_K 150, Q6_K 35, Q5_K 24, Q8_0 1, F32 232  (original: Q4_K 132, Q6_K 22, Q5_K 48, Q8_0 48, F32 177 -> different quant mix, not just +head).
Size delta vs original 5.68 GB: +0.19 GB.
