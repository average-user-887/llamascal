# CARD-1 backend sampling (`-bs`) - builder A receipt (gate 1, code evidence)
Builder A, 2026-11-24 ~21:50 CET. Source: ~/p104-exp/cardA @ p104-v11 (b098f6f), branch c1 (no patch). The benchmark (G2) is run by the orchestrator's Haiku runner; numbers go in the section below when they arrive.

Hypothesis: with a backend sampler set for the slot, llama_context::decode skips the 248320 x f32 (0.99 MB per output row) logits D2H, which saves ~1.2 ms per output row over PCIe 1.0 x4.

## Gate 1 (dispatch) by code reading: PASS (static), runtime check still to do
- Flag: `-bs/--backend-sampling` -> `params.sampling.backend_sampling` (common/arg.cpp:2290).
- Server, per task (tools/server/server-context.cpp:2014-2027): `llama_set_sampler(ctx_tgt, slot.id, chain)` when backend_sampling && !(n_probs>0 && !post_sampling_probs) && task != TTS; otherwise nullptr (CPU path).
- common_sampler_init (common/sampling.cpp:430-440) clears backend_sampling when a grammar (json_schema, grammar, tool-call lazy grammar) or a reasoning-budget sampler is present: these requests go back to the CPU path. That is a silent per-request fallback (LOG_WRN), not an error.
- llama_context::set_sampler (src/llama-context.cpp:1695-1745): offloads when the chain has backend_init/backend_apply; else LOG_WRN "cannot be offloaded" and CPU path.
- Chain offload is a prefix (src/llama-sampler.cpp:734-810): samplers run on the backend up to the first one without backend support. Neutral samplers (dry 0, xtc 0, typical 1, top-n-sigma off, penalties 1.0) are replaced by "empty" samplers that do support the backend, so the server default chain at temp 0 (greedy) and the card's sampled client (temp 0.7, top_p 0.95, top_k 20) both offload fully. Non-neutral DRY/XTC/typical/top-n-sigma would cut the prefix there (CPU does the rest on the backend-reduced candidates).
- Raw copy skip: src/llama-context.cpp:2412 `needs_raw_logits(ubatch, sampling.samplers)` returns false when every output row's sequence has a backend sampler, so the `ggml_backend_tensor_get_async(... n_outputs*n_vocab*4)` is skipped.
- MTP verify: common_sampler_sample_and_accept_n -> common_sampler_sample per idx -> `llama_get_sampled_token_ith` returns the backend token, set_logits uses the sampled candidates, no `llama_get_logits_ith`. So verify works with the backend chain; no patch needed by reading.
- Remaining raw-logits users: get_token_probabilities (n_probs pre-sampling, already excluded), classification labels (server-context.cpp:2480, not a sampling task). Neither is in the serving workloads.
- Runtime check for the runner (no GPU from me): B server logs must NOT contain "cannot be offloaded" or "backend sampling is not compatible"; text identical A vs B (mtpab).

## Production caveat
Honcho uses json_schema -> grammar -> CPU path for those requests (no gain, no harm). The share of grammar requests decides the real-world gain.

## G2 benchmark numbers
M2 (runs/cd-c1-mtp-summary.txt, v11-bin, MTP n_max 2, np1, A = no -bs, B = -bs), wall tok/s:
| arm | all | agent | chat | code | copy | draft_acc |
|---|---|---|---|---|---|---|
| A1 | 37.76 | 28.10 | 43.20 | 47.44 | 32.29 | 2136/2627 |
| B1 | 39.73 | 27.88 | 46.65 | 50.69 | 33.68 | 2136/2627 |
| B2 | 39.75 | 27.85 | 46.82 | 50.79 | 33.55 | 2136/2627 |
| A2 | 37.22 | 27.59 | 43.72 | 45.67 | 31.90 | 2136/2627 |
Mean B/A: all +6.0% (39.74 / 37.49); agent +0.05%, chat +7.8%, code +9.0%, copy +4.6%. Text identical 16/16.
Gate-1 runtime evidence: B server logs have 4 "backend sampling is not compatible with grammar, disabling" lines (the agent workload's tool-call grammar), no "cannot be offloaded". So the agent workload ran on the CPU path, which is why it shows no gain, as the code reading predicted.
S arm (runs/cd-c1S-mtp-summary.txt): not available at the time of writing.

Verdict (gate 1): PASS by code; overall verdict depends on G2 (threshold M2 B >= +2% mean, no workload < -1%).

## Verdict (analyst)
Threshold: M2 mean >= +2%, no workload < -1%; gate 2 text identical; S arm; VRAM; thermal.
- M2 (runs/cd-c1-mtp-summary.txt): mean A 37.49, B 39.74 => +6.0%; workloads agent +0.05%, chat +7.8%, code +9.0%, copy +4.6% (none < -1%). Text identical 16/16, draft_acc identical 2136/2627. ABBA drift: A1 vs A2 = 1.45% (37.76 vs 37.22; above the 1% flag, but the +6.0% effect is 4x larger); B1 vs B2 0.05%.
- S arm (runs/cd-c1S-mtp-summary.txt, plain model): A 29.945, B 31.235 => +4.31% (drift A 0.63%, B 0.74%); the SUMMARY of the builder said 'not available' - it is now; text identical 16/16. Per workload B/A: agent +0.0%, chat +5.9%, code +5.7%, copy +4.0% (computed from the summary).
- VRAM: max 7369 MiB both arms (delta 0). Max temp A 51 C, B 56 C.
- Gate 1 runtime: 4 'backend sampling is not compatible with grammar' lines (agent workload tool grammar -> CPU path), no 'cannot be offloaded'. Sampled-request (temp 0.7) no-error arm: not seen in the receipts I read (not verified).
PASS (M2 +6.0% >= +2%, S +4.3%). Ships: serving flag -bs only (no code change). Real-world gain depends on grammar share (Honcho json_schema requests fall back to CPU).
