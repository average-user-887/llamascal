# CARD-1 backend sampling: drop the 1 MB logits D2H per output row (flag first, patch only if needed)
Question: Does `-bs/--backend-sampling` (exists at 5bacb7c, off by default) remove the raw-logits copy on the target context and so speed up S and MTP serving over PCIe 1.0 x4?
Hypothesis: logits row = 248320 x 4 B = 0.99 MB; at ~0.8 GB/s practical that is ~1.2 ms per output = 4.6% of a 25.9 ms S token. A step of MTP verify outputs 3 rows (~3.6 ms of ~47 ms = 7.7%). llama-context skips the raw copy when a backend sampler is set (src/llama-context.cpp:2412 `needs_raw_logits`). Expected: S +3-5%, M2 wall +4-8%. The GPU top-k/top-p over 248k must cost < 0.3 ms or the gain is gone.
Files: flag only: `tools/server/server-context.cpp:2014` (per-task `backend_sampling`, disabled for grammar, n_probs and TTS), `common/sampling.cpp:430-640`, `src/llama-context.cpp:2121-2520`. Patch only if the flag breaks MTP verify (common_sampler_sample_and_accept_n with backend chains) or still copies logits: then (a) keep the raw copy only for the rows that need it, (b) if the backend chain lacks a sampler the serving defaults use (penalties, min_p), fall back per task and log it.
Control: v11-bin, MTP GGUF, serving config np1 -c 49152 f16 KV, MTP n_max 2 (A = no -bs, B = same binary with -bs).
Gates: (1) dispatch: server log of B shows the backend sampler attached (SLT_TRC "sampler chain") and no full logits copy: a one-shot `nsys` is not available, so check `llama_get_logits` stays unused, or run the profiler bin from CARD-0 and compare host gap per step; (2) correctness: temperature-0 text identical A vs B on all 4 workloads (mtpab "text identical"); (3) numerics: n/a at temp 0. Run one extra B arm with a sampled request (client copy with temperature 0.7, top_p 0.95, top_k 20 under cards/bench/) to show it does not error; (4) speed: mtpab ABBA M2 A vs B, threshold B >= +2% mean wall tok/s and no workload < -1%; also S arm (`--a-args "" --b-args "-bs"` with the plain model) for the S decode effect; (5) VRAM delta < 100 MiB, thermal ok.
Haiku benchmark (no build, ~30 GPU min):
```
cd ~/p104-bench; A=~/p104-exp/v11-bin
MTPB_ARGS="--spec-type draft-mtp --spec-draft-n-max 2 -bs" cards/bench/cardbench.sh c1 $A mtp
MTPMODEL=$HOME/models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf MTPA_ARGS="" MTPB_ARGS="-bs" cards/bench/cardbench.sh c1S $A mtp
cat runs/cd-c1-mtp-summary.txt runs/cd-c1S-mtp-summary.txt
```
Budget: builder 0.5 h (read the code path, confirm gate 1); patch iteration only if the flag fails: +2 h; GPU 30 min; max 2 iterations.
Bundling: flag goes into serving.env (`-bs`) for v13 on its own, so it needs no code merge; a patch, if needed, ships in the "host/server" series together with CARD-2.
Stop rule: if B < +1% or text differs at temp 0, stop (leave the flag off); json_schema/grammar requests fall back to CPU sampling anyway, so record what share of production requests use grammar (Honcho uses json_schema) before deploying.
