# Changes to the container scripts (first real execution, 2026-11-23)
1. Dockerfile: first build (a96dce0, 21 min, 307/353 steps) FAILED at link of llama-bench: `libcuda.so.1 needed by libggml-cuda.so not found`
   (devel image only ships the stub libcuda.so under /usr/local/cuda/lib64/stubs). Fix: symlink stubs/libcuda.so.1 and pass
   -Wl,-rpath-link,<stubs> in EXE/SHARED linker flags (stub is link-time only; runtime gets the real libcuda from the nvidia runtime).
   Nothing else in the build was wrong up to that point (apt, patch step, cmake configure, all nvcc compiles fine, no OOM).
2. run-experiment.sh: trip threshold was hardcoded; added env GUARD_TRIP (integer, may only LOWER the 85 C limit) so the kill path can be tested without heating the card.
3. All scripts call plain `docker`; in a shell without the docker group run them as `sg docker -c './script ...'` (no script change needed).
4. Host-side parity runs of llama-bench need CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<P104 UUID> (otherwise the P620 is picked up and OOMs); container is immune (only one GPU visible). See c1/parity.sh.
5. run-experiment.sh: trip path killed once and then waited; if the container was not created yet the kill was a no-op and the job ran to completion (seen: GUARD_TRIP=20 dry test, secs=61). Now retries docker kill until the docker client exits.

# 2026-11-24 (s1) swap-service.sh parametrised
swap-service.sh: model/alias/ctx/np/flags/mem cap are now env (MODEL ALIAS CTX NP SRV_FLAGS MEM) or optional container/serving.env; defaults =
approved serving config (Qwen3.5-9B-MTP Q4_K_M, alias qwen35-9b-mtp-q4km, -c 49152 -np 1 f16 KV, MTP draft n_max 2, --cache-ram 1024, MEM 5g).
Rollback logic unchanged; DRY=1 works. Backup of the old script: swap-service.sh.bak-20261124. Smoke test uses the new alias.
- s2 fix: smoke chat failed on first swap (Qwen3.5 thinks by default; 32 tokens of reasoning, empty content). Smoke now sends chat_template_kwargs enable_thinking=false, max_tokens 64.
- 2026-11-24 queue: jobs 05-08 moved to np1 MTP -c 49152 serving config (see LEDGER s3-jobs); analyze_all a_prod arms renamed.

# 2026-11-25 (release c2+bs)
- run-experiment.sh: env XENV="K=V K=V" passes extra `-e` to the container (backup run-experiment.sh.bak-pre-xenv).
- swap-service.sh: env SRV_ENV="K=V ..." passes extra container env to p104-svc and the spare-port candidate; SRV_FLAGS already carries server flags (e.g. -bs). DRY=1 verified.
