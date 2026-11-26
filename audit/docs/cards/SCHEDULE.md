# Night schedule 2026-11-24 -> 25 (planner). Clock check: host `date` = 21:16 CET, not 11:30. This plan starts at 21:30. Hard cutoff 07:00.

## Decision needed from the orchestrator now
PAUSE tonight's 22:30 p104-gpuq.timer run (`systemctl --user stop p104-gpuq.timer`, restart it tomorrow). The queue ran this morning (07:45). Its two NOT RUN KLD jobs and the pp d0 re-check are covered by CARD-0/4 receipts. If it is not paused, it owns the GPU 22:30-~01:30: then run only cards 0A+1 before 22:30, cards 2/3/4 between 01:30 and 03:30, and drop 5/6/7 tonight (see fallback below).

## Resources and rules
- GPU: one job at a time via flock gpu-bench-p104.lock. Every Haiku run goes through cards/bench/cardbench.sh / card0a.sh, which (a) hold cpu-build.lock for the whole run, so no build overlaps a benchmark, (b) stop p104-svc via gpu-window.sh and restart it on exit, (c) use gguard (thermal trip -> exit 2 -> stop the night's GPU work and tell the orchestrator).
- CPU builds: `flock ~/.local/state/lab-exp/cpu-build.lock nice -n 19 make -j2 <targets>`, targets only (ggml-cuda, llama-server, llama-bench, llama-perplexity, test-backend-ops, llama-quantize as needed). No ccache on the host, so the first build of a new worktree is ~20 min at -j2: create only TWO worktrees, one per builder, and switch branches inside them (do not make a worktree per card).
  - Builder A: ~/p104-exp/cardA (branches c1, c2, c5 from tag p104-v11); bins to ~/p104-exp/c1-bin, c2-bin, c5-bin.
  - Builder B: ~/p104-exp/cardB (branches c0prof, c36 (cards 3+6), c7); bins to c0-prof-bin, c3-bin/c6-bin/c36-bin, c7-bin.
- Quota rules: max 2 Opus builders at once (A and B). Builder briefs = card file path + 3 lines (worktree, branch, bin dir, "stop at the card's stop rule, write a receipt line to LEDGER and runs/<card>-progress.md"). Haiku runners: one shell command per run (the card's "Haiku benchmark" block), return stdout + receipt paths, no interpretation. The orchestrator compares numbers against the card's predeclared threshold.
- A builder never waits idle on the GPU: while its bin is queued, it codes the next card in its list.

## GPU timeline (serial; times are planned starts; each slot includes window stop/start ~2 min)
| Slot | Time | Job (Haiku) | GPU min | Prereq |
|---|---|---|---|---|
| G1 | 21:30-21:58 | CARD-0A `cards/bench/card0a.sh` | 25 | none (v11-bin) |
| G2 | 21:58-22:30 | CARD-1 flag ABBA (2 cardbench calls) | 30 | none |
| C1 | 22:30-23:15 | CPU only: full builds cardA, then cardB (serialised by the lock, ~20 min each); builder B's profiler build (c0prof) is the first cardB build | - | builders have coded since 21:35 |
| C2 | 23:15-23:40 | CPU only: CARD-4 llama-quantize x2 (~10 min each) + ggufinfo check | - | llama-quantize built in cardA during C1 |
| G3 | 23:40-00:00 | CARD-0B profiler run | 15 | c0-prof-bin |
| G4 | 00:00-00:45 | CARD-4 KLD + speed (c4kldB, c4speed.sh) | 45 | outQ5K files |
| C3 | 00:45-01:00 | incremental builds c2-bin, c3-bin (or c36) | - | code done |
| G5 | 01:00-01:35 | CARD-2 (test, mtp, n_max 3 arm) | 35 | c2-bin |
| G6 | 01:35-02:20 | CARD-3 (+6 if ready as c36-bin) test, kld, bench, mtp | 45 | c3-bin |
| C4 | 02:20-02:35 | incremental builds: c5-bin, iteration-2 bins of 2/3/6 | - | |
| G7 | 02:35-03:20 | CARD-5 (test, q8_0/f16 bench at 16k, q8_0 serving at 98k) or iteration 2 of the best card so far | 45 | c5-bin |
| C5 | 03:20-03:45 | INTEGRATION: builder A merges winners into branch p104-v13 (order: 3/6 series, 5, 7, then 2; flag 1 and file 4 are serving.env items), host build v13-bin (~15 min incremental in cardA) | - | verdicts in |
| G8 | 03:45-04:00 | v13-bin smoke: `cards/bench/cardbench.sh v13 ~/p104-exp/v13-bin test,kld` | 15 | v13-bin |
| C6 | 04:00-04:25 | image build: `container/build.sh p104-v13` (~21-25 min -j2, holds cpu-build.lock) | - | must start by 04:00 |
| G9 | 04:30-05:30 | FINAL ABBA in the container path, v11 image vs v13 image (container/run-experiment.sh, k3validate pattern: llama-bench d0/d8192 pp512/pp2048 + mtpab M2 np1 on 4 workloads + optional np2) with the chosen serving.env (flags, model file) on the B side | 60 | image |
| - | 05:30-06:30 | slack: one re-run if any final number has ABBA drift > 1%, else KLD on the final model file with the larger wiki chunk count | <=60 | |
| - | 06:30-07:00 | serving restored on v11 image (gpu-window start), receipts summarised. A swap to v13 is a separate human/orchestrator GO (swap-service.sh), not part of the night | - | |
Total GPU ~5.3 h of 9.5 h; the slack covers one card overrun.

## Coding plan (parallel, CPU-only, while the GPU runs)
- 21:35-22:30 builder A: CARD-1 code reading (gate 1 evidence, no build), then CARD-2 coding + the CPU pre-gate (tokenizer coverage; uses the existing v11 llama-tokenize? none in v11-bin: build `llama-tokenize` in C1). Builder B: CARD-0B profiler extension, then CARD-3 coding.
- 22:30-01:00 A: CARD-2 finish, CARD-4 script (c4speed.sh); B: CARD-3, then CARD-6 in the same branch (c36).
- 01:00-03:20 A: CARD-5 phase A (fattn); B: CARD-6 finish / iteration 2 of 3; CARD-7 only if B is idle by 01:30 and CARD-0B shows MMQ >= 50% of prefill.
- 03:20 builder B stops (quota); builder A does the integration.

## Fallback if the 22:30 queue is NOT paused
G1+G2 before 22:30 (as above). 22:30-01:30 queue. 01:30-02:05 CARD-4 speed, 02:05-02:40 CARD-2, 02:40-03:25 CARD-3, integration 03:25, image build 04:00, final ABBA 04:30-05:30. Cards 5/6/7: coding continues, GPU validation next night.

## Bundling / integration plan
Series in p104-v13 (on top of p104-v11): (1) decode-fusion = CARD-3 + CARD-6 (ggml-cuda.cu, mmvq.cu, norm.cu, gated_delta_net.cu); (2) long-context = CARD-5 (fattn*); (3) prefill = CARD-7 (mmq-config, mmq); (4) mtp-host = CARD-2 (+ CARD-1 patch only if the flag needed one). Non-code: CARD-1 `-bs` and CARD-4 `-outQ5K` model in env/serving.env. Only cards that passed their own gate 4 go in. One rebuild, one image, one final ABBA vs v11; export the patch as ~/p104-bench/patches/v13-*.patch (same format as k3-v11). If the final ABBA shows v13 < sum of parts by > 3 pp, bisect next night, not tonight.
