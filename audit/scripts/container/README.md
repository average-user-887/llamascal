# P104 container workflow (z4-p104)

Rule: experiments run in Docker; we swap containers, never edit live binaries. `~/llama-prod` (5bacb7c baseline) and
`~/llama-k1-20261123` (v8, current service) stay untouched as fallback until the container path is verified.

## Workflow
1. `build.sh <git-ref|patch-file>` -> image `p104-llama:<short>-<UTC date>[-N]` (never reused), record `~/p104-bench/images/<tag>.json`
   (image id, commit, dirty flag, patch sha, sha256 of llama-server and libggml-cuda). Builds from the commit only (`git archive`),
   so uncommitted worktree edits are NOT included (dirty=true is recorded as a warning). A patch file is applied (-p1) on 5bacb7c.
   CPU-only build (-j2, nice 10), no GPU. Expect ~20-40 min on 4 weak cores; do it while no timing run is active.
2. `run-experiment.sh TAG IMAGE [--purpose T] [--serve PORT] -- llama-bench -m /models/... ` (only way to run). Re-execs under
   `flock ~/.local/state/lab-exp/gpu-bench-p104.lock` (override `LOCK=`), container `p104-<TAG>`, labels `lab.*`, thermal guard from
   the host (cooldown <=45 C, `docker kill` at >=85 C or any thermal/slowdown flag), `runs/<TAG>.{trace.csv,meta.json,out}`,
   fleetreg start/finish if `FLEETREG_ENABLE=1`. Exit 2 = trip. `DRY=1` prints the command.
3. `swap-service.sh <image>` promotes: stop old, candidate on 127.0.0.1:18110 (health + alias + chat smoke), restart as
   `p104-svc` :18100 (`--restart unless-stopped`), re-verify; any failure restores the previous server (container or the systemd
   user unit) and exits 1. `images/CURRENT` and `images/PREVIOUS` updated. Downtime is two model loads (8 GiB VRAM cannot hold two servers).
   First run migrates from the systemd unit: it stops `llama-p104.service` (never disables it; disable only after verifying).

## Fleet dashboard mapping
Convention `flock <gpu bench_lock> docker run --name <title>`: lock holder argv shows `run-experiment.sh`, docker name `p104-<TAG>`.
It only becomes visible once the the lab orchestrator host side registers the lock for gpu p104 via `labslot` (registry `bench_lock` is null today, and
the dashboard reads /proc/locks on the lab orchestrator host). fleetreg.py's detached holder blocks on our lock until we exit, then ends: harmless.

## Build parity with host (CMakeCache of ~/p104-exp/wt/build)
Same: Release, 61-real, GGML_CUDA_GRAPHS=ON, BUILD_SHARED_LIBS=ON, gcc-12 host compiler, FA quants default, SCHED_MAX_COPIES=4.
Different (deliberate): CUDA 12.9.1 vs 12.0.140 (nvcc codegen for sm_61 may differ slightly: the parity test below decides);
GGML_NATIVE OFF with explicit AVX/AVX2/FMA/F16C/BMI2/AVX512 ON (host W-2123 has them; reproducible); NCCL OFF (host cache says ON
but NCCL_LIBRARY was NOTFOUND, so effectively off); ccache OFF; Ninja generator; LLAMA_BUILD_COMMIT set explicitly
(no .git in context); LLAMA_BUILD_TESTS not forced (only 4 targets built). Runtime: Ubuntu 24.04 glibc/libstdc++, non-root uid 1000.

## Status
Verified without docker: `bash -n` on all scripts, DRY=1 output of run-experiment/build/swap, image tags exist on Docker Hub
(nvidia/cuda 12.9.1-{base,devel,runtime}-ubuntu24.04, pushed 2025-07-25). shellcheck/hadolint not installed (manual review).
UNTESTED (needs ROOT-SETUP.sh): the actual docker build (patch -p1 step, cmake flags, runtime libs list), `--gpus device=UUID`
selection, flock re-exec + guard kill path, jq meta in images json, swap + rollback paths, `--read-only` + `/tmp noexec` with
llama-server (if it needs exec tmp or a cache dir, adjust). Overheads expected: container start ~1-2 s, no measurable GPU
cost; memory cgroup 5g (MEM=) with models mmapped read-only; docker engine idle ~100 MB RAM.

## First 10 minutes after docker is installed
1. Re-login (docker group). `docker run --rm --gpus device=<gpu-uuid> -e CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi -L` -> only the P104.
2. `./build.sh a96dce0` (v8) in the background; check `images/*.json`.
3. Parity (GPU idle, after the other trial): `./run-experiment.sh par-ctr <img> -- llama-bench -m /models/experimental/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 5`
   vs host `~/llama-k1-20261123/llama-bench` same args via gguard: pp and tg must agree within 1%.
4. `./swap-service.sh <img>`; check `curl :18100/health`, `docker ps`, `images/CURRENT`. Keep the systemd unit as fallback.
