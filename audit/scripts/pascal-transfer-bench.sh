#!/usr/bin/env bash
# pascal-transfer-bench.sh <GPU_UUID_OR_INDEX> <TAG_PREFIX> [BIN_A] [BIN_B]
# Cross-card Pascal (sm_61) verification & ABBA operator transfer benchmark.
# Safely targets only the selected GPU UUID/index without stopping p104-svc unless P104-100 is targeted.
set -euo pipefail

GPU_ARG="${1:?usage: pascal-transfer-bench.sh <GPU_UUID_OR_INDEX> <TAG_PREFIX> [BIN_A] [BIN_B]}"
TAG="${2:?usage: pascal-transfer-bench.sh <GPU_UUID_OR_INDEX> <TAG_PREFIX> [BIN_A] [BIN_B]}"
BIN_A="${3:-$HOME/p104-exp/v11-bin}"
BIN_B="${4:-$HOME/p104-exp/llamascal-bin}"

L="$HOME/.local/state/lab-exp"
mkdir -p "$L" "$HOME/p104-bench/runs"

# Never overlap CPU builds with GPU benchmarks; lock per target GPU.
if [ -z "${PTB_LOCKED:-}" ]; then
  PTB_LOCKED=1 exec flock "$L/cpu-build.lock" "$0" "$GPU_ARG" "$TAG" "$BIN_A" "$BIN_B"
fi

# Resolve UUID if an integer GPU index was passed
if [[ "$GPU_ARG" =~ ^[0-9]+$ ]]; then
  UUID="$(nvidia-smi -i "$GPU_ARG" --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '[:space:]')"
else
  UUID="$GPU_ARG"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$UUID"
export GGUARD_GPU="$UUID"

cd "$HOME/p104-bench"
G="flock $L/gpu-bench-${TAG}.lock python3 -I $HOME/p104-bench/gguard.py"

echo "=== Pascal Transfer Bench: gpu=$UUID tag=$TAG ==="
nvidia-smi -i "$UUID" --query-gpu=name,pci.bus_id,compute_cap,memory.total,temperature.gpu,power.limit --format=csv

echo "--- [1/3] Dispatch & Operator Correctness (test-backend-ops q[456]_K) ---"
LD_LIBRARY_PATH="$BIN_B" $G "${TAG}-test" -- \
  "$BIN_B/test-backend-ops" test -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_FUSION -p "q[456]_K"
grep -h "tests passed\|FAIL" "runs/${TAG}-test.out" | head -5

echo "--- [2/3] Operator Perf ABBA (decode n=1..3 + prefill MMQ n=512) ---"
FILTER="type_a=q[456]_K,type_b=f32,m=(4096|8192|12288),n=(1|2|3|512),k=4096,"
for arm in A1:${BIN_A} B1:${BIN_B} B2:${BIN_B} A2:${BIN_A}; do
  name="${arm%%:*}"
  bin="${arm#*:}"
  LD_LIBRARY_PATH="$bin" $G "${TAG}-op-${name}" -- \
    "$bin/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "$FILTER" >/dev/null
  echo "$(date +%T) ${TAG}-op-${name} rc=0"
done

echo "--- [3/3] ABBA Summary (us/run ; GB/s of weights) ---"
python3 -I tools/gbsn.py "runs/${TAG}-op-A1.out" "runs/${TAG}-op-B1.out" "runs/${TAG}-op-B2.out" "runs/${TAG}-op-A2.out"
