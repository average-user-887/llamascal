#!/usr/bin/env bash
# run-experiment.sh TAG IMAGE [--purpose TEXT] [--serve PORT] -- <command inside image, e.g. llama-bench -m /models/...>
# The ONLY way experiments run. Fleetdash convention: `flock <gpu bench_lock> docker run --name <title> ...`
# (we re-exec ourselves under flock, so the lock holder's argv shows run-experiment.sh; the docker --name is p104-<TAG>).
# Env: LOCK (default ~/.local/state/lab-exp/gpu-bench-p104.lock)  MEM (5g)  CPUS (4)  SHM (512m)  FLEETREG_ENABLE=1  DRY=1  ALLOW_PORT=1
# Thermal guard (host nvidia-smi, bus <pci-bus-id>): cooldown <=COOL_C before start; docker kill on >=TRIP_C (limits from ~/p104-bench/guard.env: 88/83 warn/50/900 s; env COOL_C overrides) or any
# hw_thermal/sw_thermal/hw_slowdown flag. Writes runs/TAG.{trace.csv,meta.json,out}. Exit 2 = thermal trip, 3 = cooldown timeout.
set -euo pipefail
TAG="${1:?usage: run-experiment.sh TAG IMAGE [--purpose T] [--serve PORT] -- cmd...}"; IMAGE="${2:?image}"; shift 2
PURPOSE="experiment"; PORT=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in --purpose) PURPOSE="$2"; shift 2;; --serve) PORT="$2"; shift 2;; *) echo "unknown opt $1" >&2; exit 64;; esac
done
[ "${1:-}" = "--" ] || { echo "missing -- before the container command" >&2; exit 64; }; shift
[ $# -gt 0 ] || { echo "empty container command" >&2; exit 64; }
CMD=("$@")
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad TAG (use [A-Za-z0-9_.-])" >&2; exit 64; }
UUID="${UUID:-<gpu-uuid>}"   # P104-100. Never the P620.
BUS="${BUS:-$UUID}"; gv() { sed -n "s/^$1=\([0-9][0-9]*\)[[:space:]]*$/\1/p" "${GUARD_ENV:-$HOME/p104-bench/guard.env}" 2>/dev/null | tail -1; }
TRIP_C="$(gv TRIP_C)"; TRIP_C="${TRIP_C:-88}"; WARN_C="$(gv WARN_C)"; WARN_C="${WARN_C:-83}"; COOL_T="$(gv COOL_TIMEOUT_S)"; COOL_T="${COOL_T:-900}"
COOL_C="${COOL_C:-$(gv COOL_C)}"; COOL_C="${COOL_C:-50}"; [[ "$COOL_C" =~ ^[0-9]+$ ]] || { echo "COOL_C must be an integer" >&2; exit 64; }
LIM="${GUARD_TRIP:-$TRIP_C}"; [[ "$LIM" =~ ^[0-9]+$ ]] && [ "$LIM" -le "$TRIP_C" ] || { echo "GUARD_TRIP must be an integer <= $TRIP_C" >&2; exit 64; }   # lower-only override, for dry tests
LOCK="${LOCK:-$HOME/.local/state/lab-exp/gpu-bench-p104.lock}"
RUNS="$HOME/p104-bench/runs"; NAME="p104-$TAG"
if [ -n "$PORT" ]; then
  if ! { [ "$PORT" -ge 18199 ] && [ "$PORT" -le 18499 ]; } 2>/dev/null && [ "${ALLOW_PORT:-0}" != 1 ]; then
    echo "port $PORT outside 18199..18499 (set ALLOW_PORT=1 to override)" >&2; exit 64; fi
  [ "$PORT" != 18100 ] || [ "${ALLOW_PORT:-0}" = 1 ] || { echo "18100 is the production port (use swap-service.sh)" >&2; exit 64; }
fi
LBL=(--label lab.host=z4-p104 --label lab.gpu=p104 --label "lab.experiment=$TAG" --label "lab.purpose=$PURPOSE" \
     --label lab.owner=p104-orchestrator --label "lab.image=$IMAGE")
RUN=(docker run --rm --init --name "$NAME" --gpus "device=$UUID" -e CUDA_DEVICE_ORDER=PCI_BUS_ID
     --read-only --tmpfs /tmp:rw,noexec,size=256m -e HOME=/tmp --cap-drop ALL --security-opt no-new-privileges
     --user "$(id -u):$(id -g)" -v "$HOME/models:/models:ro" -v "$RUNS:/runs"
     --memory "${MEM:-5g}" --memory-swap "${MEM:-5g}" --cpus "${CPUS:-4}" --shm-size "${SHM:-512m}" --pids-limit 512
     "${LBL[@]}")
[ -n "$PORT" ] && RUN+=(-p "127.0.0.1:$PORT:$PORT")
for kv in ${XENV:-}; do RUN+=(-e "$kv"); done   # extra container env, space separated K=V (2026-11-25 release)
RUN+=("$IMAGE" "${CMD[@]}")
if [ "${DRY:-0}" = 1 ]; then
  echo "DRY lock=$LOCK"; echo "DRY flock $LOCK $(printf '%q ' "${RUN[@]}")"; exit 0; fi

mkdir -p "$(dirname "$LOCK")" "$RUNS"
if [ -z "${P104_LOCKED:-}" ]; then
  echo "waiting for GPU lock $LOCK ..." >&2
  P104_LOCKED=1 exec flock "$LOCK" "$0" "$TAG" "$IMAGE" --purpose "$PURPOSE" ${PORT:+--serve "$PORT"} -- "${CMD[@]}"
fi
# ---- from here we hold the lock (inherited by children; docker CLI does not outlive us)
FR() { [ "${FLEETREG_ENABLE:-0}" = 1 ] && timeout 5 python3 -I "$HOME/p104-bench/fleetreg.py" "$@" </dev/null >/dev/null 2>&1 || true; }
Q="temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_power_cap"
samp() { nvidia-smi -i "$BUS" --query-gpu="$Q" --format=csv,noheader,nounits 2>/dev/null; }
DPID=""; RC=1; TRIP=""; WARNED=""
cleanup() {
  docker kill "$NAME" >/dev/null 2>&1 || true
  [ -n "$DPID" ] && { kill "$DPID" 2>/dev/null || true; wait "$DPID" 2>/dev/null || true; }
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  FR finish --tag "$TAG" --result "rc=$RC${TRIP:+ TRIP}"
}
trap cleanup EXIT; trap 'RC=130; exit 130' INT TERM HUP
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then echo "container $NAME already exists" >&2; RC=64; exit 64; fi
if [ -n "$(docker ps -q --filter label=lab.gpu=p104)" ]; then echo "another p104 container is running; refusing" >&2; RC=64; exit 64; fi
T0=$SECONDS
while :; do
  S="$(samp)" || S=""; t="${S%%,*}"
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "cannot read GPU temperature" >&2; RC=3; exit 3; }
  [ "$t" -le "$COOL_C" ] && break
  [ $((SECONDS-T0)) -gt "$COOL_T" ] && { echo "cooldown timeout: $S" >&2; RC=3; exit 3; }
  sleep 5
done
START="$S"; FR start --tag "$TAG" --purpose "$PURPOSE" --gpu-uuid "$UUID" --pid $$ ${PORT:+--port "$PORT"}
echo "t,$Q" > "$RUNS/$TAG.trace.csv"
"${RUN[@]}" > "$RUNS/$TAG.out" 2>&1 &
DPID=$!; TS=$(date +%s.%N); MX=0
while kill -0 "$DPID" 2>/dev/null; do
  S="$(samp)" || S=""
  if [ -n "$S" ]; then
    printf '%.1f,%s\n' "$(echo "$(date +%s.%N) - $TS" | bc)" "$S" >> "$RUNS/$TAG.trace.csv"
    IFS=, read -r T _ _ _ _ _ H1 H2 H3 _ <<<"${S// /}"
    [[ "$T" =~ ^[0-9]+$ ]] && [ "$T" -gt "$MX" ] && MX=$T
    [[ "$T" =~ ^[0-9]+$ ]] && [ "$T" -ge "$WARN_C" ] && [ -z "$WARNED" ] && { WARNED=1; echo "WARN temp ${T}C >= ${WARN_C}C" >&2; }
    if { [[ "$T" =~ ^[0-9]+$ ]] && [ "$T" -ge "$LIM" ]; } || [ "$H1" = Active ] || [ "$H2" = Active ] || [ "$H3" = Active ]; then
      TRIP="$S"; while kill -0 "$DPID" 2>/dev/null; do docker kill "$NAME" >/dev/null 2>&1 || true; sleep 0.3; done; break   # retry: the container may not exist yet on the first sample
    fi
  fi
  sleep 0.5
done
wait "$DPID" 2>/dev/null && RC=0 || RC=$?; DPID=""
END="$(samp || true)"
jq -n --arg tag "$TAG" --arg image "$IMAGE" --arg cmd "${CMD[*]}" --arg start "$START" --arg end "$END" --arg trip "$TRIP" \
  --argjson mx "$MX" --argjson rc "$RC" --argjson secs "$((SECONDS-T0))" --arg utc "$(date -u +%FT%TZ)" --arg name "$NAME" --argjson trip_c "$LIM" --argjson warn_c "$WARN_C" --argjson cool_c "$COOL_C" --argjson warned "${WARNED:-0}" --argjson start_temp "${START%%,*}" \
  '{tag:$tag,image:$image,container:$name,cmd:$cmd,start:$start,end:$end,max_temp:$mx,trip:(if $trip=="" then null else $trip end),rc:$rc,secs:$secs,utc:$utc,trip_c:$trip_c,warn_c:$warn_c,cool_c:$cool_c,warn:($warned==1),start_temp:$start_temp}' > "$RUNS/$TAG.meta.json"
[ -n "$TRIP" ] && { echo "TRIP $TRIP" >&2; RC=2; exit 2; }
exit "$RC"
