#!/usr/bin/env bash
# swap-service.sh <image>   Promote an image to the serving slot (container p104-svc on :18100). Exit non-zero on failure,
# after automatically restoring the previous server (previous container image, or the systemd user unit llama-p104).
# NOTE blue/green is not possible with both copies resident: the P104 has 8 GiB and one server (Q4_K_M + 24k ctx KV) fills it.
# So the sequence is: validate candidate on spare port (old stopped) -> restart it as p104-svc:18100. Downtime = 2 model loads
# (second one is page-cache warm). DRY=1 prints commands only. Env: SPARE_PORT (18110), HEALTH_TIMEOUT (240), LOCK, UNIT,
# MEM (cgroup cap, 5g), MODEL, ALIAS, CTX, NP, SRV_FLAGS, SERVING_ENV (see "Serving config" below).
set -euo pipefail
NEW="${1:?usage: swap-service.sh <image>}"
UUID="${UUID:-<gpu-uuid>}"; SPARE="${SPARE_PORT:-18110}"; PROD=18100; UNIT="${UNIT:-llama-p104.service}"
LOCK="${LOCK:-$HOME/.local/state/lab-exp/gpu-bench-p104.lock}"; IMG="$HOME/p104-bench/images"; TMO="${HEALTH_TIMEOUT:-240}"
# Serving config (defaults = approved 2026-11-24 config: 1 slot + MTP, -c 49152). Override via env or the optional file
# $SERVING_ENV (default container/serving.env, sourced if present): MODEL (path inside the container), ALIAS, CTX, NP, MEM,
# SRV_FLAGS (whole flag string after -m/--alias/--host/--port; if set it replaces the defaults, CTX/NP then unused).
[ -f "${SERVING_ENV:-$(dirname "$(readlink -f "$0")")/serving.env}" ] && . "${SERVING_ENV:-$(dirname "$(readlink -f "$0")")/serving.env}"
MODEL="${MODEL:-/models/experimental/qwen35-9b-mtp/Qwen3.5-9B-MTP-Q4_K_M.gguf}"; ALIAS="${ALIAS:-qwen35-9b-mtp-q4km}"
CTX="${CTX:-49152}"; NP="${NP:-1}"
SRV_FLAGS="${SRV_FLAGS:--ngl 99 -c $CTX -np $NP -fa on -ctk f16 -ctv f16 -ub 512 -t 4 --cache-ram 1024 --jinja --metrics --no-warmup --spec-type draft-mtp --spec-draft-n-max 2}"
SRV_ENV="${SRV_ENV:-}"   # extra container env, space separated K=V (e.g. "P104_MTP_DRAFT_VOCAB=65536"); 2026-11-25
ENV_ARGS=(); for kv in $SRV_ENV; do ENV_ARGS+=(-e "$kv"); done
# shellcheck disable=SC2206
SRV_ARGS=(llama-server -m "$MODEL" --alias "$ALIAS" $SRV_FLAGS --host 0.0.0.0 --port "$PROD")
run() { if [ "${DRY:-0}" = 1 ]; then echo "DRY $(printf '%q ' "$@")"; else "$@"; fi; }
start_ctr() {  # name host_bind host_port restart image
  run docker run -d --name "$1" --init --gpus "device=$UUID" -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${ENV_ARGS[@]}" \
    --read-only --tmpfs /tmp:rw,noexec,size=256m -e HOME=/tmp --cap-drop ALL --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" -v "$HOME/models:/models:ro" --memory "${MEM:-5g}" --memory-swap "${MEM:-5g}" --cpus 4 --shm-size 512m --pids-limit 512 \
    --restart "$4" --label lab.host=z4-p104 --label lab.gpu=p104 --label "lab.experiment=$1" --label lab.purpose=serving \
    --label lab.owner=p104-orchestrator --label "lab.image=$5" -p "$2:$3:$PROD" "$5" "${SRV_ARGS[@]}"
}
wait_health() {  # port
  [ "${DRY:-0}" = 1 ] && { echo "DRY wait /health on :$1"; return 0; }
  local i; for ((i=0;i<TMO;i+=3)); do curl -fsS -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 3; done; return 1; }
smoke() {  # port: alias listed + one chat completion
  [ "${DRY:-0}" = 1 ] && { echo "DRY smoke :$1"; return 0; }
  curl -fsS -m 10 "http://127.0.0.1:$1/v1/models" | jq -e --arg a "$ALIAS" '[.data[].id]|index($a)!=null' >/dev/null || return 1
  curl -fsS -m 120 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | jq -e '.choices[0].finish_reason!=null and (.choices[0].message|(.content//.reasoning_content//"")|length>0)' >/dev/null; }

if [ -z "${P104_LOCKED:-}" ] && [ "${DRY:-0}" != 1 ]; then
  mkdir -p "$(dirname "$LOCK")"; P104_LOCKED=1 exec flock "$LOCK" "$0" "$NEW"; fi
[ "${DRY:-0}" = 1 ] || docker image inspect "$NEW" >/dev/null
if [ "${DRY:-0}" != 1 ] && [ -n "$(docker ps -q --filter label=lab.gpu=p104 --filter label=lab.purpose=experiment)" ]; then
  echo "an experiment container is running; refusing" >&2; exit 64; fi

# ---- what is serving now?
OLD_KIND=none; OLD_IMG=""
if [ "${DRY:-0}" != 1 ] && docker inspect p104-svc >/dev/null 2>&1; then
  OLD_KIND=container; OLD_IMG="$(docker inspect -f '{{.Config.Image}}' p104-svc)"
elif [ "${DRY:-0}" = 1 ] || systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then OLD_KIND=systemd; OLD_IMG="systemd:$UNIT"; fi
echo "old=$OLD_KIND ($OLD_IMG) new=$NEW"

stop_old() {
  case "$OLD_KIND" in
    container) run docker stop -t 30 p104-svc; run docker rename p104-svc p104-svc-old ;;
    systemd) run systemctl --user stop "$UNIT" ;;
  esac; }
restore_old() {
  echo "ROLLBACK to $OLD_KIND $OLD_IMG" >&2
  docker rm -f p104-svc-new p104-svc >/dev/null 2>&1 || true
  case "$OLD_KIND" in
    container) docker rename p104-svc-old p104-svc 2>/dev/null || true
               docker update --restart unless-stopped p104-svc >/dev/null 2>&1 || true; docker start p104-svc ;;
    systemd) systemctl --user start "$UNIT" ;;
  esac
  [ "$OLD_KIND" = none ] || wait_health "$PROD" || echo "WARNING: old server not healthy after rollback" >&2; }
fail() { echo "FAILED: $1" >&2; [ "${DRY:-0}" = 1 ] || restore_old; exit 1; }

run docker rm -f p104-svc-new 2>/dev/null || true
docker rm -f p104-svc-old >/dev/null 2>&1 || true   # leftover from a previous swap
stop_old
# ---- candidate on the spare port (localhost only), restart=no
start_ctr p104-svc-new 127.0.0.1 "$SPARE" no "$NEW" || fail "candidate start"
wait_health "$SPARE" || { [ "${DRY:-0}" = 1 ] || docker logs --tail 30 p104-svc-new >&2 || true; fail "candidate /health timeout"; }
smoke "$SPARE" || fail "candidate smoke test (alias/chat completion)"
run docker rm -f p104-svc-new
# ---- production name and port
start_ctr p104-svc 0.0.0.0 "$PROD" unless-stopped "$NEW" || fail "production start"
wait_health "$PROD" || fail "production /health timeout"
smoke "$PROD" || fail "production smoke test"
if [ "${DRY:-0}" != 1 ]; then
  mkdir -p "$IMG"; [ -f "$IMG/CURRENT" ] && cp "$IMG/CURRENT" "$IMG/PREVIOUS" || echo "$OLD_IMG" > "$IMG/PREVIOUS"
  [ "$OLD_KIND" = systemd ] && echo "$OLD_IMG" > "$IMG/PREVIOUS"
  echo "$NEW" > "$IMG/CURRENT"
  docker rm -f p104-svc-old >/dev/null 2>&1 || true
fi
echo "OK: $NEW serving on :$PROD (container p104-svc)"
# Restart policy: docker "unless-stopped" restarts on crash and at boot (docker.service is enabled). If a systemd-managed
# unit is preferred, wrap: ExecStart=/usr/bin/docker start -a p104-svc / ExecStop=/usr/bin/docker stop p104-svc as a user unit
# (needs docker group), and drop --restart. Keep the old host unit DISABLED once p104-svc is live or both fight for :18100/VRAM.
