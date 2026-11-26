#!/usr/bin/env bash
# gpu-window.sh {start|stop}   (run via: sg docker -c './gpu-window.sh stop')
#   stop  : docker stop p104-svc (frees the P104; container and its restart policy stay; records state)
#   start : docker start p104-svc and wait for /health (HEALTH_TIMEOUT, default 240 s)
# Typical: gpu-window.sh stop; run-experiment.sh ...; gpu-window.sh start
set -euo pipefail
N=p104-svc; TMO="${HEALTH_TIMEOUT:-240}"
case "${1:-}" in
  stop)
    docker inspect "$N" >/dev/null 2>&1 || { echo "no container $N" >&2; exit 1; }
    docker stop -t 30 "$N" >/dev/null
    for i in $(seq 1 30); do [ -z "$(nvidia-smi -i <pci-bus-id> --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
    U="$(nvidia-smi -i <pci-bus-id> --query-gpu=memory.used --format=csv,noheader,nounits)"
    echo "window open: $N stopped, GPU memory used ${U} MiB"; [ "$U" -lt 600 ] || { echo "GPU not free" >&2; exit 1; } ;;
  start)
    docker start "$N" >/dev/null
    for ((i=0;i<TMO;i+=3)); do curl -fsS -m 3 http://127.0.0.1:18100/health >/dev/null 2>&1 && { echo "window closed: $N healthy"; exit 0; }; sleep 3; done
    echo "FAILED: $N not healthy in ${TMO}s" >&2; docker logs --tail 20 "$N" >&2; exit 1 ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 64 ;;
esac
