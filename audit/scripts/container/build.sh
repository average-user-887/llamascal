#!/usr/bin/env bash
# build.sh <git-ref|patch-file>   Builds p104-llama:<short>-<UTC date>[-N]; never reuses a tag. No GPU used.
#   git-ref    : any ref in ~/p104-exp/wt's repo (e.g. 6768e5b, p104-exp)
#   patch-file : a patch (-p1) applied on top of BASE (default 5bacb7c), e.g. ~/p104-bench/patches/k1-v10-mmvq-pascal-kquant.patch
# DRY=1 prints plan only.
set -euo pipefail
ARG="${1:?usage: build.sh <git-ref|patch-file>}"
REPO="${REPO:-$HOME/p104-exp/wt}"; BASE="${BASE:-5bacb7c}"; JOBS="${JOBS:-2}"
HERE="$(cd "$(dirname "$0")" && pwd)"; IMG="$HOME/p104-bench/images"; mkdir -p "$IMG"
PATCH=""; REF="$ARG"
if [ -f "$ARG" ]; then PATCH="$(readlink -f "$ARG")"; REF="$BASE"; fi
COMMIT="$(git -C "$REPO" rev-parse --verify "$REF^{commit}")"; SHORT="${COMMIT:0:7}"
DIRTY=false
if [ -z "$PATCH" ] && [ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT" ] && [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then DIRTY=true; fi
[ -n "$PATCH" ] && SHORT="$SHORT-p$(sha256sum "$PATCH" | cut -c1-6)"
$DIRTY && echo "note: worktree has uncommitted changes; image is built from the COMMIT only (dirty=true recorded)" >&2
DATE="$(date -u +%Y%m%d)"; TAG="p104-llama:$SHORT-$DATE"; n=1
if [ "${DRY:-0}" = 1 ]; then n=1; else
  while docker image inspect "$TAG" >/dev/null 2>&1; do n=$((n+1)); TAG="p104-llama:$SHORT-$DATE-$n"; done
fi
CTX="$(mktemp -d "${TMPDIR:-/tmp}/p104ctx.XXXXXX")"; trap 'rm -rf "$CTX"' EXIT
mkdir "$CTX/src"; git -C "$REPO" archive "$COMMIT" | tar -x -C "$CTX/src"
if [ -n "$PATCH" ]; then cp "$PATCH" "$CTX/patch.diff"; else : > "$CTX/patch.diff"; fi
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
CMD=(docker build --build-arg "GIT_COMMIT=$COMMIT" --build-arg "JOBS=$JOBS" -t "$TAG" "$CTX")
if [ "${DRY:-0}" = 1 ]; then echo "DRY tag=$TAG commit=$COMMIT patch=${PATCH:-none} dirty=$DIRTY"; echo "DRY ${CMD[*]}"; exit 0; fi
nice -n 10 "${CMD[@]}"
ID="$(docker image inspect -f '{{.Id}}' "$TAG")"
H="$(docker run --rm --entrypoint cat "$TAG" /app/HASHES.txt)"
J="${TAG#p104-llama:}"
jq -n --arg tag "$TAG" --arg id "$ID" --arg commit "$COMMIT" --argjson dirty "$DIRTY" --arg patch "${PATCH:-}" \
  --arg patch_sha "$([ -n "$PATCH" ] && sha256sum "$PATCH" | cut -d' ' -f1 || true)" --arg hashes "$H" \
  --arg utc "$(date -u +%FT%TZ)" '{tag:$tag,image_id:$id,git_commit:$commit,dirty:$dirty,patch:$patch,patch_sha256:$patch_sha,built_utc:$utc,
   hashes:($hashes|split("\n")|map(select(length>0)|split("  ")|{(.[1]):.[0]})|add)}' > "$IMG/$J.json"
echo "built $TAG ($ID) -> $IMG/$J.json"
