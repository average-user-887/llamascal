#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${1:?usage: dso_witness.sh <bin_dir> <binary>}"
BIN_NAME="${2:-llama-bench}"
BIN_REAL="$(readlink -f "$BIN_DIR")"
TARGET="$BIN_REAL/$BIN_NAME"

if [ ! -x "$TARGET" ]; then
  echo "ERROR: executable not found: $TARGET" >&2
  exit 1
fi

echo "BIN=$TARGET sha256=$(sha256sum "$TARGET" | awk '{print $1}')"

LDD_OUT="$(LD_LIBRARY_PATH="$BIN_REAL" ldd "$TARGET")"
FAIL=0
while IFS= read -r line; do
  case "$line" in
    *libggml*|*libllama*|*libmtmd*)
      path="$(echo "$line" | sed -n 's/.*=> \([^ ]*\) (.*/\1/p')"
      if [ -z "$path" ] || [ ! -e "$path" ]; then
        echo "ERROR: unresolved project DSO: $line" >&2
        FAIL=1
        continue
      fi
      rpath="$(readlink -f "$path")"
      case "$rpath" in
        "$BIN_REAL"/*) ;;
        *)
          echo "ERROR: DSO escaped BIN_DIR ($BIN_REAL): $rpath" >&2
          FAIL=1
          ;;
      esac
      sha="$(sha256sum "$rpath" | awk '{print $1}')"
      name="$(basename "$rpath")"
      echo "DSO=$name dir=$(basename "$BIN_REAL") sha256=$sha"
      ;;
  esac
done <<< "$LDD_OUT"

exit $FAIL
