#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${VAST_TRELLIS_INSTALL_PATH:-/usr/local/bin/vast-trellis2}"
LIB_DIR="${VAST_TRELLIS_LIB_DIR:-/usr/local/lib/vast-trellis2}"

[[ -f "$ROOT_DIR/vast-trellis2" ]] || { echo "ERROR: vast-trellis2 missing" >&2; exit 1; }
[[ -f "$ROOT_DIR/vast-trellis2-doctor" ]] || { echo "ERROR: vast-trellis2-doctor missing" >&2; exit 1; }
[[ -f "$ROOT_DIR/scripts/patch_trellis.py" ]] || { echo "ERROR: patch utility missing" >&2; exit 1; }
[[ -f "$ROOT_DIR/scripts/convert_dinov3_local.py" ]] || { echo "ERROR: DINO converter missing" >&2; exit 1; }

mkdir -p "$LIB_DIR/scripts" "$(dirname "$DEST")"
install -m 0755 "$ROOT_DIR/vast-trellis2" "$DEST"
install -m 0755 "$ROOT_DIR/vast-trellis2-doctor" "$LIB_DIR/vast-trellis-doctor"
install -m 0755 "$ROOT_DIR/scripts/patch_trellis.py" "$LIB_DIR/scripts/patch_trellis.py"
install -m 0755 "$ROOT_DIR/scripts/convert_dinov3_local.py" "$LIB_DIR/scripts/convert_dinov3_local.py"

BASE="${VAST_TRELLIS_BASE:-/workspace}"
[[ -d "$BASE" && -w "$BASE" ]] || BASE="$HOME"
mkdir -p "$BASE/.vast-trellis2/logs" "$BASE/.vast-trellis2/stages" "$BASE/huggingface"

cat <<EOF
Installed Vast TRELLIS.2 Wizard

  launcher: $DEST
  library:  $LIB_DIR

Start here:
  vast-trellis2 wizard

Useful commands:
  vast-trellis2 detect
  vast-trellis2 doctor
  vast-trellis2 doctor --fix
  vast-trellis2 auth
  vast-trellis2 download
  vast-trellis2 run
  vast-trellis2 start
  vast-trellis2 status
  vast-trellis2 tunnel

The wizard detects effective cgroup RAM (not just host RAM), GPU VRAM/compute
capability, CUDA Toolkit, and CPU count before selecting build concurrency.
EOF
