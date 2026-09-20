#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${VAST_TRELLIS_INSTALL_PATH:-/usr/local/bin/vast-trellis2}"
LIB_DIR="${VAST_TRELLIS_LIB_DIR:-/usr/local/lib/vast-trellis2}"

required=(
  "$ROOT_DIR/vast-trellis2"
  "$ROOT_DIR/vast-trellis2-doctor"
  "$ROOT_DIR/lib/common.sh"
  "$ROOT_DIR/lib/install.sh"
  "$ROOT_DIR/lib/download.sh"
  "$ROOT_DIR/lib/runtime.sh"
  "$ROOT_DIR/lib/lifecycle.sh"
  "$ROOT_DIR/scripts/patch_trellis.py"
  "$ROOT_DIR/scripts/convert_dinov3_local.py"
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: required file missing: $f" >&2; exit 1; }
done

mkdir -p "$LIB_DIR/lib" "$LIB_DIR/scripts" "$(dirname "$DEST")"
install -m 0755 "$ROOT_DIR/vast-trellis2" "$DEST"
install -m 0755 "$ROOT_DIR/vast-trellis2-doctor" "$LIB_DIR/vast-trellis-doctor"
install -m 0644 "$ROOT_DIR/lib/common.sh" "$LIB_DIR/lib/common.sh"
install -m 0644 "$ROOT_DIR/lib/install.sh" "$LIB_DIR/lib/install.sh"
install -m 0644 "$ROOT_DIR/lib/download.sh" "$LIB_DIR/lib/download.sh"
install -m 0644 "$ROOT_DIR/lib/runtime.sh" "$LIB_DIR/lib/runtime.sh"
install -m 0644 "$ROOT_DIR/lib/lifecycle.sh" "$LIB_DIR/lib/lifecycle.sh"
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
  vast-trellis2 install
  vast-trellis2 doctor
  vast-trellis2 doctor --fix
  vast-trellis2 auth
  vast-trellis2 download
  vast-trellis2 download --direct
  vast-trellis2 hf-speedtest
  vast-trellis2 run
  vast-trellis2 start
  vast-trellis2 stop
  vast-trellis2 status
  vast-trellis2 tunnel

The wizard detects effective cgroup RAM (not just host RAM), GPU VRAM/compute
capability, CUDA Toolkit, CPU count and safe CUDA build concurrency.
The lifecycle manager also discovers orphaned TRELLIS app.py processes so old
GPU/RAM-heavy servers can be stopped even if their original PID file was lost.
EOF
