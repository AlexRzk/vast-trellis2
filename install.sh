#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/vast-trellis2"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: vast-trellis2 executable not found next to install.sh" >&2
  exit 1
fi

chmod +x "$SRC"

TARGET_DIR=/usr/local/bin
if [[ ! -d "$TARGET_DIR" || ! -w "$TARGET_DIR" ]]; then
  TARGET_DIR="$HOME/.local/bin"
  mkdir -p "$TARGET_DIR"
fi

TARGET="$TARGET_DIR/vast-trellis2"
ln -sf "$SRC" "$TARGET"

echo "Installed: $TARGET -> $SRC"

case ":$PATH:" in
  *":$TARGET_DIR:"*) ;;
  *)
    echo
    echo "Add this directory to PATH for future shells:"
    echo "  export PATH=\"$TARGET_DIR:\$PATH\""
    ;;
esac

echo
echo "Next step:"
echo "  $TARGET wizard"
