#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/Dev_tools/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[run] Virtual environment not found. Run scripts/setup_unified_env.sh first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Ensure Qt can find its platform plugins (notably "cocoa" on macOS)
# Determine the plugin directory directly from the PySide6 install
QT_PLUGIN_PATH="$(python - <<'PY'
from pathlib import Path
import PySide6

plugin_path = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
print(plugin_path)
PY
)"
if [ -n "$QT_PLUGIN_PATH" ]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH"
fi

python "$REPO_ROOT/Dev_tools/unified_tool.py"
