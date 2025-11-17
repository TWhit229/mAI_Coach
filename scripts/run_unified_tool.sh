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
QT_PLUGIN_PATH="$(python - <<'PY'
from PySide6.QtCore import QLibraryInfo
try:
    from PySide6.QtCore import LibraryPath
except ImportError:
    LibraryPath = None

if LibraryPath is None:
    path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
else:
    path = QLibraryInfo.path(LibraryPath.PluginsPath)
print(path)
PY
)"
if [ -n "$QT_PLUGIN_PATH" ]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH"
fi

python "$REPO_ROOT/Dev_tools/unified_tool.py"
