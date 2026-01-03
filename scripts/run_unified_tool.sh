#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/Dev_tools/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[run] Virtual environment not found. Run scripts/setup_unified_env.sh first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Ensure Qt picks up its plugins/frameworks from the PySide6 wheel installation
eval "$(
python - <<'PY'
from pathlib import Path
import PySide6
import shlex

qt_root = Path(PySide6.__file__).resolve().parent / "Qt"
plugin_root = qt_root / "plugins"
platform_plugin_dir = plugin_root / "platforms"
lib_dir = qt_root / "lib"

def emit(name, path):
    print(f"{name}={shlex.quote(str(path))}")

emit("QT_BASE_DIR", qt_root)
emit("QT_PLUGIN_DIR", plugin_root)
emit("QT_PLATFORM_PLUGIN_DIR", platform_plugin_dir)
emit("QT_LIB_DIR", lib_dir)
PY
)"

if [[ "$(uname)" == "Darwin" ]] && command -v chflags >/dev/null; then
    # PySide wheels sometimes mark the bundled Qt tree as "hidden", which causes
    # Qt's plugin scanner to skip the dylibs entirely. Clear that flag eagerly.
    chflags -R nohidden "$QT_BASE_DIR" 2>/dev/null || true
fi

if [ -d "$QT_PLUGIN_DIR" ]; then
    if [ -n "${QT_PLUGIN_PATH:-}" ]; then
        export QT_PLUGIN_PATH="$QT_PLUGIN_DIR:$QT_PLUGIN_PATH"
    else
        export QT_PLUGIN_PATH="$QT_PLUGIN_DIR"
    fi
fi

if [ -d "$QT_PLATFORM_PLUGIN_DIR" ]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLATFORM_PLUGIN_DIR"
fi

if [ -d "$QT_LIB_DIR" ]; then
    if [ -n "${DYLD_FRAMEWORK_PATH:-}" ]; then
        export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$DYLD_FRAMEWORK_PATH"
    else
        export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR"
    fi
    if [ -n "${DYLD_LIBRARY_PATH:-}" ]; then
        export DYLD_LIBRARY_PATH="$QT_LIB_DIR:$DYLD_LIBRARY_PATH"
    else
        export DYLD_LIBRARY_PATH="$QT_LIB_DIR"
    fi
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-cocoa}"

python "$REPO_ROOT/Dev_tools/unified_tool.py"
