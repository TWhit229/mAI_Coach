#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/Dev_tools/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[run] Virtual environment not found. Run scripts/setup_unified_env.sh first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
python "$REPO_ROOT/Dev_tools/unified_tool.py"
