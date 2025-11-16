#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/Dev_tools/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "[setup] Reusing existing virtual environment at $VENV_DIR"
else
    echo "[setup] Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/Dev_tools/requirements.txt"

echo "[setup] Environment ready. Launch the tool with scripts/run_unified_tool.sh"
