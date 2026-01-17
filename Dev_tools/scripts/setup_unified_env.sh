#!/bin/bash
set -euo pipefail

DEV_TOOLS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$DEV_TOOLS_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "[setup] Reusing existing virtual environment at $VENV_DIR"
else
    echo "[setup] Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$DEV_TOOLS_DIR/requirements.txt"

echo "[setup] Environment ready. Launch the tool with scripts/run_unified_tool.sh"
