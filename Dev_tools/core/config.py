"""Shared configuration constants for Dev_tools."""

import sys
from pathlib import Path

# Assume this file is in Dev_tools/core/config.py
# DEV_ROOT should be Dev_tools/

if getattr(sys, "frozen", False):
    # Running in a PyInstaller bundle
    # Default to the expected location in Documents so data persists
    DEV_ROOT = Path.home() / "Documents" / "mAI_Coach" / "Dev_tools"
else:
    DEV_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DEV_ROOT / "data"
MODEL_DIR = DEV_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
