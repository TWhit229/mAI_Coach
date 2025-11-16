#!/usr/bin/env python3
"""
Cross-platform bootstrapper for the Dev_tools suite.

Run: python setup_env.py
- Detects the current OS
- Creates (or reuses) ./Dev_tools/.venv
- Installs requirements.txt inside that venv
- Prints activation instructions for the detected shell
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

IS_WINDOWS = os.name == "nt"


def run(cmd, env=None):
    """Run a subprocess and raise on failure after basic argument validation."""
    if not isinstance(cmd, (list, tuple)):
        raise TypeError("Command must be a list/tuple of arguments.")

    dangerous_chars = {";", "&", "|", "$", "`", ">", "<"}
    sanitized = []
    for arg in cmd:
        if isinstance(arg, Path):
            arg_str = str(arg)
        elif isinstance(arg, str):
            arg_str = arg
        else:
            raise TypeError(f"Command argument is not a string/path: {arg!r}")
        if any(ch in arg_str for ch in dangerous_chars):
            raise ValueError(f"Unsafe character in command argument: {arg_str!r}")
        sanitized.append(arg_str)

    print(f"[setup] Running: {' '.join(sanitized)}")
    subprocess.run(sanitized, cwd=ROOT, env=env, check=True)


def ensure_venv():
    if VENV_DIR.exists():
        print(f"[setup] Reusing existing virtual environment at {VENV_DIR}")
        return
    print(f"[setup] Creating virtual environment at {VENV_DIR}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python3"


def install_requirements():
    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Virtual environment python not found at: {py}")
    if not REQUIREMENTS.exists():
        raise SystemExit(f"Missing requirements file: {REQUIREMENTS}")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def activation_hint():
    shell = os.environ.get("SHELL", "")
    if IS_WINDOWS:
        powershell = "powershell" in (shell or "").lower()
        if powershell:
            cmd = ".venv\\Scripts\\Activate.ps1"
        else:
            cmd = ".venv\\Scripts\\activate.bat"
        print(f"\n[setup] Activate the environment with:\n    {cmd}")
    else:
        # best guess between bash/zsh/fish
        if "fish" in shell:
            cmd = "source .venv/bin/activate.fish"
        elif "csh" in shell:
            cmd = "source .venv/bin/activate.csh"
        else:
            cmd = "source .venv/bin/activate"
        print(f"\n[setup] Activate the environment with:\n    {cmd}")


def main():
    print(f"[setup] Host OS: {platform.system()} ({platform.platform()})")
    ensure_venv()
    install_requirements()
    activation_hint()
    print("\n[setup] Done. Launch the tool suite with:")
    print("    python tool_suite.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
