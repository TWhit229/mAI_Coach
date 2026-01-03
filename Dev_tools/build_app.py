#!/usr/bin/env python3
"""
Build script for mAI Coach Tools using PyInstaller.
"""

import sys
import shutil
import subprocess
from pathlib import Path

# Base directory: Dev_tools
BASE_DIR = Path(__file__).resolve().parent

def run_build():
    entry_point = BASE_DIR / "unified_tool.py"
    
    # Clean previous builds
    shutil.rmtree(BASE_DIR / "build", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "dist", ignore_errors=True)
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "mAI Coach Tools",
        "--windowed",  # Mac .app
        "--icon", str(BASE_DIR / "app_icon.icns"),  # App icon
        "--noconfirm",
        "--clean",
        # Ensure we pick up the core and ui packages
        "--paths", str(BASE_DIR),
        # Include data directories
        "--add-data", f"{BASE_DIR / 'models'}:models",
        "--add-data", f"{BASE_DIR / 'data'}:data",
        # Hidden imports often missed by PyInstaller analysis
        "--hidden-import", "PySide6",
        "--hidden-import", "numpy",
        "--hidden-import", "cv2",
        "--hidden-import", "mediapipe",
        "--hidden-import", "mediapipe.tasks",
        "--hidden-import", "mediapipe.tasks.python",
        "--hidden-import", "mediapipe.tasks.python.vision",
        "--hidden-import", "mediapipe.tasks.cc",
        "--collect-all", "mediapipe",
        # Main entry script
        str(entry_point)
    ]
    
    print("Running PyInstaller...")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    
    if result.returncode == 0:
        print("\nBuild successful!")
        app_path = BASE_DIR / "dist" / "mAI Coach Tools.app"
        if app_path.exists():
            print(f"App Bundle: {app_path}")
    else:
        print("\nBuild failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_build()
