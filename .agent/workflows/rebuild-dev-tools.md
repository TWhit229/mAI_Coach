---
description: Rebuild the mAI Coach Tools.app after making changes to Dev_tools
---
# Rebuild Dev Tools App

After making changes to any Python files in `Dev_tools/`, run this to rebuild the .app bundle.

## Quick Rebuild

// turbo
1. Navigate to Dev_tools directory:
```bash
cd /Users/whitney/Documents/mAI_Coach/Dev_tools
```

// turbo
2. Activate the virtual environment and run the build script:
```bash
source .venv/bin/activate && python build_app.py
```

3. The new app will be created at `Dev_tools/dist/mAI Coach Tools.app`

## Notes

- The build takes ~3-4 minutes because it collects all dependencies (mediapipe, PyTorch, etc.)
- Code signing warnings are normal for development and can be ignored
- If you get "No module named PyInstaller", run: `pip install pyinstaller`
