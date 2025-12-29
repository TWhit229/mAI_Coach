#!/usr/bin/env python3
"""
Unified Developer Tool for mAI Coach.
Refactored into a modular structure (Dev_tools/ui, Dev_tools/core).

Modules:
  - ui/main.py: Main entry point and window
  - ui/admin.py: Configuration, tagging, and training
  - ui/labeler.py: Video labeling
  - ui/video_cut.py: Video splitting
  - ui/pose_tuner.py: Pose tracking visualization

Shared:
  - core/config.py
  - core/metrics.py
  - core/utils.py
  - core/video.py
  - core/training.py
  - label_config.py
"""

from ui.main import main

if __name__ == "__main__":
    main()
