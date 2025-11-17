# mAI_Coach

mAI_Coach is a research prototype that brings real-time motion capture to strength training. The iOS app streams camera frames through MediaPipe pose models to flag issues during a bench press, while the companion desktop tooling (a unified PySide6 app) helps the team curate labeled video datasets for future models.

## Highlights
- **On-device inference:** SwiftUI app uses MediaPipe Tasks (`pose_landmarker_lite.task`) for low-latency landmark detection without sending video to the cloud.
- **Coach session overlay:** Bench session screen mirrors the front/back camera and draws pose landmarks live so athletes can self-check form.
- **Tooling for coaches & annotators:** Python/PySide6 utilities speed up video trimming, labeling, and overlay tuning to keep datasets consistent with the in-app experience.

## Repository layout
| Path | Purpose |
| --- | --- |
| `App Core/` | Xcode workspace, SwiftUI sources, resources, and MediaPipe task file for the iOS coach app. |
| `Dev_tools/` | Python virtual-environment setup plus labeling/tuning utilities that share the same pose models and label config. |
| `Documentation/` | Requirements documents and CONTRIBUTING guide. |
| `LICENSE.md` | Proprietary license terms for this project. |

## iOS coach app
### Requirements
- macOS with Xcode 15 (or newer) and the latest iOS SDK.
- A device or simulator running iOS 17+ (real hardware recommended for camera + neural processing performance).
- MediaPipe task file already supplied at `App Core/Resources/pose_landmarker_lite.task`.

### Build & run
1. Open the workspace: `open "App Core/mAICoach.xcworkspace"`.
2. Select the `mAICoach` scheme and your target device/simulator.
3. Run from Xcode. When the Bench session screen appears, grant camera permission to see live pose overlays.

### Customization notes
- The pose thresholds (confidence/tracking) live in `PoseLandmarkerService.swift`. Adjust them if you swap in heavier MediaPipe models.
- Additional lift types can be added by extending `CoachView` navigation targets and providing the relevant session views.

## Developer tool suite (Python)
All helper scripts now live inside the single-window PySide6 app at `Dev_tools/unified_tool.py`. Quick start:

```bash
# create/refresh the Dev_tools/.venv virtual environment
./scripts/setup_unified_env.sh

# launch the unified GUI
./scripts/run_unified_tool.sh
```

The window exposes the labeler (issue tagging, metadata forms, per-frame annotations), the automatic clipper, and the pose-tuner grid. Configuration lives in `Dev_tools/label_config.json`; edit it from inside the tool or by hand if you need to version-control label changes.

### Archived utilities

Legacy Tkinter scripts (`auto_cut_video.py`, `multi_video_pose_tuner.py`, `pose_tasks_overlay_tuner.py`, etc.) were removed from the active workflow. If you need to reference them, check the git history prior to the unified tool rollout.

## Documentation
- `Documentation/Requirements*.docx` captures the functional scope for CS462.
- `Documentation/CONTRIBUTING.md` describes how to file issues, branch, and submit changes.

## License
This repository is covered by the proprietary license in `LICENSE.md`. Do not redistribute outside the authorized team.

## Maintainers
- Travis Whitney - 830-832-4722 | whitnetr@oregonstate.edu
- Cole Seifert - 514-294-3114 | seiferco@oregonstate.edu
