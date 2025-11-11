# mAI_Coach

mAI_Coach is a research prototype that brings real-time motion capture to strength training. The iOS app streams camera frames through MediaPipe pose models to flag issues during a bench press, while the companion desktop tooling helps the team curate labeled video datasets for future models.

## Highlights
- **On-device inference:** SwiftUI app uses MediaPipe Tasks (`pose_landmarker_lite.task`) for low-latency landmark detection without sending video to the cloud.
- **Coach session overlay:** Bench session screen mirrors the front/back camera, draws pose landmarks live, and exposes quick camera toggles so athletes can self-check form.
- **Tooling for coaches & annotators:** Python/Tkinter utilities speed up video trimming, labeling, and overlay tuning to keep datasets consistent with the in-app experience.

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
All helper scripts live in `Dev_tools/` (see the folder README for details). Quick start:

```bash
cd Dev_tools
python setup_env.py      # creates .venv and installs requirements
source .venv/bin/activate
python tool_suite.py     # launches the unified GUI
```

From the launcher you can open:
- **Data Labeler (`bench_labeler.py`)** for tagging reps/issues with synchronized video + pose overlays.
- **Multi-video & single-video tuners** to dial in MediaPipe parameters across clips.
- **Auto Cut Video** to batch-export rep segments (requires system `ffmpeg`).

All tools expect the `.task` model files and `label_config.json` to remain beside the scripts; version-control any edits to the label config so the app and tools stay in sync.

## Documentation
- `Documentation/Requirements*.docx` captures the functional scope for CS462.
- `Documentation/CONTRIBUTING.md` describes how to file issues, branch, and submit changes.

## License
This repository is covered by the proprietary license in `LICENSE.md`. Do not redistribute outside the authorized team.

## Maintainers
- Travis Whitney - 830-832-4722 | whitnetr@oregonstate.edu
- Cole Seifert - 514-294-3114 | seiferco@oregonstate.edu
