# mAI_Coach

### Real-time motion capture for strength training.

mAI_Coach is a research prototype that brings advanced computer vision to the weight room. By running Google's MediaPipe pose detection models directly on your iPhone, it provides instant feedback on your form—no cloud uploads, no latency, just results.

Accompanying the app is a powerful desktop suite for data curation, helping researchers label video datasets and fine-tune models to match real-world hygiene.

## Key Features

-   **Privacy-First AI**: Runs `pose_landmarker_lite.task` entirely on-device. Your workout video never leaves your phone.
-   **Live Augmentation**: Mirrors your camera feed with a real-time skeletal overlay, correcting your rep depth and path in the moment.
-   **Research-Grade Tooling**: A unified PySide6 desktop application for annotating video, trimming clips, and training custom classification models.

## Repository Structure

| Path | Description |
| :--- | :--- |
| **[`App Core/`](App%20Core/README.md)** | **The iOS Application.** Xcode workspace, SwiftUI views, and MediaPipe inference logic. |
| **[`Dev_tools/`](Dev_tools/README.md)** | **The Researcher's Toolkit.** A Python/Qt app for labeling data, visualizing landmarks, and training the bench press classifier. |
| `Documentation/` | Project requirements and contributing guidelines. |

## Quick Start

### 📱 I want to run the iOS App
Building the app requires a Mac with Xcode 15+.
1.  Navigate to `App Core/`.
2.  Run `pod install`.
3.  Open `mAICoach.xcworkspace` and hit Run on your device.

👉 **[See full iOS instructions](App%20Core/README.md)**

### 🖥️ I want to run the Developer Tools
The tools require Python 3.10+ and ffmpeg.
1.  Run `./scripts/setup_unified_env.sh` to build the environment.
2.  Run `./scripts/run_unified_tool.sh` to launch the GUI.

👉 **[See full Dev Tools instructions](Dev_tools/README.md)**

## Maintainers

-   **Travis Whitney** (`whitnetr@oregonstate.edu`)
-   **Cole Seifert** (`seiferco@oregonstate.edu`)

## License
Proprietary. See [`LICENSE.md`](LICENSE.md).
