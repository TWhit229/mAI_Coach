# mAI_Coach iOS App

This folder contains the source code for the mAI_Coach iOS application, which provides real-time motion capture and feedback for strength training using on-device MediaPipe inference.

## Prerequisites

- **Xcode**: Version 15.0 or later.
- **iOS SDK**: iOS 17.0 or later.
- **CocoaPods**: Required for managing dependencies.

## Getting Started

### 1. Installation

This project uses CocoaPods to manage dependencies (specifically MediaPipe tasks). If you haven't already, install the pods:

```bash
# From the App Core directory
pod install
```

> **Note**: The `Pods/` directory is not tracked in git to save space. You must run `pod install` to generate the workspace and download dependencies.

### 2. Opening the Project

**Crucial**: Always open the `.xcworkspace` file, not the `.xcodeproj` file.

```bash
open "mAICoach.xcworkspace"
```

### 3. Building and Running

1.  Select the **mAICoach** scheme in Xcode.
2.  Choose your target device. **Physical devices are highly recommended** because:
    -   The Simulator does not support camera input (needed for live coaching).
    -   On-device Neural Engine (ANE) performance is significantly better than CPU emulation.
3.  Press **Run** (Cmd+R).

## Architecture

-   **`PoseLandmarkerService.swift`**: The core logic for interacting with MediaPipe's Task API. It handles frame processing and landmark stream generation.
-   **`CoachView.swift`**: The main coaching interface. It renders the camera preview and overlays the skeleton/feedback on top.
-   **`RootView.swift`**: The app's entry point and navigation hub.

## Troubleshooting

-   **Camera Permissions**: If the camera preview is black, ensure you have granted camera permissions in Settings > Privacy & Security > Camera > mAI Coach.
-   **"No such module" errors**: Make sure you opened the`.xcworkspace` and built the `Pods` targets.
