# Dev Tools Suite

The **Dev Tools Suite** is a unified desktop application built with PySide6 (Qt). It allows researchers to:
-   **Label** lifting videos with frame-perfect precision.
-   **Tune** MediaPipe tracking parameters in real-time.
-   **Train** lightweight MLP models for rep classification.

It shares the strict `label_config.json` and `.task` model files with the iOS app to ensure consistency between training (desktop) and inference (mobile).

## Setup & Installation

### Option A: The "Just Work" Script (Recommended)
From the repository root, run:
```bash
./scripts/setup_unified_env.sh
```
This will automatically create a virtual environment in `Dev_tools/.venv`, upgrade pip, and install all requirements.

### Option B: Manual Setup
If you prefer to manage your own environment:
```bash
cd Dev_tools
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> **Requirement**: You must have `ffmpeg` installed on your system path for the video clipper to work.
> - **macOS**: `brew install ffmpeg`
> - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org)

## Running the App

Once setup is complete, launch the dashboard from the repo root:

```bash
./scripts/run_unified_tool.sh
```

## Features Guide

### 🏷️ Labeler
Load folders of videos and their corresponding JSONs. Use the transport controls to scrub through footage, assign standard labels (defined in `label_config.json`), and modify rep metadata.

### ✂️ Video Cutter
A wrapper around ffmpeg. Mark `IN` and `OUT` points to extract specific reps or sets from long recording sessions.

### 🎛️ Pose Tuner
Visualize how different MediaPipe parameters (confidence thresholds, smoothing filters) affect landmark stability. Shows up to 4 videos in a grid for side-by-side comparison.

### 🧠 Model Trainer
1.  **Preprocess**: Convert your folder of labeled JSONs into NumPy tensors (`.npy`).
2.  **Train**: Train a multi-label MLP on those tensors.
3.  **Export**: The tool saves `.pt` (PyTorch) and `.json` (Weights) files ready for iOS deployment.

## File Structure

-   `data/`: Stores dataset JSONs and generated numpy tensors.
-   `models/`: Stores `.task` files (MediaPipe) and validation results.
-   `unified_tool.py`: The entry point for the GUI.
