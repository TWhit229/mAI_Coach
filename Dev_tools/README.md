# Dev Tools Suite

Tkinter utilities for reviewing and labeling lifting videos live inside this `Dev_tools` folder. The scripts share the same MediaPipe pose models (`pose_landmarker_*.task`) and the same label configuration (`label_config.json`). Use the Tool Suite launcher to keep everything in one place.

## Requirements

- Python 3.10+ with Tk support (standard on macOS/Linux; on Windows install the official build).
- System `ffmpeg` binary in `PATH` (needed for `auto_cut_video.py` exports). On macOS use `brew install ffmpeg`, on Windows install from https://ffmpeg.org.
- Python packages listed in `requirements.txt`.

Quick start (auto-detects Windows/macOS/Linux + shell):

```bash
cd Dev_tools
python setup_env.py
```

The script will:
1. Create (or reuse) `.venv` in this folder.
2. Install everything from `requirements.txt` inside that venv.
3. Print the exact activation command for your shell.

Prefer to do the steps manually? Skip the script and run:

```bash
cd Dev_tools
python3 -m venv .venv            # Windows: py -3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the suite

Launch the central menu:

```bash
cd Dev_tools
python tool_suite.py
```

The window exposes buttons for each tool. Clicking a button spawns that script in its own process/window, so you can run multiple utilities at the same time.

### Admin panel (labels & tags)

From the launcher, click **Open Admin Panel** to edit:

- Movement / lift names shown in the Data Labeler dropdown.
- Issue tag list used throughout the labeling workflow.

Changes are saved to `label_config.json` in this folder. You can also edit that JSON manually if needed (the file is created the first time the suite runs).

## Individual tools

- **Data Labeler (`bench_labeler.py`)**  
  Opens the labeling UI. Pick your videos and dataset directory when prompted. The tool automatically loads pose data from the paired JSON files, overlays landmarks on the video, and lets you tag issues, reps, and metadata. Playback starts automatically; use the scrubber or transport controls to review frames.

- **Multi Video Pose Tuner (`multi_video_pose_tuner.py`)**  
  Lets you preview several videos in a grid while adjusting one shared set of MediaPipe parameters (model variant, thresholds, smoothing, etc.). Ideal for quickly dialing in settings across different angles.

- **Pose Tasks Overlay Tuner (`pose_tasks_overlay_tuner.py`)**  
  Single-video tuner with detailed sliders and on-screen legends. Use this when you need granular control of overlay drawing and smoothing for a specific clip.

- **Auto Cut Video (`auto_cut_video.py`)**  
  A lightweight rep-cutting assistant. Mark bottom positions, adjust start/end padding, and export segments through `ffmpeg`. Make sure `ffmpeg` is installed before launching.

All scripts assume the `.task` model files remain beside them. If you relocate the folder, keep the files together or update the paths inside the scripts.

## Tips

- If Tk windows fail to open, ensure you are running from a desktop session (not SSH without X forwarding).
- When upgrading Python, reinstall the virtual environment and rerun `pip install -r requirements.txt`.
- Keep `label_config.json` under version control so new label/tag definitions are shared with your team.
