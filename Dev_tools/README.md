# Dev Tools Suite

PySide6 (Qt) utilities for reviewing and labeling lifting videos live inside this `Dev_tools` folder. The app shares the same MediaPipe pose models (`pose_landmarker_*.task`) and label configuration (`label_config.json`) used by the iOS product.

## Requirements

- Python 3.10+ (Qt/PySide6 ships with the app; Tk is only required for archived scripts).
- System `ffmpeg` binary available in `PATH` (needed for the clip exporter). On macOS use `brew install ffmpeg`, on Windows download from https://ffmpeg.org.
- Python packages listed in `requirements.txt`.

## Environment bootstrap

From the repo root, run:

```bash
./scripts/setup_unified_env.sh
```

This creates/refreshes `Dev_tools/.venv`, upgrades `pip`, and installs the dependencies. Prefer the manual steps? Run:

```bash
cd Dev_tools
python3 -m venv .venv            # Windows: py -3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Launching the unified tool

```bash
# from the repo root
./scripts/run_unified_tool.sh
```

The PySide6 window contains:

- **Labeler** – load a set of videos + dataset JSONs, scrub with transport controls, tag issues, and edit metadata. A “Save & Finish” button appears on the last clip when the workflow is pre-seeded.
- **Video Cutter** – mark rep segments with in/out buttons, preview playback, and export clips via `ffmpeg`.
- **Pose Tuner** – preview up to four videos in a grid while tweaking MediaPipe thresholds and smoothing parameters. Inline hints explain what each slider controls.

All labels/tags live in `label_config.json`. Edit it via the admin panel tile or by hand (keep it under version control so the list stays in sync for your team).

## Archived utilities

Legacy Tkinter scripts (`auto_cut_video.py`, `multi_video_pose_tuner.py`, etc.) have been removed from the primary workflow. If you still need them for reference, check the project history prior to the `unified-tool-polish` branch.

## Tips

- If Qt windows fail to open, make sure you are running from a desktop session (not headless SSH).
- Re-run `./scripts/setup_unified_env.sh` after adjusting `requirements.txt` or upgrading Python.
- Keep the `.task` pose model files inside this folder so the GUI can find them.
