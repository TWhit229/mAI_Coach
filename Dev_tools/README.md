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

- **Labeler** – load a set of videos + dataset JSONs, scrub with transport controls, review pose metrics/tracking quality, assign form tags, and edit metadata. A “Save & Finish” button appears on the last clip when the workflow is pre-seeded.
- **Video Cutter** – mark rep segments with in/out buttons, preview playback, and export clips via `ffmpeg`.
- **Pose Tuner** – preview up to four videos in a grid while tweaking MediaPipe thresholds and smoothing parameters. Inline hints explain what each slider controls.
- **Dataset + Model Trainer** – preprocess labeled JSONs into NumPy tensors and train the small MLP using per-lift presets (bench preset included; add more lifts as needed).

Data/layout: dataset assets live under `Dev_tools/data/` (JSONs, tensors) and models under `Dev_tools/models/` (MediaPipe pose tasks, bench MLP weights/scaler/meta).

All labels/tags live in `label_config.json`. Edit it via the admin panel tile or by hand (keep it under version control so the list stays in sync for your team).

## Dataset preprocessing script

Need NumPy feature tensors for model training? Use the Dataset + Model tab in the unified tool (saves presets per lift) or run the CLI:

```bash
python Dev_tools/bench_dataset_tool.py --mode preprocess \
  --dataset_dir path/to/jsons \
  --output_prefix bench_v1
```

If you omit the flags, the script will pop up a folder picker and prompt you for an output prefix. It generates:

- `<prefix>_X.npy` – feature matrix (12 features per rep)
- `<prefix>_y.npy` – multi-hot label vectors following `ALL_TAGS`
- `<prefix>_meta.json` – feature names, tag names, and dataset stats

Only label-friendly (tracking quality ≥ 0.5, not marked unreliable) reps are included.

## MLP training script

Once you have `bench_v1_X.npy`/`bench_v1_y.npy`/`bench_v1_meta.json`, train a tiny classifier from the unified tool tab or via CLI:

```bash
python Dev_tools/bench_dataset_tool.py --mode train \
  --data_prefix bench_v1 \
  --output_prefix bench_mlp_v1 \
  --epochs 200 \
  --batch_size 32
```

The script uses PyTorch + scikit-learn (train/dev split, scaling, metrics) and saves the model weights, scaler parameters, and metadata JSON beside the prefix you provide. Legacy single-purpose scripts (`preprocess_bench_dataset.py`, `train_bench_mlp.py`) now live in `Dev_tools/archive/`; prefer the unified tool tab or `bench_dataset_tool.py`.

## Archived utilities

Legacy Tkinter scripts (`auto_cut_video.py`, `multi_video_pose_tuner.py`, etc.) have been removed from the primary workflow. If you still need them for reference, check the project history prior to the `unified-tool-polish` branch.

## Tips

- If Qt windows fail to open, make sure you are running from a desktop session (not headless SSH).
- Re-run `./scripts/setup_unified_env.sh` after adjusting `requirements.txt` or upgrading Python.
- Keep the `.task` pose model files inside this folder so the GUI can find them.
