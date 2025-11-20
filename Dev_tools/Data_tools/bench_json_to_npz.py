#!/usr/bin/env python3
"""
Convert labeled bench JSON reps into normalized numpy arrays for model training.

- Input: one or more JSON files exported by the unified labeling tool
- Output: one .npz per input, with:
    features: (T, D) float32   # normalized upper-body coordinates per frame
    mask:     (T,)  float32    # 1 if pose is present and valid, else 0
    labels:   (T, N_TAGS) float32  # per-frame tag event indicators (0/1)
    fps:      float32          # frames per second for this rep
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as e:
    raise SystemExit(
        f"Failed to import tkinter: {e}\nYour Python may not have Tk support."
    )

try:
    from label_config import load_label_config
except ModuleNotFoundError:
    dev_tools_dir = Path(__file__).resolve().parents[1]
    sys.path.append(str(dev_tools_dir))
    from label_config import load_label_config


# -------------------- Config: which landmarks we keep -------------------------

# MediaPipe landmark indices we care about for bench (upper body + hips)
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_PINKY = 17
R_PINKY = 18
L_INDEX = 19
R_INDEX = 20
L_THUMB = 21
R_THUMB = 22
L_HIP = 23
R_HIP = 24

BENCH_LANDMARK_IDS = [
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_THUMB,
    R_THUMB,
    L_INDEX,
    R_INDEX,
    L_PINKY,
    R_PINKY,
    L_HIP,
    R_HIP,
]


def _load_tag_options() -> List[str]:
    cfg = load_label_config()
    tags = cfg.get("tags") or []
    if not tags:
        raise ValueError("label_config.json defines no tag options.")
    return tags


TAG_OPTIONS = _load_tag_options()
TAG_INDEX = {name: i for i, name in enumerate(TAG_OPTIONS)}


# -------------------- Per-frame feature extraction ---------------------------


def extract_frame_features(frame: Dict) -> Tuple[np.ndarray, float]:
    """
    Convert one frame record from a labeled JSON export into a normalized feature vector.

    Normalization:
    - If pose is missing: return zeros, mask=0.
    - Else:
        * Compute mid-shoulder = average of L/R shoulder.
        * Compute shoulder_width = distance between shoulders in (x,y).
        * Subtract mid-shoulder from all (x,y,z) -> center on chest.
        * Divide all coords by shoulder_width -> scale-invariant.
        * Flatten [x,y,z] for each BENCH_LANDMARK_IDS into a single vector.

    Returns:
        features: (D,) float32
        mask:     1.0 if valid, 0.0 otherwise
    """
    if not frame.get("pose_present") or not frame.get("landmarks"):
        D = len(BENCH_LANDMARK_IDS) * 3
        return np.zeros(D, dtype=np.float32), 0.0

    lms = frame["landmarks"]
    try:
        ls = lms[L_SHOULDER]
        rs = lms[R_SHOULDER]
    except (IndexError, TypeError, KeyError):
        D = len(BENCH_LANDMARK_IDS) * 3
        return np.zeros(D, dtype=np.float32), 0.0

    # mid-shoulder center
    mid_x = 0.5 * (float(ls["x"]) + float(rs["x"]))
    mid_y = 0.5 * (float(ls["y"]) + float(rs["y"]))
    mid_z = 0.5 * (float(ls["z"]) + float(rs["z"]))

    dx = float(ls["x"]) - float(rs["x"])
    dy = float(ls["y"]) - float(rs["y"])
    shoulder_width = np.sqrt(dx * dx + dy * dy)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    feats: List[float] = []
    for idx in BENCH_LANDMARK_IDS:
        try:
            lm = lms[idx]
        except (IndexError, TypeError):
            # If something is missing, pad zeros for this landmark
            feats.extend([0.0, 0.0, 0.0])
            continue
        x = (float(lm["x"]) - mid_x) / shoulder_width
        y = (float(lm["y"]) - mid_y) / shoulder_width
        z = (float(lm.get("z", 0.0)) - mid_z) / shoulder_width
        feats.extend([x, y, z])

    return np.asarray(feats, dtype=np.float32), 1.0


def build_label_matrix(num_frames: int, tag_events: List[Dict]) -> np.ndarray:
    """
    Build a (T, N_TAGS) label matrix from tag_events list.

    For now we treat each event as a "spike" at its frame_index:
    labels[frame_index, tag_index] = 1.
    """
    T = num_frames
    N = len(TAG_OPTIONS)
    labels = np.zeros((T, N), dtype=np.float32)

    if not tag_events:
        return labels

    for evt in tag_events:
        issue = evt.get("issue")
        fi = evt.get("frame_index")
        if issue is None or fi is None:
            continue
        idx = TAG_INDEX.get(issue)
        if idx is None:
            continue
        if 0 <= fi < T:
            labels[int(fi), idx] = 1.0

    return labels


# -------------------- Per-file processing ------------------------------------


def process_json_file(path: Path) -> dict:
    """Load one labeled JSON (from the unified tool) and convert to arrays."""
    with path.open("r") as f:
        data = json.load(f)

    if data.get("tracking_unreliable"):
        raise ValueError(
            f"{path.name} marked tracking_unreliable; skipping for training."
        )

    frames = data.get("frames", [])
    T = len(frames)
    if T == 0:
        raise ValueError(f"{path} has no frames.")

    D = len(BENCH_LANDMARK_IDS) * 3
    features = np.zeros((T, D), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)

    for t, frame in enumerate(frames):
        feats_t, m_t = extract_frame_features(frame)
        features[t] = feats_t
        mask[t] = m_t

    tag_events = data.get("tag_events") or data.get("issue_events", [])
    labels = build_label_matrix(T, tag_events)

    fps = float(data.get("fps", 30.0))
    rep_tags = data.get("tags") or data.get("issues") or []
    metrics = data.get("metrics") or {}

    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "fps": fps,
        "rep_tags": rep_tags,
        "metrics": metrics,
    }


# -------------------- Main (GUI file pickers) --------------------------------


def main():
    # 1) Pick JSON input files
    root = tk.Tk()
    root.withdraw()
    json_paths = filedialog.askopenfilenames(
        title="Select labeled bench JSON files",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not json_paths:
        messagebox.showinfo("No files", "No JSON files selected. Exiting.")
        return
    json_paths = [Path(p) for p in json_paths]

    # 2) Pick output directory
    outdir_str = filedialog.askdirectory(
        title="Choose output folder for processed NPZ files"
    )
    root.destroy()
    if not outdir_str:
        print("No output folder selected. Exiting.")
        return
    outdir = Path(outdir_str)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Selected {len(json_paths)} JSON files.")
    print(f"[INFO] Output dir: {outdir}")

    # 3) Process each JSON -> NPZ
    for jpath in json_paths:
        try:
            res = process_json_file(jpath)
        except Exception as e:
            print(f"[ERROR] Failed on {jpath.name}: {e}")
            continue

        metrics = res.get("metrics") or {}
        metric_names: List[str] = []
        metric_values: List[float] = []
        for key, value in metrics.items():
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_names.append(key)
                metric_values.append(float(value))
        rep_tags = res.get("rep_tags") or []
        rep_tags_arr = np.array(rep_tags, dtype="<U32")

        out_path = outdir / f"{jpath.stem}_bench.npz"
        np.savez_compressed(
            out_path,
            features=res["features"],
            mask=res["mask"],
            labels=res["labels"],
            fps=res["fps"],
            tag_names=np.array(TAG_OPTIONS),
            rep_tags=rep_tags_arr,
            metrics_names=np.array(metric_names, dtype="<U32"),
            metrics_values=np.array(metric_values, dtype=np.float32),
        )
        print(
            f"[OK] {jpath.name} -> {out_path.name} "
            f"frames={res['features'].shape[0]}, D={res['features'].shape[1]}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
