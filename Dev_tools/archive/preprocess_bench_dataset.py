#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
except Exception:  # pragma: no cover - tkinter may be unavailable in some envs
    tk = None
    filedialog = None
    simpledialog = None


ALL_TAGS = [
    "no_major_issues",
    "hands_too_wide",
    "hands_too_narrow",
    "grip_uneven",
    "barbell_tilted",
    "bar_depth_insufficient",
    "incomplete_lockout",
]

L_SHOULDER = 11
R_SHOULDER = 12
L_WRIST = 15
R_WRIST = 16


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess bench dataset into numpy arrays"
    )
    p.add_argument("--dataset_dir", type=str, help="Folder containing JSON files")
    p.add_argument("--output_prefix", type=str, help="Output prefix for npy/meta files")
    return p.parse_args()


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _lm_xy(landmarks, idx):
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError):
        return None
    if not isinstance(lm, dict):
        return None
    x = lm.get("x")
    y = lm.get("y")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def extract_example(rep):
    if rep.get("tracking_unreliable", False):
        return None
    metrics = rep.get("metrics") or {}
    frames = rep.get("frames") or []
    tracking_quality = float(metrics.get("tracking_quality", 0.0))
    if tracking_quality < 0.5 or not frames:
        return None

    load_lbs = float(rep.get("load_lbs") or 0.0)
    grip_ratio_median = float(metrics.get("grip_ratio_median", 0.0))
    grip_ratio_range = float(metrics.get("grip_ratio_range", 0.0))
    grip_uneven_median = float(metrics.get("grip_uneven_median", 0.0))
    grip_uneven_norm = float(metrics.get("grip_uneven_norm", 0.0))
    bar_tilt_median_deg = float(metrics.get("bar_tilt_median_deg", 0.0))
    bar_tilt_deg_max = float(metrics.get("bar_tilt_deg_max", 0.0))
    tracking_bad_ratio = float(metrics.get("tracking_bad_ratio", 0.0))

    wrist_y_vals = []
    for frec in frames:
        if not frec or not frec.get("pose_present"):
            continue
        lms = frec.get("landmarks")
        if not lms:
            continue
        ls = _lm_xy(lms, L_SHOULDER)
        rs = _lm_xy(lms, R_SHOULDER)
        lw = _lm_xy(lms, L_WRIST)
        rw = _lm_xy(lms, R_WRIST)
        if not (ls and rs and lw and rw):
            continue
        shoulder_width = _dist(ls, rs)
        if shoulder_width <= 1e-6:
            continue
        chest_y = 0.5 * (ls[1] + rs[1])
        lw_y_norm = (lw[1] - chest_y) / shoulder_width
        rw_y_norm = (rw[1] - chest_y) / shoulder_width
        avg_y = 0.5 * (lw_y_norm + rw_y_norm)
        wrist_y_vals.append(avg_y)

    if wrist_y_vals:
        wrist_y_min = float(min(wrist_y_vals))
        wrist_y_max = float(max(wrist_y_vals))
        wrist_y_range = wrist_y_max - wrist_y_min
    else:
        wrist_y_min = wrist_y_max = wrist_y_range = 0.0

    features = [
        load_lbs,
        grip_ratio_median,
        grip_ratio_range,
        grip_uneven_median,
        grip_uneven_norm,
        bar_tilt_median_deg,
        bar_tilt_deg_max,
        tracking_bad_ratio,
        tracking_quality,
        wrist_y_min,
        wrist_y_max,
        wrist_y_range,
    ]

    tags = set(rep.get("tags") or [])
    labels = [1 if tag in tags else 0 for tag in ALL_TAGS]

    return features, labels


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    prefix = args.output_prefix

    if dataset_dir is None and filedialog is not None:
        root = tk.Tk()
        root.withdraw()
        dataset_dir = filedialog.askdirectory(title="Select dataset folder")
        root.destroy()
    if not dataset_dir:
        dataset_dir = input("Dataset directory: ").strip()

    if prefix is None:
        if simpledialog is not None:
            root = tk.Tk()
            root.withdraw()
            prefix = simpledialog.askstring(
                "Output prefix", "Enter output prefix (e.g., bench_v1)"
            )
            root.destroy()
        if not prefix:
            prefix = input("Output prefix: ").strip()

    if not dataset_dir:
        print("No dataset directory provided; exiting.")
        return
    if not prefix:
        print("No output prefix provided; exiting.")
        return

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    json_files = sorted(dataset_dir.glob("*.json"))
    X = []
    Y = []
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"Skipping {path.name}: failed to load JSON ({exc})")
            continue
        example = extract_example(data)
        if example is None:
            continue
        feats, labels = example
        X.append(feats)
        Y.append(labels)
    if not X:
        print("No valid examples found.")
        return
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    np.save(prefix + "_X.npy", X)
    np.save(prefix + "_y.npy", Y)
    meta = {
        "feature_names": [
            "load_lbs",
            "grip_ratio_median",
            "grip_ratio_range",
            "grip_uneven_median",
            "grip_uneven_norm",
            "bar_tilt_median_deg",
            "bar_tilt_deg_max",
            "tracking_bad_ratio",
            "tracking_quality",
            "wrist_y_min",
            "wrist_y_max",
            "wrist_y_range",
        ],
        "tags": ALL_TAGS,
        "num_examples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "num_tags": int(Y.shape[1]),
    }
    meta_path = prefix + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved X to {prefix}_X.npy with shape {X.shape}")
    print(f"Saved y to {prefix}_y.npy with shape {Y.shape}")
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    main()
