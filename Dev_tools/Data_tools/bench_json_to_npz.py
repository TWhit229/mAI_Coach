#!/usr/bin/env python3
"""
Convert labeled bench JSON reps into normalized numpy arrays for model training.

- Input: one or more JSON files produced by bench_labeler.py
- Output: one .npz per input, with:
    features: (T, D) float32   # normalized upper-body coordinates per frame
    mask:     (T,)  float32    # 1 if pose is present and valid, else 0
    labels:   (T, N_ISSUES) float32  # per-frame issue event indicators (0/1)
    fps:      float32          # frames per second for this rep
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as e:
    raise SystemExit(
        f"Failed to import tkinter: {e}\n"
        "Your Python may not have Tk support."
    )

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

# Must match the ISSUE_OPTIONS list in bench_labeler.py
ISSUE_OPTIONS = [
    "01_angle_issues",
    "02_tracking_unreliable",
    "03_no_major_issues",
    "04_hands_too_wide",
    "05_hands_too_narrow",
    "06_grip_uneven",
    "07_body_placement_too_forward",
    "08_body_placement_too_backward",
    "09_bar_path_too_forward",
    "10_bar_path_too_backward",
    "11_elbows_flared",
    "12_elbows_flared_bottom",
    "13_elbows_flared_lockout",
    "14_elbows_tucked",
    "15_elbows_tucked_excessive",
    "16_pause_too_long",
    "17_pause_too_short",
    "18_no_pause_on_chest",
    "19_bar_depth_insufficient",
    "20_bar_depth_excessive",
    "21_barbell_tilt_right",
    "22_barbell_tilt_left",
    "23_uneven_lockout",
    "24_hips_off_bench",
    "25_descent_too_fast",
    "26_bounce_off_chest",
    "27_incomplete_lockout",
    "28_wrists_tilted_forward",
    "29_wrists_tilted_backward",
    "30_right_elbow_in",
    "31_left_elbow_in",
    "32_right_elbow_out",
    "33_left_elbow_out",
]

ISSUE_INDEX = {name: i for i, name in enumerate(ISSUE_OPTIONS)}


# -------------------- Per-frame feature extraction ---------------------------

def extract_frame_features(frame: Dict) -> Tuple[np.ndarray, float]:
    """
    Convert one frame record from bench_labeler JSON into a normalized feature vector.

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


def build_label_matrix(num_frames: int, issue_events: List[Dict]) -> np.ndarray:
    """
    Build a (T, N_ISSUES) label matrix from issue_events list.

    For now we treat each issue event as a "spike" at its frame_index:
    labels[frame_index, issue_index] = 1.
    """
    T = num_frames
    N = len(ISSUE_OPTIONS)
    labels = np.zeros((T, N), dtype=np.float32)

    if not issue_events:
        return labels

    for evt in issue_events:
        issue = evt.get("issue")
        fi = evt.get("frame_index")
        if issue is None or fi is None:
            continue
        idx = ISSUE_INDEX.get(issue)
        if idx is None:
            continue
        if 0 <= fi < T:
            labels[int(fi), idx] = 1.0

    return labels


# -------------------- Per-file processing ------------------------------------

def process_json_file(path: Path) -> dict:
    """Load one bench_labeler JSON and convert to arrays."""
    with path.open("r") as f:
        data = json.load(f)

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

    issue_events = data.get("issue_events", [])
    labels = build_label_matrix(T, issue_events)

    fps = float(data.get("fps", 30.0))

    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "fps": fps,
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

        out_path = outdir / f"{jpath.stem}_bench.npz"
        np.savez_compressed(
            out_path,
            features=res["features"],
            mask=res["mask"],
            labels=res["labels"],
            fps=res["fps"],
            issue_names=np.array(ISSUE_OPTIONS),
        )
        print(
            f"[OK] {jpath.name} -> {out_path.name} "
            f"frames={res['features'].shape[0]}, D={res['features'].shape[1]}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
