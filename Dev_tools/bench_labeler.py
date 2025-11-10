#!/usr/bin/env python3
"""
Bench Rep Labeling Tool – single JSON per rep

Each video gets ONE JSON file with:
- pose frames (all landmarks for every frame)
- labels (movement, quality, issues, load_lbs, etc)
- issue_events: [{issue, frame_index, time_ms}]

Typical usage (no CLI args):
    python bench_labeler.py
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception as e:
    raise SystemExit(
        f"Failed to import mediapipe tasks: {e}\n"
        "Make sure `pip install mediapipe` is done in this environment."
    )

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    raise SystemExit(
        f"Failed to import tkinter: {e}\n"
        "Your Python may not have Tk support."
    )

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit(
        f"Failed to import Pillow (PIL): {e}\n"
        "Install with: pip install pillow"
    )

from label_config import load_label_config, ensure_config_file


# -------------------- Pose export helpers -------------------------------------

def lowpass_ema(prev: np.ndarray, curr: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or prev.shape != curr.shape:
        return curr.copy()
    return alpha * curr + (1.0 - alpha) * prev


def mp_image_from_bgr(frame_bgr: np.ndarray):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def build_landmarker(model_path: Path,
                     det: float,
                     prs: float,
                     trk: float,
                     output_seg_masks: bool = False) -> PoseLandmarker:
    base = BaseOptions(model_asset_path=str(model_path))
    options = PoseLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=det,
        min_pose_presence_confidence=prs,
        min_tracking_confidence=trk,
        output_segmentation_masks=output_seg_masks,
    )
    return PoseLandmarker.create_from_options(options)


def export_dataset_json(
    video_path: Path,
    model_path: Path,
    det: float,
    prs: float,
    trk: float,
    ema_alpha: float,
    output_seg_masks: bool,
    output_dir: Path,
    movement_name: str,
) -> Path:
    """Run pose model over a video and write ONE JSON with frames + empty labels."""
    print(f"[POSE] Processing {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = build_landmarker(model_path, det, prs, trk, output_seg_masks)

    frames_out = []
    frame_idx = 0
    prev_smoothed = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            time_ms = int(round(frame_idx * 1000.0 / fps))
            mp_img = mp_image_from_bgr(frame_bgr)
            result = landmarker.detect_for_video(mp_img, time_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = np.array(
                    [[lm.x, lm.y, lm.z, getattr(lm, "presence", 1.0)] for lm in lms],
                    dtype=np.float32,
                )
                if ema_alpha > 0.0:
                    pts = lowpass_ema(prev_smoothed, pts, ema_alpha)
                    prev_smoothed = pts
                else:
                    prev_smoothed = None

                landmarks_json = [
                    {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                        "presence": float(p[3]),
                    }
                    for p in pts
                ]
                frame_rec = {
                    "frame_index": frame_idx,
                    "time_ms": time_ms,
                    "pose_present": True,
                    "landmarks": landmarks_json,
                }
            else:
                frame_rec = {
                    "frame_index": frame_idx,
                    "time_ms": time_ms,
                    "pose_present": False,
                    "landmarks": None,
                }

            frames_out.append(frame_rec)
            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}.json"

    base = {
        "rep_id": video_path.stem,
        "video_path": str(video_path),

        # label fields (initially defaulted)
        "movement": movement_name,
        "overall_quality": None,
        "issues": [],
        "load_lbs": None,
        "rpe": None,
        "camera_angle": None,
        "lens": None,
        "issue_events": [],  # [{issue, frame_index, time_ms}]

        # pose metadata + frames
        "pose_model_file": model_path.name,
        "fps": fps,
        "min_pose_detection_confidence": det,
        "min_pose_presence_confidence": prs,
        "min_tracking_confidence": trk,
        "ema_alpha": ema_alpha,
        "output_segmentation_masks": output_seg_masks,
        "frames": frames_out,
    }

    with out_path.open("w") as f:
        json.dump(base, f, indent=2)

    print(f"[POSE] Wrote {out_path}")
    return out_path


# -------------------- Overlay drawing ----------------------------------------

# Upper-body indices
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

UPPER_IDS = [
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
]

UPPER_LINES = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_WRIST, L_THUMB),
    (L_WRIST, L_INDEX),
    (L_WRIST, L_PINKY),
    (R_WRIST, R_THUMB),
    (R_WRIST, R_INDEX),
    (R_WRIST, R_PINKY),
]


def to_px(x: float, y: float, W: int, H: int):
    return int(round(x * W)), int(round(y * H))


def draw_upper_body_overlay(frame_bgr, landmarks: List[Dict[str, float]]):
    """Draw chest/arms/hands + mid-shoulder chest point."""
    H, W = frame_bgr.shape[:2]
    pts = np.array(
        [[lm["x"], lm["y"], lm["z"], lm.get("presence", 1.0)] for lm in landmarks],
        dtype=np.float32,
    )

    ls = pts[L_SHOULDER]
    rs = pts[R_SHOULDER]
    mid = (ls + rs) / 2.0

    mx, my = to_px(mid[0], mid[1], W, H)
    lsh_x, lsh_y = to_px(ls[0], ls[1], W, H)
    rsh_x, rsh_y = to_px(rs[0], rs[1], W, H)

    cv2.line(frame_bgr, (mx, my), (lsh_x, lsh_y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame_bgr, (mx, my), (rsh_x, rsh_y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, (mx, my), 3, (255, 255, 255), -1, cv2.LINE_AA)

    for a, b in UPPER_LINES:
        pa = pts[a]
        pb = pts[b]
        xa, ya = to_px(pa[0], pa[1], W, H)
        xb, yb = to_px(pb[0], pb[1], W, H)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), (255, 255, 255), 2, cv2.LINE_AA)

    for idx in UPPER_IDS:
        p = pts[idx]
        x, y = to_px(p[0], p[1], W, H)
        cv2.circle(frame_bgr, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


# -------------------- Label options & model presets ---------------------------

ensure_config_file()
_LABEL_CFG = load_label_config()


def _safe_opts(key, fallback):
    opts = _LABEL_CFG.get(key) or fallback
    return opts if isinstance(opts, list) and opts else fallback


MOVEMENT_OPTIONS = _safe_opts("movements", ["traditional_bench"])

QUALITY_OPTIONS = ["1", "2", "3", "4", "5"]

# RPE 1.0 .. 10.0 in 0.5 steps
RPE_OPTIONS = [f"{x / 2:.1f}" for x in range(2, 21)]  # 2 -> 1.0, 20 -> 10.0

CAMERA_ANGLE_OPTIONS = [
    "front",
    "front_45",
    "side",
    "rear_45",
    "rear",
    "overhead",
    "unknown",
]

LENS_OPTIONS = [
    "0.5",
    "1.0",
    "2.0",
    "3.0",
    "5.0",
    "other",
    "unknown",
]

# Numbered issue codes for consistency
ISSUE_OPTIONS = _safe_opts(
    "issues",
    [
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
    ],
)

# Per-movement model presets so we can add squats / deadlifts later.
MOVEMENT_MODEL_PRESETS = {
    "traditional_bench": {
        "model_variant": "full",
        "det": 0.50,
        "prs": 0.70,
        "trk": 0.70,
        "ema": 0.25,
        "seg": False,
    },
}


# -------------------- Labeling UI --------------------------------------------

class LabelerApp:
    def __init__(self, root, video_paths: List[Path], dataset_dir: Path):
        self.root = root
        self.video_paths = video_paths
        self.dataset_dir = dataset_dir

        self.current_index = 0
        self.dataset = None

        # all frames for current video (BGR images)
        self.frames_bgr: List[np.ndarray] = []
        self.total_frames = 0
        self.current_frame = 0   # index into frames_bgr

        self.playing = False     # start paused
        self.fps = 30.0
        self.playback_speed = 1.0
        self.tk_img = None

        self._updating_scale = False  # avoid scrub recursion
        self._scrub_release_id = None

        self.root.title("Bench Labeler (single JSON per rep)")
        self.build_ui()
        self.load_video(0)
        self.root.after(0, self.play_loop)

    # ---------------- UI building ----------------

    def build_ui(self):
        self.root.geometry("1280x720")

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: video
        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(video_frame)
        controls.pack(fill=tk.X)

        # transport controls
        self.back_frame_button = ttk.Button(
            controls, text="⟵ Frame", command=lambda: self.step_frames(-1)
        )
        self.back_frame_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.back_half_button = ttk.Button(
            controls, text="-0.5s", command=lambda: self.step_seconds(-0.5)
        )
        self.back_half_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.play_button = ttk.Button(
            controls, text="Play", command=self.toggle_play
        )
        self.play_button.pack(side=tk.LEFT, padx=5, pady=2)

        self.fwd_half_button = ttk.Button(
            controls, text="+0.5s", command=lambda: self.step_seconds(0.5)
        )
        self.fwd_half_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.fwd_frame_button = ttk.Button(
            controls, text="Frame ⟶", command=lambda: self.step_frames(1)
        )
        self.fwd_frame_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.info_label = ttk.Label(controls, text="Video 0 / 0")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # playback speed control
        ttk.Label(controls, text="Speed:").pack(side=tk.RIGHT, padx=(5, 0))
        self.playback_speed_var = tk.StringVar(value="1.0x")
        self.speed_cb = ttk.Combobox(
            controls,
            textvariable=self.playback_speed_var,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            state="readonly",
            width=6,
        )
        self.speed_cb.pack(side=tk.RIGHT, padx=(0, 5), pady=2)
        self.speed_cb.bind("<<ComboboxSelected>>", self.on_speed_change)

        # Scrubber slider (frame index)
        self.scrub_scale = tk.Scale(
            video_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_scrub,
        )
        self.scrub_scale.pack(fill=tk.X, padx=5, pady=2)

        # Right: form
        form = ttk.Frame(main, padding=10)
        form.grid(row=0, column=1, sticky="nsew")

        row = 0

        def add(label_text, widget):
            nonlocal row
            ttk.Label(form, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            row += 1

        form.columnconfigure(1, weight=1)

        # rep_id
        self.rep_id_var = tk.StringVar()
        self.rep_id_entry = ttk.Entry(form, textvariable=self.rep_id_var)
        add("rep_id:", self.rep_id_entry)

        # movement
        self.movement_var = tk.StringVar(value=MOVEMENT_OPTIONS[0])
        self.movement_cb = ttk.Combobox(
            form, textvariable=self.movement_var,
            values=MOVEMENT_OPTIONS, state="readonly"
        )
        add("movement:", self.movement_cb)

        # overall_quality
        self.quality_var = tk.StringVar(value="4")
        self.quality_cb = ttk.Combobox(
            form, textvariable=self.quality_var,
            values=QUALITY_OPTIONS, state="readonly"
        )
        add("overall_quality:", self.quality_cb)

        # load_lbs
        self.load_var = tk.StringVar()
        self.load_entry = ttk.Entry(form, textvariable=self.load_var)
        add("load_lbs:", self.load_entry)

        # rpe
        self.rpe_var = tk.StringVar(value="8.0")
        self.rpe_cb = ttk.Combobox(
            form, textvariable=self.rpe_var,
            values=RPE_OPTIONS, state="readonly"
        )
        add("RPE:", self.rpe_cb)

        # camera_angle
        self.camera_var = tk.StringVar(value="front_45")
        self.camera_cb = ttk.Combobox(
            form, textvariable=self.camera_var,
            values=CAMERA_ANGLE_OPTIONS, state="readonly"
        )
        add("camera_angle:", self.camera_cb)

        # lens
        self.lens_var = tk.StringVar(value="0.5")
        self.lens_cb = ttk.Combobox(
            form, textvariable=self.lens_var,
            values=LENS_OPTIONS, state="readonly"
        )
        add("lens:", self.lens_cb)

        # issues (multi-select, overall per rep)
        ttk.Label(form, text="issues (multi-select):").grid(
            row=row, column=0, sticky="nw", pady=4
        )
        self.issues_listbox = tk.Listbox(
            form, selectmode=tk.MULTIPLE, height=10, exportselection=False
        )
        for opt in ISSUE_OPTIONS:
            self.issues_listbox.insert(tk.END, opt)
        self.issues_listbox.grid(row=row, column=1, sticky="nsew", pady=4)
        form.rowconfigure(row, weight=1)
        row += 1

        # Issue timing controls
        ttk.Label(form, text="Tag issue at current frame:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.tag_issue_var = tk.StringVar(value=ISSUE_OPTIONS[0])
        self.tag_issue_cb = ttk.Combobox(
            form, textvariable=self.tag_issue_var,
            values=ISSUE_OPTIONS, state="readonly"
        )
        self.tag_issue_cb.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(form, text="Issue events:").grid(
            row=row, column=0, sticky="nw", pady=4
        )
        self.tags_listbox = tk.Listbox(form, height=8, exportselection=False)
        self.tags_listbox.grid(row=row, column=1, sticky="nsew", pady=4)
        form.rowconfigure(row, weight=1)
        row += 1

        tag_btn_frame = ttk.Frame(form)
        tag_btn_frame.grid(row=row, column=0, columnspan=2, pady=4, sticky="ew")
        self.add_tag_button = ttk.Button(
            tag_btn_frame,
            text="Add tag @ current frame",
            command=self.add_issue_tag,
        )
        self.add_tag_button.pack(side=tk.LEFT, padx=5)
        self.remove_tag_button = ttk.Button(
            tag_btn_frame,
            text="Remove selected tag",
            command=self.remove_selected_tag,
        )
        self.remove_tag_button.pack(side=tk.LEFT, padx=5)
        row += 1

        # navigation buttons
        btn_frame = ttk.Frame(form)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=8, sticky="ew")

        self.prev_button = ttk.Button(btn_frame, text="Previous", command=self.prev_video)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.save_next_button = ttk.Button(btn_frame, text="Save + Next", command=self.save_and_next)
        self.save_next_button.pack(side=tk.RIGHT, padx=5)

    # ---------------- video / dataset loading ----------------

    def load_video(self, index: int):
        # Clamp index
        index = max(0, min(index, len(self.video_paths) - 1))
        self.current_index = index
        vpath = self.video_paths[index]

        # Load all frames for this video into memory
        print(f"[UI] Loading frames for {vpath}")
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video: {vpath}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

        self.frames_bgr = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frames_bgr.append(frame)
        cap.release()

        self.total_frames = len(self.frames_bgr)
        if self.total_frames == 0:
            messagebox.showerror("Error", f"No frames in video: {vpath}")
            return

        # Load dataset JSON
        dataset_path = self.dataset_dir / f"{vpath.stem}.json"
        if not dataset_path.exists():
            messagebox.showerror("Error", f"Missing dataset JSON: {dataset_path}")
            self.dataset = None
        else:
            with dataset_path.open("r") as f:
                self.dataset = json.load(f)

        if self.dataset is not None and "issue_events" not in self.dataset:
            self.dataset["issue_events"] = []

        self.current_frame = 0
        self.playing = True
        self.play_button.config(text="Pause")

        # Slider range
        self.scrub_scale.config(from_=0, to=max(0, self.total_frames - 1))

        # Form values
        self.rep_id_var.set(self.dataset.get("rep_id", vpath.stem) if self.dataset else vpath.stem)
        self.load_form_from_dataset()

        self.info_label.config(text=f"Video {self.current_index + 1} / {len(self.video_paths)}")

        # Show first frame immediately
        self.show_frame(self.current_frame)

    def load_form_from_dataset(self):
        d = self.dataset or {}
        self.movement_var.set(d.get("movement") or MOVEMENT_OPTIONS[0])

        q = d.get("overall_quality")
        self.quality_var.set(str(q) if q is not None else "4")

        ll = d.get("load_lbs")
        self.load_var.set("" if ll is None else str(ll))

        rpe = d.get("rpe")
        self.rpe_var.set(str(rpe) if rpe is not None else "8.0")

        cam = d.get("camera_angle")
        self.camera_var.set(cam if cam else "front_45")

        lens = d.get("lens")
        self.lens_var.set(lens if lens else "0.5")

        # overall issues
        self.issues_listbox.selection_clear(0, tk.END)
        issues = d.get("issues") or []
        for i, opt in enumerate(ISSUE_OPTIONS):
            if opt in issues:
                self.issues_listbox.selection_set(i)

        self.refresh_tags_listbox()

    # ---------------- frame rendering ----------------

    def show_frame(self, fi: int):
        """Render frame fi with overlay, update slider/index."""
        if not self.frames_bgr or self.total_frames <= 0:
            return

        fi = max(0, min(fi, self.total_frames - 1))
        self.current_frame = fi

        frame_bgr = self.frames_bgr[fi].copy()

        # Overlay pose if we have it
        if self.dataset and "frames" in self.dataset:
            frames = self.dataset["frames"]
            if 0 <= fi < len(frames):
                frec = frames[fi]
                if frec.get("pose_present") and frec.get("landmarks"):
                    draw_upper_body_overlay(frame_bgr, frec["landmarks"])

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.tk_img)

        # Update slider without triggering on_scrub
        self._updating_scale = True
        self.scrub_scale.set(fi)
        if self._scrub_release_id is None:
            self._scrub_release_id = self.root.after_idle(self._release_scrub_lock)


    def _release_scrub_lock(self):
        self._updating_scale = False
        self._scrub_release_id = None

    # ---------------- scrubber / playback --------------------

    def on_scrub(self, value):
        """User moved the time slider; jump to that frame and pause."""
        if not self.frames_bgr or self.total_frames <= 0:
            return
        if self._updating_scale:
            return
        try:
            idx = int(float(value))
        except ValueError:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.playing = False
        self.play_button.config(text="Play")
        self.show_frame(idx)

    def on_speed_change(self, event=None):
        txt = self.playback_speed_var.get().replace("x", "")
        try:
            speed = float(txt)
        except ValueError:
            speed = 1.0
        if speed <= 0:
            speed = 1.0
        self.playback_speed = speed

    def step_frames(self, delta: int):
        """Step forward/backward by a number of frames and pause."""
        if self.total_frames <= 0:
            return
        new_index = self.current_frame + int(delta)
        new_index = max(0, min(new_index, self.total_frames - 1))
        self.playing = False
        self.play_button.config(text="Play")
        self.show_frame(new_index)

    def step_seconds(self, seconds: float):
        """Step forward/backward by a time offset in seconds."""
        if self.total_frames <= 0 or self.fps <= 0:
            return
        delta_frames = int(round(seconds * self.fps))
        if delta_frames == 0:
            delta_frames = 1 if seconds > 0 else -1
        self.step_frames(delta_frames)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")

    def play_loop(self):
        # adjust delay by playback speed
        speed = self.playback_speed if self.playback_speed > 0 else 1.0
        if self.fps > 0:
            delay = max(1, int(1000 / (self.fps * speed)))
        else:
            delay = 33

        if self.playing and self.frames_bgr and self.total_frames > 0:
            next_frame = self.current_frame + 1
            if next_frame >= self.total_frames:
                # end of video -> pause on last frame
                next_frame = self.total_frames - 1
                self.playing = False
                self.play_button.config(text="Play")
            self.show_frame(next_frame)

        self.root.after(delay, self.play_loop)

    # ---------------- tag list helpers ----------------------

    def refresh_tags_listbox(self):
        self.tags_listbox.delete(0, tk.END)
        if not self.dataset:
            return
        events = self.dataset.get("issue_events") or []
        for evt in events:
            issue = evt.get("issue", "?")
            fi = evt.get("frame_index", "?")
            t = evt.get("time_ms", "?")
            self.tags_listbox.insert(tk.END, f"f={fi} t={t}ms  {issue}")

    def add_issue_tag(self):
        if not self.dataset or "frames" not in self.dataset:
            return
        frames = self.dataset["frames"]
        if not frames:
            return

        fi = max(0, min(self.current_frame, len(frames) - 1))
        time_ms = frames[fi].get(
            "time_ms",
            int(round(fi * 1000.0 / (self.dataset.get("fps") or 30.0))),
        )
        issue = self.tag_issue_var.get()

        if "issue_events" not in self.dataset or self.dataset["issue_events"] is None:
            self.dataset["issue_events"] = []

        self.dataset["issue_events"].append(
            {
                "issue": issue,
                "frame_index": int(fi),
                "time_ms": int(time_ms),
            }
        )
        self.refresh_tags_listbox()

    def remove_selected_tag(self):
        if not self.dataset or "issue_events" not in self.dataset:
            return
        sel = list(self.tags_listbox.curselection())
        if not sel:
            return
        idx = sel[0]
        events = self.dataset["issue_events"]
        if 0 <= idx < len(events):
            events.pop(idx)
        self.refresh_tags_listbox()

    # ---------------- navigation & saving -------------------

    def prev_video(self):
        if self.current_index > 0:
            self.load_video(self.current_index - 1)

    def save_and_next(self):
        if self.dataset is None:
            return

        vpath = self.video_paths[self.current_index]

        try:
            load_val = float(self.load_var.get()) if self.load_var.get() else None
        except ValueError:
            messagebox.showerror("Error", "load_lbs must be a number.")
            return

        try:
            rpe_val = float(self.rpe_var.get())
        except ValueError:
            messagebox.showerror("Error", "RPE must be a number.")
            return

        selected_indices = list(self.issues_listbox.curselection())
        issues = [ISSUE_OPTIONS[i] for i in selected_indices]

        # update dataset
        self.dataset["rep_id"] = self.rep_id_var.get()
        self.dataset["video_path"] = str(vpath)
        self.dataset["movement"] = self.movement_var.get()
        self.dataset["overall_quality"] = int(self.quality_var.get())
        self.dataset["issues"] = issues
        self.dataset["load_lbs"] = load_val
        self.dataset["rpe"] = rpe_val
        self.dataset["camera_angle"] = self.camera_var.get()
        self.dataset["lens"] = self.lens_var.get()
        if "issue_events" not in self.dataset or self.dataset["issue_events"] is None:
            self.dataset["issue_events"] = []

        dataset_path = self.dataset_dir / f"{vpath.stem}.json"
        with dataset_path.open("w") as f:
            json.dump(self.dataset, f, indent=2)

        print(f"[LABEL] Saved {dataset_path}")

        if self.current_index < len(self.video_paths) - 1:
            self.load_video(self.current_index + 1)
        else:
            messagebox.showinfo("Done", "Last video labeled.")
            self.root.quit()



# -------------------- main ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory with rep videos (optional).",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Explicit video paths (optional, overrides --input_dir).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for combined JSON outputs (optional; if blank, folder picker is used).",
    )
    parser.add_argument(
        "--movement",
        type=str,
        default="traditional_bench",
        choices=MOVEMENT_OPTIONS,
        help="Movement preset to use for pose model settings.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="full",
        choices=["lite", "full", "heavy"],
        help="Pose model variant (may be overridden by --movement preset).",
    )
    # tuned defaults (may be overridden by movement preset)
    parser.add_argument("--det", type=float, default=0.50)
    parser.add_argument("--prs", type=float, default=0.70)
    parser.add_argument("--trk", type=float, default=0.70)
    parser.add_argument("--ema", type=float, default=0.25)
    parser.add_argument("--seg", action="store_true")

    args = parser.parse_args()

    # Apply movement preset to model settings
    preset = MOVEMENT_MODEL_PRESETS.get(args.movement, {})
    if preset:
        args.model_variant = preset.get("model_variant", args.model_variant)
        args.det = preset.get("det", args.det)
        args.prs = preset.get("prs", args.prs)
        args.trk = preset.get("trk", args.trk)
        args.ema = preset.get("ema", args.ema)
        if preset.get("seg", False):
            args.seg = True

    script_dir = Path(__file__).resolve().parent
    model_files = {
        "lite": script_dir / "pose_landmarker_lite.task",
        "full": script_dir / "pose_landmarker_full.task",
        "heavy": script_dir / "pose_landmarker_heavy.task",
    }
    model_path = model_files[args.model_variant]
    if not model_path.exists():
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            "Make sure the .task files are in the same folder as this script."
        )

    # 1) Collect videos (CLI or file picker)
    videos: List[Path] = []

    if args.videos:
        videos = [Path(v) for v in args.videos]
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        if not in_dir.is_dir():
            raise SystemExit(f"Input dir does not exist: {in_dir}")
        exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
        for p in sorted(in_dir.iterdir()):
            if p.suffix.lower() in exts:
                videos.append(p)
    else:
        root = tk.Tk()
        root.withdraw()
        filepaths = filedialog.askopenfilenames(
            title="Select rep videos",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not filepaths:
            raise SystemExit("No videos selected.")
        videos = [Path(p) for p in filepaths]

    if not videos:
        raise SystemExit("No videos found.")

    print(f"[INFO] Found {len(videos)} videos.")
    print(f"[INFO] Movement preset: {args.movement}")
    print(f"[INFO] Using model: {args.model_variant} ({model_path.name})")
    print(f"[INFO] det={args.det} prs={args.prs} trk={args.trk} ema={args.ema} seg={args.seg}")

    # 2) Choose output folder for JSONs (unless provided)
    if args.output_dir:
        dataset_dir = Path(args.output_dir)
    else:
        root = tk.Tk()
        root.withdraw()
        outdir = filedialog.askdirectory(title="Choose folder for JSON outputs")
        root.destroy()
        if not outdir:
            raise SystemExit("No output folder selected.")
        dataset_dir = Path(outdir)

    # 3) Ensure each video has a dataset JSON (pose frames)
    for v in videos:
        out_path = dataset_dir / f"{v.stem}.json"
        if out_path.exists():
            print(f"[POSE] Skipping {v.name}, JSON already exists.")
            continue
        export_dataset_json(
            v,
            model_path,
            args.det,
            args.prs,
            args.trk,
            args.ema,
            args.seg,
            dataset_dir,
            args.movement,
        )

    # 4) Launch labeling GUI
    root = tk.Tk()
    app = LabelerApp(root, videos, dataset_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
