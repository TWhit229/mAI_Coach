#!/usr/bin/env python3
"""Unified PySide6 workspace for all Dev_tools utilities."""

from __future__ import annotations

import gc
import concurrent.futures
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import PySide6
from PySide6 import QtCore, QtGui, QtWidgets

from label_config import ensure_config_file, load_label_config, save_label_config

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    PoseLandmarker = PoseLandmarkerOptions = RunningMode = BaseOptions = None
    _HAS_MEDIAPIPE = False

# Ensure Qt can locate the platform plugins (notably "cocoa" on macOS)
_PLUGIN_DIR = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins" / "platforms"
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_PLUGIN_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")


# --- Shared helpers copied from the legacy Tk labeler -----------------------


def lowpass_ema(
    prev: Optional[np.ndarray], curr: np.ndarray, alpha: float
) -> np.ndarray:
    if prev is None or prev.shape != curr.shape:
        return curr.copy()
    return alpha * curr + (1.0 - alpha) * prev


def draw_upper_body_overlay(
    frame_bgr: np.ndarray,
    landmarks: List[Dict[str, float]],
    allowed_parts: Optional[List[str]] = None,
):
    pts = np.array([[p["x"], p["y"]] for p in landmarks], dtype=np.float32)
    H, W = frame_bgr.shape[:2]

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(x * W), int(y * H)

    UPPER_IDS = [11, 12, 13, 14, 15, 16, 23, 24]
    UPPER_LINES: List[tuple[int, int]] = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
    ]

    BODY_PART_GROUPS = {
        "hands": {"points": {15, 16}, "lines": []},
        "wrists": {"points": {15, 16}, "lines": []},
        "forearms": {"points": {13, 14, 15, 16}, "lines": [(13, 15), (14, 16)]},
        "upper_arms": {"points": {11, 12, 13, 14}, "lines": [(11, 13), (12, 14)]},
        "shoulders": {"points": {11, 12}, "lines": [(11, 12)]},
        "torso": {"points": {11, 12, 23, 24}, "lines": [(11, 23), (12, 24), (23, 24)]},
        "hips": {"points": {23, 24}, "lines": []},
        "legs": {"points": set(), "lines": []},
        "full_body": {"points": set(UPPER_IDS), "lines": list(UPPER_LINES)},
    }

    if not allowed_parts:
        allowed_points = set(UPPER_IDS)
        allowed_lines: List[tuple[int, int]] = list(UPPER_LINES)
    else:
        allowed_points = set()
        allowed_lines = []
        expanded = set(allowed_parts)
        if "full_body" in expanded:
            allowed_points = set(UPPER_IDS)
            allowed_lines = list(UPPER_LINES)
        else:
            for name in expanded:
                group = BODY_PART_GROUPS.get(name)
                if not group:
                    continue
                allowed_points.update(group["points"])
                allowed_lines.extend(group["lines"])
            if not allowed_points:
                allowed_points = set(UPPER_IDS)
            if not allowed_lines:
                allowed_lines = list(UPPER_LINES)

    for a, b in allowed_lines:
        pa = to_px(*pts[a])
        pb = to_px(*pts[b])
        cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    for idx in allowed_points:
        cx, cy = to_px(*pts[idx])
        cv2.circle(frame_bgr, (cx, cy), 4, (255, 255, 0), -1)


ensure_config_file()


def load_label_options():
    cfg = load_label_config()
    movements = cfg.get("movements") or []
    tags = cfg.get("tags") or []
    raw_settings = cfg.get("movement_settings") or {}
    merged_settings: Dict[str, Dict] = {}
    for name in movements:
        defaults = default_movement_settings(name)
        stored = raw_settings.get(name) or {}
        merged = defaults.copy()
        merged.update(stored)
        merged_settings[name] = merged
    return movements, tags, merged_settings


QUALITY_OPTIONS = ["1", "2", "3", "4", "5"]
RPE_OPTIONS = [f"{x / 2:.1f}" for x in range(2, 21)]
CAMERA_ANGLE_OPTIONS = [
    "front",
    "front_45",
    "side",
    "rear_45",
    "rear",
    "overhead",
    "unknown",
]
LENS_OPTIONS = ["0.5", "1.0", "2.0", "3.0", "5.0", "other", "unknown"]
MODEL_VARIANTS = ["lite", "full", "heavy"]
BODY_PART_OPTIONS = [
    "hands",
    "wrists",
    "forearms",
    "upper_arms",
    "shoulders",
    "torso",
    "hips",
    "legs",
    "full_body",
]
BENCH_DEFAULT_PARTS = ["hands", "forearms", "upper_arms", "shoulders"]

POSE_MODEL_PATHS = {
    variant: Path(__file__).resolve().parent / f"pose_landmarker_{variant}.task"
    for variant in MODEL_VARIANTS
}

L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26

TRACKING_KEYPOINT_IDS = [
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
]

GRIP_WIDE_THRESHOLD = 2.1
GRIP_NARROW_THRESHOLD = 1.2
GRIP_UNEVEN_THRESHOLD = 0.10
BAR_TILT_THRESHOLD_DEG = 5.0
TRACKING_BAD_RATIO_MAX = 0.20
TRACKING_VISIBILITY_THRESHOLD = 0.7
TRACKING_VISIBLE_FRACTION = 0.7
DEFAULT_OK_TAG = "no_major_issues"

ROTATION_OPTIONS: List[Tuple[str, Optional[int]]] = [
    ("Auto (metadata)", None),
    ("0°", 0),
    ("90° CW", 90),
    ("180°", 180),
    ("270° CCW", 270),
]


def _rotation_option_index(degrees: Optional[int]) -> int:
    for idx, (_, val) in enumerate(ROTATION_OPTIONS):
        if val == degrees:
            return idx
    return 0


def _rotation_value_from_index(index: int) -> Optional[int]:
    if 0 <= index < len(ROTATION_OPTIONS):
        return ROTATION_OPTIONS[index][1]
    return None


def _video_rotation_degrees(path: Path) -> int:
    """Return clockwise rotation (0/90/180/270) based on ffprobe metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    try:
        value = int(result.stdout.strip() or 0)
    except ValueError:
        value = 0
    value %= 360
    if value not in (0, 90, 180, 270):
        value = 0
    return value


def _rotate_frame_if_needed(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def _pose_model_path(variant: str) -> Path:
    key = (variant or "full").lower()
    path = POSE_MODEL_PATHS.get(key) or POSE_MODEL_PATHS.get("full")
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Missing pose model for variant '{variant}'. Expected {path}."
        )
    return path


def _mp_image_from_bgr(frame_bgr: np.ndarray):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def run_pose_landmarks_on_video(
    video_path: Path,
    fps: float,
    settings: Dict,
    model_path: Path,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    rotation: int = 0,
) -> List[Dict]:
    """Run MediaPipe pose tracking over in-memory frames."""
    if not _HAS_MEDIAPIPE:
        raise RuntimeError("mediapipe is not available in this environment.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(settings.get("det", 0.5)),
        min_pose_presence_confidence=float(settings.get("prs", 0.7)),
        min_tracking_confidence=float(settings.get("trk", 0.7)),
        output_segmentation_masks=bool(settings.get("seg", False)),
    )
    landmarker = PoseLandmarker.create_from_options(options)
    fps_val = fps if fps and fps > 0 else 30.0
    ema_alpha = float(settings.get("ema", 0.0) or 0.0)
    prev_smoothed: Optional[np.ndarray] = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    results: List[Dict] = []

    try:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if rotation:
                frame_bgr = _rotate_frame_if_needed(frame_bgr, rotation)
            mp_image = _mp_image_from_bgr(frame_bgr)
            time_ms = int(round(idx * 1000.0 / fps_val))
            result = landmarker.detect_for_video(mp_image, time_ms)
            if result.pose_landmarks:
                pts = np.array(
                    [
                        [
                            lm.x,
                            lm.y,
                            lm.z,
                            getattr(lm, "presence", getattr(lm, "visibility", 1.0)),
                        ]
                        for lm in result.pose_landmarks[0]
                    ],
                    dtype=np.float32,
                )
                if ema_alpha > 0.0:
                    pts = lowpass_ema(prev_smoothed, pts, ema_alpha)
                    prev_smoothed = pts
                else:
                    prev_smoothed = None
                landmarks = [
                    {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                        "presence": float(p[3]),
                    }
                    for p in pts
                ]
                frame_rec = {
                    "frame_index": idx,
                    "time_ms": time_ms,
                    "pose_present": True,
                    "landmarks": landmarks,
                }
            else:
                prev_smoothed = None
                frame_rec = {
                    "frame_index": idx,
                    "time_ms": time_ms,
                    "pose_present": False,
                    "landmarks": None,
                }
            results.append(frame_rec)
            idx += 1
            if progress_cb and not progress_cb(idx, total or idx):
                raise RuntimeError("Pose tracking canceled")
    finally:
        cap.release()
        landmarker.close()

    return results


def _landmark_xy(
    landmarks: Sequence[Dict[str, float]], idx: int
) -> Optional[Tuple[float, float]]:
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


def _landmark_xyz(
    landmarks: Sequence[Dict[str, float]], idx: int
) -> Optional[Tuple[float, float, float]]:
    """
    Return (x, y, z) for the given landmark index, or None if unavailable.
    """
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError):
        return None
    if not isinstance(lm, dict):
        return None
    x = lm.get("x")
    y = lm.get("y")
    z = lm.get("z")
    if x is None or y is None or z is None:
        return None
    try:
        return float(x), float(y), float(z)
    except (TypeError, ValueError):
        return None


def _landmark_presence(
    landmarks: Sequence[Dict[str, float]], idx: int
) -> float:
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError):
        return 0.0
    if not isinstance(lm, dict):
        return 0.0
    val = lm.get("presence", lm.get("visibility"))
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _dist3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_rep_metrics(frames: Sequence[Dict]) -> Dict[str, float]:
    """Compute aggregate grip and tracking metrics for a rep."""
    metrics: Dict[str, float] = {}
    if not frames:
        metrics["tracking_bad_frames"] = 0
        metrics["tracking_total_frames"] = 0
        metrics["tracking_bad_ratio"] = 1.0
        metrics["tracking_quality"] = 0.0
        return metrics

    grip_ratios: List[float] = []
    grip_uneven_vals: List[float] = []
    bar_tilts: List[float] = []

    total_frames = len(frames)
    bad_frames = 0

    for frame in frames:
        if not frame or not frame.get("pose_present"):
            bad_frames += 1
            continue
        landmarks = frame.get("landmarks")
        if not landmarks:
            bad_frames += 1
            continue

        ls3 = _landmark_xyz(landmarks, L_SHOULDER)
        rs3 = _landmark_xyz(landmarks, R_SHOULDER)
        lw3 = _landmark_xyz(landmarks, L_WRIST)
        rw3 = _landmark_xyz(landmarks, R_WRIST)
        ls = _landmark_xy(landmarks, L_SHOULDER)
        rs = _landmark_xy(landmarks, R_SHOULDER)
        lw = _landmark_xy(landmarks, L_WRIST)
        rw = _landmark_xy(landmarks, R_WRIST)
        shoulder_width = None
        if ls3 and rs3:
            shoulder_width = _dist3d(ls3, rs3)

        if (
            shoulder_width
            and shoulder_width > 1e-6
            and lw3
            and rw3
            and lw
            and rw
        ):
            grip_width = _dist3d(lw3, rw3)
            grip_ratios.append(grip_width / shoulder_width)
            shoulder_center_x = 0.5 * (ls[0] + rs[0])
            left_offset = abs(lw[0] - shoulder_center_x)
            right_offset = abs(rw[0] - shoulder_center_x)
            grip_uneven_vals.append(abs(left_offset - right_offset) / shoulder_width)

        if lw and rw:
            dx = rw[0] - lw[0]
            dy = rw[1] - lw[1]
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = min(angle, abs(180.0 - angle))
            bar_tilts.append(angle)

        visible = 0
        for idx in TRACKING_KEYPOINT_IDS:
            if _landmark_presence(landmarks, idx) >= TRACKING_VISIBILITY_THRESHOLD:
                visible += 1
        visible_fraction = (
            visible / len(TRACKING_KEYPOINT_IDS) if TRACKING_KEYPOINT_IDS else 0.0
        )
        if visible_fraction < TRACKING_VISIBLE_FRACTION:
            bad_frames += 1

    # Summary stats for grip width (ratio of grip / shoulder)
    if grip_ratios:
        grip_median = float(statistics.median(grip_ratios))
        grip_min = float(min(grip_ratios))
        grip_max = float(max(grip_ratios))
        grip_range = grip_max - grip_min

        metrics["grip_ratio_median"] = grip_median
        metrics["grip_ratio_min"] = grip_min
        metrics["grip_ratio_max"] = grip_max
        metrics["grip_ratio_range"] = grip_range

        # Backwards-compatible alias for UI / older code
        metrics["grip_ratio"] = grip_median

    # Summary stats for grip unevenness (normalized by shoulder width)
    if grip_uneven_vals:
        uneven_median = float(statistics.median(grip_uneven_vals))
        uneven_min = float(min(grip_uneven_vals))
        uneven_max = float(max(grip_uneven_vals))

        metrics["grip_uneven_median"] = uneven_median
        metrics["grip_uneven_min"] = uneven_min
        metrics["grip_uneven_max"] = uneven_max

        # Backwards-compatible alias: we treat "norm" as the worst-case unevenness
        metrics["grip_uneven_norm"] = uneven_max

    # Summary stats for bar tilt
    if bar_tilts:
        tilt_median = float(statistics.median(bar_tilts))
        tilt_min = float(min(bar_tilts))
        tilt_max = float(max(bar_tilts))
        metrics["bar_tilt_median_deg"] = tilt_median
        metrics["bar_tilt_min_deg"] = tilt_min
        metrics["bar_tilt_max_deg"] = tilt_max

        # Backwards-compatible aliases
        metrics["bar_tilt_deg"] = tilt_median
        metrics["bar_tilt_deg_max"] = tilt_max

    metrics["tracking_total_frames"] = total_frames
    metrics["tracking_bad_frames"] = bad_frames
    denom = total_frames if total_frames > 0 else 1
    bad_ratio = bad_frames / denom
    metrics["tracking_bad_ratio"] = float(bad_ratio)
    metrics["tracking_quality"] = float(max(0.0, min(1.0, 1.0 - bad_ratio)))
    return metrics


def compute_frame_grip_metrics(
    landmarks: Sequence[Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute single-frame grip ratio, grip unevenness, and bar tilt (degrees)
    using the same logic as compute_rep_metrics, but for one frame only.
    Returns a dict with keys: 'grip_ratio', 'grip_uneven_norm', 'bar_tilt_deg'.
    """
    ls3 = _landmark_xyz(landmarks, L_SHOULDER)
    rs3 = _landmark_xyz(landmarks, R_SHOULDER)
    lw3 = _landmark_xyz(landmarks, L_WRIST)
    rw3 = _landmark_xyz(landmarks, R_WRIST)
    ls = _landmark_xy(landmarks, L_SHOULDER)
    rs = _landmark_xy(landmarks, R_SHOULDER)
    lw = _landmark_xy(landmarks, L_WRIST)
    rw = _landmark_xy(landmarks, R_WRIST)

    metrics: Dict[str, float] = {}
    shoulder_width = None
    if ls3 and rs3:
        shoulder_width = _dist3d(ls3, rs3)

    if (
        shoulder_width
        and shoulder_width > 1e-6
        and lw3
        and rw3
        and lw
        and rw
    ):
        grip_width = _dist3d(lw3, rw3)
        metrics["grip_ratio"] = grip_width / shoulder_width

        shoulder_center_x = 0.5 * (ls[0] + rs[0])
        left_offset = abs(lw[0] - shoulder_center_x)
        right_offset = abs(rw[0] - shoulder_center_x)
        metrics["grip_uneven_norm"] = (
            abs(left_offset - right_offset) / shoulder_width
        )

    if lw and rw:
        dx = rw[0] - lw[0]
        dy = rw[1] - lw[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        metrics["bar_tilt_deg"] = min(angle, abs(180.0 - angle))

    return metrics


def suggest_auto_tags(
    metrics: Dict[str, float],
    tracking_unreliable: bool,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Return suggested tags based on computed metrics."""
    tags: List[str] = []
    thresholds = thresholds or {}
    wide_thresh = thresholds.get("grip_wide_threshold", GRIP_WIDE_THRESHOLD)
    narrow_thresh = thresholds.get("grip_narrow_threshold", GRIP_NARROW_THRESHOLD)
    uneven_thresh = thresholds.get("grip_uneven_threshold", GRIP_UNEVEN_THRESHOLD)
    tilt_thresh = thresholds.get("bar_tilt_threshold", BAR_TILT_THRESHOLD_DEG)
    grip_ratio_median = metrics.get("grip_ratio")
    if grip_ratio_median is not None:
        if grip_ratio_median >= wide_thresh:
            tags.append("hands_too_wide")
        elif grip_ratio_median <= narrow_thresh:
            tags.append("hands_too_narrow")
    grip_uneven_med = metrics.get("grip_uneven_median")
    if grip_uneven_med is not None and grip_uneven_med >= uneven_thresh:
        tags.append("grip_uneven")
    tilt = metrics.get("bar_tilt_deg_max")
    if tilt is not None and tilt >= tilt_thresh:
        tags.append("barbell_tilted")
    if not tags and not tracking_unreliable:
        tags.append("no_major_issues")
    return tags


def default_movement_settings(name: str = "") -> Dict:
    lower = name.lower()
    body_parts = (
        BENCH_DEFAULT_PARTS.copy() if "bench" in lower else BODY_PART_OPTIONS.copy()
    )
    return {
        "model": "full",
        "det": 0.5,
        "prs": 0.7,
        "trk": 0.7,
        "ema": 0.25,
        "seg": False,
        "grip_wide_threshold": GRIP_WIDE_THRESHOLD,
        "grip_narrow_threshold": GRIP_NARROW_THRESHOLD,
        "grip_uneven_threshold": GRIP_UNEVEN_THRESHOLD,
        "bar_tilt_threshold": BAR_TILT_THRESHOLD_DEG,
        "body_parts": body_parts,
    }


# --- Video session ----------------------------------------------------------


class VideoSession(QtCore.QObject):
    dataset_loaded = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.video_paths: List[Path] = []
        self.dataset_dir: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self._last_capture_index = -1
        self.current_video_path: Optional[Path] = None
        self.current_rotation = 0
        self.current_dataset: Optional[Dict] = None
        self.current_index = -1
        self.fps = 30.0

    @property
    def has_video(self) -> bool:
        return self.cap is not None and self.frame_count > 0

    @property
    def total_frames(self) -> int:
        return self.frame_count

    def set_video_list(self, paths: List[Path]):
        self.video_paths = paths
        self.current_index = -1
        self._release_capture()
        self.frame_count = 0
        self.current_video_path = None
        self.current_rotation = 0

    def set_dataset_dir(self, path: Path):
        self.dataset_dir = path

    def _release_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._last_capture_index = -1
        self.current_rotation = 0
        self.current_rotation_override = None

    def _count_frames(self, path: Path) -> int:
        temp = cv2.VideoCapture(str(path))
        if not temp.isOpened():
            return 0
        count = 0
        while True:
            ok, _ = temp.read()
            if not ok:
                break
            count += 1
        temp.release()
        return count

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if not self.cap or self.frame_count <= 0:
            return None
        index = max(0, min(index, self.frame_count - 1))
        if index != self._last_capture_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last_capture_index = index
        rotation = self.current_rotation
        if self.current_dataset:
            override = self.current_dataset.get("rotation_override_degrees")
            if override is not None:
                rotation = override
        if rotation:
            frame = _rotate_frame_if_needed(frame, rotation)
        return frame

    def __del__(self):
        self._release_capture()

    def load_index(self, index: int) -> bool:
        if not self.video_paths:
            return False
        index = max(0, min(index, len(self.video_paths) - 1))
        self.current_index = index
        vpath = self.video_paths[index]

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {vpath}")

        fps_val = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_frames(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"No frames found in video: {vpath}")

        self._release_capture()
        self.cap = cap
        self.current_video_path = vpath
        self.frame_count = frame_count
        self._last_capture_index = -1
        self.current_rotation = _video_rotation_degrees(vpath)
        self.fps = fps_val if fps_val and fps_val > 0 else 30.0

        dataset = None
        if self.dataset_dir:
            dataset_path = self.dataset_dir / f"{vpath.stem}.json"
            if dataset_path.exists():
                dataset = json.loads(dataset_path.read_text())

        if not dataset:
            dataset = self._blank_dataset(vpath)
        else:
            fps_meta = dataset.get("fps")
            if isinstance(fps_meta, (int, float)) and fps_meta > 0:
                self.fps = fps_meta

        frames_meta = dataset.get("frames") or []
        if len(frames_meta) != self.frame_count:
            frames_meta = [
                self._blank_frame_meta(i, self.fps) for i in range(self.frame_count)
            ]
            dataset["frames"] = frames_meta

        dataset = self._normalize_dataset(dataset)
        self.current_dataset = dataset
        self.dataset_loaded.emit()
        return True

    def _blank_dataset(self, vpath: Path) -> Dict:
        movements, _, _ = load_label_options()
        movement = movements[0] if movements else ""
        return self._normalize_dataset(
            {
                "rep_id": vpath.stem,
                "video_path": str(vpath),
                "movement": movement,
                "overall_quality": None,
                "tags": [],
                "load_lbs": None,
                "rpe": 1.0,
                "camera_angle": "front",
                "lens": "0.5",
                "tag_events": [],
                "tracking_unreliable": False,
                "metrics": {},
                "fps": self.fps,
                "video_rotation_degrees": self.current_rotation,
                "rotation_override_degrees": None,
                "frames": [
                    self._blank_frame_meta(i, self.fps) for i in range(self.frame_count)
                ],
            }
        )

    @staticmethod
    def _blank_frame_meta(fi: int, fps: float) -> Dict:
        fps = fps if fps > 0 else 30.0
        return {
            "frame_index": fi,
            "time_ms": int(round(fi * 1000.0 / fps)),
            "pose_present": False,
            "landmarks": None,
        }

    def _normalize_dataset(self, dataset: Dict) -> Dict:
        tags = dataset.get("tags")
        if tags is None:
            tags = dataset.get("issues") or []
        dataset["tags"] = tags
        dataset.pop("issues", None)

        if "tag_events" not in dataset:
            legacy = dataset.get("issue_events")
            if legacy is not None:
                dataset["tag_events"] = legacy
            else:
                dataset["tag_events"] = []
        dataset.pop("issue_events", None)

        dataset.setdefault("tracking_unreliable", False)
        dataset.setdefault("tracking_manual_override", False)
        dataset.setdefault("metrics", {})
        dataset.setdefault("fps", self.fps)
        dataset.setdefault("video_rotation_degrees", self.current_rotation)
        dataset.setdefault("rotation_override_degrees", None)
        dataset.setdefault(
            "frames",
            [self._blank_frame_meta(i, self.fps) for i in range(self.frame_count)],
        )
        return dataset

    def save_current_dataset(self):
        if not (self.dataset_dir and self.current_dataset and self.current_index >= 0):
            return
        target = self.dataset_dir / f"{self.video_paths[self.current_index].stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.current_dataset, indent=2))

    def __del__(self):
        self._release_capture()


# --- Labeler view ----------------------------------------------------------


class MovementDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, name: str = "", settings: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Movement Settings")
        layout = QtWidgets.QFormLayout(self)

        self.name_edit = QtWidgets.QLineEdit(name)
        layout.addRow("Movement name", self.name_edit)

        cfg = settings or default_movement_settings(name)
        self.model_box = QtWidgets.QComboBox()
        self.model_box.addItems(MODEL_VARIANTS)
        idx = self.model_box.findText(cfg.get("model", "full"))
        if idx >= 0:
            self.model_box.setCurrentIndex(idx)
        layout.addRow("Model variant", self.model_box)

        self.det_spin = QtWidgets.QDoubleSpinBox()
        self.det_spin.setRange(0.1, 1.0)
        self.det_spin.setSingleStep(0.05)
        self.det_spin.setValue(cfg.get("det", 0.5))
        self.prs_spin = QtWidgets.QDoubleSpinBox()
        self.prs_spin.setRange(0.1, 1.0)
        self.prs_spin.setSingleStep(0.05)
        self.prs_spin.setValue(cfg.get("prs", 0.7))
        self.trk_spin = QtWidgets.QDoubleSpinBox()
        self.trk_spin.setRange(0.1, 1.0)
        self.trk_spin.setSingleStep(0.05)
        self.trk_spin.setValue(cfg.get("trk", 0.7))
        self.ema_spin = QtWidgets.QDoubleSpinBox()
        self.ema_spin.setRange(0.0, 1.0)
        self.ema_spin.setSingleStep(0.05)
        self.ema_spin.setValue(cfg.get("ema", 0.25))
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setChecked(bool(cfg.get("seg", False)))
        self.grip_wide_spin = QtWidgets.QDoubleSpinBox()
        self.grip_wide_spin.setRange(1.0, 5.0)
        self.grip_wide_spin.setSingleStep(0.05)
        self.grip_wide_spin.setValue(cfg.get("grip_wide_threshold", GRIP_WIDE_THRESHOLD))
        self.grip_narrow_spin = QtWidgets.QDoubleSpinBox()
        self.grip_narrow_spin.setRange(0.1, 3.0)
        self.grip_narrow_spin.setSingleStep(0.05)
        self.grip_narrow_spin.setValue(
            cfg.get("grip_narrow_threshold", GRIP_NARROW_THRESHOLD)
        )
        self.grip_uneven_spin = QtWidgets.QDoubleSpinBox()
        self.grip_uneven_spin.setRange(0.0, 1.0)
        self.grip_uneven_spin.setSingleStep(0.01)
        self.grip_uneven_spin.setValue(
            cfg.get("grip_uneven_threshold", GRIP_UNEVEN_THRESHOLD)
        )
        self.bar_tilt_spin = QtWidgets.QDoubleSpinBox()
        self.bar_tilt_spin.setRange(0.0, 45.0)
        self.bar_tilt_spin.setSingleStep(0.5)
        self.bar_tilt_spin.setValue(
            cfg.get("bar_tilt_threshold", BAR_TILT_THRESHOLD_DEG)
        )

        layout.addRow("det threshold", self.det_spin)
        layout.addRow("prs threshold", self.prs_spin)
        layout.addRow("trk threshold", self.trk_spin)
        layout.addRow("EMA alpha", self.ema_spin)
        layout.addRow(self.seg_check)
        layout.addRow("Grip wide threshold", self.grip_wide_spin)
        layout.addRow("Grip narrow threshold", self.grip_narrow_spin)
        layout.addRow("Grip uneven threshold", self.grip_uneven_spin)
        layout.addRow("Bar tilt threshold", self.bar_tilt_spin)

        self.body_part_checks = []
        parts_box = QtWidgets.QGroupBox("Body parts to display")
        parts_layout = QtWidgets.QGridLayout(parts_box)
        selected_parts = set(cfg.get("body_parts") or [])
        for i, part in enumerate(BODY_PART_OPTIONS):
            chk = QtWidgets.QCheckBox(part)
            chk.setChecked(part in selected_parts)
            self.body_part_checks.append(chk)
            row, col = divmod(i, 2)
            parts_layout.addWidget(chk, row, col)
        layout.addRow(parts_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        if not self.name_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self, "Missing name", "Enter a movement name."
            )
            return
        super().accept()

    def values(self):
        name = self.name_edit.text().strip()
        body_parts = [
            chk.text() for chk in self.body_part_checks if chk.isChecked()
        ] or BODY_PART_OPTIONS.copy()
        return name, {
            "model": self.model_box.currentText(),
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
             "grip_wide_threshold": self.grip_wide_spin.value(),
             "grip_narrow_threshold": self.grip_narrow_spin.value(),
             "grip_uneven_threshold": self.grip_uneven_spin.value(),
             "bar_tilt_threshold": self.bar_tilt_spin.value(),
            "body_parts": body_parts,
        }


class AdminPanel(QtWidgets.QWidget):
    config_saved = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        main_layout = QtWidgets.QVBoxLayout(self)

        header_row = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel(
            "Manage lifts and form tags. Saving updates label_config.json."
        )
        header_row.addWidget(header_label)
        header_row.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        header_row.addWidget(self.home_button)
        main_layout.addLayout(header_row)

        splitter = QtWidgets.QHBoxLayout()
        main_layout.addLayout(splitter)

        # Movements section
        move_group = QtWidgets.QGroupBox("Lifts / Exercises")
        splitter.addWidget(move_group, stretch=2)
        move_layout = QtWidgets.QVBoxLayout(move_group)
        self.movement_list = QtWidgets.QListWidget()
        self.movement_list.currentTextChanged.connect(self._show_movement_settings)
        move_layout.addWidget(self.movement_list)

        move_btn_row = QtWidgets.QHBoxLayout()
        move_layout.addLayout(move_btn_row)
        add_move = QtWidgets.QPushButton("Add")
        add_move.clicked.connect(self._add_movement)
        edit_move = QtWidgets.QPushButton("Edit")
        edit_move.clicked.connect(self._edit_movement)
        del_move = QtWidgets.QPushButton("Remove")
        del_move.clicked.connect(self._remove_movement)
        move_btn_row.addWidget(add_move)
        move_btn_row.addWidget(edit_move)
        move_btn_row.addWidget(del_move)

        self.movement_info = QtWidgets.QTextEdit()
        self.movement_info.setReadOnly(True)
        move_layout.addWidget(self.movement_info)

        # Tags section
        tag_group = QtWidgets.QGroupBox("Form tags")
        splitter.addWidget(tag_group, stretch=1)
        tag_layout = QtWidgets.QVBoxLayout(tag_group)
        self.tag_list = QtWidgets.QListWidget()
        tag_layout.addWidget(self.tag_list)

        tag_btn_row = QtWidgets.QHBoxLayout()
        tag_layout.addLayout(tag_btn_row)
        add_tag = QtWidgets.QPushButton("Add")
        add_tag.clicked.connect(self._add_tag)
        edit_tag = QtWidgets.QPushButton("Edit")
        edit_tag.clicked.connect(self._edit_tag)
        del_tag = QtWidgets.QPushButton("Remove")
        del_tag.clicked.connect(self._remove_tag)
        tag_btn_row.addWidget(add_tag)
        tag_btn_row.addWidget(edit_tag)
        tag_btn_row.addWidget(del_tag)

        # Bottom actions
        btn_row = QtWidgets.QHBoxLayout()
        main_layout.addLayout(btn_row)
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self._load_from_file)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_to_file)
        btn_row.addWidget(reload_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)

        self.movements: List[str] = []
        self.tags: List[str] = []
        self.movement_settings: Dict[str, Dict] = {}
        self._load_from_file()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def _load_from_file(self):
        cfg = load_label_config()
        self.movements = cfg.get("movements") or []
        self.tags = cfg.get("tags") or []
        raw_settings = cfg.get("movement_settings") or {}
        self.movement_settings = {}
        for name in self.movements:
            merged = default_movement_settings(name)
            merged.update(raw_settings.get(name) or {})
            self.movement_settings[name] = merged

        self.movement_list.clear()
        self.movement_list.addItems(self.movements)
        self.tag_list.clear()
        self.tag_list.addItems(self.tags)
        self.movement_info.clear()

    def _save_to_file(self):
        movements = [
            self.movement_list.item(i).text() for i in range(self.movement_list.count())
        ]
        tags = [self.tag_list.item(i).text() for i in range(self.tag_list.count())]
        settings = {
            name: self.movement_settings.get(name, default_movement_settings(name))
            for name in movements
        }
        save_label_config(
            {
                "movements": movements,
                "tags": tags,
                "movement_settings": settings,
            }
        )
        QtWidgets.QMessageBox.information(self, "Saved", "Configuration updated.")
        self.config_saved.emit()

    def _add_movement(self):
        dialog = MovementDialog(self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            name, settings = dialog.values()
            if name in self.movements:
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate", f"{name} already exists."
                )
                return
            self.movements.append(name)
            self.movement_settings[name] = settings
            self.movement_list.addItem(name)

    def _edit_movement(self):
        item = self.movement_list.currentItem()
        if not item:
            return
        old_name = item.text()
        settings = self.movement_settings.get(
            old_name, default_movement_settings(old_name)
        )
        dialog = MovementDialog(self, name=old_name, settings=settings)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_name, new_settings = dialog.values()
            if new_name != old_name and new_name in self.movements:
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate", f"{new_name} already exists."
                )
                return
            idx = self.movements.index(old_name)
            self.movements[idx] = new_name
            del self.movement_settings[old_name]
            self.movement_settings[new_name] = new_settings
            item.setText(new_name)

    def _remove_movement(self):
        item = self.movement_list.currentItem()
        if not item:
            return
        name = item.text()
        self.movements.remove(name)
        self.movement_settings.pop(name, None)
        self.movement_list.takeItem(self.movement_list.row(item))
        self.movement_info.clear()

    def _show_movement_settings(self, name: str):
        if not name:
            self.movement_info.clear()
            return
        settings = self.movement_settings.get(name, default_movement_settings(name))
        pretty = json.dumps(settings, indent=2)
        self.movement_info.setPlainText(pretty)

    def _add_tag(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add form tag", "Tag name")
        if ok and text.strip():
            value = text.strip()
            self.tags.append(value)
            self.tag_list.addItem(value)

    def _edit_tag(self):
        item = self.tag_list.currentItem()
        if not item:
            return
        current = item.text()
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Edit form tag", "Tag name", text=current
        )
        if ok and text.strip():
            value = text.strip()
            index = self.tag_list.row(item)
            self.tags[index] = value
            item.setText(value)

    def _remove_tag(self):
        item = self.tag_list.currentItem()
        if not item:
            return
        index = self.tag_list.row(item)
        self.tag_list.takeItem(index)
        if 0 <= index < len(self.tags):
            self.tags.pop(index)


class HomePage(QtWidgets.QWidget):
    requested_admin = QtCore.Signal()
    requested_cutting = QtCore.Signal()
    requested_labeling = QtCore.Signal()
    requested_pose = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "Welcome to the mAI Coach tool suite.\nChoose an option to get started.",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        button_grid = QtWidgets.QGridLayout()
        layout.addLayout(button_grid)

        def add_button(row, col, text, signal):
            btn = QtWidgets.QPushButton(text)
            btn.setMinimumHeight(80)
            button_grid.addWidget(btn, row, col)
            btn.clicked.connect(signal)

        add_button(0, 0, "Admin Controls", self.requested_admin.emit)
        add_button(0, 1, "Video Cutting", self.requested_cutting.emit)
        add_button(1, 0, "Video Labeling", self.requested_labeling.emit)
        add_button(1, 1, "Pose Tuning", self.requested_pose.emit)
        layout.addStretch(1)


class WorkflowPlaceholderPage(QtWidgets.QWidget):
    def __init__(self, title: str, description: str):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        layout = QtWidgets.QVBoxLayout(self)
        header_row = QtWidgets.QHBoxLayout()
        header = QtWidgets.QLabel(title)
        header.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        header_row.addWidget(self.home_button)
        layout.addLayout(header_row)

        self.description_label = QtWidgets.QLabel(
            description, alignment=QtCore.Qt.AlignCenter
        )
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        layout.addWidget(QtWidgets.QLabel("Selected files:"))
        self.file_list = QtWidgets.QListWidget()
        layout.addWidget(self.file_list, stretch=1)

    def set_files(self, files: List[Path]):
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(str(f))

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()


class LabelerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.tags, self.movement_settings = load_label_options()
        self.current_body_parts = BODY_PART_OPTIONS.copy()
        self._rotation_lock_value: Optional[int] = None
        self._default_weight_lbs = 0
        self._inputs_locked = False
        self._auto_finish = False
        self.session = VideoSession()
        self.session.dataset_loaded.connect(self._on_dataset_loaded)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        self.play_timer.start(30)

        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._pose_job_active = False

        self._build_ui()
        self._update_timer_interval()
        self._update_live_frame_metrics(None)

    # UI ------------------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        top_bar = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Videos")
        self.load_btn.clicked.connect(self._pick_videos)
        self.dataset_btn = QtWidgets.QPushButton("Choose Dataset Folder")
        self.dataset_btn.clicked.connect(self._pick_dataset_dir)
        top_bar.addWidget(self.load_btn)
        top_bar.addWidget(self.dataset_btn)
        top_bar.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top_bar.addWidget(self.home_button)
        root.addLayout(top_bar)

        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, stretch=1)

        # left: video + controls
        left = QtWidgets.QVBoxLayout()
        content.addLayout(left, stretch=3)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 315)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #333;")
        left.addWidget(self.video_label, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        left.addLayout(controls)

        def mk_btn(text, slot):
            self._add_btn(controls, text, slot)

        mk_btn("⟵ Frame", lambda: self._step_frames(-1))
        mk_btn("-0.5s", lambda: self._step_seconds(-0.5))
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        mk_btn("Replay", self._replay)
        mk_btn("+0.5s", lambda: self._step_seconds(0.5))
        mk_btn("Frame ⟶", lambda: self._step_frames(1))

        controls.addStretch(1)

        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_box)

        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        left.addWidget(self.scrubber)

        self.pose_refresh_btn = QtWidgets.QPushButton("Re-run Pose Tracker")
        self.pose_refresh_btn.setToolTip(
            "Generate or refresh pose overlays for the current video."
        )
        self.pose_refresh_btn.setEnabled(False)
        self.pose_refresh_btn.clicked.connect(
            lambda: self._ensure_pose_data(force=True)
        )
        left.addWidget(self.pose_refresh_btn)

        # right: form
        right_column = QtWidgets.QVBoxLayout()
        content.addLayout(right_column, stretch=2)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_column.addWidget(right_scroll)
        right_widget = QtWidgets.QWidget()
        right_scroll.setWidget(right_widget)
        right = QtWidgets.QVBoxLayout(right_widget)

        form = QtWidgets.QFormLayout()
        right.addLayout(form)

        self.rep_id = QtWidgets.QLineEdit()
        form.addRow("rep_id", self.rep_id)

        self.movement_cb = QtWidgets.QComboBox()
        self.movement_cb.setEditable(True)
        self.movement_cb.addItems(self.movements)
        self.movement_cb.currentTextChanged.connect(self._on_movement_changed)
        form.addRow("movement", self.movement_cb)

        meta_widget = QtWidgets.QWidget()
        meta_grid = QtWidgets.QGridLayout(meta_widget)
        meta_grid.setHorizontalSpacing(8)
        meta_grid.setVerticalSpacing(2)
        meta_grid.setContentsMargins(0, 0, 0, 0)

        self.load_spin = QtWidgets.QSpinBox()
        self.load_spin.setRange(0, 2000)
        self.load_spin.setSuffix(" lbs")
        self.load_spin.setSingleStep(5)
        self.load_spin.valueChanged.connect(self._on_load_spin_changed)
        self.load_spin.valueChanged.connect(self._on_load_spin_changed)

        self.rpe_cb = QtWidgets.QComboBox()
        self.rpe_cb.addItems(RPE_OPTIONS)
        self.rpe_cb.setCurrentText("1.0")

        self.camera_cb = QtWidgets.QComboBox()
        self.camera_cb.addItems(CAMERA_ANGLE_OPTIONS)

        self.lens_cb = QtWidgets.QComboBox()
        self.lens_cb.addItems(LENS_OPTIONS)

        meta_fields = [
            ("load_lbs", self.load_spin),
            ("RPE", self.rpe_cb),
            ("camera_angle", self.camera_cb),
            ("lens", self.lens_cb),
        ]

        row = 0
        col = 0
        for label_text, widget in meta_fields[:3]:
            label = QtWidgets.QLabel(label_text)
            meta_grid.addWidget(label, row, col)
            meta_grid.addWidget(widget, row, col + 1)
            col += 2

        row = 1
        col = 0
        for label_text, widget in meta_fields[3:]:
            label = QtWidgets.QLabel(label_text)
            meta_grid.addWidget(label, row, col)
            meta_grid.addWidget(widget, row, col + 1)
            col += 2

        form.addRow(meta_widget)

        self.rotation_combo = QtWidgets.QComboBox()
        for label, _ in ROTATION_OPTIONS:
            self.rotation_combo.addItem(label)
        self.rotation_combo.currentIndexChanged.connect(self._on_rotation_combo_changed)
        self.rotation_combo.setToolTip(
            "Override the video's rotation when metadata is wrong."
        )
        rotation_widget = QtWidgets.QWidget()
        rotation_layout = QtWidgets.QHBoxLayout(rotation_widget)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.addWidget(self.rotation_combo)
        self.rotation_lock_cb = QtWidgets.QCheckBox("Keep for next videos")
        self.rotation_lock_cb.setToolTip(
            "When enabled, this rotation setting becomes the default for future videos."
        )
        self.rotation_lock_cb.stateChanged.connect(self._on_rotation_lock_toggled)
        rotation_layout.addWidget(self.rotation_lock_cb)
        rotation_layout.addStretch(1)
        form.addRow("Rotation override", rotation_widget)

        form.addRow(QtWidgets.QLabel("Form tags (Cmd/Ctrl or Shift to multi-select):"))
        self.tag_list = QtWidgets.QListWidget()
        self.tag_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tag_list.setMinimumHeight(160)
        self.tag_list.addItems(self.tags)
        self._ensure_default_tag_entry()
        self.tag_list.itemSelectionChanged.connect(self._on_tag_selection_changed)
        form.addRow(self.tag_list)

        tag_event_row = QtWidgets.QHBoxLayout()
        self.tag_picker = QtWidgets.QComboBox()
        self.tag_picker.addItems(self.tags)
        tag_event_row.addWidget(self.tag_picker)
        add_tag = QtWidgets.QPushButton("Add tag @ frame")
        add_tag.clicked.connect(self._add_tag_event)
        tag_event_row.addWidget(add_tag)
        right.addLayout(tag_event_row)

        self.lift_tools_box = QtWidgets.QGroupBox("Lift-specific tools")
        lift_layout = QtWidgets.QVBoxLayout(self.lift_tools_box)
        right.addWidget(self.lift_tools_box)

        self.live_metrics_box = QtWidgets.QGroupBox(
            "Live frame pose (current frame)"
        )
        live_layout = QtWidgets.QFormLayout(self.live_metrics_box)
        live_layout.setContentsMargins(8, 8, 8, 8)
        live_layout.setVerticalSpacing(2)
        live_layout.setHorizontalSpacing(6)
        self.live_grip_label = QtWidgets.QLabel("n/a")
        self.live_uneven_label = QtWidgets.QLabel("n/a")
        self.live_tilt_label = QtWidgets.QLabel("n/a")
        live_layout.addRow("Grip width", self.live_grip_label)
        live_layout.addRow("Grip unevenness", self.live_uneven_label)
        live_layout.addRow("Bar tilt", self.live_tilt_label)
        lift_layout.addWidget(self.live_metrics_box)

        self.metrics_box = QtWidgets.QGroupBox("Pose metrics & tracking")
        self.metrics_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        metrics_layout = QtWidgets.QVBoxLayout(self.metrics_box)
        table = QtWidgets.QTableWidget(3, 4)
        table.setHorizontalHeaderLabels(["Metric", "Min", "Median", "Max"])
        table.setVerticalHeaderLabels(["", "", ""])
        for row, name in enumerate(["Grip width", "Grip unevenness", "Bar tilt"]):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            table.setItem(row, 0, item)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(QtCore.Qt.NoFocus)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setFixedHeight(
            table.horizontalHeader().sizeHint().height()
            + table.verticalHeader().length()
            + table.frameWidth() * 2
        )
        self.metrics_table = table
        metrics_layout.addWidget(table)
        self.tracking_quality_label = QtWidgets.QLabel("Tracking quality: –")
        metrics_layout.addWidget(self.tracking_quality_label)

        self.tracking_checkbox = QtWidgets.QCheckBox(
            "Mark tracking unreliable (exclude from training)"
        )
        self.tracking_checkbox.stateChanged.connect(
            self._on_tracking_checkbox_changed
        )
        metrics_layout.addWidget(self.tracking_checkbox)

        self.tracking_auto_label = QtWidgets.QLabel("Auto suggestion: –")
        self.tracking_auto_label.setToolTip(
            "Shows whether the automatic rules think tracking is OK or unreliable."
        )
        metrics_layout.addWidget(self.tracking_auto_label)

        self.tracking_auto_button = QtWidgets.QPushButton("Use auto suggestion")
        self.tracking_auto_button.clicked.connect(self._use_tracking_auto)
        self.tracking_auto_button.setEnabled(False)
        self.tracking_auto_button.setToolTip(
            "Restore the auto-detected tracking flag if you overrode it manually."
        )
        metrics_layout.addWidget(self.tracking_auto_button)

        lift_layout.addWidget(self.metrics_box)
        self.lift_placeholder = QtWidgets.QLabel(
            "No lift-specific tools configured for this movement yet."
        )
        self.lift_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lift_placeholder.setStyleSheet("color: #777; font-style: italic;")
        lift_layout.addWidget(self.lift_placeholder)

        self.tag_events = QtWidgets.QListWidget()
        right.addWidget(self.tag_events, stretch=1)

        remove_tag = QtWidgets.QPushButton("Remove selected tag event")
        remove_tag.clicked.connect(self._remove_tag_event)
        right.addWidget(remove_tag)

        right.addStretch(1)
        nav = QtWidgets.QHBoxLayout()
        right_column.addLayout(nav)
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda: self._load_relative(-1))
        nav.addWidget(self.prev_btn)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save_dataset)
        nav.addWidget(self.save_btn)
        self.next_btn = QtWidgets.QPushButton("Save + Next")
        self.next_btn.clicked.connect(lambda: self._load_relative(+1, save=True))
        nav.addWidget(self.next_btn)
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()
        self._update_nav_buttons()

    def refresh_label_options(self):
        self.movements, self.tags, self.movement_settings = load_label_options()
        current_move = self.movement_cb.currentText()
        self.movement_cb.blockSignals(True)
        self.movement_cb.clear()
        self.movement_cb.addItems(self.movements)
        if current_move:
            idx = self.movement_cb.findText(current_move)
            if idx >= 0:
                self.movement_cb.setCurrentIndex(idx)
            else:
                self.movement_cb.setEditText(current_move)
        self.movement_cb.blockSignals(False)

        if self.session.current_dataset:
            selected = set(self.session.current_dataset.get("tags", []))
        else:
            selected = set()
        self.tag_list.clear()
        self.tag_list.addItems(self.tags)
        self._ensure_default_tag_entry()
        self._apply_tag_selection(list(selected))

        current_tag = self.tag_picker.currentText()
        self.tag_picker.blockSignals(True)
        self.tag_picker.clear()
        self.tag_picker.addItems(self.tags)
        idx = self.tag_picker.findText(current_tag)
        if idx >= 0:
            self.tag_picker.setCurrentIndex(idx)
        self.tag_picker.blockSignals(False)
        self._update_body_part_preview()
        self._enforce_default_tag_rule()

    def _on_movement_changed(self, name: str):
        if self.session.current_dataset is not None:
            self.session.current_dataset["movement"] = name
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()

    def _resolve_movement_settings(self, movement: str) -> Optional[Dict]:
        if not movement:
            return None
        movement = movement.strip()
        if not movement:
            return None
        direct = self.movement_settings.get(movement)
        if direct:
            return direct
        lower = movement.lower()
        for key, value in self.movement_settings.items():
            if key.lower() == lower:
                return value
        return None

    def _movement_settings_for(self, movement: str) -> Dict:
        settings = self._resolve_movement_settings(movement)
        if not settings:
            settings = default_movement_settings(movement)
        return settings

    def _update_body_part_preview(self):
        movement = self.movement_cb.currentText().strip()
        settings = self._movement_settings_for(movement)
        self.current_body_parts = settings.get("body_parts") or BODY_PART_OPTIONS.copy()

    def _add_btn(self, layout: QtWidgets.QHBoxLayout, text: str, slot):
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(slot)
        layout.addWidget(btn)

    def _ensure_default_tag_entry(self):
        if not self.tag_list.findItems(DEFAULT_OK_TAG, QtCore.Qt.MatchExactly):
            self.tag_list.addItem(DEFAULT_OK_TAG)
        if hasattr(self, "tag_picker"):
            if self.tag_picker.findText(DEFAULT_OK_TAG) < 0:
                self.tag_picker.addItem(DEFAULT_OK_TAG)

    def _toggle_tag_selection(self, tag: str, selected: bool):
        items = self.tag_list.findItems(tag, QtCore.Qt.MatchExactly)
        if not items:
            return
        self.tag_list.blockSignals(True)
        items[0].setSelected(selected)
        self.tag_list.blockSignals(False)

    def _enforce_default_tag_rule(self):
        selected_items = self.tag_list.selectedItems()
        names = {item.text() for item in selected_items}
        if not names:
            self._toggle_tag_selection(DEFAULT_OK_TAG, True)
            return
        if DEFAULT_OK_TAG in names and len(names) > 1:
            self._toggle_tag_selection(DEFAULT_OK_TAG, False)

    def _on_tag_selection_changed(self):
        self._enforce_default_tag_rule()

    def _movement_name_for_tools(self) -> str:
        if self.session.current_dataset:
            movement = self.session.current_dataset.get("movement")
            if movement:
                return movement
        return self.movement_cb.currentText()

    @staticmethod
    def _lift_uses_bench_tools(name: Optional[str]) -> bool:
        if not name:
            return False
        return "bench" in name.lower()

    def _thresholds_for_movement(self, movement: str) -> Dict[str, float]:
        settings = self._movement_settings_for(movement)
        return {
            "grip_wide_threshold": settings.get(
                "grip_wide_threshold", GRIP_WIDE_THRESHOLD
            ),
            "grip_narrow_threshold": settings.get(
                "grip_narrow_threshold", GRIP_NARROW_THRESHOLD
            ),
            "grip_uneven_threshold": settings.get(
                "grip_uneven_threshold", GRIP_UNEVEN_THRESHOLD
            ),
            "bar_tilt_threshold": settings.get(
                "bar_tilt_threshold", BAR_TILT_THRESHOLD_DEG
            ),
        }

    def _refresh_lift_specific_tools(self):
        movement = self._movement_name_for_tools()
        use_bench = self._lift_uses_bench_tools(movement)
        for widget in (self.live_metrics_box, self.metrics_box):
            widget.setVisible(use_bench)
        self.lift_placeholder.setVisible(not use_bench)
        self.lift_tools_box.setVisible(True)
        if not use_bench:
            self._update_live_frame_metrics(None)

    def _apply_tag_selection(self, tags: Optional[Sequence[str]]):
        selected = set(tags or [])
        if not self.tag_list:
            return
        self._ensure_default_tag_entry()
        for tag in selected:
            if not self.tag_list.findItems(tag, QtCore.Qt.MatchExactly):
                self.tag_list.addItem(tag)
            if self.tag_picker.findText(tag) < 0:
                self.tag_picker.addItem(tag)
        for i in range(self.tag_list.count()):
            item = self.tag_list.item(i)
            item.setSelected(item.text() in selected)
        self._enforce_default_tag_rule()

    def _compute_metrics_for_current_dataset(self, auto_apply_if_empty: bool) -> List[str]:
        dataset = self.session.current_dataset
        if not dataset:
            self._update_metrics_panel({}, False)
            self._update_tracking_controls(False, False)
            return []
        frames = dataset.get("frames") or []
        dataset.setdefault("tracking_manual_override", False)
        metrics = compute_rep_metrics(frames)
        dataset["metrics"] = metrics

        movement = dataset.get("movement") or self.movement_cb.currentText()
        thresholds = self._thresholds_for_movement(movement)

        total_frames = len(frames)
        bad_frames = int(metrics.get("tracking_bad_frames", 0))
        auto_flag = bool(metrics.get("tracking_bad_ratio", 1.0) > TRACKING_BAD_RATIO_MAX)
        dataset["tracking_auto_recommended"] = auto_flag
        manual_override = bool(dataset.get("tracking_manual_override", False))
        if not manual_override or "tracking_unreliable" not in dataset:
            dataset["tracking_unreliable"] = auto_flag

        tracking_flag = bool(dataset.get("tracking_unreliable", False))
        auto_tags = suggest_auto_tags(metrics, tracking_flag, thresholds)
        dataset["auto_tags"] = auto_tags
        if auto_apply_if_empty and not dataset.get("tags"):
            dataset["tags"] = list(auto_tags)

        self._update_metrics_panel(metrics, tracking_flag)
        self._update_tracking_controls(tracking_flag, auto_flag)
        return auto_tags

    def _update_metrics_panel(self, metrics: Dict[str, float], tracking_flag: bool):
        if not hasattr(self, "metrics_table"):
            return

        def fmt_ratio(value: Optional[float], suffix: str = "") -> str:
            if value is None:
                return "n/a"
            return f"{value:.2f}{suffix}"

        def fmt_degrees(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value:.1f}°"

        def set_cell(row: int, col: int, text: str):
            item = self.metrics_table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.metrics_table.setItem(row, col, item)
            item.setText(text)

        set_cell(0, 1, fmt_ratio(metrics.get("grip_ratio_min"), " × shoulder"))
        set_cell(0, 2, fmt_ratio(metrics.get("grip_ratio"), " × shoulder"))
        set_cell(0, 3, fmt_ratio(metrics.get("grip_ratio_max"), " × shoulder"))
        set_cell(1, 1, fmt_ratio(metrics.get("grip_uneven_min")))
        set_cell(1, 2, fmt_ratio(metrics.get("grip_uneven_median")))
        set_cell(1, 3, fmt_ratio(metrics.get("grip_uneven_norm")))
        set_cell(2, 1, fmt_degrees(metrics.get("bar_tilt_min_deg")))
        set_cell(2, 2, fmt_degrees(metrics.get("bar_tilt_deg")))
        set_cell(2, 3, fmt_degrees(metrics.get("bar_tilt_deg_max")))
        quality = metrics.get("tracking_quality")
        bad_frames = int(metrics.get("tracking_bad_frames", 0) or 0)
        total_frames = int(metrics.get("tracking_total_frames", 0) or 0)
        if quality is None:
            quality_text = "n/a"
        else:
            quality_text = f"{quality:.2f} (bad {bad_frames}/{total_frames})"
        self.tracking_quality_label.setText(f"Tracking quality: {quality_text}")

        status_text = "Tracking unreliable" if tracking_flag else "Tracking OK"
        self.metrics_box.setTitle(f"Pose metrics & tracking – {status_text}")

    def _update_tracking_controls(self, tracking_flag: bool, auto_flag: bool):
        dataset = self.session.current_dataset or {}
        manual_override = bool(dataset.get("tracking_manual_override", False))
        auto_text = "unreliable" if auto_flag else "ok"
        if manual_override:
            auto_text += " (manual override)"
        if hasattr(self, "tracking_checkbox"):
            self.tracking_checkbox.blockSignals(True)
            self.tracking_checkbox.setChecked(tracking_flag)
            self.tracking_checkbox.blockSignals(False)
        if hasattr(self, "tracking_auto_label"):
            self.tracking_auto_label.setText(f"Auto suggestion: {auto_text}")
        if hasattr(self, "tracking_auto_button"):
            self.tracking_auto_button.setEnabled(True)
        self._update_live_frame_metrics(None)

    def _on_tracking_checkbox_changed(self, state: int):
        dataset = self.session.current_dataset
        if not dataset:
            return
        checked = state == QtCore.Qt.Checked
        dataset["tracking_unreliable"] = checked
        auto_flag = dataset.get("tracking_auto_recommended")
        dataset["tracking_manual_override"] = (
            auto_flag is None or bool(auto_flag) != checked
        )
        self._update_tracking_controls(checked, bool(auto_flag))

    def _use_tracking_auto(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        dataset["tracking_manual_override"] = False
        auto_flag = bool(dataset.get("tracking_auto_recommended", False))
        dataset["tracking_unreliable"] = auto_flag
        self._update_tracking_controls(auto_flag, auto_flag)

    def _update_live_frame_metrics(self, frame_rec: Optional[Dict]):
        if not self._lift_uses_bench_tools(self._movement_name_for_tools()):
            for lbl in (
                self.live_grip_label,
                self.live_uneven_label,
                self.live_tilt_label,
            ):
                lbl.setText("n/a")
            return
        if (
            not frame_rec
            or not frame_rec.get("pose_present")
            or not frame_rec.get("landmarks")
        ):
            for lbl in (
                self.live_grip_label,
                self.live_uneven_label,
                self.live_tilt_label,
            ):
                lbl.setText("n/a")
            return

        metrics = compute_frame_grip_metrics(frame_rec["landmarks"])
        grip_ratio = metrics.get("grip_ratio")
        if grip_ratio is not None:
            self.live_grip_label.setText(f"{grip_ratio:.2f} × shoulder width")
        else:
            self.live_grip_label.setText("n/a")

        grip_uneven = metrics.get("grip_uneven_norm")
        if grip_uneven is not None:
            self.live_uneven_label.setText(
                f"{grip_uneven:.2f} × shoulder width"
            )
        else:
            self.live_uneven_label.setText("n/a")

        tilt = metrics.get("bar_tilt_deg")
        if tilt is not None:
            self.live_tilt_label.setText(f"{tilt:.1f}°")
        else:
            self.live_tilt_label.setText("n/a")

    def _set_rotation_combo_value(self, degrees: Optional[int]):
        target_idx = _rotation_option_index(degrees)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(target_idx)
        self.rotation_combo.blockSignals(False)

    def _current_rotation_override(self) -> Optional[int]:
        return _rotation_value_from_index(self.rotation_combo.currentIndex())

    def _on_rotation_lock_toggled(self, state: int):
        if state == QtCore.Qt.Checked:
            self._rotation_lock_value = self._current_rotation_override()
            dataset = self.session.current_dataset
            if dataset and dataset.get("rotation_override_degrees") is None:
                dataset["rotation_override_degrees"] = self._rotation_lock_value
                self._render_frame(self.current_frame)
        else:
            self._rotation_lock_value = None

    def _on_rotation_combo_changed(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        override = self._current_rotation_override()
        dataset["rotation_override_degrees"] = override
        if self.rotation_lock_cb.isChecked():
            self._rotation_lock_value = override
        self._render_frame(self.current_frame)

    def _on_load_spin_changed(self, value: int):
        self._default_weight_lbs = value

    # Data -----------------------------------------------------------------
    def _pick_videos(self):
        self._inputs_locked = False
        self._auto_finish = False
        self.load_btn.setEnabled(True)
        self.dataset_btn.setEnabled(True)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        paths = [Path(f) for f in files]
        self.session.set_video_list(paths)
        self._load_by_index(0)

    def _pick_dataset_dir(self):
        self._inputs_locked = False
        self.dataset_btn.setEnabled(True)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if not folder:
            return
        self.session.set_dataset_dir(Path(folder))
        if self.session.current_index >= 0:
            self.session.save_current_dataset()

    def load_labeler_inputs(self, videos: List[Path], dataset_dir: Optional[Path]):
        if videos:
            self.session.set_video_list(videos)
            self._inputs_locked = True
            self._auto_finish = True
            self.load_btn.setEnabled(False)
        if dataset_dir:
            self.session.set_dataset_dir(dataset_dir)
            self.dataset_btn.setEnabled(False)
        if videos:
            self._load_by_index(0)
        else:
            self._update_body_part_preview()
            self._update_nav_buttons()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def _load_by_index(self, index: int):
        try:
            if self.session.load_index(index):
                self.current_frame = 0
                self.playing = False
                self._speed_residual = 0.0
                self.scrubber.setRange(0, max(0, self.session.total_frames - 1))
                self._update_form_from_dataset()
                self._update_timer_interval()
                self._render_frame(0)
                self._update_nav_buttons()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

    def _load_relative(self, delta: int, save: bool = False):
        if save:
            self._save_dataset()
        total = len(self.session.video_paths)
        if total == 0 or self.session.current_index < 0:
            return
        if delta > 0 and self._auto_finish and self.session.current_index >= total - 1:
            if self._home_cb:
                self._home_cb()
            return
        new_index = self.session.current_index + delta
        new_index = max(0, min(new_index, total - 1))
        if new_index == self.session.current_index:
            return
        self._load_by_index(new_index)

    def _on_dataset_loaded(self):
        self._update_form_from_dataset()
        self._render_frame(0)
        self.pose_refresh_btn.setEnabled(True)
        if not self._has_pose_overlay():
            self._ensure_pose_data(force=False)

    def _update_form_from_dataset(self):
        d = self.session.current_dataset or {}
        movement_val = d.get("movement") or ""
        if self.movements:
            self.movement_cb.setCurrentText(movement_val or self.movements[0])
        else:
            self.movement_cb.setEditText(movement_val)
        self.rep_id.setText(str(d.get("rep_id", "")))
        oq = d.get("overall_quality")
        self.quality_cb.setCurrentText(str(oq) if oq else "3")
        if d.get("load_lbs") is not None:
            ll = float(d.get("load_lbs") or 0)
            self._default_weight_lbs = ll
        else:
            ll = self._default_weight_lbs
            d["load_lbs"] = ll
        self.load_spin.blockSignals(True)
        self.load_spin.setValue(float(ll))
        self.load_spin.blockSignals(False)
        self.rpe_cb.setCurrentText(str(d.get("rpe", "1.0")))
        self.camera_cb.setCurrentText(d.get("camera_angle") or "front")
        self.lens_cb.setCurrentText(d.get("lens") or "0.5")
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()

        override = d.get("rotation_override_degrees")
        if (
            override is None
            and self.rotation_lock_cb.isChecked()
            and self._rotation_lock_value is not None
        ):
            override = self._rotation_lock_value
            d["rotation_override_degrees"] = override
        self._set_rotation_combo_value(override)

        existing_tags = d.get("tags") or []
        auto_tags = self._compute_metrics_for_current_dataset(
            auto_apply_if_empty=not existing_tags
        )
        active_tags = d.get("tags") or auto_tags
        self._apply_tag_selection(active_tags)

        self._refresh_tag_events()

    def _refresh_tag_events(self):
        self.tag_events.clear()
        if not self.session.current_dataset:
            return
        for evt in self.session.current_dataset.get("tag_events", []):
            txt = (
                f"frame={evt.get('frame_index', '?')} "
                f"time={evt.get('time_ms', '?')}ms  {evt.get('issue')}"
            )
            self.tag_events.addItem(txt)

    def _has_pose_overlay(self) -> bool:
        dataset = self.session.current_dataset or {}
        frames = dataset.get("frames") or []
        for rec in frames:
            if rec and rec.get("pose_present") and rec.get("landmarks"):
                return True
        return False

    def _update_nav_buttons(self):
        total = len(self.session.video_paths)
        idx = self.session.current_index
        has_video = total > 0 and idx >= 0
        finish = self._auto_finish and has_video and idx >= total - 1
        self.prev_btn.setEnabled(has_video and idx > 0)
        self.save_btn.setEnabled(has_video)
        self.next_btn.setEnabled(has_video)
        self.next_btn.setText("Save & Finish" if finish else "Save + Next")

    # Playback --------------------------------------------------------------
    def _render_frame(self, index: int):
        if not self.session.has_video:
            return
        index = max(0, min(index, self.session.total_frames - 1))
        self.current_frame = index
        frame = self.session.get_frame(index)
        if frame is None:
            return
        frame = frame.copy()
        dataset = self.session.current_dataset
        frame_rec = None
        if dataset and 0 <= index < len(dataset.get("frames", [])):
            frame_rec = dataset["frames"][index]
            if frame_rec.get("pose_present") and frame_rec.get("landmarks"):
                draw_upper_body_overlay(
                    frame, frame_rec["landmarks"], self.current_body_parts
                )

        self._update_live_frame_metrics(frame_rec)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        scaled = qt_image.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(index)
        self.scrubber.blockSignals(False)

    def _advance_frame(self):
        if not (self.playing and self.session.has_video):
            return
        step = 1
        if self.playback_speed > 1.0:
            whole = int(self.playback_speed)
            frac = self.playback_speed - whole
            step = max(1, whole)
            self._speed_residual += frac
            if self._speed_residual >= 1.0:
                step += 1
                self._speed_residual -= 1.0
        next_index = self.current_frame + step
        if next_index >= self.session.total_frames:
            next_index = self.session.total_frames - 1
            self.playing = False
            self.play_btn.setText("Play")
        self._render_frame(next_index)

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._update_timer_interval()

    def _replay(self):
        self.playing = True
        self.play_btn.setText("Pause")
        self._update_timer_interval()
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if not self.session.has_video:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self._render_frame(self.current_frame + delta)

    def _step_seconds(self, seconds: float):
        fps = self.session.fps if self.session.fps > 0 else 30.0
        delta_frames = int(round(seconds * fps))
        if delta_frames == 0:
            delta_frames = 1 if seconds > 0 else -1
        self._step_frames(delta_frames)

    def _change_speed(self, text: str):
        try:
            self.playback_speed = max(0.1, float(text.replace("x", "")))
        except ValueError:
            self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._update_timer_interval()

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _scrubbed(self, value: int):
        if self.session.has_video:
            self._render_frame(value)

    def _update_timer_interval(self):
        fps = self.session.fps if self.session.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)

    # Tagging ---------------------------------------------------------
    def _add_tag_event(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        frames = dataset.get("frames") or []
        if not frames:
            return
        fi = max(0, min(self.current_frame, len(frames) - 1))
        tag = self.tag_picker.currentText()
        time_ms = frames[fi].get(
            "time_ms", int(round(fi * 1000.0 / (dataset.get("fps") or 30.0)))
        )
        dataset.setdefault("tag_events", []).append(
            {"issue": tag, "frame_index": int(fi), "time_ms": int(time_ms)}
        )
        self._refresh_tag_events()

    def _remove_tag_event(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        sel = self.tag_events.selectedIndexes()
        if not sel:
            return
        idx = sel[0].row()
        events = dataset.get("tag_events", [])
        if 0 <= idx < len(events):
            events.pop(idx)
            self._refresh_tag_events()

    # Saving ---------------------------------------------------------
    def _save_dataset(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        dataset["rep_id"] = self.rep_id.text()
        dataset["movement"] = self.movement_cb.currentText().strip()
        dataset["overall_quality"] = int(self.quality_cb.currentText())
        dataset["load_lbs"] = int(self.load_spin.value())
        dataset["rpe"] = float(self.rpe_cb.currentText())
        dataset["camera_angle"] = self.camera_cb.currentText()
        dataset["lens"] = self.lens_cb.currentText()
        selected = [item.text() for item in self.tag_list.selectedItems()]
        if not selected:
            selected = [DEFAULT_OK_TAG]
        dataset["tags"] = selected
        self.session.save_current_dataset()
        self._reload_current_dataset_from_disk()
        QtWidgets.QMessageBox.information(self, "Saved", "Dataset JSON saved.")

    def _ensure_pose_data(self, force: bool = False):
        if self._pose_job_active:
            return
        if not self.session.current_dataset or not self.session.has_video:
            return
        if not force and self._has_pose_overlay():
            return
        if not _HAS_MEDIAPIPE:
            QtWidgets.QMessageBox.warning(
                self,
                "Pose overlay unavailable",
                "mediapipe is not installed. Run scripts/setup_unified_env.sh first.",
            )
            return

        movement = (
            self.session.current_dataset.get("movement")
            or self.movement_cb.currentText()
        )
        settings = self._movement_settings_for(movement)
        try:
            model_path = _pose_model_path(settings.get("model", "full"))
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.critical(self, "Pose tracker", str(exc))
            return

        if not force:
            prompt = (
                "No pose overlay data was found for this video.\n"
                "Run the pose tracker now to enable overlays?"
            )
        else:
            prompt = (
                "Re-run the pose tracker for this video?\n"
                "Existing pose results will be replaced."
            )
        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        default = QtWidgets.QMessageBox.Yes
        resp = QtWidgets.QMessageBox.question(
            self, "Pose Tracker", prompt, buttons, default
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        total_frames = self.session.total_frames
        if total_frames == 0:
            QtWidgets.QMessageBox.warning(
                self, "Pose tracker", "Video frames are not loaded yet."
            )
            return

        progress = QtWidgets.QProgressDialog(
            "Running pose tracker...",
            "Cancel",
            0,
            total_frames,
            self,
        )
        progress.setWindowTitle("Pose Tracker")
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        def progress_cb(done: int, total: int) -> bool:
            progress.setMaximum(max(total, 1))
            progress.setValue(done)
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        pose_frames: Optional[List[Dict]] = None
        self._pose_job_active = True
        try:
            video_path = self.session.current_video_path
            if not video_path:
                raise RuntimeError("Current video path is unknown.")
            rotation_deg = self.session.current_rotation
            override = self.session.current_dataset.get("rotation_override_degrees")
            if override is not None:
                rotation_deg = override
            pose_frames = run_pose_landmarks_on_video(
                video_path,
                self.session.fps,
                settings,
                model_path,
                progress_cb,
                rotation=rotation_deg,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "canceled" in message.lower():
                QtWidgets.QMessageBox.information(
                    self, "Pose tracker", "Pose tracking was canceled."
                )
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Pose tracker failed", message or "Unknown error."
                )
        except Exception as exc:  # pragma: no cover - mediapipe runtime errors
            QtWidgets.QMessageBox.critical(self, "Pose tracker failed", str(exc))
        finally:
            progress.close()
            self._pose_job_active = False

        if not pose_frames:
            return

        dataset = self.session.current_dataset
        dataset["frames"] = pose_frames
        dataset["fps"] = self.session.fps
        dataset["pose_model_file"] = model_path.name
        dataset["min_pose_detection_confidence"] = float(settings.get("det", 0.5))
        dataset["min_pose_presence_confidence"] = float(settings.get("prs", 0.7))
        dataset["min_tracking_confidence"] = float(settings.get("trk", 0.7))
        dataset["ema_alpha"] = float(settings.get("ema", 0.0))
        dataset["output_segmentation_masks"] = bool(settings.get("seg", False))
        self.session.save_current_dataset()
        had_tags = bool(dataset.get("tags"))
        auto_tags = self._compute_metrics_for_current_dataset(
            auto_apply_if_empty=not had_tags
        )
        self._apply_tag_selection(dataset.get("tags") or auto_tags)
        self._render_frame(self.current_frame)
        if force:
            QtWidgets.QMessageBox.information(
                self, "Pose tracker", "Pose overlays refreshed."
            )


class VideoCutView(QtWidgets.QWidget):
    TARGET_HEIGHT = 720
    TARGET_FPS = 30

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        self.videos: List[Path] = []
        self.current_index = -1
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._last_capture_index = -1
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.play_timer.timeout.connect(self._advance_frame)
        self.play_timer.start(30)
        self.cuts: Dict[Path, List[tuple[int, int]]] = {}
        self.output_dir: Optional[Path] = None
        self.next_clip_start_ms: Dict[Path, float] = {}
        self.split_overlap_ms = 100  # milliseconds of overlap between clips
        self.rotation_overrides: Dict[Path, Optional[int]] = {}
        self.rotation_lock_value: Optional[int] = None
        self.current_rotation_override: Optional[int] = None
        self.current_rotation = 0

        self._build_ui()
        self._change_speed("1.0x")
        self._update_nav_buttons()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Video Cutter")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        top.addWidget(title)
        top.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top.addWidget(self.home_button)
        root.addLayout(top)

        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, stretch=1)

        # Left: video and controls
        left = QtWidgets.QVBoxLayout()
        content.addLayout(left, stretch=3)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #333;")
        left.addWidget(self.video_label, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        left.addLayout(controls)

        def ctrl_btn(text, slot):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        ctrl_btn("⟵ Frame", lambda: self._step_frames(-1))
        ctrl_btn("-0.5s", lambda: self._step_seconds(-0.5))
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        ctrl_btn("Replay", self._replay)
        ctrl_btn("+0.5s", lambda: self._step_seconds(0.5))
        ctrl_btn("Frame ⟶", lambda: self._step_frames(1))

        controls.addStretch(1)
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_box)

        mark_row = QtWidgets.QHBoxLayout()
        left.addLayout(mark_row)
        self.split_btn = QtWidgets.QPushButton("Mark Split")
        self.split_btn.setToolTip(
            "End current clip at the playhead and start the next one with 0.1s overlap."
        )
        self.split_btn.clicked.connect(self._mark_split)
        self.split_btn.setEnabled(False)
        mark_row.addWidget(self.split_btn)
        mark_row.addStretch(1)

        rotation_row = QtWidgets.QHBoxLayout()
        left.addLayout(rotation_row)
        rotation_row.addWidget(QtWidgets.QLabel("Rotation:"))
        self.rotation_combo = QtWidgets.QComboBox()
        for label, _ in ROTATION_OPTIONS:
            self.rotation_combo.addItem(label)
        self.rotation_combo.currentIndexChanged.connect(
            self._on_cutter_rotation_changed
        )
        self.rotation_combo.setToolTip(
            "Override the displayed/exported orientation for this clip."
        )
        rotation_row.addWidget(self.rotation_combo)
        self.rotation_lock_cb = QtWidgets.QCheckBox("Keep for all videos")
        self.rotation_lock_cb.setToolTip(
            "When checked, the selected rotation is applied automatically to every new video."
        )
        self.rotation_lock_cb.stateChanged.connect(
            self._on_cutter_rotation_lock_toggled
        )
        rotation_row.addWidget(self.rotation_lock_cb)
        rotation_row.addStretch(1)

        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        left.addWidget(self.scrubber)

        self.status_label = QtWidgets.QLabel("")
        left.addWidget(self.status_label)

        # Right side: cuts list and options
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        content.addWidget(right_scroll, stretch=2)
        right_widget = QtWidgets.QWidget()
        right_scroll.setWidget(right_widget)
        right = QtWidgets.QVBoxLayout(right_widget)

        clips_header = QtWidgets.QHBoxLayout()
        clips_header.addWidget(QtWidgets.QLabel("Marked clips"))
        clips_header.addStretch(1)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.setToolTip("Remove the highlighted clip from the list.")
        remove_btn.clicked.connect(self._remove_selected_cut)
        clear_btn = QtWidgets.QPushButton("Clear clips")
        clear_btn.setToolTip("Delete all clips for this video.")
        clear_btn.clicked.connect(self._clear_cuts)
        clips_header.addWidget(remove_btn)
        clips_header.addWidget(clear_btn)
        right.addLayout(clips_header)

        self.cut_list = QtWidgets.QListWidget()
        right.addWidget(self.cut_list, stretch=1)

        pad_row = QtWidgets.QHBoxLayout()
        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 2000)
        self.pad_spin.setValue(120)
        self.pad_spin.setSuffix(" ms pad")
        pad_row.addWidget(QtWidgets.QLabel("Padding:"))
        pad_row.addWidget(self.pad_spin)
        right.addLayout(pad_row)

        nav = QtWidgets.QHBoxLayout()
        self.prev_video_btn = QtWidgets.QPushButton("Previous Video")
        self.prev_video_btn.clicked.connect(lambda: self._load_relative(-1))
        nav.addWidget(self.prev_video_btn)
        nav.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save Clips")
        self.save_btn.clicked.connect(self._save_current_video)
        nav.addWidget(self.save_btn)
        self.save_next_btn = QtWidgets.QPushButton("Save & Next")
        self.save_next_btn.clicked.connect(self._save_and_advance)
        nav.addWidget(self.save_next_btn)
        right.addLayout(nav)

    def _release_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._last_capture_index = -1

    def _count_frames(self, path: Path) -> int:
        temp = cv2.VideoCapture(str(path))
        if not temp.isOpened():
            return 0
        count = 0
        while True:
            ok, _ = temp.read()
            if not ok:
                break
            count += 1
        temp.release()
        return count

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if not self.cap or self.frame_count <= 0:
            return None
        idx = max(0, min(idx, self.frame_count - 1))
        if idx != self._last_capture_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last_capture_index = idx
        rotation = (
            self.current_rotation_override
            if self.current_rotation_override is not None
            else self.current_rotation
        )
        if rotation:
            frame = _rotate_frame_if_needed(frame, rotation)
        return frame

    def _on_cutter_rotation_changed(self, state: int):
        path = self._current_video_path()
        if not path:
            return
        value = _rotation_value_from_index(self.rotation_combo.currentIndex())
        self.rotation_overrides[path] = value
        self.current_rotation_override = value
        if self.rotation_lock_cb.isChecked():
            self.rotation_lock_value = value
        self._render_frame(self.current_frame)

    def _on_cutter_rotation_lock_toggled(self, state: int):
        if state == QtCore.Qt.Checked:
            self.rotation_lock_value = self.current_rotation_override
            if (
                self.rotation_lock_value is not None
                and self._current_video_path() is not None
            ):
                self.rotation_overrides[self._current_video_path()] = (
                    self.rotation_lock_value
                )
                self.current_rotation_override = self.rotation_lock_value
                self._render_frame(self.current_frame)
        else:
            self.rotation_lock_value = None

    def start_new_session(
        self, videos: Optional[List[Path]] = None, output_dir: Optional[Path] = None
    ) -> bool:
        self.videos = [Path(v) for v in (videos or [])]
        self.output_dir = Path(output_dir) if output_dir else None
        self.current_index = -1
        self._release_capture()
        self.frame_count = 0
        self.rotation_overrides = {}
        self.cuts = {path: [] for path in self.videos}
        self.next_clip_start_ms = {}
        self.cut_list.clear()
        self.video_label.clear()
        self.status_label.setText("Select videos to begin.")
        self.split_btn.setEnabled(False)
        self.scrubber.setEnabled(False)
        self.scrubber.setRange(0, 0)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(0)
        self.rotation_combo.blockSignals(False)
        self.current_rotation_override = None
        self._update_nav_buttons()
        if not self._prompt_for_inputs():
            return False
        if not self.videos:
            return False
        self._load_video(0)
        return True

    def _prompt_for_inputs(self) -> bool:
        if not self.videos and not self._choose_videos():
            return False
        if not self.output_dir and not self._choose_output_folder():
            return False
        return True

    def _choose_videos(self) -> bool:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return False
        paths = [Path(f) for f in files]
        self.videos = paths
        self.cuts = {path: [] for path in self.videos}
        self.next_clip_start_ms = {}
        self.current_index = -1
        self._release_capture()
        self.frame_count = 0
        self.rotation_overrides = {}
        return True

    def _choose_output_folder(self) -> bool:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path.home())
        )
        if not folder:
            return False
        self.output_dir = Path(folder)
        self.status_label.setText(f"Output folder: {self.output_dir}")
        return True

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_cutting_inputs(self, videos: List[Path]):
        self.start_new_session(videos)

    def _load_relative(self, delta: int):
        if not self.videos:
            return
        if self.current_index < 0:
            return
        new_index = max(0, min(self.current_index + delta, len(self.videos) - 1))
        if new_index == self.current_index:
            return
        self._load_video(new_index)

    def _load_video(self, index: int):
        if not self.videos:
            return
        index = max(0, min(index, len(self.videos) - 1))
        if index == self.current_index:
            return
        path = self.videos[index]
        self.cuts.setdefault(path, [])
        self.next_clip_start_ms.setdefault(path, 0.0)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open {path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_frames(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", f"No frames in {path}")
            cap.release()
            return
        fps = fps if fps and fps > 0 else 30.0
        self._release_capture()
        self.cap = cap
        self.current_index = index
        self.fps = fps
        self.frame_count = frame_count
        self.current_frame = 0
        self.current_rotation = _video_rotation_degrees(path)
        override = self.rotation_overrides.get(path)
        if (
            override is None
            and self.rotation_lock_cb.isChecked()
            and self.rotation_lock_value is not None
        ):
            override = self.rotation_lock_value
            self.rotation_overrides[path] = override
        self.current_rotation_override = override
        target_idx = _rotation_option_index(override)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(target_idx)
        self.rotation_combo.blockSignals(False)
        self._speed_residual = 0.0
        self.scrubber.setRange(0, max(0, self.frame_count - 1))
        self.scrubber.setEnabled(True)
        self.split_btn.setEnabled(True)
        self.playing = False
        self.play_btn.setText("Play")
        self._update_timer_interval()
        self._render_frame(0)
        self._refresh_cut_list()
        status = f"Loaded {path.name} – {self.frame_count} frames @ {self.fps:.2f} fps"
        if self.output_dir:
            status += f" | Output: {self.output_dir}"
        self.status_label.setText(status)
        self._update_nav_buttons()

    def _render_frame(self, idx: int):
        if not self.cap or self.frame_count <= 0:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        frame = self._read_frame(idx)
        if frame is None:
            return
        # _read_frame already applies rotation override; nothing more to do
        self.current_frame = idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        scaled = image.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(idx)
        self.scrubber.blockSignals(False)

    def _mark_split(self):
        path = self._current_video_path()
        if not path or self.frame_count <= 0:
            return
        current_ms = int(round(self._current_time_ms()))
        start_ms = int(round(self.next_clip_start_ms.get(path, 0.0)))
        if current_ms - start_ms < 50:
            self.status_label.setText("Split point must be after the current clip start.")
            return
        clips = self.cuts.setdefault(path, [])
        clips.append((start_ms, current_ms))
        next_start = max(current_ms - self.split_overlap_ms, 0)
        self.next_clip_start_ms[path] = next_start
        self.status_label.setText(
            f"Split at {current_ms / 1000:.2f}s. Next clip starts at {next_start / 1000:.2f}s"
        )
        self._refresh_cut_list()

    def _refresh_cut_list(self):
        self.cut_list.clear()
        path = self._current_video_path()
        if not path:
            return
        clips = self.cuts.get(path, [])
        for idx, (start, end) in enumerate(clips, 1):
            self.cut_list.addItem(
                f"{idx}. {start / 1000:.2f}s -> {end / 1000:.2f}s (len {(end - start) / 1000:.2f}s)"
            )
        self._update_next_clip_start()

    def _current_video_path(self) -> Optional[Path]:
        if not self.videos or self.current_index < 0:
            return None
        return self.videos[self.current_index]

    def _update_next_clip_start(self):
        path = self._current_video_path()
        if not path:
            return
        clips = self.cuts.get(path) or []
        if clips:
            start = max(float(clips[-1][1]) - self.split_overlap_ms, 0.0)
        else:
            start = 0.0
        self.next_clip_start_ms[path] = start

    def _update_nav_buttons(self):
        has_video = bool(self.videos) and self.current_index >= 0
        can_go_prev = has_video and self.current_index > 0
        total = len(self.videos)
        is_last = has_video and self.current_index >= total - 1
        self.prev_video_btn.setEnabled(can_go_prev)
        self.save_btn.setEnabled(has_video)
        self.save_next_btn.setEnabled(has_video)
        self.save_next_btn.setText("Save & Exit" if is_last else "Save & Next")

    def _remove_selected_cut(self):
        path = self._current_video_path()
        if not path:
            return
        row = self.cut_list.currentRow()
        if row < 0:
            return
        clips = self.cuts.get(path, [])
        if row < len(clips):
            clips.pop(row)
            self._refresh_cut_list()

    def _clear_cuts(self):
        path = self._current_video_path()
        if not path:
            return
        self.cuts[path] = []
        self._refresh_cut_list()

    def _save_current_video(self) -> bool:
        video = self._current_video_path()
        if not video:
            return False
        clips = self.cuts.get(video) or []
        if not clips:
            QtWidgets.QMessageBox.information(
                self, "No clips", "Mark at least one split before saving."
            )
            return False
        if not self.output_dir and not self._choose_output_folder():
            return False
        assert self.output_dir is not None
        pad = self.pad_spin.value()
        success = self._export_video_clips(video, self.output_dir, pad)
        if success:
            self.fps = float(self.TARGET_FPS)
            self.status_label.setText(
                f"Saved {len(clips)} clip(s) for {video.stem} → {self.output_dir}"
            )
        return success

    def _save_and_advance(self):
        if not self._save_current_video():
            return
        if self.current_index >= len(self.videos) - 1:
            self._go_home()
        else:
            self._load_relative(+1)

    def _export_video_clips(self, video: Path, out_dir: Path, pad_ms: int) -> bool:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = self.cuts.get(video, [])
        if not clips:
            return False
        rotation = self._effective_rotation_for_video(video)
        errors = False
        for idx, (start, end) in enumerate(clips, 1):
            s = max(0, start - pad_ms)
            e = end + pad_ms
            stem = video.stem
            out_path = out_dir / f"{stem}_clip{idx:02d}.mp4"
            if not self._run_ffmpeg(video, out_path, s, e, rotation):
                errors = True
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Export errors",
                f"One or more clips failed to export for {video.name}.",
            )
            return False
        QtWidgets.QMessageBox.information(
            self, "Clips saved", f"Exported {len(clips)} clip(s) for {video.name}."
        )
        return True

    def _effective_rotation_for_video(self, video: Path) -> int:
        override = self.rotation_overrides.get(video)
        if override is not None:
            return int(override or 0)
        if video == self._current_video_path():
            return int(self.current_rotation or 0)
        return _video_rotation_degrees(video)

    @staticmethod
    def _rotation_filter_chain(rotation: int) -> List[str]:
        chain: List[str] = []
        if rotation == 90:
            chain.append("transpose=1")
        elif rotation == 180:
            chain.append("transpose=1")
            chain.append("transpose=1")
        elif rotation == 270:
            chain.append("transpose=2")
        return chain

    def _run_ffmpeg(
        self, src: Path, dst: Path, start_ms: int, end_ms: int, rotation: int = 0
    ) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotation = rotation or 0
        vf_parts = self._rotation_filter_chain(rotation)
        vf_parts.append(f"scale=-2:{self.TARGET_HEIGHT}")
        vf = ",".join(vf_parts)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-to",
                f"{end_ms / 1000:.3f}",
                "-i",
                str(src),
                "-vf",
                vf,
                "-r",
                str(self.TARGET_FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(dst),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(
                self,
                "FFmpeg not found",
                "FFmpeg is not installed or not available on PATH. "
                "Install ffmpeg and try exporting again.",
            )
            return False

        if result.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "FFmpeg error",
                f"FFmpeg failed to export clip from {src.name}. "
                "Verify the source video and FFmpeg installation.",
            )
            return False
        return True

    def _current_time_ms(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return (self.current_frame / (self.fps if self.fps > 0 else 30.0)) * 1000.0

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frame(self):
        if not (self.playing and self.frame_count > 0 and self.cap):
            return
        step = 1
        if self.playback_speed > 1.0:
            whole = int(self.playback_speed)
            frac = self.playback_speed - whole
            step = max(1, whole)
            self._speed_residual += frac
            if self._speed_residual >= 1.0:
                step += 1
                self._speed_residual -= 1.0
        next_idx = self.current_frame + step
        if next_idx >= self.frame_count:
            next_idx = self.frame_count - 1
            self.playing = False
            self.play_btn.setText("Play")
        self._render_frame(next_idx)

    def _replay(self):
        if self.frame_count <= 0:
            return
        self.playing = True
        self.play_btn.setText("Pause")
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if self.frame_count <= 0:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self._render_frame(self.current_frame + delta)

    def _step_seconds(self, seconds: float):
        fps = self.fps if self.fps > 0 else 30.0
        delta = int(round(seconds * fps))
        if delta == 0:
            delta = 1 if seconds > 0 else -1
        self._step_frames(delta)

    def _change_speed(self, text: str):
        try:
            self.playback_speed = max(0.1, float(text.replace("x", "")))
        except ValueError:
            self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._update_timer_interval()

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _scrubbed(self, value: int):
        if self.frame_count > 0:
            self._render_frame(value)

    def __del__(self):
        self._release_capture()

    def _update_timer_interval(self):
        fps = self.fps if self.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)


class PoseTunerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.tags, self.movement_settings = load_label_options()
        self.dataset_dir: Optional[Path] = None
        self.video_entries: List[Dict] = []
        self.video_slots: List[Dict] = []
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frames)
        self.play_timer.start(30)
        self.playing = False
        self.playback_speed = 1.0
        self._home_cb: Optional[Callable[[], None]] = None
        self._pose_job_active = False
        self.max_slots = 1

        self.pose_dirty = False
        self._loading_pose_settings = False
        self._pose_executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
        )
        self._pose_future: Optional[concurrent.futures.Future] = None
        self._build_ui()

    def _release_entry_caps(self, clear_labels: bool = True):
        if self._pose_future and not self._pose_future.done():
            self._pose_future.cancel()
        self._pose_future = None
        self._pose_job_active = False
        for entry in self.video_entries:
            cap = entry.get("cap")
            if cap:
                cap.release()
        self.video_entries.clear()
        self.pose_dirty = False
        if clear_labels:
            for slot in self.video_slots:
                label = slot.get("label")
                if isinstance(label, QtWidgets.QLabel):
                    label.clear()
                combo = slot.get("combo")
                if combo:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
                    combo.setEnabled(False)
                button = slot.get("button")
                if button:
                    button.setEnabled(False)
                slot["entry_index"] = None

    @staticmethod
    def _count_video_frames(path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        return count

    def _read_entry_frame(self, entry: Dict, idx: int) -> Optional[np.ndarray]:
        cap = entry.get("cap")
        frame_count = entry.get("frame_count", 0)
        if not cap or frame_count <= 0:
            return None
        idx = max(0, min(idx, frame_count - 1))
        last_idx = entry.get("_last_index", -1)
        if idx != last_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        entry["_last_index"] = idx
        rotation = entry.get("rotation", 0)
        if rotation:
            frame = _rotate_frame_if_needed(frame, rotation)
        return frame

    def _on_slot_rotation_changed(self, slot: Dict):
        entry_idx = slot.get("entry_index")
        if entry_idx is None or not (0 <= entry_idx < len(self.video_entries)):
            return
        entry = self.video_entries[entry_idx]
        value = _rotation_value_from_index(slot["combo"].currentIndex())
        entry["rotation_override"] = value
        entry["rotation"] = (
            value if value is not None else entry.get("detected_rotation", 0)
        )
        entry["_last_index"] = -1
        self._save_entry_rotation_override(entry)
        self._render_entry(entry)
        self._schedule_pose_rerun()

    def _save_entry_rotation_override(self, entry: Dict):
        json_path: Path = entry.get("json_path")
        if not json_path:
            return
        data = {}
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                data = {}
        data["rotation_override_degrees"] = entry.get("rotation_override")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.load_videos_btn = QtWidgets.QPushButton("Load Videos")
        self.load_videos_btn.clicked.connect(self._pick_videos)
        self.dataset_btn = QtWidgets.QPushButton("Choose Dataset Folder")
        self.dataset_btn.clicked.connect(self._pick_dataset_dir)
        top.addWidget(self.load_videos_btn)
        top.addWidget(self.dataset_btn)
        top.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top.addWidget(self.home_button)
        root.addLayout(top)

        layout = QtWidgets.QHBoxLayout()
        root.addLayout(layout, stretch=1)

        # Left: controls
        control_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(control_panel, stretch=1)

        form = QtWidgets.QFormLayout()
        control_panel.addLayout(form)

        def add_setting_row(
            label_text: str, widget: QtWidgets.QWidget, description: str
        ):
            form.addRow(label_text, widget)
            hint = QtWidgets.QLabel(description)
            hint.setStyleSheet("color: #888; font-size: 11px;")
            hint.setWordWrap(True)
            form.addRow("", hint)

        self.movement_cb = QtWidgets.QComboBox()
        self.movement_cb.setEditable(True)
        self.movement_cb.addItems(self.movements)
        self.movement_cb.currentTextChanged.connect(self._load_settings_for_movement)
        add_setting_row(
            "Movement",
            self.movement_cb,
            "Pick a movement preset to load or edit settings.",
        )
        self.model_cb = QtWidgets.QComboBox()
        self.model_cb.addItems(MODEL_VARIANTS)
        self.model_cb.currentTextChanged.connect(
            lambda _: self._on_model_variant_changed()
        )
        add_setting_row(
            "Model",
            self.model_cb,
            "Choose which MediaPipe pose model variant to run.",
        )

        self.det_spin = QtWidgets.QDoubleSpinBox()
        self.det_spin.setRange(0.1, 1.0)
        self.det_spin.setSingleStep(0.05)
        self.det_spin.setToolTip(
            "Detection confidence threshold (higher = fewer detections)."
        )
        self.det_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.prs_spin = QtWidgets.QDoubleSpinBox()
        self.prs_spin.setRange(0.1, 1.0)
        self.prs_spin.setSingleStep(0.05)
        self.prs_spin.setValue(0.7)
        self.prs_spin.setToolTip(
            "Pose score threshold used when filtering landmark quality."
        )
        self.prs_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.trk_spin = QtWidgets.QDoubleSpinBox()
        self.trk_spin.setRange(0.1, 1.0)
        self.trk_spin.setSingleStep(0.05)
        self.trk_spin.setValue(0.7)
        self.trk_spin.setToolTip(
            "Tracker confidence threshold before dropping a track."
        )
        self.trk_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.ema_spin = QtWidgets.QDoubleSpinBox()
        self.ema_spin.setRange(0.0, 1.0)
        self.ema_spin.setSingleStep(0.05)
        self.ema_spin.setValue(0.25)
        self.ema_spin.setToolTip(
            "Exponential moving average amount for smoothing landmarks."
        )
        self.ema_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setToolTip("Overlay segmentation masks when available.")
        self.seg_check.stateChanged.connect(self._schedule_pose_rerun)
        add_setting_row(
            "det",
            self.det_spin,
            "Detection confidence threshold (higher removes weaker detections).",
        )
        add_setting_row(
            "prs",
            self.prs_spin,
            "Pose score threshold applied when keeping landmark results.",
        )
        add_setting_row(
            "trk",
            self.trk_spin,
            "Tracking score threshold before the subject is re-detected.",
        )
        add_setting_row(
            "ema",
            self.ema_spin,
            "Smoothing factor for the exponential moving average applied to pose data.",
        )
        form.addRow(self.seg_check)
        seg_hint = QtWidgets.QLabel(
            "Overlay segmentation masks for the model (slightly slower)."
        )
        seg_hint.setStyleSheet("color: #888; font-size: 11px;")
        seg_hint.setWordWrap(True)
        form.addRow("", seg_hint)

        body_group = QtWidgets.QGroupBox("Body parts to display")
        control_panel.addWidget(body_group)
        body_layout = QtWidgets.QGridLayout(body_group)
        self.body_part_checks = []
        for i, part in enumerate(BODY_PART_OPTIONS):
            chk = QtWidgets.QCheckBox(part)
            chk.setChecked(True)
            chk.stateChanged.connect(self._refresh_all_frames)
            self.body_part_checks.append(chk)
            row, col = divmod(i, 2)
            body_layout.addWidget(chk, row, col)
        body_hint = QtWidgets.QLabel(
            "Select which body segments should remain highlighted on the overlays."
        )
        body_hint.setStyleSheet("color: #888; font-size: 11px;")
        body_hint.setWordWrap(True)
        control_panel.addWidget(body_hint)

        save_row = QtWidgets.QHBoxLayout()
        control_panel.addLayout(save_row)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        save_as_btn = QtWidgets.QPushButton("Save As")
        save_as_btn.clicked.connect(self._save_as)
        save_row.addWidget(save_btn)
        save_row.addWidget(save_as_btn)

        control_panel.addStretch(1)

        # Right: video grid and controls
        video_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(video_panel, stretch=2)

        self.video_grid = QtWidgets.QGridLayout()
        video_panel.addLayout(self.video_grid)
        self.video_slots = []
        for idx in range(self.max_slots):
            slot_widget = QtWidgets.QWidget()
            slot_layout = QtWidgets.QVBoxLayout(slot_widget)
            slot_layout.setContentsMargins(0, 0, 0, 0)
            frame_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            frame_label.setStyleSheet("background: #111; border: 1px solid #333;")
            frame_label.setMinimumSize(420, 260)
            slot_layout.addWidget(frame_label)
            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Rotation:"))
            combo = QtWidgets.QComboBox()
            for label_text, _ in ROTATION_OPTIONS:
                combo.addItem(label_text)
            combo.setEnabled(False)
            controls.addWidget(combo, stretch=1)
            rerun_btn = QtWidgets.QPushButton("Re-run Pose")
            rerun_btn.setEnabled(False)
            controls.addWidget(rerun_btn)
            slot_layout.addLayout(controls)
            row, col = divmod(idx, 2)
            self.video_grid.addWidget(slot_widget, row, col)
            slot = {
                "widget": slot_widget,
                "label": frame_label,
                "combo": combo,
                "button": rerun_btn,
                "entry_index": None,
            }
            combo.currentIndexChanged.connect(
                lambda _, s=slot: self._on_slot_rotation_changed(s)
            )
            rerun_btn.clicked.connect(
                lambda _, s=slot: self._rerun_pose_for_slot(s)
            )
            self.video_slots.append(slot)

        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(self.speed_box)
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.sliderMoved.connect(self._scrub_to)
        controls.addWidget(self.scrubber)
        video_panel.addLayout(controls)

        self.status_label = QtWidgets.QLabel("")
        video_panel.addWidget(self.status_label)
        self.global_rerun_btn = QtWidgets.QPushButton("Re-run Pose Tracker")
        self.global_rerun_btn.clicked.connect(self._rerun_pose_for_loaded_entries)
        video_panel.addWidget(self.global_rerun_btn)
        self.refresh_movements()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_pose_inputs(self, videos: List[Path], dataset_dir: Optional[Path]):
        if dataset_dir:
            self.dataset_dir = dataset_dir
            self.status_label.setText(f"Dataset dir: {self.dataset_dir}")
        elif not self.dataset_dir:
            self.status_label.setText("Dataset dir: not set (pose data kept in memory)")
        if videos:
            self._load_videos(videos[: self.max_slots])
        else:
            self.refresh_movements()

    def _pick_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select a video",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        self._load_videos([Path(files[0])])

    def _pick_dataset_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if folder:
            self.dataset_dir = Path(folder)
            self.status_label.setText(f"Dataset dir: {self.dataset_dir}")
            self.refresh_movements()

    def refresh_movements(self):
        self.movements, self.tags, self.movement_settings = load_label_options()
        current = self.movement_cb.currentText()
        self.movement_cb.blockSignals(True)
        self.movement_cb.clear()
        self.movement_cb.addItems(self.movements)
        if current:
            idx = self.movement_cb.findText(current)
            if idx >= 0:
                self.movement_cb.setCurrentIndex(idx)
            else:
                self.movement_cb.setEditText(current)
        self.movement_cb.blockSignals(False)
        self._load_settings_for_movement(self.movement_cb.currentText())

    def _load_videos(self, videos: List[Path]):
        if not videos:
            return
        self._release_entry_caps()
        for idx, path in enumerate(videos[: self.max_slots]):
            entry = self._load_single_video(path)
            if entry:
                slot = self.video_slots[idx]
                entry["slot"] = slot
                self.video_entries.append(entry)
                slot["entry_index"] = len(self.video_entries) - 1
                slot["label"].clear()
                slot["combo"].blockSignals(True)
                slot["combo"].setCurrentIndex(
                    _rotation_option_index(entry.get("rotation_override"))
                )
                slot["combo"].blockSignals(False)
                slot["combo"].setEnabled(True)
                slot["button"].setEnabled(True)
        if self.video_entries:
            self.scrubber.setEnabled(True)
            self.scrubber.setValue(0)
            self.playing = False
            self.play_btn.setText("Play")
            self._refresh_all_frames()
            if any(not entry.get("pose") for entry in self.video_entries):
                self._schedule_pose_rerun(
                    message="Pose data missing – click Re-run Pose Tracker."
                )

    def _load_single_video(self, path: Path) -> Optional[Dict]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Video error", f"Failed to open {path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_video_frames(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            cap.release()
            return None
        json_path = None
        pose_frames = None
        rotation_override = None
        if self.dataset_dir:
            json_path = self.dataset_dir / f"{path.stem}.json"
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text())
                    pose_frames = data.get("frames", [])
                    rotation_override = data.get("rotation_override_degrees")
                except Exception:
                    pose_frames = None
                    rotation_override = None
        detected_rotation = _video_rotation_degrees(path)
        rotation = rotation_override if rotation_override is not None else detected_rotation
        return {
            "path": path,
            "cap": cap,
            "frame_count": frame_count,
            "pose": pose_frames,
            "fps": fps,
            "current_frame": 0,
            "next_time": time.perf_counter(),
            "_last_index": -1,
            "rotation": rotation,
            "detected_rotation": detected_rotation,
            "rotation_override": rotation_override,
            "json_path": json_path,
        }

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frames(self):
        if not (self.playing and self.video_entries):
            return
        for entry in self.video_entries:
            fps = entry["fps"]
            interval = 1.0 / (fps * max(self.playback_speed, 0.1))
            now = time.perf_counter()
            if now >= entry.get("next_time", 0):
                frame_count = entry.get("frame_count", 0)
                if frame_count <= 0:
                    continue
                entry["current_frame"] = min(
                    frame_count - 1, entry["current_frame"] + 1
                )
                entry["next_time"] = now + interval
                self._render_entry(entry)

    def _render_entry(self, entry: Dict):
        frame_count = entry.get("frame_count", 0)
        if frame_count <= 0:
            return
        idx = max(0, min(entry["current_frame"], frame_count - 1))
        frame = self._read_entry_frame(entry, idx)
        if frame is None:
            return
        frame = frame.copy()
        pose_frames = entry.get("pose")
        allowed = self._selected_body_parts()
        if pose_frames and idx < len(pose_frames):
            frec = pose_frames[idx]
            if frec and frec.get("pose_present") and frec.get("landmarks"):
                draw_upper_body_overlay(frame, frec["landmarks"], allowed)
        else:
            if allowed and "full_body" not in allowed:
                # Dim the frame if a subset of body parts was requested but no pose
                # landmarks exist for this frame. This gives users a visual hint that
                # the overlay cannot be drawn for the selected regions yet.
                frame[:] = frame * 0.8
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        slot = entry.get("slot")
        if not slot:
            return
        label: QtWidgets.QLabel = slot["label"]
        scaled = image.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(QtGui.QPixmap.fromImage(scaled))

    def _refresh_all_frames(self):
        for entry in self.video_entries:
            self._render_entry(entry)

    def _selected_body_parts(self) -> List[str]:
        selected = [chk.text() for chk in self.body_part_checks if chk.isChecked()]
        return selected or BODY_PART_OPTIONS.copy()

    def _on_model_variant_changed(self):
        self._schedule_pose_rerun(message="Pose model changed – click Re-run Pose Tracker.")

    def _change_speed(self, text: str):
        try:
            self.playback_speed = float(text.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0

    def _scrub_to(self, value: int):
        if not self.video_entries:
            return
        ratio = value / 1000.0
        for entry in self.video_entries:
            frame_count = entry.get("frame_count", 0)
            if frame_count > 0:
                entry["current_frame"] = int(ratio * (frame_count - 1))
                entry["next_time"] = time.perf_counter()
                self._render_entry(entry)

    def _current_pose_settings(self) -> Dict:
        return {
            "model": self.model_cb.currentText() or "full",
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
        }

    def _schedule_pose_rerun(self, delay_ms: int = 200, message: Optional[str] = None):
        if (
            not self.video_entries
            or not self.dataset_dir
            or self._loading_pose_settings
        ):
            return
        overlay_cleared = False
        for entry in self.video_entries:
            if entry.get("pose"):
                entry["pose"] = None
                entry["_last_index"] = -1
                overlay_cleared = True
        if overlay_cleared:
            self._refresh_all_frames()
            gc.collect()
        already_dirty = self.pose_dirty
        self.pose_dirty = True
        if not already_dirty or message:
            self.status_label.setText(
                message or "Pose settings changed – click Re-run Pose Tracker."
            )


    def _rerun_pose_for_slot(self, slot: Dict):
        entry_idx = slot.get("entry_index")
        if entry_idx is None or not (0 <= entry_idx < len(self.video_entries)):
            return
        self._rerun_pose_for_entries([self.video_entries[entry_idx]])

    def _rerun_pose_for_loaded_entries(self, auto: bool = False):
        if not self.video_entries:
            return
        self._rerun_pose_for_entries(self.video_entries, auto=auto)

    def _rerun_pose_for_entries(self, entries: List[Dict], auto: bool = False):
        if not entries or self._pose_job_active:
            return
        settings = self._current_pose_settings()
        try:
            model_name = settings.get("model", "full")
            model_path = _pose_model_path(model_name)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.critical(self, "Pose tracker", str(exc))
            return
        entry = entries[0]
        if entry not in self.video_entries:
            return
        entry_idx = self.video_entries.index(entry)
        self.global_rerun_btn.setEnabled(False)
        self._pose_job_active = True
        rotation = entry.get("rotation_override")
        if rotation is None:
            rotation = entry.get("detected_rotation", 0)
        self.status_label.setText(
            f"Running pose tracker for {entry['path'].name}..."
        )

        if not self._pose_executor:
            self._pose_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        future = self._pose_executor.submit(
            PoseTunerView._pose_worker_task,
            entry["path"],
            entry["fps"],
            settings,
            model_path,
            rotation,
        )
        self._pose_future = future

        def _schedule_finished(fut, idx=entry_idx):
            QtCore.QTimer.singleShot(
                0, lambda: self._on_pose_worker_finished(idx, fut)
            )

        future.add_done_callback(_schedule_finished)

    def _on_pose_worker_finished(self, entry_idx: int, future):
        if self._pose_future is not future:
            return
        self._pose_future = None
        self._pose_job_active = False
        self.global_rerun_btn.setEnabled(True)
        try:
            success, pose_frames, error = future.result()
        except Exception as exc:
            success = False
            pose_frames = None
            error = str(exc)
        if not (0 <= entry_idx < len(self.video_entries)):
            return
        entry = self.video_entries[entry_idx]
        if not success or not pose_frames:
            self.pose_dirty = True
            self.status_label.setText("Pose refresh failed")
            if error:
                QtWidgets.QMessageBox.critical(
                    self, "Pose tracker failed", f"{entry['path'].name}: {error}"
                )
            return
        entry["pose"] = pose_frames
        self.pose_dirty = False
        self.status_label.setText("Pose overlays refreshed.")
        self._save_entry_dataset(entry, pose_frames)
        self._render_entry(entry)

    @staticmethod
    def _pose_worker_task(
        video_path: Path,
        fps: float,
        settings: Dict,
        model_path: Path,
        rotation: int,
    ):
        try:
            pose_frames = run_pose_landmarks_on_video(
                video_path,
                fps,
                settings,
                model_path,
                rotation=rotation,
            )
            return True, pose_frames, ""
        except Exception as exc:
            return False, None, str(exc)

    def _save_entry_dataset(self, entry: Dict, pose_frames: List[Dict]):
        json_path: Path = entry.get("json_path")
        if not json_path:
            return
        data = {}
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                data = {}
        data["frames"] = pose_frames
        data["fps"] = entry.get("fps", 30.0)
        data["rotation_override_degrees"] = entry.get("rotation_override")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2))

    def _save_settings(self):
        movement = self.movement_cb.currentText().strip()
        if not movement:
            QtWidgets.QMessageBox.warning(self, "Movement", "Enter a movement name.")
            return
        self._persist_settings(movement)
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Settings saved for {movement}."
        )

    def _save_as(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Save As", "New movement name")
        if ok and text.strip():
            movement = text.strip()
            if movement not in self.movements:
                self.movements.append(movement)
                self.movement_settings[movement] = default_movement_settings(movement)
                self.movement_cb.addItem(movement)
            self.movement_cb.setCurrentText(movement)
            self._persist_settings(movement)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Settings saved for {movement}."
            )

    def _persist_settings(self, movement: str):
        settings = {
            "model": self.model_cb.currentText() or "full",
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
            "body_parts": self._selected_body_parts(),
        }
        self.movement_settings[movement] = settings
        save_label_config(
            {
                "movements": self.movements,
                "tags": self.tags,
                "movement_settings": self.movement_settings,
            }
        )

    def _load_settings_for_movement(self, name: str):
        settings = self.movement_settings.get(name)
        if not settings:
            settings = default_movement_settings(name)
        self._loading_pose_settings = True
        try:
            self.det_spin.blockSignals(True)
            self.prs_spin.blockSignals(True)
            self.trk_spin.blockSignals(True)
            self.ema_spin.blockSignals(True)
            self.model_cb.blockSignals(True)
            self.seg_check.blockSignals(True)

            self.det_spin.setValue(settings.get("det", 0.5))
            self.prs_spin.setValue(settings.get("prs", 0.7))
            self.trk_spin.setValue(settings.get("trk", 0.7))
            self.ema_spin.setValue(settings.get("ema", 0.25))
            model_value = settings.get("model", "full")
            idx = self.model_cb.findText(model_value)
            if idx < 0:
                idx = 0
            self.model_cb.setCurrentIndex(idx)
            self.seg_check.setChecked(settings.get("seg", False))
        finally:
            self.det_spin.blockSignals(False)
            self.prs_spin.blockSignals(False)
            self.trk_spin.blockSignals(False)
            self.ema_spin.blockSignals(False)
            self.model_cb.blockSignals(False)
            self.seg_check.blockSignals(False)
            self._loading_pose_settings = False
        parts = set(settings.get("body_parts") or BODY_PART_OPTIONS)
        for chk in self.body_part_checks:
            chk.setChecked(chk.text() in parts)
        self._refresh_all_frames()

    def __del__(self):
        try:
            if self._pose_executor:
                self._pose_executor.shutdown(wait=False)
        except Exception:
            pass
        self._release_entry_caps(clear_labels=False)


# --- Main window ------------------------------------------------------------


class UnifiedToolWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mAI Coach Tools")
        self.resize(1600, 900)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.admin_page = AdminPanel()
        self.labeler_page = LabelerView()
        self.cutting_page = VideoCutView()
        self.pose_page = PoseTunerView()
        self.admin_page.set_home_callback(self.show_home)
        self.labeler_page.set_home_callback(self.show_home)
        self.cutting_page.set_home_callback(self.show_home)
        self.pose_page.set_home_callback(self.show_home)

        for page in [
            self.home_page,
            self.admin_page,
            self.labeler_page,
            self.cutting_page,
            self.pose_page,
        ]:
            self.stack.addWidget(page)

        self.home_page.requested_admin.connect(lambda: self.show_page(self.admin_page))
        self.home_page.requested_labeling.connect(self.start_labeling_workflow)
        self.home_page.requested_cutting.connect(self.start_cutting_workflow)
        self.home_page.requested_pose.connect(self.start_pose_workflow)

        self.admin_page.config_saved.connect(self.labeler_page.refresh_label_options)
        self.admin_page.config_saved.connect(self._reload_pose_settings)

        self.show_home()

    def show_home(self):
        self.stack.setCurrentWidget(self.home_page)

    def show_page(self, widget: QtWidgets.QWidget):
        self.stack.setCurrentWidget(widget)

    def _select_videos(self) -> List[Path]:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        return [Path(f) for f in files] if files else []

    def start_labeling_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        dataset_dir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if not dataset_dir_str:
            return
        dataset_dir = Path(dataset_dir_str)
        self.labeler_page.load_labeler_inputs(videos, dataset_dir)
        self.show_page(self.labeler_page)

    def start_cutting_workflow(self):
        self.show_page(self.cutting_page)
        if not self.cutting_page.start_new_session():
            self.show_home()

    def start_pose_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        self.pose_page.load_pose_inputs(videos, None)
        self.show_page(self.pose_page)

    def _reload_pose_settings(self):
        self.pose_page.refresh_movements()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = UnifiedToolWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
