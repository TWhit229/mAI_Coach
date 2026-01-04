"""Video processing and MediaPipe helpers."""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    _HAS_MEDIAPIPE = False

from core.utils import lowpass_ema
from core.config import DEV_ROOT

POSE_MODEL_PATHS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}




def _pose_model_path(variant: str = "full") -> Path:
    """Return absolute path to a MediaPipe pose model."""
    filename = POSE_MODEL_PATHS.get(variant, "pose_landmarker_full.task")
    
    # Check PyInstaller bundle location first
    if getattr(sys, 'frozen', False):
        # Running as frozen app - data is in _MEIPASS
        bundle_dir = Path(sys._MEIPASS)  # type: ignore
        bundled_path = bundle_dir / "models" / filename
        if bundled_path.exists():
            return bundled_path
    
    # Check DEV_ROOT/models
    models_dir = DEV_ROOT / "models"
    path = models_dir / filename
    if path.exists():
        return path
    
    # Fallback to current working dir
    alt = Path(filename)
    if alt.exists():
        return alt
    return path


def video_rotation_degrees(path: Path) -> int:
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


def rotate_frame_if_needed(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def preprocess_video_for_pose(
    video_path: Path, target_height: int = 720, target_fps: float = 15.0
) -> Tuple[Path, float]:
    """Downscale/cap FPS to speed up pose tracking."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else target_fps
    out_fps = target_fps if target_fps and target_fps > 0 else fps

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"No frames found in {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h, w = frame.shape[:2]
    scale = target_height / float(h) if h else 1.0
    new_h = max(1, int(round(h * scale)))
    new_w = max(2, int(round(w * scale)))
    # Width must be even for many codecs.
    if new_w % 2:
        new_w += 1

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="pose_pre_")
    os.close(fd)
    tmp = Path(tmp_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, out_fps, (new_w, new_h))
    if not writer.isOpened():
        cap.release()
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Failed to create temporary writer for preprocessing.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Preprocess output is empty.")
    return tmp, out_fps


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
                frame_bgr = rotate_frame_if_needed(frame_bgr, rotation)
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


def detect_best_rotation(
    video_path: Path,
    settings: Dict,
    model_path: Path,
    sample_count: int = 5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Sample frames at different rotations and return the one with best pose detection.
    
    Tests 0°, 90°, 180°, 270° rotations and returns the rotation that produces
    the most confident pose detections.
    """
    if not _HAS_MEDIAPIPE:
        return 0
    
    video_path = Path(video_path)
    model_path = Path(model_path)
    if not model_path.exists():
        return 0
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames < sample_count:
        sample_count = max(1, total_frames)
    
    # Sample frames evenly distributed through the video
    sample_indices = [int(i * total_frames / (sample_count + 1)) for i in range(1, sample_count + 1)]
    
    sample_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            sample_frames.append(frame)
    cap.release()
    
    if not sample_frames:
        return 0
    
    rotations = [0, 90, 180, 270]
    rotation_scores: Dict[int, float] = {}
    
    for rotation in rotations:
        if progress_cb:
            progress_cb(f"Testing rotation {rotation}°...")
        
        # Create landmarker for this rotation test
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=float(settings.get("det", 0.5)),
            min_pose_presence_confidence=float(settings.get("prs", 0.5)),
            min_tracking_confidence=float(settings.get("trk", 0.5)),
        )
        landmarker = PoseLandmarker.create_from_options(options)
        
        total_score = 0.0
        detections = 0
        orientation_valid_count = 0
        
        try:
            for frame in sample_frames:
                rotated = rotate_frame_if_needed(frame, rotation)
                mp_image = _mp_image_from_bgr(rotated)
                result = landmarker.detect(mp_image)
                
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    
                    # Check body orientation - nose (0) should be above hips (23, 24)
                    # In normalized coords, y increases downward, so nose.y < hip.y is correct
                    nose_y = landmarks[0].y if len(landmarks) > 0 else 0.5
                    left_hip_y = landmarks[23].y if len(landmarks) > 23 else 0.5
                    right_hip_y = landmarks[24].y if len(landmarks) > 24 else 0.5
                    avg_hip_y = (left_hip_y + right_hip_y) / 2
                    
                    # For portrait video, check if nose is above hips (correct orientation)
                    # Allow some tolerance for lying down positions
                    orientation_valid = nose_y < avg_hip_y + 0.15
                    if orientation_valid:
                        orientation_valid_count += 1
                    
                    # Calculate average confidence from key body landmarks
                    # Use shoulders (11, 12), hips (23, 24), and wrists (15, 16)
                    key_indices = [11, 12, 15, 16, 23, 24]
                    confidences = []
                    for i in key_indices:
                        if i < len(landmarks):
                            lm = landmarks[i]
                            conf = getattr(lm, 'presence', getattr(lm, 'visibility', 0.5))
                            confidences.append(conf)
                    if confidences:
                        total_score += sum(confidences) / len(confidences)
                        detections += 1
        finally:
            landmarker.close()
        
        # Score is average confidence, weighted by detection rate AND orientation validity
        if detections > 0:
            avg_conf = total_score / detections
            detection_rate = detections / len(sample_frames)
            orientation_rate = orientation_valid_count / detections
            # Orientation is critical - if most frames have wrong orientation, heavily penalize
            rotation_scores[rotation] = avg_conf * detection_rate * (0.2 + 0.8 * orientation_rate)
        else:
            rotation_scores[rotation] = 0.0
    
    # Return rotation with highest score
    if not rotation_scores:
        return 0
    
    best_rotation = max(rotation_scores, key=rotation_scores.get)  # type: ignore
    
    # Only use non-zero rotation if it's significantly better
    if best_rotation != 0 and rotation_scores[0] > 0:
        improvement = rotation_scores[best_rotation] / rotation_scores[0]
        if improvement < 1.2:  # Less than 20% improvement, stick with 0
            return 0
    
    return best_rotation

