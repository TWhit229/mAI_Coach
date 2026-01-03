"""Shared logic for pose metrics and feature extraction."""

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# --- Constants --------------------------------------------------------------

L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16

# Tracking quality config
TRACKING_KEYPOINT_IDS = [
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
]
TRACKING_VISIBILITY_THRESHOLD = 0.5
TRACKING_VISIBLE_FRACTION = 1.0  # require all keypoints to be visible
TRACKING_BAD_RATIO_MAX = 0.5

# --- Geometry Helpers -------------------------------------------------------


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _dist3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _landmark_xy(landmarks: List[Dict], idx: int) -> Optional[Tuple[float, float]]:
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError, KeyError):
        return None
    if not isinstance(lm, dict):
        return None
    x = lm.get("x")
    y = lm.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _landmark_xyz(
    landmarks: List[Dict], idx: int
) -> Optional[Tuple[float, float, float]]:
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError, KeyError):
        return None
    if not isinstance(lm, dict):
        return None
    x = lm.get("x")
    y = lm.get("y")
    z = lm.get("z")
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def _landmark_presence(landmarks: List[Dict], idx: int) -> float:
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


# --- Metrics Logic ----------------------------------------------------------


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
        metrics["grip_ratio"] = grip_median

    # Summary stats for grip unevenness (normalized by shoulder width)
    if grip_uneven_vals:
        uneven_median = float(statistics.median(grip_uneven_vals))
        uneven_min = float(min(grip_uneven_vals))
        uneven_max = float(max(grip_uneven_vals))

        metrics["grip_uneven_median"] = uneven_median
        metrics["grip_uneven_min"] = uneven_min
        metrics["grip_uneven_max"] = uneven_max
        metrics["grip_uneven_norm"] = uneven_max

    # Summary stats for bar tilt
    if bar_tilts:
        tilt_median = float(statistics.median(bar_tilts))
        tilt_min = float(min(bar_tilts))
        tilt_max = float(max(bar_tilts))
        metrics["bar_tilt_median_deg"] = tilt_median
        metrics["bar_tilt_min_deg"] = tilt_min
        metrics["bar_tilt_max_deg"] = tilt_max
        metrics["bar_tilt_deg"] = tilt_median
        metrics["bar_tilt_deg_max"] = tilt_max

    metrics["tracking_total_frames"] = total_frames
    metrics["tracking_bad_frames"] = bad_frames
    denom = total_frames if total_frames > 0 else 1
    bad_ratio = bad_frames / denom
    metrics["tracking_bad_ratio"] = float(bad_ratio)
    metrics["tracking_quality"] = float(max(0.0, min(1.0, 1.0 - bad_ratio)))
    return metrics


def suggest_auto_tags(
    metrics: Dict[str, float],
    tracking_unreliable: bool,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Return suggested tags based on computed metrics."""
    tags: List[str] = []
    thresholds = thresholds or {}
    # Defaults should match label_config if not provided
    wide_thresh = thresholds.get("grip_wide_threshold", 2.1)
    narrow_thresh = thresholds.get("grip_narrow_threshold", 1.2)
    uneven_thresh = thresholds.get("grip_uneven_threshold", 0.10)
    tilt_thresh = thresholds.get("bar_tilt_threshold", 5.0)

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


def metrics_to_features(rep_data: Dict) -> Optional[List[float]]:
    """Convert a rep dictionary (from JSON) into a feature vector."""
    if rep_data.get("tracking_unreliable", False):
        return None
    
    # Ensure metrics are present or recompute them? 
    # For now, assume they are either in the JSON or we could recompute if missing.
    # But usually 'metrics' is populated. logic in extract_example accessed it directly.
    metrics = rep_data.get("metrics") or {}
    frames = rep_data.get("frames") or []
    
    tracking_quality = float(metrics.get("tracking_quality", 0.0))
    if tracking_quality < 0.5 or not frames:
        return None

    # Compute additional frame-level stats (wrist Y) that aren't in standard metrics
    wrist_y_vals = []
    for frec in frames:
        if not frec or not frec.get("pose_present"):
            continue
        lms = frec.get("landmarks")
        if not lms:
            continue
        ls = _landmark_xy(lms, L_SHOULDER)
        rs = _landmark_xy(lms, R_SHOULDER)
        lw = _landmark_xy(lms, L_WRIST)
        rw = _landmark_xy(lms, R_WRIST)
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

    return [
        float(rep_data.get("load_lbs") or 0.0),
        float(metrics.get("grip_ratio_median", 0.0)),
        float(metrics.get("grip_ratio_range", 0.0)),
        float(metrics.get("grip_uneven_median", 0.0)),
        float(metrics.get("grip_uneven_norm", 0.0)),
        float(metrics.get("bar_tilt_median_deg", 0.0)),
        float(metrics.get("bar_tilt_deg_max", 0.0)),
        float(metrics.get("tracking_bad_ratio", 0.0)),
        tracking_quality,
        wrist_y_min,
        wrist_y_max,
        wrist_y_range,
    ]
