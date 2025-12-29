"""Shared generic utilities."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


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


ROTATION_OPTIONS: List[Tuple[str, Optional[int]]] = [
    ("Auto (metadata)", None),
    ("0°", 0),
    ("90° CW", 90),
    ("180°", 180),
    ("270° CCW", 270),
]


def rotation_option_index(degrees: Optional[int]) -> int:
    if degrees is None:
        return 0
    degrees %= 360
    for i, (_, val) in enumerate(ROTATION_OPTIONS):
        if val == degrees:
            return i
    return 0


def rotation_value_from_index(index: int) -> Optional[int]:
    if 0 <= index < len(ROTATION_OPTIONS):
        return ROTATION_OPTIONS[index][1]
    return None
