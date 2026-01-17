"""Shared generic utilities."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


# MARK: - One Euro Filter
# Adaptive low-pass filter: smooth when slow, responsive when fast.
# Based on: https://cristal.univ-lille.fr/~casiez/1euro/

class OneEuroFilter:
    """Adaptive low-pass filter for smoothing noisy signals."""
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)
    
    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1.0 - a) * x_prev
    
    def filter(self, value: float, timestamp: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = value
            self.t_prev = timestamp
            return value
        
        t_e = timestamp - self.t_prev
        if t_e <= 0:
            return self.x_prev
        
        # Derivative estimation
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (value - self.x_prev) / t_e
        dx_smoothed = self._exponential_smoothing(a_d, dx, self.dx_prev)
        self.dx_prev = dx_smoothed
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)
        a = self._smoothing_factor(t_e, cutoff)
        
        x_filtered = self._exponential_smoothing(a, value, self.x_prev)
        self.x_prev = x_filtered
        self.t_prev = timestamp
        
        return x_filtered


class LandmarkSmoother:
    """Manages One Euro Filters for all landmark coordinates with outlier rejection and skeleton constraints."""
    
    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    def __init__(self, max_jump_fraction: float = 0.15):
        self.filters_x: List[OneEuroFilter] = []
        self.filters_y: List[OneEuroFilter] = []
        self.filters_z: List[OneEuroFilter] = []
        self.previous_landmarks: Optional[np.ndarray] = None
        self.max_jump_fraction = max_jump_fraction
        
        # Skeleton length calibration
        self.calibrated_left_upper_arm: Optional[float] = None
        self.calibrated_right_upper_arm: Optional[float] = None
        self.calibration_samples: List[Tuple[float, float]] = []  # (left, right) lengths
        self.calibration_complete = False
        self.min_calibration_samples = 5  # Number of good frames to calibrate
    
    def reset(self):
        self.filters_x.clear()
        self.filters_y.clear()
        self.filters_z.clear()
        self.previous_landmarks = None
        self.calibrated_left_upper_arm = None
        self.calibrated_right_upper_arm = None
        self.calibration_samples.clear()
        self.calibration_complete = False
    
    def _distance_3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate 3D distance between two landmark points."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def _calibrate_skeleton(self, landmarks: np.ndarray) -> None:
        """Collect arm length samples when visibility is good for calibration."""
        if self.calibration_complete:
            return
        
        # Check if key landmarks have good visibility
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_elbow = landmarks[self.LEFT_ELBOW]
        right_elbow = landmarks[self.RIGHT_ELBOW]
        
        # Require high visibility for calibration
        min_vis = 0.8
        if (left_shoulder[3] >= min_vis and left_elbow[3] >= min_vis and
            right_shoulder[3] >= min_vis and right_elbow[3] >= min_vis):
            
            left_len = self._distance_3d(left_shoulder, left_elbow)
            right_len = self._distance_3d(right_shoulder, right_elbow)
            
            # Sanity check: arm lengths should be reasonable
            if 0.05 < left_len < 0.5 and 0.05 < right_len < 0.5:
                self.calibration_samples.append((left_len, right_len))
                
                if len(self.calibration_samples) >= self.min_calibration_samples:
                    # Use median for robustness
                    left_samples = [s[0] for s in self.calibration_samples]
                    right_samples = [s[1] for s in self.calibration_samples]
                    self.calibrated_left_upper_arm = sorted(left_samples)[len(left_samples) // 2]
                    self.calibrated_right_upper_arm = sorted(right_samples)[len(right_samples) // 2]
                    self.calibration_complete = True
    
    def _apply_skeleton_constraints(self, landmarks: np.ndarray) -> np.ndarray:
        """Correct shoulder positions when visibility drops using calibrated arm lengths."""
        if not self.calibration_complete:
            return landmarks
        
        corrected = landmarks.copy()
        
        # Correct left shoulder if visibility is poor but elbow is good
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        left_elbow = landmarks[self.LEFT_ELBOW]
        
        if left_shoulder[3] < 0.6 and left_elbow[3] >= 0.6:
            # Shoulder visibility is poor - use skeleton constraint
            current_len = self._distance_3d(left_shoulder, left_elbow)
            if current_len > 0.01 and self.calibrated_left_upper_arm:
                # Calculate how much the arm length deviates from calibrated
                ratio = self.calibrated_left_upper_arm / current_len
                
                # Only correct if the arm appears too short (shoulder drifted toward elbow)
                if ratio > 1.1:  # More than 10% shorter than calibrated
                    # Move shoulder away from elbow to restore correct length
                    direction = np.array([
                        left_shoulder[0] - left_elbow[0],
                        left_shoulder[1] - left_elbow[1],
                        left_shoulder[2] - left_elbow[2]
                    ])
                    norm = np.linalg.norm(direction)
                    if norm > 0.001:
                        direction = direction / norm
                        # New shoulder position
                        new_shoulder = np.array([
                            left_elbow[0] + direction[0] * self.calibrated_left_upper_arm,
                            left_elbow[1] + direction[1] * self.calibrated_left_upper_arm,
                            left_elbow[2] + direction[2] * self.calibrated_left_upper_arm,
                            left_shoulder[3]  # Keep original visibility
                        ])
                        corrected[self.LEFT_SHOULDER] = new_shoulder
        
        # Correct right shoulder similarly
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        right_elbow = landmarks[self.RIGHT_ELBOW]
        
        if right_shoulder[3] < 0.6 and right_elbow[3] >= 0.6:
            current_len = self._distance_3d(right_shoulder, right_elbow)
            if current_len > 0.01 and self.calibrated_right_upper_arm:
                ratio = self.calibrated_right_upper_arm / current_len
                
                if ratio > 1.1:
                    direction = np.array([
                        right_shoulder[0] - right_elbow[0],
                        right_shoulder[1] - right_elbow[1],
                        right_shoulder[2] - right_elbow[2]
                    ])
                    norm = np.linalg.norm(direction)
                    if norm > 0.001:
                        direction = direction / norm
                        new_shoulder = np.array([
                            right_elbow[0] + direction[0] * self.calibrated_right_upper_arm,
                            right_elbow[1] + direction[1] * self.calibrated_right_upper_arm,
                            right_elbow[2] + direction[2] * self.calibrated_right_upper_arm,
                            right_shoulder[3]
                        ])
                        corrected[self.RIGHT_SHOULDER] = new_shoulder
        
        return corrected
    
    def smooth(self, landmarks: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """
        Smooth landmarks with One Euro Filter + outlier rejection + skeleton constraints.
        
        Args:
            landmarks: Nx4 array of [x, y, z, visibility]
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            Smoothed Nx4 array
        """
        timestamp = timestamp_ms / 1000.0
        
        # Initialize filters if needed
        if len(self.filters_x) != len(landmarks):
            self.filters_x = [OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0) for _ in range(len(landmarks))]
            self.filters_y = [OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0) for _ in range(len(landmarks))]
            self.filters_z = [OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0) for _ in range(len(landmarks))]
        
        # Calibrate skeleton lengths during good visibility
        self._calibrate_skeleton(landmarks)
        
        # Apply skeleton constraints to correct bad detections
        landmarks = self._apply_skeleton_constraints(landmarks)
        
        # Calculate shoulder width for outlier detection
        shoulder_width = 0.2  # default fallback
        if len(landmarks) > 12:
            ls = landmarks[11]  # left shoulder
            rs = landmarks[12]  # right shoulder
            dx = ls[0] - rs[0]
            dy = ls[1] - rs[1]
            shoulder_width = max(0.05, math.sqrt(dx * dx + dy * dy))
        max_jump = shoulder_width * self.max_jump_fraction
        
        smoothed = np.zeros_like(landmarks)
        
        for i in range(len(landmarks)):
            lm = landmarks[i]
            use_x, use_y, use_z = lm[0], lm[1], lm[2]
            vis = lm[3] if len(lm) > 3 else 1.0
            
            # Outlier rejection: if landmark jumped too far with low visibility, use previous
            if self.previous_landmarks is not None and i < len(self.previous_landmarks):
                prev = self.previous_landmarks[i]
                dx = abs(lm[0] - prev[0])
                dy = abs(lm[1] - prev[1])
                jump = math.sqrt(dx * dx + dy * dy)
                
                if jump > max_jump and vis < 0.7:
                    # Large jump with low visibility = likely bad detection
                    use_x, use_y, use_z = prev[0], prev[1], prev[2]
            
            # Apply One Euro Filter
            filtered_x = self.filters_x[i].filter(use_x, timestamp)
            filtered_y = self.filters_y[i].filter(use_y, timestamp)
            filtered_z = self.filters_z[i].filter(use_z, timestamp)
            
            # Visibility-weighted blending: low visibility = trust previous more
            if self.previous_landmarks is not None and i < len(self.previous_landmarks) and vis < 0.8:
                vis_weight = max(0.3, min(1.0, vis))
                smooth_weight = 1.0 - (vis_weight * 0.5)  # 0.5 to 0.85
                prev = self.previous_landmarks[i]
                
                output_x = filtered_x * (1.0 - smooth_weight) + prev[0] * smooth_weight
                output_y = filtered_y * (1.0 - smooth_weight) + prev[1] * smooth_weight
                output_z = filtered_z * (1.0 - smooth_weight) + prev[2] * smooth_weight
            else:
                output_x, output_y, output_z = filtered_x, filtered_y, filtered_z
            
            smoothed[i] = [output_x, output_y, output_z, vis]
        
        self.previous_landmarks = smoothed.copy()
        return smoothed


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
