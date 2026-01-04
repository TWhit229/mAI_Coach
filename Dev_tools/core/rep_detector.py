"""Rep detection algorithm - ported from iOS BenchInferenceEngine.swift.

This module detects rep boundaries from pose landmark data by tracking
elbow angles through a state machine.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Landmark indices (MediaPipe)
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16


class RepPhase(Enum):
    """State machine phases for rep detection."""
    IDLE = "idle"           # Arms extended (top)
    DESCENDING = "descending"  # Arms bending
    BOTTOM = "bottom"       # Arms bent (chest)
    ASCENDING = "ascending"  # Arms straightening


@dataclass
class RepBoundary:
    """Detected rep with frame indices."""
    start_frame: int
    end_frame: int
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None


def _angle_3d(
    first: Tuple[float, float, float],
    mid: Tuple[float, float, float], 
    last: Tuple[float, float, float]
) -> float:
    """Calculate 3D angle at the middle point in degrees."""
    v1x = first[0] - mid[0]
    v1y = first[1] - mid[1]
    v1z = first[2] - mid[2]
    
    v2x = last[0] - mid[0]
    v2y = last[1] - mid[1]
    v2z = last[2] - mid[2]
    
    dot = v1x * v2x + v1y * v2y + v1z * v2z
    mag1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    mag2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    
    if mag1 * mag2 < 1e-6:
        return 0.0
    
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_val) * 180.0 / math.pi


def _get_landmark_xyz(
    landmarks: List[Dict], idx: int
) -> Optional[Tuple[float, float, float]]:
    """Extract x, y, z from a landmark dict."""
    if not landmarks or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if not isinstance(lm, dict):
        return None
    x = lm.get("x")
    y = lm.get("y")
    z = lm.get("z", 0.0)  # z might be missing, default to 0
    if x is None or y is None:
        return None
    return (float(x), float(y), float(z))


def _compute_avg_elbow_angle(landmarks: List[Dict]) -> Optional[float]:
    """Compute average elbow angle from left and right arms."""
    if not landmarks or len(landmarks) <= max(L_WRIST, R_WRIST):
        return None
    
    # Left arm: shoulder -> elbow -> wrist
    ls = _get_landmark_xyz(landmarks, L_SHOULDER)
    le = _get_landmark_xyz(landmarks, L_ELBOW)
    lw = _get_landmark_xyz(landmarks, L_WRIST)
    
    # Right arm: shoulder -> elbow -> wrist
    rs = _get_landmark_xyz(landmarks, R_SHOULDER)
    re = _get_landmark_xyz(landmarks, R_ELBOW)
    rw = _get_landmark_xyz(landmarks, R_WRIST)
    
    angles = []
    if ls and le and lw:
        angles.append(_angle_3d(ls, le, lw))
    if rs and re and rw:
        angles.append(_angle_3d(rs, re, rw))
    
    if not angles:
        return None
    return sum(angles) / len(angles)


def _get_wrist_position(landmarks: List[Dict]) -> Optional[float]:
    """Get average wrist Y position (lower Y = higher in frame for bench press).
    
    For bench press lying down, we actually want the X position relative
    to shoulders to track arm extension. But for safety, we'll use elbow angle
    combined with a simple position check.
    """
    if not landmarks or len(landmarks) <= max(L_WRIST, R_WRIST):
        return None
    
    lw = _get_landmark_xyz(landmarks, L_WRIST)
    rw = _get_landmark_xyz(landmarks, R_WRIST)
    
    positions = []
    if lw:
        positions.append(lw[1])  # Y coordinate
    if rw:
        positions.append(rw[1])  # Y coordinate
    
    if not positions:
        return None
    return sum(positions) / len(positions)


class RepDetector:
    """State machine for detecting rep boundaries from pose data.
    
    This mirrors the iOS RepDetector implementation for bench press:
    - Starts in IDLE (arms extended, angle > lockout_angle)
    - Transitions to DESCENDING when angle drops below lockout - buffer
    - Transitions to BOTTOM when angle < bottom_angle
    - Transitions to ASCENDING when angle starts increasing
    - Rep completes when angle returns above lockout_angle
    """
    
    def __init__(
        self,
        lockout_angle: float = 150.0,
        bottom_angle: float = 100.0,
        cooldown_ms: int = 800,
    ):
        self.lockout_angle = lockout_angle
        self.bottom_angle = bottom_angle
        self.cooldown_ms = cooldown_ms
        
        self.phase = RepPhase.IDLE
        self.current_start_frame: Optional[int] = None
        self.current_start_time: Optional[int] = None
        self.last_rep_time: int = 0
    
    def reset(self):
        """Reset the detector state."""
        self.phase = RepPhase.IDLE
        self.current_start_frame = None
        self.current_start_time = None
        self.last_rep_time = 0
    
    def process_frame(
        self,
        frame_idx: int,
        landmarks: List[Dict],
        timestamp_ms: Optional[int] = None
    ) -> Optional[RepBoundary]:
        """Process a single frame and return a RepBoundary if a rep completed.
        
        Args:
            frame_idx: Index of the current frame
            landmarks: List of landmark dicts with x, y, z, visibility
            timestamp_ms: Optional timestamp in milliseconds
            
        Returns:
            RepBoundary if a rep just completed, None otherwise
        """
        avg_angle = _compute_avg_elbow_angle(landmarks)
        if avg_angle is None:
            return None
        
        t = timestamp_ms or 0
        buffer = 5  # Reduced buffer to avoid threshold overlap
        
        if self.phase == RepPhase.IDLE:
            # Waiting for descent - start when angle drops below threshold
            if avg_angle < (self.lockout_angle - buffer):
                self.phase = RepPhase.DESCENDING
                self.current_start_frame = frame_idx
                self.current_start_time = t
        
        elif self.phase == RepPhase.DESCENDING:
            # Arms bending - wait for bottom
            if avg_angle < self.bottom_angle:
                self.phase = RepPhase.BOTTOM
            # Abort if we go back up without hitting bottom
            elif avg_angle > self.lockout_angle:
                self.phase = RepPhase.IDLE
                self.current_start_frame = None
                self.current_start_time = None
        
        elif self.phase == RepPhase.BOTTOM:
            # At bottom - wait for ascent
            if avg_angle > (self.bottom_angle + buffer):
                self.phase = RepPhase.ASCENDING
        
        elif self.phase == RepPhase.ASCENDING:
            # Pushing up - check for lockout (rep complete)
            if avg_angle > self.lockout_angle:
                # Rep completed!
                if (t - self.last_rep_time) > self.cooldown_ms:
                    boundary = RepBoundary(
                        start_frame=self.current_start_frame or 0,
                        end_frame=frame_idx,
                        start_time_ms=self.current_start_time,
                        end_time_ms=t,
                    )
                    self._reset_for_next_rep(t)
                    return boundary
                else:
                    # Too fast / double count - reset without returning
                    self._reset_for_next_rep(t)
            # Abort if we drop back down
            elif avg_angle < self.bottom_angle:
                self.phase = RepPhase.BOTTOM
        
        return None
    
    def _reset_for_next_rep(self, t: int):
        """Reset state for the next rep."""
        self.phase = RepPhase.IDLE
        self.current_start_frame = None
        self.current_start_time = None
        self.last_rep_time = t


def detect_reps_in_frames(
    frames: List[Dict],
    lockout_angle: float = 140.0,
    bottom_angle: float = 130.0,
    cooldown_ms: int = 300,
    min_rep_frames: int = 80,  # ~2.7 seconds at 30fps
) -> List[RepBoundary]:
    """Detect rep boundaries using rise-after-valley algorithm.
    
    This algorithm detects rep boundaries by:
    1. Finding valleys (local minima) in the elbow angle signal (rep bottoms)
    2. For each valley, detecting when the angle rises by a threshold amount
    3. That rising point marks the start of the next rep
    
    This works well because cuts should happen when the bar starts rising,
    which is a distinct signal regardless of absolute angle values.
    
    Args:
        frames: List of frame dicts with 'landmarks' and 'time_ms'
        lockout_angle: Not used (kept for API compat)
        bottom_angle: Not used (kept for API compat)
        cooldown_ms: Not used (kept for API compat)
        min_rep_frames: Minimum frames between cuts (~2.7 sec)
        
    Returns:
        List of RepBoundary objects
    """
    if not frames:
        return []
    
    import numpy as np
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # Fallback if scipy not available
        return []
    
    # Step 1: Extract elbow angles for all frames
    raw_angles: List[Optional[float]] = []
    for frame in frames:
        landmarks = frame.get("landmarks", [])
        angle = _compute_avg_elbow_angle(landmarks) if landmarks else None
        raw_angles.append(angle)
    
    # Convert to numpy array, filling None with NaN
    angles = np.array([a if a is not None else np.nan for a in raw_angles])
    n = len(angles)
    
    # Interpolate NaN values
    nans = np.isnan(angles)
    if nans.all():
        return []
    
    if nans.any():
        x = np.arange(n)
        angles[nans] = np.interp(x[nans], x[~nans], angles[~nans])
    
    # Step 2: Heavy smoothing (15-frame window) to reduce noise
    kernel = np.ones(15) / 15
    angles_smooth = np.convolve(angles, kernel, mode='same')
    
    # Step 3: Find valleys (local minima = rep bottoms) using scipy
    # Invert signal to find minima as peaks
    valleys, _ = find_peaks(-angles_smooth, prominence=5, distance=35)
    
    if len(valleys) < 2:
        return []
    
    # Step 4: For each valley, find when angle rises by threshold amount
    # This marks the start of the ascent (start of next rep)
    rise_threshold = 7  # Must rise 7 degrees from valley bottom
    cuts = []
    
    for v in valleys:
        valley_angle = angles_smooth[v]
        # Look forward up to 50 frames (~1.7 sec) for the rise
        for j in range(v, min(v + 50, n)):
            if angles_smooth[j] > valley_angle + rise_threshold:
                # Found the rise point - add as cut if enough gap
                if not cuts or (j - cuts[-1]) >= min_rep_frames:
                    cuts.append(j)
                break
    
    # Step 5: Add video start if first cut is late and video starts with high angle
    if cuts and cuts[0] > 90 and angles_smooth[0] > 145:
        cuts = [0] + cuts
    
    if len(cuts) < 2:
        return []
    
    # Step 6: Create RepBoundary objects
    reps: List[RepBoundary] = []
    for i in range(len(cuts) - 1):
        start_frame = int(cuts[i])
        end_frame = int(cuts[i + 1])
        
        # Skip if too short
        if end_frame - start_frame < min_rep_frames:
            continue
        
        # Get timestamps
        start_ms = frames[start_frame].get("time_ms", 0) or 0
        end_ms = frames[end_frame].get("time_ms", 0) or 0
        
        reps.append(RepBoundary(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        ))
    
    return reps


def _find_direction_changes(
    values: List[Optional[float]], 
    min_gap: int = 10
) -> List[int]:
    """Find frames where signal changes from increasing to decreasing.
    
    These are the peaks/turning points where hands go from moving up
    to starting to move down (lockout position).
    """
    turning_points = []
    n = len(values)
    
    # Look for sign changes in the derivative
    i = min_gap
    while i < n - min_gap:
        curr = values[i]
        if curr is None:
            i += 1
            continue
        
        # Check if this is a local maximum by looking at neighbors
        # Look back: was the trend increasing?
        prev_val = None
        for j in range(i - 1, max(0, i - min_gap) - 1, -1):
            if values[j] is not None:
                prev_val = values[j]
                break
        
        # Look forward: is the trend decreasing?
        next_val = None
        for j in range(i + 1, min(n, i + min_gap + 1)):
            if values[j] is not None:
                next_val = values[j]
                break
        
        if prev_val is not None and next_val is not None:
            # This is a turning point if: prev < curr AND next < curr
            if prev_val < curr and next_val < curr:
                # Ensure minimum gap from last turning point
                if not turning_points or (i - turning_points[-1]) >= min_gap:
                    turning_points.append(i)
                    i += min_gap  # Skip ahead to avoid detecting same peak
                    continue
        
        i += 1
    
    return turning_points


def _smooth_angles(angles: List[Optional[float]], window: int = 5) -> List[Optional[float]]:
    """Apply simple moving average smoothing to reduce noise."""
    result = []
    half_w = window // 2
    
    for i in range(len(angles)):
        # Gather valid values in window
        vals = []
        for j in range(max(0, i - half_w), min(len(angles), i + half_w + 1)):
            if angles[j] is not None:
                vals.append(angles[j])
        
        if vals:
            result.append(sum(vals) / len(vals))
        else:
            result.append(None)
    
    return result


def _find_local_maxima(
    values: List[Optional[float]], 
    min_prominence: float = 10.0,
    min_distance: int = 10
) -> List[int]:
    """Find local maxima (peaks) in the signal.
    
    Args:
        values: List of angle values (may contain None)
        min_prominence: Minimum height above surrounding values to count as peak
        min_distance: Minimum frames between peaks
        
    Returns:
        List of frame indices where peaks occur
    """
    peaks = []
    n = len(values)
    
    for i in range(min_distance, n - min_distance):
        val = values[i]
        if val is None:
            continue
        
        # Check if this is higher than neighbors in both directions
        is_peak = True
        left_min = val
        right_min = val
        
        # Check left side
        for j in range(max(0, i - min_distance), i):
            v = values[j]
            if v is not None:
                if v >= val:
                    is_peak = False
                    break
                left_min = min(left_min, v)
        
        if not is_peak:
            continue
            
        # Check right side
        for j in range(i + 1, min(n, i + min_distance + 1)):
            v = values[j]
            if v is not None:
                if v > val:  # Use > instead of >= to handle ties
                    is_peak = False
                    break
                right_min = min(right_min, v)
        
        if not is_peak:
            continue
        
        # Check prominence (how much higher than surrounding valleys)
        prominence = val - max(left_min, right_min)
        if prominence < min_prominence:
            continue
        
        # Ensure minimum distance from previous peak
        if peaks and (i - peaks[-1]) < min_distance:
            # Keep the higher peak
            if val > (values[peaks[-1]] or 0):
                peaks[-1] = i
            continue
        
        peaks.append(i)
    
    return peaks

