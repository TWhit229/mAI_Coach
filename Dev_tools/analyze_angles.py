#!/usr/bin/env python3
"""Analyze the angle data around expected cut times to understand the pattern."""

import csv
import numpy as np
from typing import List, Tuple

# Ground truth cut times from user's manual cuts (in seconds)
EXPECTED_CUTS = [0, 3.8, 6.2, 11.8, 21.0, 24.5, 31.2, 34.8, 38.9, 42.1, 46.6, 50.1]

def load_csv(path: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Load CSV and return frames, times_ms, raw_angles, smooth_angles."""
    frames = []
    times_ms = []
    raw_angles = []
    smooth_angles = []
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            times_ms.append(float(row['time_ms']))
            raw_angles.append(float(row['angle_raw']))
            smooth_angles.append(float(row['angle_smooth']))
    
    return frames, times_ms, raw_angles, smooth_angles


def find_frame_for_time(times_ms: List[float], target_s: float) -> int:
    """Find the frame index closest to target time in seconds."""
    target_ms = target_s * 1000
    min_diff = float('inf')
    best_frame = 0
    for i, t in enumerate(times_ms):
        diff = abs(t - target_ms)
        if diff < min_diff:
            min_diff = diff
            best_frame = i
    return best_frame


def main():
    csv_path = '/Users/whitney/Documents/mAI_Coach/Dev_tools/data/PXL_20251111_210527952_angles_debug.csv'
    
    print("Loading CSV...")
    frames, times_ms, raw_angles, smooth_angles = load_csv(csv_path)
    
    angles = np.array(raw_angles)
    
    # Analyze each expected cut point
    print("\n" + "=" * 80)
    print("ANALYZING EXPECTED CUT POINTS")
    print("=" * 80)
    
    for cut_time in EXPECTED_CUTS:
        frame_idx = find_frame_for_time(times_ms, cut_time)
        
        # Get angle at cut point and surrounding context
        window = 15  # frames before/after
        start_idx = max(0, frame_idx - window)
        end_idx = min(len(angles), frame_idx + window + 1)
        
        window_angles = angles[start_idx:end_idx]
        center_angle = angles[frame_idx] if frame_idx < len(angles) else 0
        
        # Calculate velocity around this point
        if frame_idx > 0 and frame_idx < len(angles) - 1:
            velocity = angles[frame_idx + 1] - angles[frame_idx - 1]
        else:
            velocity = 0
        
        print(f"\nCut at {cut_time:.1f}s (frame {frame_idx}):")
        print(f"  Angle: {center_angle:.1f}° (window: {window_angles.min():.1f}° - {window_angles.max():.1f}°)")
        print(f"  Velocity: {velocity/2:.1f}°/frame")
        print("  Context: ", end="")
        
        # Show if this looks like a peak, valley, or transitional
        if center_angle > 155:
            print("HIGH angle (lockout)")
        elif center_angle < 135:
            print("LOW angle (bottom)")
        else:
            print("MID angle (transitional)")
    
    # Now let's look for patterns in the valleys
    print("\n" + "=" * 80)
    print("FINDING ALL VALLEYS (local minima < 138°)")
    print("=" * 80)
    
    # Heavy smoothing
    kernel = np.ones(15) / 15
    angles_smooth = np.convolve(angles, kernel, mode='same')
    
    valleys = []
    window = 20
    for i in range(window, len(angles_smooth) - window):
        if angles_smooth[i] > 138:
            continue
        
        # Check if local minimum
        is_min = True
        for j in range(i - window, i + window + 1):
            if j != i and angles_smooth[j] < angles_smooth[i]:
                is_min = False
                break
        
        if is_min:
            t = times_ms[i] / 1000.0
            if not valleys or (t - valleys[-1][1]) > 1.0:
                valleys.append((i, t, angles_smooth[i]))
    
    print(f"Found {len(valleys)} valleys:")
    for frame, t, angle in valleys:
        print(f"  {t:.1f}s: {angle:.1f}°")
    
    # Try midpoints between consecutive valleys
    print("\n" + "=" * 80)
    print("MIDPOINTS BETWEEN VALLEYS")
    print("=" * 80)
    
    if len(valleys) >= 2:
        midpoints = []
        for i in range(len(valleys) - 1):
            mid_t = (valleys[i][1] + valleys[i + 1][1]) / 2
            # Find angle at midpoint
            mid_frame = find_frame_for_time(times_ms, mid_t)
            mid_angle = angles[mid_frame]
            midpoints.append((mid_t, mid_angle))
            print(f"  {mid_t:.1f}s: {mid_angle:.1f}°")
        
        # Compare to expected
        print("\nComparison to expected:")
        for exp in EXPECTED_CUTS:
            closest = None
            min_diff = float('inf')
            for mid_t, _ in midpoints:
                if abs(mid_t - exp) < min_diff:
                    min_diff = abs(mid_t - exp)
                    closest = mid_t
            
            status = "✓" if min_diff < 1.5 else "✗"
            print(f"  Expected {exp:.1f}s -> closest midpoint: {closest:.1f}s (diff: {min_diff:.1f}s) {status}")


if __name__ == '__main__':
    main()
