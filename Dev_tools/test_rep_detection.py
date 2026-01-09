#!/usr/bin/env python3
"""Final test with fixed start detection."""

import csv
import numpy as np
from typing import List

EXPECTED_CUTS = [0, 3.8, 6.2, 11.8, 21.0, 24.5, 31.2, 34.8, 38.9, 42.1, 46.6, 50.1]

def load_csv(path: str):
    frames, times_ms, raw_angles, smooth_angles = [], [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            times_ms.append(float(row['time_ms']))
            raw_angles.append(float(row['angle_raw']))
            smooth_angles.append(float(row['angle_smooth']))
    return frames, times_ms, raw_angles, smooth_angles


def evaluate(detected: List[float], expected: List[float], tolerance: float = 1.5) -> dict:
    matched = 0
    matched_exp = set()
    for det in detected:
        for i, exp in enumerate(expected):
            if i not in matched_exp and abs(det - exp) <= tolerance:
                matched += 1
                matched_exp.add(i)
                break
    
    precision = matched / len(detected) if detected else 0
    recall = matched / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    missed = [EXPECTED_CUTS[i] for i in range(len(expected)) if i not in matched_exp]
    
    return {'matched': matched, 'f1': f1, 'missed': missed, 'detected': len(detected)}


def detect_final(times_ms: List[float], raw_angles: List[float],
                 min_gap: float = 1.8,
                 prominence: int = 6,
                 valley_thresh: float = 145.0) -> List[float]:
    """
    Final optimized algorithm:
    1. Check if video starts with rising angle -> add 0.0 as first cut
    2. Find peaks with prominence check  
    3. Find valley midpoints
    4. Combine all and merge with minimum gap
    """
    angles = np.array(raw_angles)
    n = len(angles)
    
    kernel = np.ones(11) / 11
    angles_smooth = np.convolve(angles, kernel, mode='same')
    
    all_cuts = []
    
    # --- Include video start if angle is rising in first 2 seconds ---
    # The video might start at or near lockout position
    # Check if early part has an angle peak
    early_max = np.max(angles_smooth[:60])  # First 2 seconds
    early_max_idx = np.argmax(angles_smooth[:60])
    
    # If max is in first 2 seconds and is relatively high, start from 0
    if early_max > 160 and early_max_idx < 60:
        # Check if angle goes up initially (video starts mid-rep)
        if angles_smooth[5] > angles_smooth[0]:
            all_cuts.append(('start', 0.0))
    
    # --- Find prominent peaks ---
    window = 18
    for i in range(window, n - window):
        val = angles_smooth[i]
        
        is_max = True
        for j in range(i - window, i + window + 1):
            if j != i and angles_smooth[j] > val:
                is_max = False
                break
        
        if not is_max:
            continue
        
        left_min = min(angles_smooth[max(0, i-window):i])
        right_min = min(angles_smooth[i+1:min(n, i+window+1)])
        left_prom = val - left_min
        right_prom = val - right_min
        
        if left_prom >= prominence and right_prom >= prominence:
            t = times_ms[i] / 1000.0
            all_cuts.append(('peak', t))
    
    # --- Find valleys and midpoints ---
    valleys = []
    valley_window = 15
    for i in range(valley_window, n - valley_window):
        if angles_smooth[i] > valley_thresh:
            continue
        is_min = True
        for j in range(i - valley_window//2, i + valley_window//2 + 1):
            if j != i and angles_smooth[j] < angles_smooth[i]:
                is_min = False
                break
        if is_min:
            t = times_ms[i] / 1000.0
            if not valleys or (t - valleys[-1]) > 1.0:
                valleys.append(t)
    
    for i in range(len(valleys) - 1):
        mid = (valleys[i] + valleys[i+1]) / 2
        all_cuts.append(('midpoint', mid))
    
    # Sort by time and merge with minimum gap
    all_cuts.sort(key=lambda x: x[1])
    
    final_cuts = []
    for cut_type, t in all_cuts:
        if not final_cuts or (t - final_cuts[-1]) > min_gap:
            final_cuts.append(t)
    
    return final_cuts


def main():
    csv_path = '/Users/whitney/Documents/mAI_Coach/Dev_tools/data/PXL_20251111_210527952_angles_debug.csv'
    frames, times_ms, raw_angles, smooth_angles = load_csv(csv_path)
    
    print("Expected:", [f"{t:.1f}s" for t in EXPECTED_CUTS])
    print()
    
    # Debug: show first few angles
    angles = np.array(raw_angles)
    kernel = np.ones(11) / 11
    angles_smooth = np.convolve(angles, kernel, mode='same')
    print(f"First 10 smoothed angles: {[f'{a:.1f}' for a in angles_smooth[:10]]}")
    print(f"Early max (first 2s): {np.max(angles_smooth[:60]):.1f} at frame {np.argmax(angles_smooth[:60])}")
    print()
    
    cuts = detect_final(times_ms, raw_angles)
    result = evaluate(cuts, EXPECTED_CUTS)
    
    print("FINAL ALGORITHM:")
    print(f"  Detected: {result['detected']}, Matched: {result['matched']}/12, F1: {result['f1']:.2f}")
    print(f"  Cuts: {[f'{t:.1f}s' for t in cuts]}")
    print(f"  Missed: {[f'{t:.1f}s' for t in result['missed']]}")


if __name__ == '__main__':
    main()
