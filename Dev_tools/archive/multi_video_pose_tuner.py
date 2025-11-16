#!/usr/bin/env python3
"""
Multi-Video Pose Tuner (grid of videos, shared settings)

- Uses MediaPipe Tasks Pose Landmarker (VIDEO mode).
- Expects these model files in the SAME folder as this script:
    pose_landmarker_lite.task
    pose_landmarker_full.task
    pose_landmarker_heavy.task
- Lets you run multiple videos in parallel (grid layout).
- All videos share the same settings, controlled by sliders:
    * model_variant: 0=lite, 1=full, 2=heavy (default full)
    * upper_body_only: 0=full body, 1=chest/arms/hands only
    * min_pose_detection
    * min_pose_presence
    * min_tracking
    * output_seg_masks
    * EMA_alpha (extra smoothing)
    * circle_radius
    * thickness

Usage:
    python multi_video_pose_tuner.py
      -> opens file dialog, choose N videos

    python multi_video_pose_tuner.py --videos vid1.mp4 vid2.mp4 ...

Keys:
    q          Quit
    SPACE      Pause/Resume
    s          Save PNG snapshot of the grid
    e          Export current settings to pose_tuner_settings.json
"""

import argparse
import math
import json
import time
from pathlib import Path

import cv2
import numpy as np

# --- Mediapipe Tasks import ---------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception as e:
    mp = None
    _mp_import_err = e

WIN_NAME = "Multi Pose Tuner"


# --- Optional multi-file picker ----------------------------------------------
def pick_files_dialog_multi():
    """Try to open a native file-dialog that lets you choose multiple videos."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        fnames = filedialog.askopenfilenames(
            title="Choose video files",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        # fnames is typically a tuple of strings
        return list(fnames)
    except Exception:
        return []


# --- Landmark indices (BlazePose full body) ----------------------------------
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
    # chest / shoulders
    (L_SHOULDER, R_SHOULDER),
    # arms
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    # hands: wrist to thumb/index/pinky
    (L_WRIST, L_THUMB),
    (L_WRIST, L_INDEX),
    (L_WRIST, L_PINKY),
    (R_WRIST, R_THUMB),
    (R_WRIST, R_INDEX),
    (R_WRIST, R_PINKY),
]


# --- Helpers ------------------------------------------------------------------
def lowpass_ema(prev, curr, alpha: float):
    if prev is None or prev.shape != curr.shape:
        return curr.copy()
    return alpha * curr + (1.0 - alpha) * prev


def to_px(p, W, H):
    x = int(round(float(p[0]) * W))
    y = int(round(float(p[1]) * H))
    return x, y


def draw_upper_body(frame_bgr, pts, circle_radius=2, thickness=2):
    """Draw chest/arms/hands + mid-shoulder chest point."""
    H, W = frame_bgr.shape[:2]

    ls = pts[L_SHOULDER]
    rs = pts[R_SHOULDER]
    mid = (ls + rs) / 2.0

    mx, my = to_px(mid, W, H)
    lsh_x, lsh_y = to_px(ls, W, H)
    rsh_x, rsh_y = to_px(rs, W, H)

    # chest point + lines to shoulders
    cv2.line(
        frame_bgr, (mx, my), (lsh_x, lsh_y), (255, 255, 255), thickness, cv2.LINE_AA
    )
    cv2.line(
        frame_bgr, (mx, my), (rsh_x, rsh_y), (255, 255, 255), thickness, cv2.LINE_AA
    )
    cv2.circle(frame_bgr, (mx, my), circle_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)

    # lines for upper body + hands
    for a, b in UPPER_LINES:
        pa = pts[a]
        pb = pts[b]
        xa, ya = to_px(pa, W, H)
        xb, yb = to_px(pb, W, H)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), (255, 255, 255), thickness, cv2.LINE_AA)

    # joints
    for idx in UPPER_IDS:
        x, y = to_px(pts[idx], W, H)
        cv2.circle(frame_bgr, (x, y), circle_radius, (255, 255, 255), -1, cv2.LINE_AA)


def draw_full_pose(frame_bgr, lmlist, connections, circle_radius=2, thickness=2):
    du = mp.solutions.drawing_utils
    spec = du.DrawingSpec(thickness=thickness, circle_radius=circle_radius)
    du.draw_landmarks(
        frame_bgr,
        lmlist,
        connections,
        landmark_drawing_spec=spec,
        connection_drawing_spec=spec,
    )


def mp_image_from_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def make_norm_landmark_list_from_pts(pts):
    from mediapipe.framework.formats import landmark_pb2

    return landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(
                x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=float(p[3])
            )
            for p in pts
        ]
    )


# --- Main ---------------------------------------------------------------------
def main():
    if mp is None:
        raise SystemExit(
            f"Mediapipe import failed: {_mp_import_err}\nInstall with: pip install mediapipe"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Video paths (if omitted, a file dialog will let you choose multiple).",
    )
    args = parser.parse_args()

    if args.videos:
        video_paths = [str(p) for p in args.videos]
    else:
        video_paths = pick_files_dialog_multi()

    if not video_paths:
        raise SystemExit("No videos chosen. Pass --videos or pick files in the dialog.")

    # Open all videos
    caps = []
    names = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"[WARN] Failed to open {p}")
            continue
        caps.append(cap)
        names.append(Path(p).name)

    if not caps:
        raise SystemExit("None of the videos could be opened.")

    num_videos = len(caps)
    print(f"[INFO] Loaded {num_videos} videos.")

    # Grid layout
    MAX_CELL_W = 320
    MAX_CELL_H = 240
    cols = int(math.ceil(math.sqrt(num_videos)))
    rows = int(math.ceil(num_videos / cols))
    CELL_W, CELL_H = MAX_CELL_W, MAX_CELL_H
    MOSAIC_W = cols * CELL_W
    MOSAIC_H = rows * CELL_H

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, MOSAIC_W, MOSAIC_H)

    # Trackbars ---------------------------------------------------------------
    def add_tb(name, maxval, init=0):
        cv2.createTrackbar(name, WIN_NAME, init, maxval, lambda v: None)

    add_tb("model_variant", 2, 1)  # 0=lite,1=full,2=heavy; default full
    add_tb("upper_body_only", 1, 1)  # 1 = upper body + hands only
    add_tb("min_pose_detection_x100", 100, 60)  # 0.60
    add_tb("min_pose_presence_x100", 100, 50)  # 0.50
    add_tb("min_tracking_x100", 100, 70)  # 0.70
    add_tb("output_seg_masks", 1, 0)  # off
    add_tb("EMA_alpha_x100", 100, 25)  # 0.25
    add_tb("circle_radius", 10, 2)
    add_tb("thickness", 10, 2)

    # State -------------------------------------------------------------------
    tasks = [None] * num_videos
    prev_smoothed = [None] * num_videos
    done = [False] * num_videos
    last_frames = [None] * num_videos

    script_dir = Path(__file__).resolve().parent
    model_files = {
        0: script_dir / "pose_landmarker_lite.task",
        1: script_dir / "pose_landmarker_full.task",
        2: script_dir / "pose_landmarker_heavy.task",
    }

    last_conf_tuple = None
    paused = False
    frame_idx = 0
    t_last = time.time()
    fps_smooth = 0.0

    def build_task(conf_tuple):
        mv, det_pos, pres_pos, trk_pos, seg_pos = conf_tuple
        model_path = model_files.get(mv)
        if not model_path or not model_path.exists():
            raise SystemExit(
                f"Model file not found for variant {mv} at: {model_path}\n"
                f"Expected .task files in same folder as this script."
            )
        base = BaseOptions(model_asset_path=str(model_path))
        options = PoseLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det_pos / 100.0,
            min_pose_presence_confidence=pres_pos / 100.0,
            min_tracking_confidence=trk_pos / 100.0,
            output_segmentation_masks=bool(seg_pos),
        )
        return PoseLandmarker.create_from_options(options)

    try:
        while True:
            # Read slider values
            model_variant = cv2.getTrackbarPos("model_variant", WIN_NAME)
            det_pos = cv2.getTrackbarPos("min_pose_detection_x100", WIN_NAME)
            pres_pos = cv2.getTrackbarPos("min_pose_presence_x100", WIN_NAME)
            trk_pos = cv2.getTrackbarPos("min_tracking_x100", WIN_NAME)
            seg_pos = cv2.getTrackbarPos("output_seg_masks", WIN_NAME)
            ema_pos = cv2.getTrackbarPos("EMA_alpha_x100", WIN_NAME)
            circle_radius = max(1, cv2.getTrackbarPos("circle_radius", WIN_NAME))
            thickness = max(1, cv2.getTrackbarPos("thickness", WIN_NAME))
            upper_only = bool(cv2.getTrackbarPos("upper_body_only", WIN_NAME))

            conf_tuple = (model_variant, det_pos, pres_pos, trk_pos, seg_pos)
            if conf_tuple != last_conf_tuple:
                # Recreate all tasks with new settings
                for t in tasks:
                    if t is not None:
                        t.close()
                tasks = [build_task(conf_tuple) for _ in range(num_videos)]
                last_conf_tuple = conf_tuple
                prev_smoothed = [None] * num_videos
                print("[INFO] Rebuilt tasks with new settings")

            # Read frames
            if not paused:
                for i, cap in enumerate(caps):
                    if done[i]:
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        done[i] = True
                        last_frames[i] = None
                    else:
                        last_frames[i] = frame
                frame_idx += 1

            if all(done) and not any(f is not None for f in last_frames):
                print("[INFO] All videos finished.")
                break

            # Create mosaic
            mosaic = np.zeros((MOSAIC_H, MOSAIC_W, 3), dtype=np.uint8)

            for i in range(num_videos):
                # choose frame
                src = last_frames[i]
                if src is None:
                    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
                    # label as finished
                    label = f"{i+1}: (done)"
                    cv2.putText(
                        cell,
                        label,
                        (10, CELL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cell = cv2.resize(
                        src, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR
                    )
                    t = tasks[i]
                    if t is not None:
                        mp_img = mp_image_from_bgr(cell)
                        ts_ms = int(caps[i].get(cv2.CAP_PROP_POS_MSEC))
                        res = t.detect_for_video(mp_img, ts_ms)

                        # segmentation per cell
                        if seg_pos and getattr(res, "segmentation_masks", None):
                            mask = res.segmentation_masks[0].numpy_view()
                            mask3 = np.dstack([mask] * 3)
                            overlay = (
                                cell * 0.3 + np.where(mask3 > 0.2, 255, 0) * 0.2
                            ).astype(np.uint8)
                            cell = cv2.addWeighted(cell, 0.7, overlay, 0.3, 0)

                        # pose
                        if res.pose_landmarks:
                            lms = res.pose_landmarks[0]
                            pts = np.array(
                                [
                                    [lm.x, lm.y, lm.z, getattr(lm, "presence", 1.0)]
                                    for lm in lms
                                ],
                                dtype=np.float32,
                            )

                            alpha = ema_pos / 100.0
                            if alpha > 0.0:
                                pts = lowpass_ema(prev_smoothed[i], pts, alpha)
                                prev_smoothed[i] = pts
                            else:
                                prev_smoothed[i] = None

                            if upper_only:
                                draw_upper_body(
                                    cell,
                                    pts,
                                    circle_radius=circle_radius,
                                    thickness=thickness,
                                )
                            else:
                                lmlist = make_norm_landmark_list_from_pts(pts)
                                draw_full_pose(
                                    cell,
                                    lmlist,
                                    mp.solutions.pose.POSE_CONNECTIONS,
                                    circle_radius=circle_radius,
                                    thickness=thickness,
                                )

                    # add small label (index + file name) at top left of the cell
                    label = f"{i+1}: {names[i][:20]}"
                    cv2.putText(
                        cell,
                        label,
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        cell,
                        label,
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # place cell into mosaic
                r = i // cols
                c = i % cols
                y0 = r * CELL_H
                x0 = c * CELL_W
                mosaic[y0 : y0 + CELL_H, x0 : x0 + CELL_W] = cell

            # FPS + HUD
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                fps_curr = 1.0 / dt
                fps_smooth = (
                    0.9 * fps_smooth + 0.1 * fps_curr if fps_smooth > 0 else fps_curr
                )

            det = det_pos / 100.0
            prs = pres_pos / 100.0
            trk = trk_pos / 100.0
            seg = bool(seg_pos)
            ema = ema_pos / 100.0
            model_name = ["lite", "full", "heavy"][max(0, min(2, model_variant))]

            hud_lines = [
                f"FPS: {fps_smooth:5.1f}  model={model_name}  videos={num_videos}",
                "SLIDERS (top -> bottom):",
                f"1 model_variant  (0=lite,1=full,2=heavy): {model_variant}",
                f"2 upper_body_only (0=full body,1=upper) : {int(upper_only)}",
                f"3 min_pose_detection                  : {det:.2f}",
                f"4 min_pose_presence                   : {prs:.2f}",
                f"5 min_tracking                        : {trk:.2f}",
                f"6 output_seg_masks                    : {int(seg)}",
                f"7 EMA_alpha                           : {ema:.2f}",
                f"8 circle_radius                       : {circle_radius}",
                f"9 thickness                           : {thickness}",
                "Keys: q=quit  SPACE=pause  s=snap  e=export-settings",
            ]

            y = 30
            for line in hud_lines:
                cv2.putText(
                    mosaic,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    mosaic,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 26

            cv2.imshow(WIN_NAME, mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                snap_path = Path(f"mosaic_snapshot_{frame_idx:06d}.png")
                cv2.imwrite(str(snap_path), mosaic)
                print(f"[Saved] {snap_path}")
            elif key == ord("e"):
                exported = {
                    "model_variant": model_name,
                    "min_pose_detection_confidence": det,
                    "min_pose_presence_confidence": prs,
                    "min_tracking_confidence": trk,
                    "output_segmentation_masks": seg,
                    "ema_alpha": ema,
                    "upper_body_only": bool(upper_only),
                }
                with open("pose_tuner_settings.json", "w") as f:
                    json.dump(exported, f, indent=2)
                print("[Exported] pose_tuner_settings.json")

    finally:
        for t in tasks:
            if t is not None:
                t.close()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
