#!/usr/bin/env python3
import argparse, subprocess, time, sys
from pathlib import Path
import cv2

# --- file picker (tk) ---
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

def pick_files(allow_multiple=True):
    if not TK_AVAILABLE:
        return []
    root = tk.Tk()
    root.withdraw()
    opts = {
        "title": "Select bench set video(s)",
        "filetypes": [
            ("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm"),
            ("All files", "*.*"),
        ],
    }
    if allow_multiple:
        files = filedialog.askopenfilenames(**opts)
    else:
        f = filedialog.askopenfilename(**opts)
        files = [f] if f else []
    root.update()
    root.destroy()
    return [Path(f) for f in files]

def cut(src, dst, s_ms, e_ms):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg","-y","-ss",f"{s_ms/1000:.3f}","-to",f"{e_ms/1000:.3f}",
        "-i",str(src),"-c","copy",str(dst)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def process_one(video: Path, out_dir: Path, pad_ms: int, speed: float):
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f"Cannot open {video}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = (n_frames / fps) * 1000 if (fps > 0 and n_frames > 0) else float("inf")

    def get_time_ms():
        return cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_frame_idx():
        return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek_ms(t_ms):
        t_ms = clamp(t_ms, 0, duration_ms if duration_ms != float("inf") else max(0, t_ms))
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        # read one frame to land cleanly on the seek target
        ok, frame = cap.read()
        return ok, frame

    def seek_frame(idx):
        idx = max(0, idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    times = []          # marked bottoms (ms)
    paused = False
    t0_ms = None
    wall0 = None
    last_frame = None

    # initial read so we have a frame even if paused immediately
    ok, frame = cap.read()
    if not ok:
        cap.release()
        print(f"{video.name}: empty/unreadable?")
        return 0

    while True:
        if not paused:
            t_ms = get_time_ms()
            if t0_ms is None:
                t0_ms = t_ms
                wall0 = time.perf_counter()

            # pace playback to timestamps
            target_elapsed = max(0.0, (t_ms - t0_ms) / 1000.0) / max(1e-6, speed)
            now_elapsed = time.perf_counter() - wall0
            wait = target_elapsed - now_elapsed
            if wait > 0:
                time.sleep(min(wait, 0.02))
            disp = frame.copy()
            last_frame = disp
            ok, next_frame = cap.read()
            if ok:
                frame = next_frame
            else:
                # reached end
                paused = True
                disp = last_frame
        else:
            # show current frame without advancing
            disp = last_frame if last_frame is not None else frame

        # HUD
        hud = (
            f"[{video.name}] SPACE=mark  U=undo  P/K=pause  ←/→=±0.2s  J/L=±0.5s  "
            f",/.=frame± (paused)  [ / ] speed  Q=finish  "
            f"speed={speed:.2f}x  marks={len(times)}"
        )
        cv2.putText(disp, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # show last mark time
        if times:
            last_s = times[-1] / 1000.0
            cv2.putText(disp, f"last mark: {last_s:.2f}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)

        cv2.imshow("Mark bottoms", disp)
        k = cv2.waitKey(1) & 0xFF

        # --- hotkeys ---
        if k == ord(' '):                       # mark bottom
            t_ms_now = get_time_ms()
            times.append(t_ms_now)
        elif k in (ord('p'), ord('k')):         # pause/play
            paused = not paused
            # reset pacing baseline when resuming
            if not paused:
                t0_ms = get_time_ms()
                wall0 = time.perf_counter()
        elif k == ord('u'):                     # undo last mark
            if times: times.pop()

        elif k in (81,):                        # LEFT arrow → rewind 0.2s
            paused = True
            ok, frame = seek_ms(get_time_ms() - 200)
            if ok: last_frame = frame
        elif k in (83,):                        # RIGHT arrow → forward 0.2s
            paused = True
            ok, frame = seek_ms(get_time_ms() + 200)
            if ok: last_frame = frame
        elif k == ord('j'):                     # J → rewind 0.5s
            paused = True
            ok, frame = seek_ms(get_time_ms() - 500)
            if ok: last_frame = frame
        elif k == ord('l'):                     # L → forward 0.5s
            paused = True
            ok, frame = seek_ms(get_time_ms() + 500)
            if ok: last_frame = frame

        elif k == ord(','):                     # single-frame back (paused)
            paused = True
            idx = get_frame_idx()
            ok, frame = seek_frame(idx - 2)     # -2 because read() advances one
            if ok: last_frame = frame
        elif k == ord('.'):                     # single-frame forward (paused)
            paused = True
            ok, frame = seek_frame(get_frame_idx())  # next frame
            if ok: last_frame = frame

        elif k == ord('['):                     # slower
            speed = max(0.05, speed * 0.8)
        elif k == ord(']'):                     # faster
            speed = min(8.0, speed * 1.25)

        elif k == ord('q') or k == 27:          # q or ESC → finish
            break

    cap.release()
    cv2.destroyAllWindows()

    # Build rep windows bottom->bottom with padding
    reps = []
    for i in range(len(times)-1):
        s = max(0, times[i] - pad_ms)
        e = times[i+1] + pad_ms
        if e - s > 200:
            reps.append((s, e))

    stem = video.stem
    for k,(s,e) in enumerate(reps,1):
        out = out_dir / f"{stem}_rep{k:02d}.mp4"
        cut(video, out, s, e)

    print(f"{video.name}: marked {len(times)} bottoms → {len(reps)} rep clips → {out_dir}")
    return len(reps)

def main():
    ap = argparse.ArgumentParser(description="Mark bench rep bottoms and auto-split clips.")
    ap.add_argument("--video", type=Path, help="Optional path to a single video (skips file picker)")
    ap.add_argument("--out_dir", type=Path, default=Path("reps_out"), help="Output directory")
    ap.add_argument("--pad_ms", type=int, default=120, help="Padding before/after each rep")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0=real-time)")
    ap.add_argument("--no_multi", action="store_true", help="Picker selects only one file")
    args = ap.parse_args()

    # gather videos
    if args.video:
        videos = [args.video]
    else:
        if not TK_AVAILABLE:
            print("tkinter not available; pass --video PATH instead.")
            sys.exit(1)
        videos = pick_files(allow_multiple=(not args.no_multi))
        if not videos:
            print("No file selected.")
            sys.exit(0)

    # process sequentially
    total_clips = 0
    for v in videos:
        total_clips += process_one(v, args.out_dir, args.pad_ms, args.speed)

    print(f"Done. Total clips exported: {total_clips}")

if __name__ == "__main__":
    main()
