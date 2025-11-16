#!/usr/bin/env python3
import argparse
import subprocess
import time
import sys
from pathlib import Path

import cv2

# --- GUI / file picker ---
try:
    import tkinter as tk
    from tkinter import filedialog

    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


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


def pick_output_dir():
    """Choose output folder via dialog; fall back to ./reps_out if cancelled or no Tk."""
    if not TK_AVAILABLE:
        out = Path("reps_out")
        out.mkdir(parents=True, exist_ok=True)
        return out

    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory(title="Select output folder for rep clips")
    root.update()
    root.destroy()

    if not dirname:
        out = Path("reps_out")
        print("No output folder selected; using ./reps_out")
    else:
        out = Path(dirname)

    out.mkdir(parents=True, exist_ok=True)
    return out


def cut(src, dst, s_ms, e_ms):
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{s_ms/1000:.3f}",
            "-to",
            f"{e_ms/1000:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            str(dst),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def process_one(
    video: Path, out_dir: Path, pad_ms: int, init_speed: float, is_last: bool
):
    if not TK_AVAILABLE or not PIL_AVAILABLE:
        print("tkinter and pillow are required for this UI.")
        return 0, False

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = 1.0 / (fps if fps > 0 else 30.0)
    duration_ms = (
        (n_frames / fps) * 1000 if (fps > 0 and n_frames > 0) else float("inf")
    )

    def get_time_ms():
        return cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_frame_idx():
        return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek_ms(t_ms):
        t_ms = clamp(
            t_ms, 0, duration_ms if duration_ms != float("inf") else max(0, t_ms)
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame = cap.read()
        return ok, frame

    def seek_frame(idx):
        idx = max(0, min(idx, max(n_frames - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    # Prime first frame so we know video size
    ok, first = cap.read()
    if not ok:
        cap.release()
        print(f"{video.name}: empty/unreadable?")
        return 0, False

    frame_h, frame_w = first.shape[:2]
    MAX_W, MAX_H = 960, 540  # display size cap
    scale = min(MAX_W / frame_w, MAX_H / frame_h, 1.0)
    disp_w, disp_h = int(frame_w * scale), int(frame_h * scale)

    times = []  # marked bottoms in ms

    # --- Tk window: video on top, controls below ---
    root = tk.Tk()
    root.title(f"Bench Rep Marker – {video.name}")

    # Grid layout: row 0 = video, row 1 = scrub, row 2 = buttons, row 3 = speed
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    video_label = tk.Label(root, bg="black")
    video_label.grid(row=0, column=0, sticky="nsew")

    # Scrubber
    scrub_var = tk.IntVar(value=0)
    scrub = tk.Scale(
        root,
        from_=0,
        to=max(n_frames - 1, 0),
        orient="horizontal",
        variable=scrub_var,
        showvalue=False,
        length=800,
        label="Scrub",
    )
    scrub.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    # Buttons row
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

    # Speed slider
    speed_var = tk.DoubleVar(value=init_speed)
    speed_scale = tk.Scale(
        root,
        from_=0.1,
        to=3.0,
        resolution=0.1,
        orient="horizontal",
        label="Playback speed (x)",
        variable=speed_var,
        length=300,
    )
    speed_scale.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

    # Playback state
    state = {
        "playing": True,
        "speed": init_speed,
        "last_frame": first,
        "quit_video": False,  # finish this video
        "abort_all": False,  # stop processing remaining videos
        "next_frame_time": time.perf_counter() + frame_interval / max(0.1, init_speed),
        "scrubbing": False,
    }

    def on_speed(val):
        state["speed"] = max(0.1, float(val))
        state["next_frame_time"] = time.perf_counter() + frame_interval / state["speed"]

    speed_scale.config(command=on_speed)

    # Scrubber events
    def scrub_press(event):
        state["scrubbing"] = True

    def scrub_release(event):
        state["scrubbing"] = False
        idx = scrub_var.get()
        state["playing"] = False
        ok2, frame2 = seek_frame(idx)
        if ok2:
            state["last_frame"] = frame2
        state["next_frame_time"] = time.perf_counter() + frame_interval / max(
            0.1, state["speed"]
        )

    scrub.bind("<ButtonPress-1>", scrub_press)
    scrub.bind("<ButtonRelease-1>", scrub_release)

    # Playback controls
    def play(event=None):
        state["playing"] = True
        state["next_frame_time"] = time.perf_counter() + frame_interval / max(
            0.1, state["speed"]
        )

    def pause(event=None):
        state["playing"] = False

    def toggle_play(event=None):
        if state["playing"]:
            pause()
        else:
            play()

    def mark(event=None):
        t_ms_now = get_time_ms()
        times.append(t_ms_now)

    def step_frames(delta):
        state["playing"] = False
        idx = get_frame_idx() + delta
        ok2, frame2 = seek_frame(idx)
        if ok2:
            state["last_frame"] = frame2
        if n_frames > 0 and not state["scrubbing"]:
            try:
                scrub_var.set(get_frame_idx())
            except tk.TclError:
                # Widget can disappear while window tears down; harmless to skip
                pass

    def jump_ms(delta_ms):
        state["playing"] = False
        cur = get_time_ms()
        ok2, frame2 = seek_ms(cur + delta_ms)
        if ok2:
            state["last_frame"] = frame2
        if n_frames > 0 and not state["scrubbing"]:
            try:
                scrub_var.set(get_frame_idx())
            except tk.TclError:
                # Widget can disappear while window tears down; harmless to skip
                pass

    def slower(event=None):
        new = max(0.1, speed_var.get() * 0.8)
        speed_var.set(new)

    def faster(event=None):
        new = min(3.0, speed_var.get() * 1.25)
        speed_var.set(new)

    def done_this_video(event=None):
        state["quit_video"] = True

    def quit_all(event=None):
        state["quit_video"] = True
        state["abort_all"] = True

    # Buttons
    tk.Button(btn_frame, text="Play", command=play).grid(
        row=0, column=0, padx=2, pady=2
    )
    tk.Button(btn_frame, text="Pause", command=pause).grid(
        row=0, column=1, padx=2, pady=2
    )

    tk.Button(btn_frame, text="<<1f", command=lambda: step_frames(-1)).grid(
        row=0, column=2, padx=2, pady=2
    )
    tk.Button(btn_frame, text=">>1f", command=lambda: step_frames(+1)).grid(
        row=0, column=3, padx=2, pady=2
    )

    tk.Button(btn_frame, text="-0.1s", command=lambda: jump_ms(-100)).grid(
        row=0, column=4, padx=2, pady=2
    )
    tk.Button(btn_frame, text="+0.1s", command=lambda: jump_ms(+100)).grid(
        row=0, column=5, padx=2, pady=2
    )

    tk.Button(btn_frame, text="-0.5s", command=lambda: jump_ms(-500)).grid(
        row=0, column=6, padx=2, pady=2
    )
    tk.Button(btn_frame, text="+0.5s", command=lambda: jump_ms(+500)).grid(
        row=0, column=7, padx=2, pady=2
    )

    tk.Button(btn_frame, text="-1s", command=lambda: jump_ms(-1000)).grid(
        row=0, column=8, padx=2, pady=2
    )
    tk.Button(btn_frame, text="+1s", command=lambda: jump_ms(+1000)).grid(
        row=0, column=9, padx=2, pady=2
    )

    tk.Button(btn_frame, text="Mark (SPACE)", command=mark).grid(
        row=0, column=10, padx=2, pady=2
    )

    # Next / Finish button
    next_label = "Finish" if is_last else "Next video"
    tk.Button(btn_frame, text=next_label, command=done_this_video).grid(
        row=0, column=11, padx=2, pady=2
    )

    # Optional: Quit-all button if user wants to stop entire batch early
    tk.Button(btn_frame, text="Quit all", fg="red", command=quit_all).grid(
        row=0, column=12, padx=2, pady=2
    )

    # Keyboard shortcuts on this same window
    root.bind("<space>", mark)
    root.bind("<Key-p>", toggle_play)
    root.bind("<Key-k>", toggle_play)
    root.bind("<Key-j>", lambda e: jump_ms(-500))
    root.bind("<Key-l>", lambda e: jump_ms(+500))
    root.bind("<Key-comma>", lambda e: step_frames(-1))
    root.bind("<Key-period>", lambda e: step_frames(+1))
    root.bind("<Key-bracketleft>", slower)
    root.bind("<Key-bracketright>", faster)
    root.bind("<Left>", lambda e: jump_ms(-200))
    root.bind("<Right>", lambda e: jump_ms(+200))
    root.bind("<Key-n>", done_this_video)  # N = next video
    root.bind("<Key-q>", quit_all)
    root.protocol("WM_DELETE_WINDOW", quit_all)

    root.focus_set()

    # --- main update loop ---
    def update():
        if state["quit_video"]:
            cap.release()
            root.destroy()
            return

        now = time.perf_counter()
        if state["playing"] and now >= state["next_frame_time"]:
            ok2, frame2 = cap.read()
            if ok2:
                state["last_frame"] = frame2
                state["next_frame_time"] = now + frame_interval / max(
                    0.1, state["speed"]
                )
            else:
                state["playing"] = False

        frame = state["last_frame"]
        if frame is not None:
            disp = frame.copy()
            hud = (
                f"[{video.name}] SPACE=mark  "
                f"play/pause, frame +/- , jumps 0.1/0.5/1.0s, "
                f"speed={state['speed']:.2f}x  marks={len(times)}"
            )
            cv2.putText(
                disp, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            if times:
                last_s = times[-1] / 1000.0
                cv2.putText(
                    disp,
                    f"last mark: {last_s:.2f}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (200, 255, 200),
                    2,
                )

            # scale to fit window max without cropping
            if scale != 1.0:
                disp = cv2.resize(disp, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk  # prevent GC
            video_label.configure(image=imgtk)

        # Keep scrubber in sync (unless user is dragging)
        if n_frames > 0 and not state["scrubbing"]:
            try:
                scrub_var.set(get_frame_idx())
            except tk.TclError:
                # Widget can disappear while window tears down; harmless to skip
                pass

        root.after(10, update)

    update()
    root.mainloop()

    # If user chose "Quit all" or closed window via X
    if state["abort_all"]:
        return 0, True

    # --- export clips for this video ---
    reps = []
    for i in range(len(times) - 1):
        s = max(0, times[i] - pad_ms)
        e = times[i + 1] + pad_ms
        if e - s > 200:
            reps.append((s, e))

    stem = video.stem
    for k, (s, e) in enumerate(reps, 1):
        out = out_dir / f"{stem}_rep{k:02d}.mp4"
        cut(video, out, s, e)

    print(
        f"{video.name}: marked {len(times)} bottoms -> {len(reps)} rep clips -> {out_dir}"
    )
    return len(reps), False


def main():
    ap = argparse.ArgumentParser(
        description="Mark bench rep bottoms and auto-split clips."
    )
    ap.add_argument(
        "--video", type=Path, help="Optional path to a single video (skips file picker)"
    )
    ap.add_argument(
        "--out_dir", type=Path, help="Output directory (omit to choose via dialog)"
    )
    ap.add_argument(
        "--pad_ms", type=int, default=120, help="Padding before/after each rep"
    )
    ap.add_argument("--speed", type=float, default=1.0, help="Initial playback speed")
    ap.add_argument(
        "--no_multi", action="store_true", help="Picker selects only one file"
    )
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

    # choose output directory
    if args.out_dir:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pick_output_dir()

    total_clips = 0
    abort_all = False
    n = len(videos)
    for idx, v in enumerate(videos):
        is_last = idx == n - 1
        clips, abort_all = process_one(v, out_dir, args.pad_ms, args.speed, is_last)
        total_clips += clips
        if abort_all:
            break

    print(f"Done. Total clips exported: {total_clips}")


if __name__ == "__main__":
    main()
