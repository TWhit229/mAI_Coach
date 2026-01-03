"""Video cutting interface."""

import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.video import video_rotation_degrees, rotate_frame_if_needed
# We need _rotation_value_from_index helper, maybe define it here or in widgets or labeler?
# It was in labeler.py, I should probably put it in core/utils or widgets if used in multiple places.
# For now, I'll copy it here or use a shared one if I moved it.
# I'll check ui/labeler.py to see if I can import it.
# Actually I put it in ui/labeler.py but not as a class method, just a helper.
# It's better to duplicate or move to shared. I'll duplicate for isolation for now.

# Also need ROTATION_OPTIONS.

ROTATION_OPTIONS: List[Tuple[str, Optional[int]]] = [
    ("Auto (metadata)", None),
    ("0°", 0),
    ("90° CW", 90),
    ("180°", 180),
    ("270° CCW", 270),
]

def _rotation_option_index(degrees: Optional[int]) -> int:
    if degrees is None:
        return 0
    degrees %= 360
    for i, (_, val) in enumerate(ROTATION_OPTIONS):
        if val == degrees:
            return i
    return 0

def _rotation_value_from_index(index: int) -> Optional[int]:
    if 0 <= index < len(ROTATION_OPTIONS):
        return ROTATION_OPTIONS[index][1]
    return None


class VideoCutView(QtWidgets.QWidget):
    TARGET_HEIGHT = 720
    TARGET_FPS = 30

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        self.videos: List[Path] = []
        self.current_index = -1
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._last_capture_index = -1
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.play_timer.timeout.connect(self._advance_frame)
        self.play_timer.start(30)
        self.cuts: Dict[Path, List[tuple[int, int]]] = {}
        self.output_dir: Optional[Path] = None
        self.next_clip_start_ms: Dict[Path, float] = {}
        self.split_overlap_ms = 100  # milliseconds of overlap between clips
        self.rotation_overrides: Dict[Path, Optional[int]] = {}
        self.rotation_lock_value: Optional[int] = None
        self.current_rotation_override: Optional[int] = None
        self.current_rotation = 0

        self._build_ui()
        self._change_speed("1.0x")
        self._update_nav_buttons()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Video Cutter")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        top.addWidget(title)
        top.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top.addWidget(self.home_button)
        root.addLayout(top)

        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, stretch=1)

        # Left: video and controls
        left = QtWidgets.QVBoxLayout()
        content.addLayout(left, stretch=3)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #333;")
        left.addWidget(self.video_label, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        left.addLayout(controls)

        def ctrl_btn(text, slot):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        ctrl_btn("⟵ Frame", lambda: self._step_frames(-1))
        ctrl_btn("-0.5s", lambda: self._step_seconds(-0.5))
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        ctrl_btn("Replay", self._replay)
        ctrl_btn("+0.5s", lambda: self._step_seconds(0.5))
        ctrl_btn("Frame ⟶", lambda: self._step_frames(1))

        controls.addStretch(1)
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_box)

        mark_row = QtWidgets.QHBoxLayout()
        left.addLayout(mark_row)
        self.split_btn = QtWidgets.QPushButton("Mark Split")
        self.split_btn.setToolTip(
            "End current clip at the playhead and start the next one with 0.1s overlap."
        )
        self.split_btn.clicked.connect(self._mark_split)
        self.split_btn.setEnabled(False)
        mark_row.addWidget(self.split_btn)
        mark_row.addStretch(1)

        rotation_row = QtWidgets.QHBoxLayout()
        left.addLayout(rotation_row)
        rotation_row.addWidget(QtWidgets.QLabel("Rotation:"))
        self.rotation_combo = QtWidgets.QComboBox()
        for label, _ in ROTATION_OPTIONS:
            self.rotation_combo.addItem(label)
        self.rotation_combo.currentIndexChanged.connect(
            self._on_cutter_rotation_changed
        )
        self.rotation_combo.setToolTip(
            "Override the displayed/exported orientation for this clip."
        )
        rotation_row.addWidget(self.rotation_combo)
        self.rotation_lock_cb = QtWidgets.QCheckBox("Keep for all videos")
        self.rotation_lock_cb.setToolTip(
            "When checked, the selected rotation is applied automatically to every new video."
        )
        self.rotation_lock_cb.stateChanged.connect(
            self._on_cutter_rotation_lock_toggled
        )
        rotation_row.addWidget(self.rotation_lock_cb)
        rotation_row.addStretch(1)

        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        left.addWidget(self.scrubber)

        self.status_label = QtWidgets.QLabel("")
        left.addWidget(self.status_label)

        # Right side: cuts list and options
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        content.addWidget(right_scroll, stretch=2)
        right_widget = QtWidgets.QWidget()
        right_scroll.setWidget(right_widget)
        right = QtWidgets.QVBoxLayout(right_widget)

        clips_header = QtWidgets.QHBoxLayout()
        clips_header.addWidget(QtWidgets.QLabel("Marked clips"))
        clips_header.addStretch(1)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.setToolTip("Remove the highlighted clip from the list.")
        remove_btn.clicked.connect(self._remove_selected_cut)
        clear_btn = QtWidgets.QPushButton("Clear clips")
        clear_btn.setToolTip("Delete all clips for this video.")
        clear_btn.clicked.connect(self._clear_cuts)
        clips_header.addWidget(remove_btn)
        clips_header.addWidget(clear_btn)
        right.addLayout(clips_header)

        self.cut_list = QtWidgets.QListWidget()
        right.addWidget(self.cut_list, stretch=1)

        pad_row = QtWidgets.QHBoxLayout()
        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 2000)
        self.pad_spin.setValue(120)
        self.pad_spin.setSuffix(" ms pad")
        pad_row.addWidget(QtWidgets.QLabel("Padding:"))
        pad_row.addWidget(self.pad_spin)
        right.addLayout(pad_row)

        nav = QtWidgets.QHBoxLayout()
        self.prev_video_btn = QtWidgets.QPushButton("Previous Video")
        self.prev_video_btn.clicked.connect(lambda: self._load_relative(-1))
        nav.addWidget(self.prev_video_btn)
        nav.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save Clips")
        self.save_btn.clicked.connect(self._save_current_video)
        nav.addWidget(self.save_btn)
        self.save_next_btn = QtWidgets.QPushButton("Save & Next")
        self.save_next_btn.clicked.connect(self._save_and_advance)
        nav.addWidget(self.save_next_btn)
        right.addLayout(nav)

    def _release_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._last_capture_index = -1

    def _count_frames(self, path: Path) -> int:
        temp = cv2.VideoCapture(str(path))
        if not temp.isOpened():
            return 0
        count = 0
        while True:
            ok, _ = temp.read()
            if not ok:
                break
            count += 1
        temp.release()
        return count

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if not self.cap or self.frame_count <= 0:
            return None
        idx = max(0, min(idx, self.frame_count - 1))
        if idx != self._last_capture_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last_capture_index = idx
        rotation = (
            self.current_rotation_override
            if self.current_rotation_override is not None
            else self.current_rotation
        )
        if rotation:
            frame = rotate_frame_if_needed(frame, rotation)
        return frame

    def _on_cutter_rotation_changed(self, state: int):
        path = self._current_video_path()
        if not path:
            return
        value = _rotation_value_from_index(self.rotation_combo.currentIndex())
        self.rotation_overrides[path] = value
        self.current_rotation_override = value
        if self.rotation_lock_cb.isChecked():
            self.rotation_lock_value = value
        self._render_frame(self.current_frame)

    def _on_cutter_rotation_lock_toggled(self, state: int):
        if state == QtCore.Qt.Checked:
            self.rotation_lock_value = self.current_rotation_override
            if (
                self.rotation_lock_value is not None
                and self._current_video_path() is not None
            ):
                self.rotation_overrides[self._current_video_path()] = (
                    self.rotation_lock_value
                )
                self.current_rotation_override = self.rotation_lock_value
                self._render_frame(self.current_frame)
        else:
            self.rotation_lock_value = None

    def start_new_session(
        self, videos: Optional[List[Path]] = None, output_dir: Optional[Path] = None
    ) -> bool:
        self.videos = [Path(v) for v in (videos or [])]
        self.output_dir = Path(output_dir) if output_dir else None
        self.current_index = -1
        self._release_capture()
        self.frame_count = 0
        self.rotation_overrides = {}
        self.cuts = {path: [] for path in self.videos}
        self.next_clip_start_ms = {}
        self.cut_list.clear()
        self.video_label.clear()
        self.status_label.setText("Select videos to begin.")
        self.split_btn.setEnabled(False)
        self.scrubber.setEnabled(False)
        self.scrubber.setRange(0, 0)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(0)
        self.rotation_combo.blockSignals(False)
        self.current_rotation_override = None
        self._update_nav_buttons()
        if not self._prompt_for_inputs():
            return False
        if not self.videos:
            return False
        self._load_video(0)
        return True

    def _prompt_for_inputs(self) -> bool:
        if not self.videos and not self._choose_videos():
            return False
        if not self.output_dir and not self._choose_output_folder():
            return False
        return True

    def _choose_videos(self) -> bool:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return False
        paths = [Path(f) for f in files]
        self.videos = paths
        self.cuts = {path: [] for path in self.videos}
        self.next_clip_start_ms = {}
        self.current_index = -1
        self._release_capture()
        self.frame_count = 0
        self.rotation_overrides = {}
        return True

    def _choose_output_folder(self) -> bool:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path.home())
        )
        if not folder:
            return False
        self.output_dir = Path(folder)
        self.status_label.setText(f"Output folder: {self.output_dir}")
        return True

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_cutting_inputs(self, videos: List[Path]):
        self.start_new_session(videos)

    def _load_relative(self, delta: int):
        if not self.videos:
            return
        if self.current_index < 0:
            return
        new_index = max(0, min(self.current_index + delta, len(self.videos) - 1))
        if new_index == self.current_index:
            return
        self._load_video(new_index)

    def _load_video(self, index: int):
        if not self.videos:
            return
        index = max(0, min(index, len(self.videos) - 1))
        if index == self.current_index:
            return
        path = self.videos[index]
        self.cuts.setdefault(path, [])
        self.next_clip_start_ms.setdefault(path, 0.0)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open {path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_frames(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", f"No frames in {path}")
            cap.release()
            return
        fps = fps if fps and fps > 0 else 30.0
        self._release_capture()
        self.cap = cap
        self.current_index = index
        self.fps = fps
        self.frame_count = frame_count
        self.current_frame = 0
        self.current_rotation = video_rotation_degrees(path)
        override = self.rotation_overrides.get(path)
        if (
            override is None
            and self.rotation_lock_cb.isChecked()
            and self.rotation_lock_value is not None
        ):
            override = self.rotation_lock_value
            self.rotation_overrides[path] = override
        self.current_rotation_override = override
        target_idx = _rotation_option_index(override)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(target_idx)
        self.rotation_combo.blockSignals(False)
        self._speed_residual = 0.0
        self.scrubber.setRange(0, max(0, self.frame_count - 1))
        self.scrubber.setEnabled(True)
        self.split_btn.setEnabled(True)
        self.playing = False
        self.play_btn.setText("Play")
        self._update_timer_interval()
        self._render_frame(0)
        self._refresh_cut_list()
        status = f"Loaded {path.name} – {self.frame_count} frames @ {self.fps:.2f} fps"
        if self.output_dir:
            status += f" | Output: {self.output_dir}"
        self.status_label.setText(status)
        self._update_nav_buttons()

    def _render_frame(self, idx: int):
        if not self.cap or self.frame_count <= 0:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        frame = self._read_frame(idx)
        if frame is None:
            return
        # _read_frame already applies rotation override; nothing more to do
        self.current_frame = idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        scaled = image.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(idx)
        self.scrubber.blockSignals(False)

    def _mark_split(self):
        path = self._current_video_path()
        if not path or self.frame_count <= 0:
            return
        current_ms = int(round(self._current_time_ms()))
        start_ms = int(round(self.next_clip_start_ms.get(path, 0.0)))
        if current_ms - start_ms < 50:
            self.status_label.setText("Split point must be after the current clip start.")
            return
        clips = self.cuts.setdefault(path, [])
        clips.append((start_ms, current_ms))
        next_start = max(current_ms - self.split_overlap_ms, 0)
        self.next_clip_start_ms[path] = next_start
        self.status_label.setText(
            f"Split at {current_ms / 1000:.2f}s. Next clip starts at {next_start / 1000:.2f}s"
        )
        self._refresh_cut_list()

    def _refresh_cut_list(self):
        self.cut_list.clear()
        path = self._current_video_path()
        if not path:
            return
        clips = self.cuts.get(path, [])
        for idx, (start, end) in enumerate(clips, 1):
            self.cut_list.addItem(
                f"{idx}. {start / 1000:.2f}s -> {end / 1000:.2f}s (len {(end - start) / 1000:.2f}s)"
            )
        self._update_next_clip_start()

    def _current_video_path(self) -> Optional[Path]:
        if not self.videos or self.current_index < 0:
            return None
        return self.videos[self.current_index]

    def _update_next_clip_start(self):
        path = self._current_video_path()
        if not path:
            return
        clips = self.cuts.get(path) or []
        if clips:
            start = max(float(clips[-1][1]) - self.split_overlap_ms, 0.0)
        else:
            start = 0.0
        self.next_clip_start_ms[path] = start

    def _update_nav_buttons(self):
        has_video = bool(self.videos) and self.current_index >= 0
        can_go_prev = has_video and self.current_index > 0
        total = len(self.videos)
        is_last = has_video and self.current_index >= total - 1
        self.prev_video_btn.setEnabled(can_go_prev)
        self.save_btn.setEnabled(has_video)
        self.save_next_btn.setEnabled(has_video)
        self.save_next_btn.setText("Save & Exit" if is_last else "Save & Next")

    def _remove_selected_cut(self):
        path = self._current_video_path()
        if not path:
            return
        row = self.cut_list.currentRow()
        if row < 0:
            return
        clips = self.cuts.get(path, [])
        if row < len(clips):
            clips.pop(row)
            self._refresh_cut_list()

    def _clear_cuts(self):
        path = self._current_video_path()
        if not path:
            return
        self.cuts[path] = []
        self._refresh_cut_list()

    def _save_current_video(self) -> bool:
        video = self._current_video_path()
        if not video:
            return False
        clips = self.cuts.get(video) or []
        if not clips:
            QtWidgets.QMessageBox.information(
                self, "No clips", "Mark at least one split before saving."
            )
            return False
        if not self.output_dir and not self._choose_output_folder():
            return False
        assert self.output_dir is not None
        pad = self.pad_spin.value()
        success = self._export_video_clips(video, self.output_dir, pad)
        if success:
            self.fps = float(self.TARGET_FPS)
            self.status_label.setText(
                f"Saved {len(clips)} clip(s) for {video.stem} → {self.output_dir}"
            )
        return success

    def _save_and_advance(self):
        if not self._save_current_video():
            return
        if self.current_index >= len(self.videos) - 1:
            self._go_home()
        else:
            self._load_relative(+1)

    def _export_video_clips(self, video: Path, out_dir: Path, pad_ms: int) -> bool:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = self.cuts.get(video, [])
        if not clips:
            return False
        rotation = self._effective_rotation_for_video(video)
        errors = False
        for idx, (start, end) in enumerate(clips, 1):
            s = max(0, start - pad_ms)
            e = end + pad_ms
            stem = video.stem
            out_path = out_dir / f"{stem}_clip{idx:02d}.mp4"
            if not self._run_ffmpeg(video, out_path, s, e, rotation):
                errors = True
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Export errors",
                f"One or more clips failed to export for {video.name}.",
            )
            return False
        QtWidgets.QMessageBox.information(
            self, "Clips saved", f"Exported {len(clips)} clip(s) for {video.name}."
        )
        return True

    def _effective_rotation_for_video(self, video: Path) -> int:
        override = self.rotation_overrides.get(video)
        if override is not None:
            return int(override or 0)
        if video == self._current_video_path():
            return int(self.current_rotation or 0)
        return video_rotation_degrees(video)

    @staticmethod
    def _rotation_filter_chain(rotation: int) -> List[str]:
        chain: List[str] = []
        if rotation == 90:
            chain.append("transpose=1")
        elif rotation == 180:
            chain.append("transpose=1")
            chain.append("transpose=1")
        elif rotation == 270:
            chain.append("transpose=2")
        return chain

    def _run_ffmpeg(
        self, src: Path, dst: Path, start_ms: int, end_ms: int, rotation: int = 0
    ) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotation = rotation or 0
        vf_parts = self._rotation_filter_chain(rotation)
        vf_parts.append(f"scale=-2:{self.TARGET_HEIGHT}")
        vf = ",".join(vf_parts)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-to",
                f"{end_ms / 1000:.3f}",
                "-i",
                str(src),
                "-vf",
                vf,
                "-r",
                str(self.TARGET_FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(dst),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(
                self,
                "FFmpeg not found",
                "FFmpeg is not installed or not available on PATH. "
                "Install ffmpeg and try exporting again.",
            )
            return False

        if result.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "FFmpeg error",
                f"FFmpeg failed to export clip from {src.name}. "
                "Verify the source video and FFmpeg installation.",
            )
            return False
        return True

    def _current_time_ms(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return (self.current_frame / (self.fps if self.fps > 0 else 30.0)) * 1000.0

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frame(self):
        if not (self.playing and self.frame_count > 0 and self.cap):
            return
        step = 1
        if self.playback_speed > 1.0:
            whole = int(self.playback_speed)
            frac = self.playback_speed - whole
            step = max(1, whole)
            self._speed_residual += frac
            if self._speed_residual >= 1.0:
                step += 1
                self._speed_residual -= 1.0
        next_idx = self.current_frame + step
        if next_idx >= self.frame_count:
            next_idx = self.frame_count - 1
            self.playing = False
            self.play_btn.setText("Play")
        self._render_frame(next_idx)

    def _replay(self):
        if self.frame_count <= 0:
            return
        self.playing = True
        self.play_btn.setText("Pause")
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if self.frame_count <= 0:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self._render_frame(self.current_frame + delta)

    def _step_seconds(self, seconds: float):
        fps = self.fps if self.fps > 0 else 30.0
        delta = int(round(seconds * fps))
        if delta == 0:
            delta = 1 if seconds > 0 else -1
        self._step_frames(delta)

    def _change_speed(self, text: str):
        try:
            self.playback_speed = float(text.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._update_timer_interval()

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _scrubbed(self, value: int):
        self._render_frame(value)

    def _update_timer_interval(self):
        fps = self.fps if self.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)
