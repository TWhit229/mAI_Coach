"""Video cutting interface."""

import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.rep_detector import detect_reps_in_frames
from core.video import video_rotation_degrees, rotate_frame_if_needed, run_pose_landmarks_on_video, _pose_model_path, detect_best_rotation
from label_config import default_movement_settings

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
        self._preview_scale = 1.0  # Scale factor for preview (computed on load)
        self._preview_end_frame: Optional[int] = None  # For clip preview auto-stop
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
        # Preview quality settings
        self.PREVIEW_HEIGHT = 480  # Scale to this height for smoother playback

        self._build_ui()
        self._setup_shortcuts()
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

        # Navigation buttons with shortcut hints
        frame_back_btn = QtWidgets.QPushButton("◀ Frame [←]")
        frame_back_btn.clicked.connect(lambda: self._step_frames(-1))
        controls.addWidget(frame_back_btn)
        
        sec_back_btn = QtWidgets.QPushButton("-0.5s [,]")
        sec_back_btn.clicked.connect(lambda: self._step_seconds(-0.5))
        controls.addWidget(sec_back_btn)
        
        self.play_btn = QtWidgets.QPushButton("Play [Space]")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        
        replay_btn = QtWidgets.QPushButton("Replay [R]")
        replay_btn.clicked.connect(self._replay)
        controls.addWidget(replay_btn)
        
        sec_fwd_btn = QtWidgets.QPushButton("+0.5s [.]")
        sec_fwd_btn.clicked.connect(lambda: self._step_seconds(0.5))
        controls.addWidget(sec_fwd_btn)
        
        frame_fwd_btn = QtWidgets.QPushButton("Frame ▶ [→]")
        frame_fwd_btn.clicked.connect(lambda: self._step_frames(1))
        controls.addWidget(frame_fwd_btn)

        controls.addStretch(1)
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_box)

        mark_row = QtWidgets.QHBoxLayout()
        left.addLayout(mark_row)
        self.split_btn = QtWidgets.QPushButton("✂ Mark Split [M]")
        self.split_btn.setToolTip(
            "End current clip at the playhead and start the next one with 0.1s overlap."
        )
        self.split_btn.clicked.connect(self._mark_split)
        self.split_btn.setEnabled(False)
        mark_row.addWidget(self.split_btn)
        
        # Auto-split section with lift type dropdown
        auto_box = QtWidgets.QGroupBox("Auto-Split (Experimental)")
        auto_layout = QtWidgets.QHBoxLayout(auto_box)
        auto_layout.setContentsMargins(8, 4, 8, 4)
        
        auto_layout.addWidget(QtWidgets.QLabel("Lift:"))
        self.lift_combo = QtWidgets.QComboBox()
        self.lift_combo.addItems(["Bench Press", "Other (manual only)"])
        self.lift_combo.setToolTip(
            "Select the lift type for rep detection.\n"
            "Only Bench Press has auto-detection. Other lifts require manual marking."
        )
        self.lift_combo.currentIndexChanged.connect(self._on_lift_changed)
        auto_layout.addWidget(self.lift_combo)
        
        self.auto_split_btn = QtWidgets.QPushButton("🔍 Detect Reps [A]")
        self.auto_split_btn.setToolTip(
            "Run pose tracking and automatically detect rep boundaries for the selected lift.\n"
            "⚠️ Experimental: Results may need manual adjustment."
        )
        self.auto_split_btn.clicked.connect(self._auto_split_by_reps)
        self.auto_split_btn.setEnabled(False)
        auto_layout.addWidget(self.auto_split_btn)
        
        mark_row.addWidget(auto_box)
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

        # Scrubber with time indicator
        scrubber_row = QtWidgets.QHBoxLayout()
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        scrubber_row.addWidget(self.scrubber, stretch=1)
        
        self.time_label = QtWidgets.QLabel("0:00.0 / 0:00.0")
        self.time_label.setMinimumWidth(120)
        self.time_label.setStyleSheet("font-family: monospace;")
        scrubber_row.addWidget(self.time_label)
        left.addLayout(scrubber_row)

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
        preview_btn = QtWidgets.QPushButton("▶ Preview")
        preview_btn.setToolTip("Preview the selected clip (plays from start to end).")
        preview_btn.clicked.connect(self._preview_selected_cut)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.setToolTip("Remove the highlighted clip from the list.")
        remove_btn.clicked.connect(self._remove_selected_cut)
        clear_btn = QtWidgets.QPushButton("Clear clips")
        clear_btn.setToolTip("Delete all clips for this video.")
        clear_btn.clicked.connect(self._clear_cuts)
        clips_header.addWidget(preview_btn)
        clips_header.addWidget(remove_btn)
        clips_header.addWidget(clear_btn)
        right.addLayout(clips_header)

        self.cut_list = QtWidgets.QListWidget()
        self.cut_list.itemDoubleClicked.connect(self._on_cut_double_clicked)
        self.cut_list.setToolTip("Double-click a clip to jump to its start.")
        right.addWidget(self.cut_list, stretch=1)

        # Output options
        options_box = QtWidgets.QGroupBox("Export Options")
        options_layout = QtWidgets.QVBoxLayout(options_box)
        
        # Force vertical checkbox
        self.force_vertical_cb = QtWidgets.QCheckBox("Force vertical (9:16) for mobile")
        self.force_vertical_cb.setToolTip(
            "Output 720x1280 videos suitable for mobile viewing."
        )
        self.force_vertical_cb.setChecked(True)  # Default to vertical
        options_layout.addWidget(self.force_vertical_cb)
        
        # Pre-rep padding
        prerep_row = QtWidgets.QHBoxLayout()
        prerep_row.addWidget(QtWidgets.QLabel("Pre-rep padding:"))
        self.prerep_spin = QtWidgets.QSpinBox()
        self.prerep_spin.setRange(0, 5000)
        self.prerep_spin.setValue(2000)  # 2 seconds default
        self.prerep_spin.setSuffix(" ms")
        self.prerep_spin.setToolTip(
            "Extend cut this many milliseconds BEFORE the rep starts (for pose model warmup)."
        )
        prerep_row.addWidget(self.prerep_spin)
        prerep_row.addStretch(1)
        options_layout.addLayout(prerep_row)
        
        # Post-rep padding
        postrep_row = QtWidgets.QHBoxLayout()
        postrep_row.addWidget(QtWidgets.QLabel("Post-rep padding:"))
        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 2000)
        self.pad_spin.setValue(500)  # 0.5 seconds default
        self.pad_spin.setSuffix(" ms")
        self.pad_spin.setToolTip("Extend cut this many milliseconds AFTER the rep ends.")
        postrep_row.addWidget(self.pad_spin)
        postrep_row.addStretch(1)
        options_layout.addLayout(postrep_row)
        
        right.addWidget(options_box)


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

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts for video navigation and editing."""
        from PySide6.QtGui import QShortcut, QKeySequence
        
        # Playback controls
        QShortcut(QKeySequence(QtCore.Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(QtCore.Qt.Key_R), self, self._replay)
        
        # Frame navigation
        QShortcut(QKeySequence(QtCore.Qt.Key_Left), self, lambda: self._step_frames(-1))
        QShortcut(QKeySequence(QtCore.Qt.Key_Right), self, lambda: self._step_frames(1))
        QShortcut(QKeySequence(QtCore.Qt.Key_Comma), self, lambda: self._step_seconds(-0.5))
        QShortcut(QKeySequence(QtCore.Qt.Key_Period), self, lambda: self._step_seconds(0.5))
        
        # Editing actions
        QShortcut(QKeySequence(QtCore.Qt.Key_M), self, self._mark_split)
        QShortcut(QKeySequence(QtCore.Qt.Key_A), self, self._auto_split_by_reps)
        
        # Save
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_current_video)

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
        
        # Downsample for preview performance
        h, w = frame.shape[:2]
        if h > self.PREVIEW_HEIGHT:
            scale = self.PREVIEW_HEIGHT / h
            new_w = int(w * scale)
            new_h = self.PREVIEW_HEIGHT
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
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
        self.auto_split_btn.setEnabled(True)

        self.playing = False
        self.play_btn.setText("Play [Space]")
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
        self._update_time_label()

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

    def _on_cut_double_clicked(self, item):
        """Jump to the start of the double-clicked cut."""
        row = self.cut_list.row(item)
        path = self._current_video_path()
        if not path:
            return
        clips = self.cuts.get(path, [])
        if row < 0 or row >= len(clips):
            return
        start_ms, end_ms = clips[row]
        # Jump to start frame
        frame_idx = int(start_ms / 1000.0 * self.fps)
        self.playing = False
        self.play_btn.setText("Play [Space]")
        self._render_frame(frame_idx)
        self.status_label.setText(f"Jumped to clip {row + 1} start: {start_ms / 1000:.2f}s")

    def _preview_selected_cut(self):
        """Preview the selected clip by playing from start to end."""
        path = self._current_video_path()
        if not path:
            return
        row = self.cut_list.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "Select a clip from the list to preview."
            )
            return
        clips = self.cuts.get(path, [])
        if row >= len(clips):
            return
        start_ms, end_ms = clips[row]
        # Jump to start and start playing
        start_frame = int(start_ms / 1000.0 * self.fps)
        end_frame = int(end_ms / 1000.0 * self.fps)
        self._render_frame(start_frame)
        # Store preview end frame so we stop there
        self._preview_end_frame = end_frame
        self.playing = True
        self.play_btn.setText("Pause [Space]")
        self.status_label.setText(f"Previewing clip {row + 1}: {start_ms / 1000:.2f}s → {end_ms / 1000:.2f}s")

    def _on_lift_changed(self, index: int):
        """Handle lift type selection change."""
        lift = self.lift_combo.currentText()
        is_bench = "Bench" in lift
        
        # Only enable auto-split for Bench Press (which has detection algorithm)
        self.auto_split_btn.setEnabled(is_bench and self.cap is not None)
        
        if not is_bench:
            self.status_label.setText("Auto-detect not available for this lift. Use Mark Split for manual cuts.")

    def _auto_split_by_reps(self):
        """Run pose tracking and auto-detect rep boundaries."""
        # Check if a supported lift is selected
        lift = self.lift_combo.currentText()
        if "Bench" not in lift:
            QtWidgets.QMessageBox.warning(
                self, "Lift Not Supported",
                "Auto-detection is only available for Bench Press.\n"
                "Use 'Mark Split' for manual cutting of other lifts."
            )
            return
        
        path = self._current_video_path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        
        # Confirm with user
        resp = QtWidgets.QMessageBox.question(
            self, "Auto-Split",
            "This will run pose tracking to detect rep boundaries.\n"
            "Existing cuts will be replaced.\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        
        # Show progress
        progress = QtWidgets.QProgressDialog(
            "Detecting best rotation...", "Cancel", 0, 100, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        # Get settings for pose tracking
        settings = default_movement_settings("bench_press")
        model_path = _pose_model_path(settings.get("model", "full"))
        
        # First, detect best rotation by sampling frames
        def rotation_progress_cb(msg: str):
            progress.setLabelText(msg)
            QtWidgets.QApplication.processEvents()
        
        try:
            best_rotation = detect_best_rotation(
                str(path), settings, model_path, 
                sample_count=5, progress_cb=rotation_progress_cb
            )
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.warning(
                self, "Rotation Detection Failed", 
                f"Could not auto-detect rotation: {e}\nUsing current rotation setting."
            )
            best_rotation = self._effective_rotation_for_video(path)
        
        # Apply the detected rotation
        if best_rotation != 0:
            self.rotation_overrides[path] = best_rotation
            self.current_rotation_override = best_rotation
            target_idx = _rotation_option_index(best_rotation)
            self.rotation_combo.blockSignals(True)
            self.rotation_combo.setCurrentIndex(target_idx)
            self.rotation_combo.blockSignals(False)
            self._render_frame(self.current_frame)
        
        # Update progress for pose tracking
        progress.setLabelText("Running pose tracker...")
        progress.setMaximum(self.frame_count)
        progress.setValue(0)
        
        # Track pose (blocking with progress updates)
        frames = []
        def progress_cb(done, total):
            progress.setValue(done)
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()
        
        try:
            frames = run_pose_landmarks_on_video(
                str(path), self.fps, settings, model_path, progress_cb, best_rotation
            )
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self, "Pose Tracking Failed", f"Error: {e}"
            )
            return
        
        progress.close()
        
        if not frames:
            QtWidgets.QMessageBox.warning(
                self, "No Frames", "Pose tracking returned no frames."
            )
            return
        
        # Detect reps using peak detection
        # Finds local maxima (lockout positions) and cuts between them
        self.status_label.setText("Detecting reps...")
        QtWidgets.QApplication.processEvents()
        
        reps = detect_reps_in_frames(frames)
        
        # Always save debug data for analysis
        from core.rep_detector import _compute_avg_elbow_angle, _smooth_angles
        angles_raw = []
        timestamps = []
        for f in frames:
            landmarks = f.get("landmarks", [])
            angle = _compute_avg_elbow_angle(landmarks) if landmarks else None
            angles_raw.append(angle)
            timestamps.append(f.get("time_ms", 0) or 0)
        
        # Smooth for analysis
        angles_smooth = _smooth_angles(angles_raw, window=21)
        
        # Save to CSV for debugging
        debug_path = path.parent / f"{path.stem}_angles_debug.csv"
        try:
            with open(debug_path, "w") as f:
                f.write("frame,time_ms,angle_raw,angle_smooth\n")
                for i, (raw, smooth, ts) in enumerate(zip(angles_raw, angles_smooth, timestamps)):
                    raw_str = f"{raw:.1f}" if raw else ""
                    smooth_str = f"{smooth:.1f}" if smooth else ""
                    f.write(f"{i},{ts},{raw_str},{smooth_str}\n")
            print(f"Debug angles saved to: {debug_path}")
        except Exception as e:
            print(f"Could not save debug: {e}")
        
        if not reps:
            # Diagnostic: calculate angle stats to help debug
            valid_angles = [a for a in angles_smooth if a is not None]
            
            diag_msg = ""
            if not valid_angles:
                diag_msg = "\n\n⚠️ No valid elbow angles could be computed!"
            else:
                min_angle = min(valid_angles)
                max_angle = max(valid_angles)
                angle_range = max_angle - min_angle
                adaptive_lockout = max_angle - (angle_range * 0.3)
                
                # Count potential lockouts
                lockout_candidates = sum(1 for a in valid_angles if a >= adaptive_lockout)
                
                diag_msg = (
                    f"\n\n📊 Diagnostics:\n"
                    f"• Valid angles: {len(valid_angles)}/{len(frames)} frames\n"
                    f"• Angle range: {min_angle:.0f}° - {max_angle:.0f}° (span: {angle_range:.0f}°)\n"
                    f"• Lockout threshold: {adaptive_lockout:.0f}°\n"
                    f"• Frames above threshold: {lockout_candidates}\n"
                    f"\n📁 Debug CSV saved to:\n{debug_path}"
                )
            
            QtWidgets.QMessageBox.information(
                self, "No Reps Detected",
                "Could not detect any complete reps.\n\n"
                f"Check the debug CSV file to analyze the angle data.{diag_msg}"
            )
            return
        
        # Convert frame indices to milliseconds and create cuts
        new_cuts = []
        for rep in reps:
            start_ms = int(rep.start_frame / self.fps * 1000)
            end_ms = int(rep.end_frame / self.fps * 1000)
            new_cuts.append((start_ms, end_ms))
        
        self.cuts[path] = new_cuts
        self._refresh_cut_list()
        
        rotation_msg = f" (rotation: {best_rotation}°)" if best_rotation != 0 else ""
        self.status_label.setText(
            f"Auto-detected {len(reps)} rep(s){rotation_msg}. Review and export when ready."
        )
        QtWidgets.QMessageBox.information(
            self, "Reps Detected",
            f"Found {len(reps)} rep(s).{rotation_msg}\n\n"
            "Review the cuts list and click 'Save Clips' to export."
        )


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
        force_vertical = self.force_vertical_cb.isChecked()
        prerep_pad = self.prerep_spin.value()
        postrep_pad = self.pad_spin.value()
        errors = False
        for idx, (start, end) in enumerate(clips, 1):
            # Apply pre-rep and post-rep padding
            s = max(0, start - prerep_pad)
            e = end + postrep_pad
            stem = video.stem
            out_path = out_dir / f"{stem}_clip{idx:02d}.mp4"
            if not self._run_ffmpeg(video, out_path, s, e, rotation, force_vertical):
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
        self, src: Path, dst: Path, start_ms: int, end_ms: int, rotation: int = 0,
        force_vertical: bool = False
    ) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotation = rotation or 0
        vf_parts = self._rotation_filter_chain(rotation)
        
        if force_vertical:
            # Output 720x1280 (9:16) for mobile
            # First scale to fit within 720x1280, then pad to exact size
            vf_parts.append("scale=720:1280:force_original_aspect_ratio=decrease")
            vf_parts.append("pad=720:1280:(ow-iw)/2:(oh-ih)/2:black")
        else:
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
                "-an",  # Remove audio for cleaner training clips
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

    def _update_time_label(self):
        """Update the time display showing current position and total duration."""
        current_sec = self._current_time_ms() / 1000.0
        total_sec = (self.frame_count / (self.fps if self.fps > 0 else 30.0))
        
        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = sec % 60
            return f"{m}:{s:04.1f}"
        
        self.time_label.setText(f"{fmt(current_sec)} / {fmt(total_sec)}")

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause [Space]" if self.playing else "Play [Space]")

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
            self.play_btn.setText("Play [Space]")
            self._preview_end_frame = None
        # Stop at preview end if set
        elif self._preview_end_frame is not None and next_idx >= self._preview_end_frame:
            next_idx = self._preview_end_frame
            self.playing = False
            self.play_btn.setText("Play [Space]")
            self.status_label.setText("Preview complete.")
            self._preview_end_frame = None
        self._render_frame(next_idx)

    def _replay(self):
        if self.frame_count <= 0:
            return
        self.playing = True
        self.play_btn.setText("Pause [Space]")
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if self.frame_count <= 0:
            return
        self.playing = False
        self.play_btn.setText("Play [Space]")
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
        self.play_btn.setText("Play [Space]")

    def _scrubbed(self, value: int):
        self._render_frame(value)

    def _update_timer_interval(self):
        fps = self.fps if self.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)
