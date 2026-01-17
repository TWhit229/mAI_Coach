"""Labeling interface."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from core.metrics import (
    TRACKING_BAD_RATIO_MAX,
    compute_frame_grip_metrics,
    compute_rep_metrics,
    suggest_auto_tags,
)
from core.rep_detector import detect_reps_in_frames
from core.video import (

    run_pose_landmarks_on_video,
    _pose_model_path, 
)
from label_config import (
    DEFAULT_OK_TAG,
    load_label_options,
    default_movement_settings,
    GRIP_WIDE_THRESHOLD,
    GRIP_NARROW_THRESHOLD,
    GRIP_UNEVEN_THRESHOLD,
    BAR_TILT_THRESHOLD_DEG,
    BODY_PART_OPTIONS,
    MODEL_VARIANTS,
)
from ui.widgets import VideoSession


# --- Constants not in shared ---
RPE_OPTIONS = [f"{x/2:.1f}" for x in range(2, 21)]
CAMERA_ANGLE_OPTIONS = [
    "front",
    "front_45",
    "side",
    "rear_45",
    "rear",
    "overhead",
    "unknown",
]
LENS_OPTIONS = ["0.5", "1.0", "2.0", "3.0", "telephoto", "wide", "ultra-wide"]
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


class MovementDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, name: str = "", settings: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Movement Settings")
        layout = QtWidgets.QFormLayout(self)

        self.name_edit = QtWidgets.QLineEdit(name)
        layout.addRow("Movement name", self.name_edit)

        cfg = settings or default_movement_settings(name)
        self.model_box = QtWidgets.QComboBox()
        self.model_box.addItems(MODEL_VARIANTS)
        idx = self.model_box.findText(cfg.get("model", "full"))
        if idx >= 0:
            self.model_box.setCurrentIndex(idx)
        layout.addRow("Model variant", self.model_box)

        self.preprocess_check = QtWidgets.QCheckBox("Preprocess (downscale/FPS cap)")
        self.preprocess_check.setChecked(bool(cfg.get("preprocess", True)))
        self.pre_height_spin = QtWidgets.QSpinBox()
        self.pre_height_spin.setRange(240, 2160)
        self.pre_height_spin.setValue(int(cfg.get("pre_height", 720)))
        self.pre_fps_spin = QtWidgets.QDoubleSpinBox()
        self.pre_fps_spin.setRange(1.0, 60.0)
        self.pre_fps_spin.setValue(float(cfg.get("pre_fps", 15.0)))
        self.pre_fps_spin.setSingleStep(1.0)
        layout.addRow(self.preprocess_check)
        layout.addRow("Target height (px)", self.pre_height_spin)
        layout.addRow("Target FPS", self.pre_fps_spin)

        self.det_spin = QtWidgets.QDoubleSpinBox()
        self.det_spin.setRange(0.1, 1.0)
        self.det_spin.setSingleStep(0.05)
        self.det_spin.setValue(cfg.get("det", 0.5))
        self.prs_spin = QtWidgets.QDoubleSpinBox()
        self.prs_spin.setRange(0.1, 1.0)
        self.prs_spin.setSingleStep(0.05)
        self.prs_spin.setValue(cfg.get("prs", 0.7))
        self.trk_spin = QtWidgets.QDoubleSpinBox()
        self.trk_spin.setRange(0.1, 1.0)
        self.trk_spin.setSingleStep(0.05)
        self.trk_spin.setValue(cfg.get("trk", 0.7))
        self.ema_spin = QtWidgets.QDoubleSpinBox()
        self.ema_spin.setRange(0.0, 1.0)
        self.ema_spin.setSingleStep(0.05)
        self.ema_spin.setValue(cfg.get("ema", 0.25))
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setChecked(bool(cfg.get("seg", False)))
        self.grip_wide_spin = QtWidgets.QDoubleSpinBox()
        self.grip_wide_spin.setRange(1.0, 5.0)
        self.grip_wide_spin.setSingleStep(0.05)
        self.grip_wide_spin.setValue(cfg.get("grip_wide_threshold", GRIP_WIDE_THRESHOLD))
        self.grip_narrow_spin = QtWidgets.QDoubleSpinBox()
        self.grip_narrow_spin.setRange(0.1, 3.0)
        self.grip_narrow_spin.setSingleStep(0.05)
        self.grip_narrow_spin.setValue(
            cfg.get("grip_narrow_threshold", GRIP_NARROW_THRESHOLD)
        )
        self.grip_uneven_spin = QtWidgets.QDoubleSpinBox()
        self.grip_uneven_spin.setRange(0.0, 1.0)
        self.grip_uneven_spin.setSingleStep(0.01)
        self.grip_uneven_spin.setValue(
            cfg.get("grip_uneven_threshold", GRIP_UNEVEN_THRESHOLD)
        )
        self.bar_tilt_spin = QtWidgets.QDoubleSpinBox()
        self.bar_tilt_spin.setRange(0.0, 45.0)
        self.bar_tilt_spin.setSingleStep(0.5)
        self.bar_tilt_spin.setValue(
            cfg.get("bar_tilt_threshold", BAR_TILT_THRESHOLD_DEG)
        )

        layout.addRow("det threshold", self.det_spin)
        layout.addRow("prs threshold", self.prs_spin)
        layout.addRow("trk threshold", self.trk_spin)
        layout.addRow("EMA alpha", self.ema_spin)
        layout.addRow(self.seg_check)
        layout.addRow("Grip wide threshold", self.grip_wide_spin)
        layout.addRow("Grip narrow threshold", self.grip_narrow_spin)
        layout.addRow("Grip uneven threshold", self.grip_uneven_spin)
        layout.addRow("Bar tilt threshold", self.bar_tilt_spin)

        self.body_part_checks = []
        parts_box = QtWidgets.QGroupBox("Body parts to display")
        parts_layout = QtWidgets.QGridLayout(parts_box)
        selected_parts = set(cfg.get("body_parts") or [])
        for i, part in enumerate(BODY_PART_OPTIONS):
            chk = QtWidgets.QCheckBox(part)
            chk.setChecked(part in selected_parts)
            self.body_part_checks.append(chk)
            row, col = divmod(i, 2)
            parts_layout.addWidget(chk, row, col)
        layout.addRow(parts_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        if not self.name_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self, "Missing name", "Enter a movement name."
            )
            return
        super().accept()

    def values(self):
        name = self.name_edit.text().strip()
        body_parts = [
            chk.text() for chk in self.body_part_checks if chk.isChecked()
        ] or BODY_PART_OPTIONS.copy()
        return name, {
            "model": self.model_box.currentText(),
            "preprocess": self.preprocess_check.isChecked(),
            "pre_height": self.pre_height_spin.value(),
            "pre_fps": self.pre_fps_spin.value(),
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
            "grip_wide_threshold": self.grip_wide_spin.value(),
            "grip_narrow_threshold": self.grip_narrow_spin.value(),
            "grip_uneven_threshold": self.grip_uneven_spin.value(),
            "bar_tilt_threshold": self.bar_tilt_spin.value(),
            "body_parts": body_parts,
        }


class LabelerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.tags, self.movement_settings = load_label_options()
        self.current_body_parts = BODY_PART_OPTIONS.copy()
        self._rotation_lock_value: Optional[int] = None
        self._default_weight_lbs = 0
        self._inputs_locked = False
        self._auto_finish = False
        self.session = VideoSession()
        self.session.dataset_loaded.connect(self._on_dataset_loaded)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        self.play_timer.start(30)

        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._pose_job_active = False
        self._pose_worker = None

        self._build_ui()
        self._update_timer_interval()
        self._update_live_frame_metrics(None)
        
        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard shortcuts for faster labeling."""
        key = event.key()
        
        # Space: Play/Pause
        if key == QtCore.Qt.Key_Space:
            self._toggle_play()
            event.accept()
            return
        
        # Left/Right arrows: Frame stepping
        if key == QtCore.Qt.Key_Left:
            self._step_frames(-1)
            event.accept()
            return
        if key == QtCore.Qt.Key_Right:
            self._step_frames(1)
            event.accept()
            return
        
        # Enter/Return: Save and advance to next
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self._load_relative(+1, save=True)
            event.accept()
            return
        
        # Escape: Clear tag selection (select only default)
        if key == QtCore.Qt.Key_Escape:
            self._clear_tag_selection()
            event.accept()
            return
        
        # Number keys 1-9: Toggle tag by index
        if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_9:
            tag_index = key - QtCore.Qt.Key_1  # 0-indexed
            self._toggle_tag_by_index(tag_index)
            event.accept()
            return
        
        # Let parent handle other keys
        super().keyPressEvent(event)

    def _toggle_tag_by_index(self, index: int):
        """Toggle a tag by its index in the list."""
        if index < 0 or index >= self.tag_list.count():
            return
        item = self.tag_list.item(index)
        if item:
            item.setSelected(not item.isSelected())
            self._enforce_default_tag_rule()

    def _clear_tag_selection(self):
        """Clear all tag selections and select only the default tag."""
        self.tag_list.clearSelection()
        self._toggle_tag_selection(DEFAULT_OK_TAG, True)


    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        top_bar = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Videos")
        self.load_btn.clicked.connect(self._pick_videos)
        self.dataset_btn = QtWidgets.QPushButton("Choose Dataset Folder")
        self.dataset_btn.clicked.connect(self._pick_dataset_dir)
        top_bar.addWidget(self.load_btn)
        top_bar.addWidget(self.dataset_btn)
        top_bar.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top_bar.addWidget(self.home_button)
        root.addLayout(top_bar)

        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, stretch=1)

        # left: video + controls
        left = QtWidgets.QVBoxLayout()
        content.addLayout(left, stretch=3)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 315)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #333;")
        left.addWidget(self.video_label, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        left.addLayout(controls)

        def mk_btn(text, slot):
            self._add_btn(controls, text, slot)

        mk_btn("⟵ Frame", lambda: self._step_frames(-1))
        mk_btn("-0.5s", lambda: self._step_seconds(-0.5))
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        mk_btn("Replay", self._replay)
        mk_btn("+0.5s", lambda: self._step_seconds(0.5))
        mk_btn("Frame ⟶", lambda: self._step_frames(1))

        controls.addStretch(1)

        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_box)

        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        left.addWidget(self.scrubber)

        self.pose_refresh_btn = QtWidgets.QPushButton("Re-run Pose Tracker")
        self.pose_refresh_btn.setToolTip(
            "Generate or refresh pose overlays for the current video."
        )
        self.pose_refresh_btn.setEnabled(False)
        self.pose_refresh_btn.clicked.connect(
            lambda: self._ensure_pose_data(force=True)
        )
        left.addWidget(self.pose_refresh_btn)

        # right: form
        right_column = QtWidgets.QVBoxLayout()
        content.addLayout(right_column, stretch=2)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_column.addWidget(right_scroll)
        right_widget = QtWidgets.QWidget()
        right_scroll.setWidget(right_widget)
        right = QtWidgets.QVBoxLayout(right_widget)

        form = QtWidgets.QFormLayout()
        right.addLayout(form)

        self.rep_id = QtWidgets.QLineEdit()
        form.addRow("rep_id", self.rep_id)

        self.movement_cb = QtWidgets.QComboBox()
        self.movement_cb.setEditable(True)
        self.movement_cb.addItems(self.movements)
        self.movement_cb.currentTextChanged.connect(self._on_movement_changed)
        form.addRow("movement", self.movement_cb)

        self.quality_cb = QtWidgets.QComboBox()
        self.quality_cb.addItems(["1", "2", "3", "4", "5"])
        self.quality_cb.setCurrentText("3")
        form.addRow("overall_quality", self.quality_cb)

        meta_widget = QtWidgets.QWidget()
        meta_grid = QtWidgets.QGridLayout(meta_widget)
        meta_grid.setHorizontalSpacing(8)
        meta_grid.setVerticalSpacing(2)
        meta_grid.setContentsMargins(0, 0, 0, 0)

        self.load_spin = QtWidgets.QSpinBox()
        self.load_spin.setRange(0, 2000)
        self.load_spin.setSuffix(" lbs")
        self.load_spin.setSingleStep(5)
        self.load_spin.valueChanged.connect(self._on_load_spin_changed)
        self.load_spin.valueChanged.connect(self._on_load_spin_changed)

        self.rpe_cb = QtWidgets.QComboBox()
        self.rpe_cb.addItems(RPE_OPTIONS)
        self.rpe_cb.setCurrentText("1.0")

        self.camera_cb = QtWidgets.QComboBox()
        self.camera_cb.addItems(CAMERA_ANGLE_OPTIONS)

        self.lens_cb = QtWidgets.QComboBox()
        self.lens_cb.addItems(LENS_OPTIONS)

        meta_fields = [
            ("load_lbs", self.load_spin),
            ("RPE", self.rpe_cb),
            ("camera_angle", self.camera_cb),
            ("lens", self.lens_cb),
        ]

        row = 0
        col = 0
        for label_text, widget in meta_fields[:3]:
            label = QtWidgets.QLabel(label_text)
            meta_grid.addWidget(label, row, col)
            meta_grid.addWidget(widget, row, col + 1)
            col += 2

        row = 1
        col = 0
        for label_text, widget in meta_fields[3:]:
            label = QtWidgets.QLabel(label_text)
            meta_grid.addWidget(label, row, col)
            meta_grid.addWidget(widget, row, col + 1)
            col += 2

        form.addRow(meta_widget)

        self.rotation_combo = QtWidgets.QComboBox()
        for label, _ in ROTATION_OPTIONS:
            self.rotation_combo.addItem(label)
        self.rotation_combo.currentIndexChanged.connect(self._on_rotation_combo_changed)
        self.rotation_combo.setToolTip(
            "Override the video's rotation when metadata is wrong."
        )
        rotation_widget = QtWidgets.QWidget()
        rotation_layout = QtWidgets.QHBoxLayout(rotation_widget)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.addWidget(self.rotation_combo)
        self.rotation_lock_cb = QtWidgets.QCheckBox("Keep for next videos")
        self.rotation_lock_cb.setToolTip(
            "When enabled, this rotation setting becomes the default for future videos."
        )
        self.rotation_lock_cb.stateChanged.connect(self._on_rotation_lock_toggled)
        rotation_layout.addWidget(self.rotation_lock_cb)
        rotation_layout.addStretch(1)
        form.addRow("Rotation override", rotation_widget)

        form.addRow(QtWidgets.QLabel("Form tags (Cmd/Ctrl or Shift to multi-select):"))
        self.tag_list = QtWidgets.QListWidget()
        self.tag_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tag_list.setMinimumHeight(160)
        self.tag_list.addItems(self.tags)
        self._ensure_default_tag_entry()
        self.tag_list.itemSelectionChanged.connect(self._on_tag_selection_changed)
        form.addRow(self.tag_list)

        tag_event_row = QtWidgets.QHBoxLayout()
        self.tag_picker = QtWidgets.QComboBox()
        self.tag_picker.addItems(self.tags)
        tag_event_row.addWidget(self.tag_picker)
        add_tag = QtWidgets.QPushButton("Add tag @ frame")
        add_tag.clicked.connect(self._add_tag_event)
        tag_event_row.addWidget(add_tag)
        right.addLayout(tag_event_row)

        self.lift_tools_box = QtWidgets.QGroupBox("Lift-specific tools")
        lift_layout = QtWidgets.QVBoxLayout(self.lift_tools_box)
        right.addWidget(self.lift_tools_box)

        self.live_metrics_box = QtWidgets.QGroupBox(
            "Live frame pose (current frame)"
        )
        live_layout = QtWidgets.QFormLayout(self.live_metrics_box)
        live_layout.setContentsMargins(8, 8, 8, 8)
        live_layout.setVerticalSpacing(2)
        live_layout.setHorizontalSpacing(6)
        self.live_grip_label = QtWidgets.QLabel("n/a")
        self.live_uneven_label = QtWidgets.QLabel("n/a")
        self.live_tilt_label = QtWidgets.QLabel("n/a")
        live_layout.addRow("Grip width", self.live_grip_label)
        live_layout.addRow("Grip unevenness", self.live_uneven_label)
        live_layout.addRow("Bar tilt", self.live_tilt_label)
        lift_layout.addWidget(self.live_metrics_box)

        self.metrics_box = QtWidgets.QGroupBox("Pose metrics & tracking")
        self.metrics_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        metrics_layout = QtWidgets.QVBoxLayout(self.metrics_box)
        table = QtWidgets.QTableWidget(3, 4)
        table.setHorizontalHeaderLabels(["Metric", "Min", "Median", "Max"])
        table.setVerticalHeaderLabels(["", "", ""])
        for row, name in enumerate(["Grip width", "Grip unevenness", "Bar tilt"]):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            table.setItem(row, 0, item)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(QtCore.Qt.NoFocus)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setFixedHeight(
            table.horizontalHeader().sizeHint().height()
            + table.verticalHeader().length()
            + table.frameWidth() * 2
        )
        self.metrics_table = table
        metrics_layout.addWidget(table)
        self.tracking_quality_label = QtWidgets.QLabel("Tracking quality: –")
        metrics_layout.addWidget(self.tracking_quality_label)

        self.tracking_checkbox = QtWidgets.QCheckBox(
            "Mark tracking unreliable (exclude from training)"
        )
        self.tracking_checkbox.stateChanged.connect(
            self._on_tracking_checkbox_changed
        )
        metrics_layout.addWidget(self.tracking_checkbox)

        self.tracking_auto_label = QtWidgets.QLabel("Auto suggestion: –")
        self.tracking_auto_label.setToolTip(
            "Shows whether the automatic rules think tracking is OK or unreliable."
        )
        metrics_layout.addWidget(self.tracking_auto_label)

        self.tracking_auto_button = QtWidgets.QPushButton("Use auto suggestion")
        self.tracking_auto_button.clicked.connect(self._use_tracking_auto)
        self.tracking_auto_button.setEnabled(False)
        self.tracking_auto_button.setToolTip(
            "Restore the auto-detected tracking flag if you overrode it manually."
        )
        metrics_layout.addWidget(self.tracking_auto_button)

        lift_layout.addWidget(self.metrics_box)
        self.lift_placeholder = QtWidgets.QLabel(
            "Select a supported lift (e.g., Bench Press) to see lift-specific tools."
        )
        self.lift_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lift_placeholder.setStyleSheet("color: #777; font-style: italic; padding: 20px;")
        lift_layout.addWidget(self.lift_placeholder)

        self.tag_events = QtWidgets.QListWidget()
        right.addWidget(self.tag_events, stretch=1)

        remove_tag = QtWidgets.QPushButton("Remove selected tag event")
        remove_tag.clicked.connect(self._remove_tag_event)
        right.addWidget(remove_tag)

        right.addStretch(1)
        
        # Progress indicator
        self.progress_label = QtWidgets.QLabel("Rep –/–")
        self.progress_label.setStyleSheet("font-weight: bold; color: #666;")
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        right_column.addWidget(self.progress_label)
        
        # Keyboard shortcuts hint
        shortcuts_label = QtWidgets.QLabel(
            "⌨️ Space: play/pause | ←→: frames | 1-9: tags | Enter: save+next | Esc: clear"
        )
        shortcuts_label.setStyleSheet("color: #888; font-size: 11px;")
        shortcuts_label.setAlignment(QtCore.Qt.AlignCenter)
        right_column.addWidget(shortcuts_label)
        
        nav = QtWidgets.QHBoxLayout()
        right_column.addLayout(nav)
        self.prev_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(lambda: self._load_relative(-1))
        nav.addWidget(self.prev_btn)
        self.save_btn = QtWidgets.QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._save_dataset)
        nav.addWidget(self.save_btn)
        self.next_btn = QtWidgets.QPushButton("Save + Next ▶")
        self.next_btn.clicked.connect(lambda: self._load_relative(+1, save=True))
        nav.addWidget(self.next_btn)
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()
        self._update_nav_buttons()

    def refresh_label_options(self):
        self.movements, self.tags, self.movement_settings = load_label_options()
        current_move = self.movement_cb.currentText()
        self.movement_cb.blockSignals(True)
        self.movement_cb.clear()
        self.movement_cb.addItems(self.movements)
        if current_move:
            idx = self.movement_cb.findText(current_move)
            if idx >= 0:
                self.movement_cb.setCurrentIndex(idx)
            else:
                self.movement_cb.setEditText(current_move)
        self.movement_cb.blockSignals(False)

        if self.session.current_dataset:
            selected = set(self.session.current_dataset.get("tags", []))
        else:
            selected = set()
        self.tag_list.clear()
        self.tag_list.addItems(self.tags)
        self._ensure_default_tag_entry()
        self._apply_tag_selection(list(selected))

        current_tag = self.tag_picker.currentText()
        self.tag_picker.blockSignals(True)
        self.tag_picker.clear()
        self.tag_picker.addItems(self.tags)
        idx = self.tag_picker.findText(current_tag)
        if idx >= 0:
            self.tag_picker.setCurrentIndex(idx)
        self.tag_picker.blockSignals(False)
        self._update_body_part_preview()
        self._enforce_default_tag_rule()

    def _on_movement_changed(self, name: str):
        if self.session.current_dataset is not None:
            self.session.current_dataset["movement"] = name
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()

    def _resolve_movement_settings(self, movement: str) -> Optional[Dict]:
        if not movement:
            return None
        movement = movement.strip()
        if not movement:
            return None
        direct = self.movement_settings.get(movement)
        if direct:
            return direct
        lower = movement.lower()
        for key, value in self.movement_settings.items():
            if key.lower() == lower:
                return value
        return None

    def _movement_settings_for(self, movement: str) -> Dict:
        settings = self._resolve_movement_settings(movement)
        if not settings:
            settings = default_movement_settings(movement)
        return settings

    def _update_body_part_preview(self):
        movement = self.movement_cb.currentText().strip()
        settings = self._movement_settings_for(movement)
        self.current_body_parts = settings.get("body_parts") or BODY_PART_OPTIONS.copy()

    def _add_btn(self, layout: QtWidgets.QHBoxLayout, text: str, slot):
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(slot)
        layout.addWidget(btn)

    def _ensure_default_tag_entry(self):
        if not self.tag_list.findItems(DEFAULT_OK_TAG, QtCore.Qt.MatchExactly):
            self.tag_list.addItem(DEFAULT_OK_TAG)
        if hasattr(self, "tag_picker"):
            if self.tag_picker.findText(DEFAULT_OK_TAG) < 0:
                self.tag_picker.addItem(DEFAULT_OK_TAG)

    def _toggle_tag_selection(self, tag: str, selected: bool):
        items = self.tag_list.findItems(tag, QtCore.Qt.MatchExactly)
        if not items:
            return
        self.tag_list.blockSignals(True)
        items[0].setSelected(selected)
        self.tag_list.blockSignals(False)

    def _enforce_default_tag_rule(self):
        selected_items = self.tag_list.selectedItems()
        names = {item.text() for item in selected_items}
        if not names:
            self._toggle_tag_selection(DEFAULT_OK_TAG, True)
            return
        if DEFAULT_OK_TAG in names and len(names) > 1:
            self._toggle_tag_selection(DEFAULT_OK_TAG, False)

    def _on_tag_selection_changed(self):
        self._enforce_default_tag_rule()

    def _movement_name_for_tools(self) -> str:
        if self.session.current_dataset:
            movement = self.session.current_dataset.get("movement")
            if movement:
                return movement
        return self.movement_cb.currentText()

    @staticmethod
    def _lift_uses_bench_tools(name: Optional[str]) -> bool:
        if not name:
            return False
        return "bench" in name.lower()

    def _thresholds_for_movement(self, movement: str) -> Dict[str, float]:
        settings = self._movement_settings_for(movement)
        return {
            "grip_wide_threshold": settings.get(
                "grip_wide_threshold", GRIP_WIDE_THRESHOLD
            ),
            "grip_narrow_threshold": settings.get(
                "grip_narrow_threshold", GRIP_NARROW_THRESHOLD
            ),
            "grip_uneven_threshold": settings.get(
                "grip_uneven_threshold", GRIP_UNEVEN_THRESHOLD
            ),
            "bar_tilt_threshold": settings.get(
                "bar_tilt_threshold", BAR_TILT_THRESHOLD_DEG
            ),
        }

    def _refresh_lift_specific_tools(self):
        """Show/hide lift-specific tools based on the current movement."""
        movement = self._movement_name_for_tools()
        use_bench = self._lift_uses_bench_tools(movement)
        
        # Update box title to show which lift tools are active
        if use_bench:
            self.lift_tools_box.setTitle("Lift-specific tools (Bench Press)")
        else:
            self.lift_tools_box.setTitle("Lift-specific tools")
        
        # Show/hide bench-specific widgets
        for widget in (self.live_metrics_box, self.metrics_box):
            widget.setVisible(use_bench)
        self.lift_placeholder.setVisible(not use_bench)
        self.lift_tools_box.setVisible(True)
        
        if not use_bench:
            self._update_live_frame_metrics(None)

    def _apply_tag_selection(self, tags: Optional[Sequence[str]]):
        selected = set(tags or [])
        if not self.tag_list:
            return
        self._ensure_default_tag_entry()
        for tag in selected:
            if not self.tag_list.findItems(tag, QtCore.Qt.MatchExactly):
                self.tag_list.addItem(tag)
            if self.tag_picker.findText(tag) < 0:
                self.tag_picker.addItem(tag)
        for i in range(self.tag_list.count()):
            item = self.tag_list.item(i)
            item.setSelected(item.text() in selected)
        self._enforce_default_tag_rule()

    def _compute_metrics_for_current_dataset(self, auto_apply_if_empty: bool) -> List[str]:
        dataset = self.session.current_dataset
        if not dataset:
            self._update_metrics_panel({}, False)
            self._update_tracking_controls(False, False)
            return []
        frames = dataset.get("frames") or []
        dataset.setdefault("tracking_manual_override", False)
        metrics = compute_rep_metrics(frames)
        dataset["metrics"] = metrics

        movement = dataset.get("movement") or self.movement_cb.currentText()
        thresholds = self._thresholds_for_movement(movement)


        auto_flag = bool(metrics.get("tracking_bad_ratio", 1.0) > TRACKING_BAD_RATIO_MAX)
        dataset["tracking_auto_recommended"] = auto_flag
        manual_override = bool(dataset.get("tracking_manual_override", False))
        if not manual_override or "tracking_unreliable" not in dataset:
            dataset["tracking_unreliable"] = auto_flag

        tracking_flag = bool(dataset.get("tracking_unreliable", False))
        auto_tags = suggest_auto_tags(metrics, tracking_flag, thresholds)
        dataset["auto_tags"] = auto_tags
        if auto_apply_if_empty and not dataset.get("tags"):
            dataset["tags"] = list(auto_tags)

        self._update_metrics_panel(metrics, tracking_flag)
        self._update_tracking_controls(tracking_flag, auto_flag)
        return auto_tags

    def _update_metrics_panel(self, metrics: Dict[str, float], tracking_flag: bool):
        if not hasattr(self, "metrics_table"):
            return

        def fmt_ratio(value: Optional[float], suffix: str = "") -> str:
            if value is None:
                return "n/a"
            return f"{value:.2f}{suffix}"

        def fmt_degrees(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value:.1f}°"

        def set_cell(row: int, col: int, text: str):
            item = self.metrics_table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.metrics_table.setItem(row, col, item)
            item.setText(text)

        set_cell(0, 1, fmt_ratio(metrics.get("grip_ratio_min"), " × shoulder"))
        set_cell(0, 2, fmt_ratio(metrics.get("grip_ratio"), " × shoulder"))
        set_cell(0, 3, fmt_ratio(metrics.get("grip_ratio_max"), " × shoulder"))
        set_cell(1, 1, fmt_ratio(metrics.get("grip_uneven_min")))
        set_cell(1, 2, fmt_ratio(metrics.get("grip_uneven_median")))
        set_cell(1, 3, fmt_ratio(metrics.get("grip_uneven_norm")))
        set_cell(2, 1, fmt_degrees(metrics.get("bar_tilt_min_deg")))
        set_cell(2, 2, fmt_degrees(metrics.get("bar_tilt_deg")))
        set_cell(2, 3, fmt_degrees(metrics.get("bar_tilt_deg_max")))
        quality = metrics.get("tracking_quality")
        bad_frames = int(metrics.get("tracking_bad_frames", 0) or 0)
        total_frames = int(metrics.get("tracking_total_frames", 0) or 0)
        if quality is None:
            quality_text = "n/a"
        else:
            quality_text = f"{quality:.2f} (bad {bad_frames}/{total_frames})"
        self.tracking_quality_label.setText(f"Tracking quality: {quality_text}")

        status_text = "Tracking unreliable" if tracking_flag else "Tracking OK"
        self.metrics_box.setTitle(f"Pose metrics & tracking – {status_text}")

    def _update_tracking_controls(self, tracking_flag: bool, auto_flag: bool):
        dataset = self.session.current_dataset or {}
        manual_override = bool(dataset.get("tracking_manual_override", False))
        auto_text = "unreliable" if auto_flag else "ok"
        if manual_override:
            auto_text += " (manual override)"
        if hasattr(self, "tracking_checkbox"):
            self.tracking_checkbox.blockSignals(True)
            self.tracking_checkbox.setChecked(tracking_flag)
            self.tracking_checkbox.blockSignals(False)
        if hasattr(self, "tracking_auto_label"):
            self.tracking_auto_label.setText(f"Auto suggestion: {auto_text}")
        if hasattr(self, "tracking_auto_button"):
            self.tracking_auto_button.setEnabled(True)
        self._update_live_frame_metrics(None)

    def _on_tracking_checkbox_changed(self, state: int):
        dataset = self.session.current_dataset
        if not dataset:
            return
        checked = state == QtCore.Qt.Checked
        dataset["tracking_unreliable"] = checked
        auto_flag = dataset.get("tracking_auto_recommended")
        dataset["tracking_manual_override"] = (
            auto_flag is None or bool(auto_flag) != checked
        )
        self._update_tracking_controls(checked, bool(auto_flag))

    def _use_tracking_auto(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        dataset["tracking_manual_override"] = False
        auto_flag = bool(dataset.get("tracking_auto_recommended", False))
        dataset["tracking_unreliable"] = auto_flag
        self._update_tracking_controls(auto_flag, auto_flag)

    def _update_live_frame_metrics(self, frame_rec: Optional[Dict]):
        if not self._lift_uses_bench_tools(self._movement_name_for_tools()):
            for lbl in (
                self.live_grip_label,
                self.live_uneven_label,
                self.live_tilt_label,
            ):
                lbl.setText("n/a")
            return
        if (
            not frame_rec
            or not frame_rec.get("pose_present")
            or not frame_rec.get("landmarks")
        ):
            for lbl in (
                self.live_grip_label,
                self.live_uneven_label,
                self.live_tilt_label,
            ):
                lbl.setText("n/a")
            return

        metrics = compute_frame_grip_metrics(frame_rec["landmarks"])
        grip_ratio = metrics.get("grip_ratio")
        if grip_ratio is not None:
            self.live_grip_label.setText(f"{grip_ratio:.2f} × shoulder width")
        else:
            self.live_grip_label.setText("n/a")

        grip_uneven = metrics.get("grip_uneven_norm")
        if grip_uneven is not None:
            self.live_uneven_label.setText(
                f"{grip_uneven:.2f} × shoulder width"
            )
        else:
            self.live_uneven_label.setText("n/a")

        tilt = metrics.get("bar_tilt_deg")
        if tilt is not None:
            self.live_tilt_label.setText(f"{tilt:.1f}°")
        else:
            self.live_tilt_label.setText("n/a")

    def _set_rotation_combo_value(self, degrees: Optional[int]):
        target_idx = _rotation_option_index(degrees)
        self.rotation_combo.blockSignals(True)
        self.rotation_combo.setCurrentIndex(target_idx)
        self.rotation_combo.blockSignals(False)

    def _current_rotation_override(self) -> Optional[int]:
        return _rotation_value_from_index(self.rotation_combo.currentIndex())

    def _on_rotation_lock_toggled(self, state: int):
        if state == QtCore.Qt.Checked:
            self._rotation_lock_value = self._current_rotation_override()
            dataset = self.session.current_dataset
            if dataset and dataset.get("rotation_override_degrees") is None:
                dataset["rotation_override_degrees"] = self._rotation_lock_value
                self._render_frame(self.current_frame)
        else:
            self._rotation_lock_value = None

    def _on_rotation_combo_changed(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        override = self._current_rotation_override()
        dataset["rotation_override_degrees"] = override
        if self.rotation_lock_cb.isChecked():
            self._rotation_lock_value = override
        self._render_frame(self.current_frame)

    def _on_load_spin_changed(self, value: int):
        self._default_weight_lbs = value

    # Data -----------------------------------------------------------------
    def _pick_videos(self):
        self._inputs_locked = False
        self._auto_finish = False
        self.load_btn.setEnabled(True)
        self.dataset_btn.setEnabled(True)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        paths = [Path(f) for f in files]
        self.session.set_video_list(paths)
        self._load_by_index(0)

    def _pick_dataset_dir(self):
        self._inputs_locked = False
        self.dataset_btn.setEnabled(True)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if not folder:
            return
        self.session.set_dataset_dir(Path(folder))
        if self.session.current_index >= 0:
            self.session.save_current_dataset()

    def load_labeler_inputs(self, videos: List[Path], dataset_dir: Optional[Path]):
        if videos:
            self.session.set_video_list(videos)
            self._inputs_locked = True
            self._auto_finish = True
            self.load_btn.setEnabled(False)
        if dataset_dir:
            self.session.set_dataset_dir(dataset_dir)
            self.dataset_btn.setEnabled(False)
        if videos:
            self._load_by_index(0)
        else:
            self._update_body_part_preview()
            self._update_nav_buttons()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def _load_by_index(self, index: int):
        try:
            if self.session.load_index(index):
                self.current_frame = 0
                self.playing = False
                self._speed_residual = 0.0
                self.scrubber.setRange(0, max(0, self.session.total_frames - 1))
                self._update_form_from_dataset()
                self._update_timer_interval()
                self._render_frame(0)
                self._update_nav_buttons()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

    def _load_relative(self, delta: int, save: bool = False):
        if save:
            self._save_dataset()
        total = len(self.session.video_paths)
        if total == 0 or self.session.current_index < 0:
            return
        if delta > 0 and self._auto_finish and self.session.current_index >= total - 1:
            if self._home_cb:
                self._home_cb()
            return
        new_index = self.session.current_index + delta
        new_index = max(0, min(new_index, total - 1))
        if new_index == self.session.current_index:
            return
        self._load_by_index(new_index)

    def _on_dataset_loaded(self):
        self._update_form_from_dataset()
        self._render_frame(0)
        self.pose_refresh_btn.setEnabled(True)
        if not self._has_pose_overlay():
            self._ensure_pose_data(force=False)

    def _update_form_from_dataset(self):
        d = self.session.current_dataset or {}
        movement_val = d.get("movement") or ""
        if self.movements:
            self.movement_cb.setCurrentText(movement_val or self.movements[0])
        else:
            self.movement_cb.setEditText(movement_val)
        self.rep_id.setText(str(d.get("rep_id", "")))
        oq = d.get("overall_quality")
        self.quality_cb.setCurrentText(str(oq) if oq else "3")
        if d.get("load_lbs") is not None:
            ll = float(d.get("load_lbs") or 0)
            self._default_weight_lbs = ll
        else:
            ll = self._default_weight_lbs
            d["load_lbs"] = ll
        self.load_spin.blockSignals(True)
        self.load_spin.setValue(float(ll))
        self.load_spin.blockSignals(False)
        self.rpe_cb.setCurrentText(str(d.get("rpe", "1.0")))
        self.camera_cb.setCurrentText(d.get("camera_angle") or "front")
        self.lens_cb.setCurrentText(d.get("lens") or "0.5")
        self._update_body_part_preview()
        self._refresh_lift_specific_tools()

        override = d.get("rotation_override_degrees")
        if (
            override is None
            and self.rotation_lock_cb.isChecked()
            and self._rotation_lock_value is not None
        ):
            override = self._rotation_lock_value
            d["rotation_override_degrees"] = override
        self._set_rotation_combo_value(override)

        existing_tags = d.get("tags") or []
        auto_tags = self._compute_metrics_for_current_dataset(
            auto_apply_if_empty=not existing_tags
        )
        active_tags = d.get("tags") or auto_tags
        self._apply_tag_selection(active_tags)

        self._refresh_tag_events()

    def _refresh_tag_events(self):
        self.tag_events.clear()
        if not self.session.current_dataset:
            return
        for evt in self.session.current_dataset.get("tag_events", []):
            txt = (
                f"frame={evt.get('frame_index', '?')} "
                f"time={evt.get('time_ms', '?')}ms  {evt.get('issue')}"
            )
            self.tag_events.addItem(txt)

    def _save_dataset(self):
        if not self.session.current_dataset:
            return
        d = self.session.current_dataset
        d["rep_id"] = self.rep_id.text()
        d["movement"] = self.movement_cb.currentText()
        d["overall_quality"] = self.quality_cb.currentText()
        d["load_lbs"] = self.load_spin.value()
        try:
            d["rpe"] = float(self.rpe_cb.currentText())
        except ValueError:
            d["rpe"] = 1.0
        d["camera_angle"] = self.camera_cb.currentText()
        d["lens"] = self.lens_cb.currentText()
        
        # Tags are updated live in the dict by _apply_tag_selection / toggle
        selected_items = self.tag_list.selectedItems()
        final_tags = [item.text() for item in selected_items]
        d["tags"] = final_tags
        
        self.session.save_current_dataset()
        self._compute_metrics_for_current_dataset(auto_apply_if_empty=False)

    def _update_nav_buttons(self):
        total = len(self.session.video_paths)
        idx = self.session.current_index
        
        # Update progress label
        if total > 0 and idx >= 0:
            self.progress_label.setText(f"Rep {idx + 1}/{total}")
        else:
            self.progress_label.setText("Rep –/–")
        
        if total <= 1:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(idx > 0)
            self.next_btn.setEnabled(idx < total - 1)
            
            if self._auto_finish and idx >= total - 1:
                 self.next_btn.setText("💾 Save + Finish")
                 self.next_btn.setEnabled(True)
            else:
                 self.next_btn.setText("Save + Next ▶")

    # Playback / Video --------------------------------------------------------
    def _render_frame(self, index: int):
        if index < 0 or index >= self.session.total_frames:
            return

        frame = self.session.get_frame(
            index,
            rotation_override=self._current_rotation_override(),
            overlays=self._pose_overlays_for_frame(index),
            draw_opts={"parts": self.current_body_parts},
        )
        if frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(
            frame.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
        )
        pix = QtGui.QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))
        
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(index)
        self.scrubber.blockSignals(False)

        frame_meta = (
            self.session.current_dataset["frames"][index]
            if (
                self.session.current_dataset
                and self.session.current_dataset.get("frames")
            )
            else None
        )
        self._update_live_frame_metrics(frame_meta)

    def _pose_overlays_for_frame(self, index: int) -> Optional[List[Dict]]:
        dataset = self.session.current_dataset
        if not dataset:
            return None
        frames = dataset.get("frames")
        if not frames or index < 0 or index >= len(frames):
            return None
        rec = frames[index]
        if rec and rec.get("pose_present"):
            return [rec]
        return None

    def _has_pose_overlay(self) -> bool:
        dataset = self.session.current_dataset
        if not dataset:
            return False
        frames = dataset.get("frames") or []
        for f in frames:
            if f.get("pose_present"):
                return True
        return False

    def _change_speed(self, text: str):
        if text.endswith("x"):
            try:
                self.playback_speed = float(text[:-1])
                self._update_timer_interval()
            except ValueError:
                pass

    def _update_timer_interval(self):
        base_ms = 1000.0 / self.session.fps
        interval = base_ms / self.playback_speed
        self.play_timer.setInterval(int(interval))

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frame(self):
        if not self.playing:
            return
        next_idx = self.current_frame + 1
        if next_idx >= self.session.total_frames:
            self.playing = False
            self.play_btn.setText("Play")
            return
        self.current_frame = next_idx
        self._render_frame(next_idx)

    def _step_frames(self, delta: int):
        self.playing = False
        self.play_btn.setText("Play")
        new_idx = max(0, min(self.current_frame + delta, self.session.total_frames - 1))
        self.current_frame = new_idx
        self._render_frame(new_idx)

    def _step_seconds(self, sec: float):
        frames = int(sec * self.session.fps)
        self._step_frames(frames)

    def _replay(self):
        self.playing = True
        self.play_btn.setText("Pause")
        self.current_frame = 0
        self._render_frame(0)

    def _scrubbed(self, value):
        self.current_frame = value
        self._render_frame(value)

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _add_tag_event(self):
        tag = self.tag_picker.currentText()
        if not tag:
            return
        if not self.session.current_dataset:
            QtWidgets.QMessageBox.warning(self, "No dataset", "No video loaded.")
            return
        ts = int(round(self.current_frame * 1000.0 / self.session.fps))
        evt = {"frame_index": self.current_frame, "time_ms": ts, "issue": tag}
        events = self.session.current_dataset.setdefault("tag_events", [])
        events.append(evt)
        events.sort(key=lambda x: x["time_ms"])
        self._refresh_tag_events()

    def _remove_tag_event(self):
        row = self.tag_events.currentRow()
        if row < 0:
            return
        if not self.session.current_dataset:
            return
        events = self.session.current_dataset.get("tag_events", [])
        if 0 <= row < len(events):
            events.pop(row)
            self._refresh_tag_events()

    # Pose Tracking (Job) -----------------------------------------------------

    def _ensure_pose_data(self, force: bool = False):
        if self._pose_job_active:
            return
        if not self.session.current_video_path:
            return

        message = (
            "No pose data found. Run pose tracker?"
            if not force
            else "Re-run pose tracker? Existing data will be replaced."
        )
        if not force:
             # Auto-run for new videos if configured? For now just confirm
             pass
        else:
             resp = QtWidgets.QMessageBox.question(
                 self, "Pose Tracker", message,
                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
             )
             if resp != QtWidgets.QMessageBox.Yes:
                 return

        progress = QtWidgets.QProgressDialog(
            "Running pose tracker...", "Cancel", 0, self.session.total_frames, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        self._pose_job_active = True
        self.pose_refresh_btn.setEnabled(False)

        # Gather settings
        movement = self._movement_name_for_tools()
        settings = self._movement_settings_for(movement)
        model_path = _pose_model_path(settings.get("model", "full"))
        
        print(f"[DEBUG] Model path: {model_path}, exists: {model_path.exists() if hasattr(model_path, 'exists') else 'N/A'}", flush=True)
        
        vid_path = self.session.current_video_path
        fps = self.session.fps
        rot = self.session.current_dataset.get("rotation_override_degrees")
        if rot is None:
             rot = self.session.current_rotation
        
        print(f"[DEBUG] vid_path: {vid_path}, fps: {fps}, rot: {rot}", flush=True)
        
        # We need to run this in a thread or process to not block UI, 
        # but unified_tool used a blocking call with progress loop. 
        # Refactoring to a thread is better, but to keep logic identical first,
        # we'll stick to blocking with QApplication.processEvents in callback,
        # OR better: use the extracted run_pose_landmarks_on_video's callback support.
        
        # Since I moved run_pose_landmarks_on_video to core/video.py, 
        # I can call it here. But it is blocking. 
        # To avoid blocking UI completely, we should run it in a thread.
        
        print("[DEBUG] Creating PoseWorker...", flush=True)
        self._pose_worker = PoseWorker(
            vid_path, fps, settings, model_path, rot
        )
        self._pose_worker.progress.connect(progress.setValue)
        
        def on_finished(results):
            self._pose_job_active = False
            progress.close()
            self.pose_refresh_btn.setEnabled(True)
            # Clean up worker reference after it completes
            self._pose_worker = None
            
            print(f"[DEBUG] Pose tracking finished. Results: {type(results)}, length: {len(results) if results else 0}")
            
            if results is None: 
                # Canceled or failed
                print("[DEBUG] Results is None - canceled or failed")
                return
            
            # Merge results
            fc = self.session.frame_count
            rl = len(results)
            cd = self.session.current_dataset
            print(f"[DEBUG] current_dataset: {cd is not None}, frame_count: {fc}, results_len: {rl}", flush=True)
            print(f"[DEBUG] fc type: {type(fc)}, rl type: {type(rl)}, comparison: {rl == fc}", flush=True)
            if cd is not None and rl == fc:
                cd["frames"] = results
                print(f"[DEBUG] Assigned {rl} frames to dataset. First frame pose_present: {results[0].get('pose_present') if results else 'N/A'}", flush=True)
                self._compute_metrics_for_current_dataset(auto_apply_if_empty=False)
                self._render_frame(self.current_frame)
            else:
                print(f"[DEBUG] Mismatch - results: {rl}, frame_count: {fc}, cd is None: {cd is None}", flush=True)
                QtWidgets.QMessageBox.warning(self, "Error", "Pose tracking result mismatch.")


        self._pose_worker.finished.connect(on_finished)
        self._pose_worker.start()

class PoseWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object)  # List[Dict] or None

    def __init__(self, vid_path, fps, settings, model_path, rotation):
        super().__init__()
        self.vid_path = vid_path
        self.fps = fps
        self.settings = settings
        self.model_path = model_path
        self.rotation = rotation
        self._canceled = False

    def run(self):
        try:
            def cb(done, total):
                self.progress.emit(done)
                return not self.isInterruptionRequested()
            
            results = run_pose_landmarks_on_video(
                self.vid_path, self.fps, self.settings, self.model_path, cb, self.rotation
            )
            self.finished.emit(results)
        except Exception as e:
            print(f"Pose worker failed: {e}")
            self.finished.emit(None)
