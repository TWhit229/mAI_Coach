"""Pose tuning interface."""

import concurrent.futures
import gc
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.utils import (
    ROTATION_OPTIONS,
    draw_upper_body_overlay,
    rotation_option_index,
    rotation_value_from_index,
)
from core.video import (
    _pose_model_path,
    preprocess_video_for_pose,
    rotate_frame_if_needed,
    run_pose_landmarks_on_video,
    video_rotation_degrees,
)
from label_config import (
    BODY_PART_OPTIONS,
    MODEL_VARIANTS,
    default_movement_settings,
    load_label_config,
    save_label_config,
)


def load_label_options():
    cfg = load_label_config()
    movements = cfg.get("movements") or []
    tags = cfg.get("tags") or []
    raw_settings = cfg.get("movement_settings") or {}
    merged_settings: Dict[str, Dict] = {}
    for name in movements:
        defaults = default_movement_settings(name)
        stored = raw_settings.get(name) or {}
        merged = defaults.copy()
        merged.update(stored)
        merged_settings[name] = merged
    return movements, tags, merged_settings


class PoseTunerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.tags, self.movement_settings = load_label_options()
        self.dataset_dir: Optional[Path] = None
        self.video_entries: List[Dict] = []
        self.video_slots: List[Dict] = []
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frames)
        self.play_timer.start(33)  # ~30fps to match phone app
        self.playing = False
        self.playback_speed = 1.0
        self._home_cb: Optional[Callable[[], None]] = None
        self._pose_job_active = False
        self.max_slots = 1

        self.pose_dirty = False
        self._loading_pose_settings = False
        self._pose_executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
        )
        self._pose_future: Optional[concurrent.futures.Future] = None
        self._preprocess_cache: Dict[Tuple[Path, int, float], Path] = {}
        self._build_ui()

    def _release_entry_caps(self, clear_labels: bool = True):
        if self._pose_future and not self._pose_future.done():
            self._pose_future.cancel()
        self._pose_future = None
        self._pose_job_active = False
        for entry in self.video_entries:
            cap = entry.get("cap")
            if cap:
                cap.release()
        self.video_entries.clear()
        self.pose_dirty = False
        self._clear_preprocess_cache()
        if clear_labels:
            for slot in self.video_slots:
                label = slot.get("label")
                if isinstance(label, QtWidgets.QLabel):
                    label.clear()
                combo = slot.get("combo")
                if combo:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
                    combo.setEnabled(False)
                button = slot.get("button")
                if button:
                    button.setEnabled(False)
                slot["entry_index"] = None

    @staticmethod
    def _count_video_frames(path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        return count

    def _read_entry_frame(self, entry: Dict, idx: int) -> Optional[np.ndarray]:
        cap = entry.get("cap")
        frame_count = entry.get("frame_count", 0)
        if not cap or frame_count <= 0:
            return None
        idx = max(0, min(idx, frame_count - 1))
        last_idx = entry.get("_last_index", -1)
        if idx != last_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        entry["_last_index"] = idx
        rotation = entry.get("rotation", 0)
        if rotation:
            frame = rotate_frame_if_needed(frame, rotation)
        return frame

    def _on_slot_rotation_changed(self, slot: Dict):
        entry_idx = slot.get("entry_index")
        if entry_idx is None or not (0 <= entry_idx < len(self.video_entries)):
            return
        entry = self.video_entries[entry_idx]
        value = rotation_value_from_index(slot["combo"].currentIndex())
        entry["rotation_override"] = value
        entry["rotation"] = (
            value if value is not None else entry.get("detected_rotation", 0)
        )
        entry["_last_index"] = -1
        self._save_entry_rotation_override(entry)
        self._render_entry(entry)
        self._schedule_pose_rerun()

    def _save_entry_rotation_override(self, entry: Dict):
        json_path: Path = entry.get("json_path")
        if not json_path:
            return
        data = {}
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                data = {}
        data["rotation_override_degrees"] = entry.get("rotation_override")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.load_videos_btn = QtWidgets.QPushButton("Load Videos")
        self.load_videos_btn.clicked.connect(self._pick_videos)
        self.dataset_btn = QtWidgets.QPushButton("Choose Dataset Folder")
        self.dataset_btn.clicked.connect(self._pick_dataset_dir)
        top.addWidget(self.load_videos_btn)
        top.addWidget(self.dataset_btn)
        top.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        top.addWidget(self.home_button)
        root.addLayout(top)

        layout = QtWidgets.QHBoxLayout()
        root.addLayout(layout, stretch=1)

        # Left: controls
        control_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(control_panel, stretch=1)

        form = QtWidgets.QFormLayout()
        control_panel.addLayout(form)

        def add_setting_row(
            label_text: str, widget: QtWidgets.QWidget, description: str
        ):
            form.addRow(label_text, widget)
            hint = QtWidgets.QLabel(description)
            hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px 0; margin-bottom: 4px;")
            hint.setWordWrap(True)
            hint.setMinimumHeight(28)  # Prevent text clipping
            form.addRow("", hint)

        self.movement_cb = QtWidgets.QComboBox()
        self.movement_cb.setEditable(True)
        self.movement_cb.addItems(self.movements)
        self.movement_cb.currentTextChanged.connect(self._load_settings_for_movement)
        add_setting_row(
            "Movement",
            self.movement_cb,
            "Pick a movement preset to load or edit settings.",
        )
        self.model_cb = QtWidgets.QComboBox()
        self.model_cb.addItems(MODEL_VARIANTS)
        self.model_cb.currentTextChanged.connect(
            lambda _: self._on_model_variant_changed()
        )
        add_setting_row(
            "Model",
            self.model_cb,
            "Choose which MediaPipe pose model variant to run.",
        )

        self.preprocess_check = QtWidgets.QCheckBox("Preprocess (faster)")
        self.preprocess_check.setToolTip(
            "Downscale and cap FPS before running pose to speed up heavy/full models."
        )
        self.preprocess_check.setChecked(True)
        self.preprocess_check.stateChanged.connect(self._schedule_pose_rerun)
        pre_row = QtWidgets.QHBoxLayout()
        pre_row.addWidget(self.preprocess_check)
        pre_row.addStretch(1)
        self.pre_height_spin = QtWidgets.QSpinBox()
        self.pre_height_spin.setRange(240, 2160)
        self.pre_height_spin.setValue(720)
        self.pre_height_spin.setSuffix(" px height")
        self.pre_height_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.pre_fps_spin = QtWidgets.QDoubleSpinBox()
        self.pre_fps_spin.setRange(1.0, 60.0)
        self.pre_fps_spin.setValue(15.0)
        self.pre_fps_spin.setSingleStep(1.0)
        self.pre_fps_spin.setSuffix(" fps")
        self.pre_fps_spin.valueChanged.connect(self._schedule_pose_rerun)
        form.addRow("Preprocess", pre_row)
        form.addRow("Target size/FPS", self.pre_height_spin)
        form.addRow("", self.pre_fps_spin)

        self.det_spin = QtWidgets.QDoubleSpinBox()
        self.det_spin.setRange(0.1, 1.0)
        self.det_spin.setSingleStep(0.05)
        self.det_spin.setToolTip(
            "Detection confidence threshold (higher = fewer detections)."
        )
        self.det_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.prs_spin = QtWidgets.QDoubleSpinBox()
        self.prs_spin.setRange(0.1, 1.0)
        self.prs_spin.setSingleStep(0.05)
        self.prs_spin.setValue(0.7)
        self.prs_spin.setToolTip(
            "Pose score threshold used when filtering landmark quality."
        )
        self.prs_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.trk_spin = QtWidgets.QDoubleSpinBox()
        self.trk_spin.setRange(0.1, 1.0)
        self.trk_spin.setSingleStep(0.05)
        self.trk_spin.setValue(0.7)
        self.trk_spin.setToolTip(
            "Tracker confidence threshold before dropping a track."
        )
        self.trk_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.ema_spin = QtWidgets.QDoubleSpinBox()
        self.ema_spin.setRange(0.0, 1.0)
        self.ema_spin.setSingleStep(0.05)
        self.ema_spin.setValue(0.25)
        self.ema_spin.setToolTip(
            "Exponential moving average amount for smoothing landmarks."
        )
        self.ema_spin.valueChanged.connect(self._schedule_pose_rerun)
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setToolTip("Overlay segmentation masks when available.")
        self.seg_check.stateChanged.connect(self._schedule_pose_rerun)
        add_setting_row(
            "det",
            self.det_spin,
            "Detection confidence threshold (higher removes weaker detections).",
        )
        add_setting_row(
            "prs",
            self.prs_spin,
            "Pose score threshold applied when keeping landmark results.",
        )
        add_setting_row(
            "trk",
            self.trk_spin,
            "Tracking score threshold before the subject is re-detected.",
        )
        add_setting_row(
            "ema",
            self.ema_spin,
            "Smoothing factor for the exponential moving average applied to pose data.",
        )
        form.addRow(self.seg_check)
        seg_hint = QtWidgets.QLabel(
            "Overlay segmentation masks for the model (slightly slower)."
        )
        seg_hint.setStyleSheet("color: #888; font-size: 11px;")
        seg_hint.setWordWrap(True)
        form.addRow("", seg_hint)

        body_group = QtWidgets.QGroupBox("Body parts to display")
        control_panel.addWidget(body_group)
        body_layout = QtWidgets.QGridLayout(body_group)
        self.body_part_checks = []
        for i, part in enumerate(BODY_PART_OPTIONS):
            chk = QtWidgets.QCheckBox(part)
            chk.setChecked(True)
            chk.stateChanged.connect(self._refresh_all_frames)
            self.body_part_checks.append(chk)
            row, col = divmod(i, 2)
            body_layout.addWidget(chk, row, col)
        body_hint = QtWidgets.QLabel(
            "Select which body segments should remain highlighted on the overlays."
        )
        body_hint.setStyleSheet("color: #888; font-size: 11px;")
        body_hint.setWordWrap(True)
        control_panel.addWidget(body_hint)

        save_row = QtWidgets.QHBoxLayout()
        control_panel.addLayout(save_row)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        save_as_btn = QtWidgets.QPushButton("Save As")
        save_as_btn.clicked.connect(self._save_as)
        save_row.addWidget(save_btn)
        save_row.addWidget(save_as_btn)

        control_panel.addStretch(1)
        self.pose_progress = QtWidgets.QProgressBar()
        self.pose_progress.setVisible(False)
        control_panel.addWidget(self.pose_progress)

        # Right: video grid and controls
        video_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(video_panel, stretch=2)

        self.video_grid = QtWidgets.QGridLayout()
        video_panel.addLayout(self.video_grid)
        self.video_slots = []
        for idx in range(self.max_slots):
            slot_widget = QtWidgets.QWidget()
            slot_layout = QtWidgets.QVBoxLayout(slot_widget)
            slot_layout.setContentsMargins(0, 0, 0, 0)
            frame_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            frame_label.setStyleSheet("background: #111; border: 1px solid #333;")
            frame_label.setMinimumSize(420, 260)
            slot_layout.addWidget(frame_label)
            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Rotation:"))
            combo = QtWidgets.QComboBox()
            for label_text, _ in ROTATION_OPTIONS:
                combo.addItem(label_text)
            combo.setEnabled(False)
            controls.addWidget(combo, stretch=1)
            rerun_btn = QtWidgets.QPushButton("Re-run Pose")
            rerun_btn.setEnabled(False)
            controls.addWidget(rerun_btn)
            slot_layout.addLayout(controls)
            row, col = divmod(idx, 2)
            self.video_grid.addWidget(slot_widget, row, col)
            slot = {
                "widget": slot_widget,
                "label": frame_label,
                "combo": combo,
                "button": rerun_btn,
                "entry_index": None,
            }
            combo.currentIndexChanged.connect(
                lambda _, s=slot: self._on_slot_rotation_changed(s)
            )
            rerun_btn.clicked.connect(
                lambda _, s=slot: self._rerun_pose_for_slot(s)
            )
            self.video_slots.append(slot)

        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        self.restart_btn = QtWidgets.QPushButton("⟲ Restart")
        self.restart_btn.setToolTip("Restart video from beginning")
        self.restart_btn.clicked.connect(self._restart_video)
        controls.addWidget(self.restart_btn)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self._change_speed)
        controls.addWidget(self.speed_box)
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.sliderMoved.connect(self._scrub_to)
        controls.addWidget(self.scrubber)
        video_panel.addLayout(controls)

        self.status_label = QtWidgets.QLabel("")
        video_panel.addWidget(self.status_label)
        self.global_rerun_btn = QtWidgets.QPushButton("Re-run Pose Tracker")
        self.global_rerun_btn.clicked.connect(self._rerun_pose_for_loaded_entries)
        video_panel.addWidget(self.global_rerun_btn)
        self.refresh_movements()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_pose_inputs(self, videos: List[Path], dataset_dir: Optional[Path]):
        if dataset_dir:
            self.dataset_dir = dataset_dir
            self.status_label.setText(f"Dataset dir: {self.dataset_dir}")
        elif not self.dataset_dir:
            self.status_label.setText("Dataset dir: not set (pose data kept in memory)")
        if videos:
            self._load_videos(videos[: self.max_slots])
        else:
            self.refresh_movements()

    def _pick_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select a video",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        self._load_videos([Path(files[0])])

    def _pick_dataset_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if folder:
            self.dataset_dir = Path(folder)
            self.status_label.setText(f"Dataset dir: {self.dataset_dir}")
            self.refresh_movements()

    def refresh_movements(self):
        self.movements, self.tags, self.movement_settings = load_label_options()
        current = self.movement_cb.currentText()
        self.movement_cb.blockSignals(True)
        self.movement_cb.clear()
        self.movement_cb.addItems(self.movements)
        if current:
            idx = self.movement_cb.findText(current)
            if idx >= 0:
                self.movement_cb.setCurrentIndex(idx)
            else:
                self.movement_cb.setEditText(current)
        self.movement_cb.blockSignals(False)
        self._load_settings_for_movement(self.movement_cb.currentText())

    def _load_videos(self, videos: List[Path]):
        if not videos:
            return
        self._release_entry_caps()
        self._clear_preprocess_cache()
        for idx, path in enumerate(videos[: self.max_slots]):
            entry = self._load_single_video(path)
            if entry:
                slot = self.video_slots[idx]
                entry["slot"] = slot
                self.video_entries.append(entry)
                slot["entry_index"] = len(self.video_entries) - 1
                slot["label"].clear()
                slot["combo"].blockSignals(True)
                slot["combo"].setCurrentIndex(
                    rotation_option_index(entry.get("rotation_override"))
                )
                slot["combo"].blockSignals(False)
                slot["combo"].setEnabled(True)
                slot["button"].setEnabled(True)
        if self.video_entries:
            self.scrubber.setEnabled(True)
            self.scrubber.setValue(0)
            self.playing = False
            self.play_btn.setText("Play")
            self._refresh_all_frames()
            if any(not entry.get("pose") for entry in self.video_entries):
                # Kick off an initial pose run automatically so overlays appear.
                self._rerun_pose_for_loaded_entries(auto=True)

    def _load_single_video(self, path: Path) -> Optional[Dict]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Video error", f"Failed to open {path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_video_frames(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame_count <= 0:
            cap.release()
            return None
        json_path = None
        pose_frames = None
        rotation_override = None
        if self.dataset_dir:
            json_path = self.dataset_dir / f"{path.stem}.json"
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text())
                    pose_frames = data.get("frames", [])
                    rotation_override = data.get("rotation_override_degrees")
                except Exception:
                    pose_frames = None
                    rotation_override = None
        detected_rotation = video_rotation_degrees(path)
        rotation = rotation_override if rotation_override is not None else detected_rotation
        return {
            "path": path,
            "cap": cap,
            "frame_count": frame_count,
            "pose": pose_frames,
            "fps": fps,
            "current_frame": 0,
            "next_time": time.perf_counter(),
            "_last_index": -1,
            "rotation": rotation,
            "detected_rotation": detected_rotation,
            "rotation_override": rotation_override,
            "json_path": json_path,
        }

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _restart_video(self):
        """Reset all videos to frame 0."""
        for entry in self.video_entries:
            entry["current_frame"] = 0
            entry["_last_index"] = -1
            entry["next_time"] = time.perf_counter()
        self.scrubber.setValue(0)
        self._refresh_all_frames()

    def _advance_frames(self):
        if not (self.playing and self.video_entries):
            return
        for entry in self.video_entries:
            fps = entry["fps"]
            interval = 1.0 / (fps * max(self.playback_speed, 0.1))
            now = time.perf_counter()
            if now >= entry.get("next_time", 0):
                frame_count = entry.get("frame_count", 0)
                if frame_count <= 0:
                    continue
                entry["current_frame"] = min(
                    frame_count - 1, entry["current_frame"] + 1
                )
                entry["next_time"] = now + interval
                self._render_entry(entry)
                # Update scrubber position
                if frame_count > 1:
                    ratio = entry["current_frame"] / (frame_count - 1)
                    self.scrubber.blockSignals(True)
                    self.scrubber.setValue(int(ratio * 1000))
                    self.scrubber.blockSignals(False)

    def _render_entry(self, entry: Dict):
        frame_count = entry.get("frame_count", 0)
        if frame_count <= 0:
            return
        idx = max(0, min(entry["current_frame"], frame_count - 1))
        frame = self._read_entry_frame(entry, idx)
        if frame is None:
            return
        frame = frame.copy()
        pose_frames = entry.get("pose")
        allowed = self._selected_body_parts()
        if pose_frames and idx < len(pose_frames):
            frec = pose_frames[idx]
            if frec and frec.get("pose_present") and frec.get("landmarks"):
                draw_upper_body_overlay(frame, frec["landmarks"], allowed)
        else:
            if allowed and "full_body" not in allowed:
                # Dim the frame if a subset of body parts was requested but no pose
                # landmarks exist for this frame. This gives users a visual hint that
                # the overlay cannot be drawn for the selected regions yet.
                frame[:] = frame * 0.8
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        slot = entry.get("slot")
        if not slot:
            return
        label: QtWidgets.QLabel = slot["label"]
        scaled = image.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(QtGui.QPixmap.fromImage(scaled))

    def _refresh_all_frames(self):
        for entry in self.video_entries:
            self._render_entry(entry)

    def _clear_preprocess_cache(self):
        for tmp in self._preprocess_cache.values():
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass
        self._preprocess_cache.clear()

    def _selected_body_parts(self) -> List[str]:
        selected = [chk.text() for chk in self.body_part_checks if chk.isChecked()]
        return selected or BODY_PART_OPTIONS.copy()

    def _on_model_variant_changed(self):
        self._schedule_pose_rerun(message="Pose model changed – click Re-run Pose Tracker.")

    def _change_speed(self, text: str):
        try:
            self.playback_speed = float(text.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0

    def _scrub_to(self, value: int):
        if not self.video_entries:
            return
        ratio = value / 1000.0
        for entry in self.video_entries:
            frame_count = entry.get("frame_count", 0)
            if frame_count > 0:
                entry["current_frame"] = int(ratio * (frame_count - 1))
                entry["next_time"] = time.perf_counter()
                self._render_entry(entry)

    def _current_pose_settings(self) -> Dict:
        return {
            "model": self.model_cb.currentText() or "full",
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
            "preprocess": self.preprocess_check.isChecked(),
            "pre_height": self.pre_height_spin.value(),
            "pre_fps": self.pre_fps_spin.value(),
        }

    def _schedule_pose_rerun(self, delay_ms: int = 200, message: Optional[str] = None):
        if (
            not self.video_entries
            or not self.dataset_dir
            or self._loading_pose_settings
        ):
            return
        overlay_cleared = False
        for entry in self.video_entries:
            if entry.get("pose"):
                entry["pose"] = None
                entry["_last_index"] = -1
                overlay_cleared = True
        if overlay_cleared:
            self._refresh_all_frames()
            gc.collect()
        already_dirty = self.pose_dirty
        self.pose_dirty = True
        if not already_dirty or message:
            self.status_label.setText(
                message or "Pose settings changed – click Re-run Pose Tracker."
            )


    def _rerun_pose_for_slot(self, slot: Dict):
        entry_idx = slot.get("entry_index")
        if entry_idx is None or not (0 <= entry_idx < len(self.video_entries)):
            return
        self._rerun_pose_for_entries([self.video_entries[entry_idx]])

    def _rerun_pose_for_loaded_entries(self, auto: bool = False):
        if not self.video_entries:
            return
        self._rerun_pose_for_entries(self.video_entries, auto=auto)

    def _rerun_pose_for_entries(self, entries: List[Dict], auto: bool = False):
        if not entries or self._pose_job_active:
            return
        settings = self._current_pose_settings()
        try:
            model_name = settings.get("model", "full")
            model_path = _pose_model_path(model_name)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.critical(self, "Pose tracker", str(exc))
            return
        entry = entries[0]
        if entry not in self.video_entries:
            return
        entry_idx = self.video_entries.index(entry)
        total_frames = entry.get("frame_count", 0)
        self.global_rerun_btn.setEnabled(False)
        self._pose_job_active = True
        rotation = entry.get("rotation_override")
        if rotation is None:
            rotation = entry.get("detected_rotation", 0)
        self.status_label.setText(
            f"Running pose tracker for {entry['path'].name}..."
        )

        source_path = entry["path"]
        source_fps = entry["fps"]
        temp_path: Optional[Path] = None
        if settings.get("preprocess"):
            try:
                temp_path, source_fps = preprocess_video_for_pose(
                    source_path,
                    int(settings.get("pre_height", 720)),
                    float(settings.get("pre_fps", 15.0)),
                )
                source_path = temp_path
            except Exception as exc:
                self.pose_dirty = True
                self.status_label.setText("Preprocess failed")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Preprocess failed",
                    f"{entry['path'].name}: {exc}",
                )
                self._pose_job_active = False
                self.global_rerun_btn.setEnabled(True)
                return

        progress = QtWidgets.QProgressDialog(
            "Running pose tracker...",
            "Cancel",
            0,
            max(total_frames, 1),
            self,
        )
        progress.setWindowTitle("Pose Tracker")
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        def progress_cb(done: int, total: int) -> bool:
            progress.setMaximum(max(total, 1))
            progress.setValue(done)
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        pose_frames: Optional[List[Dict]] = None
        try:
            pose_frames = run_pose_landmarks_on_video(
                source_path,
                source_fps,
                settings,
                model_path,
                progress_cb=progress_cb,
                rotation=rotation,
            )
        except Exception as exc:
            self.pose_dirty = True
            self.status_label.setText("Pose refresh failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Pose tracker failed",
                f"{entry['path'].name}: {exc}",
            )
        finally:
            progress.close()
            self._pose_job_active = False
            self.global_rerun_btn.setEnabled(True)

        if not pose_frames:
            return

        if not (0 <= entry_idx < len(self.video_entries)):
            return
        entry = self.video_entries[entry_idx]
        entry["pose"] = pose_frames
        self.pose_dirty = False
        self.status_label.setText("Pose overlays refreshed.")
        self._save_entry_dataset(entry, pose_frames)
        self._render_entry(entry)

    def _save_entry_dataset(self, entry: Dict, pose_frames: List[Dict]):
        json_path: Path = entry.get("json_path")
        if not json_path:
            return
        data = {}
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                data = {}
        data["frames"] = pose_frames
        data["fps"] = entry.get("fps", 30.0)
        data["rotation_override_degrees"] = entry.get("rotation_override")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2))

    def _save_settings(self):
        movement = self.movement_cb.currentText().strip()
        if not movement:
            QtWidgets.QMessageBox.warning(self, "Movement", "Enter a movement name.")
            return
        self._persist_settings(movement)
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Settings saved for {movement}."
        )

    def _save_as(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Save As", "New movement name")
        if ok and text.strip():
            movement = text.strip()
            if movement not in self.movements:
                self.movements.append(movement)
                self.movement_settings[movement] = default_movement_settings(movement)
                self.movement_cb.addItem(movement)
            self.movement_cb.setCurrentText(movement)
            self._persist_settings(movement)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Settings saved for {movement}."
            )

    def _persist_settings(self, movement: str):
        settings = {
            "model": self.model_cb.currentText() or "full",
            "preprocess": self.preprocess_check.isChecked(),
            "pre_height": self.pre_height_spin.value(),
            "pre_fps": self.pre_fps_spin.value(),
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
            "body_parts": self._selected_body_parts(),
        }
        self.movement_settings[movement] = settings
        save_label_config(
            {
                "movements": self.movements,
                "tags": self.tags,
                "movement_settings": self.movement_settings,
            }
        )

    def _load_settings_for_movement(self, name: str):
        settings = self.movement_settings.get(name)
        if not settings:
            settings = default_movement_settings(name)
        self._loading_pose_settings = True
        try:
            self.det_spin.blockSignals(True)
            self.prs_spin.blockSignals(True)
            self.trk_spin.blockSignals(True)
            self.ema_spin.blockSignals(True)
            self.model_cb.blockSignals(True)
            self.seg_check.blockSignals(True)
            self.preprocess_check.blockSignals(True)
            self.pre_height_spin.blockSignals(True)
            self.pre_fps_spin.blockSignals(True)

            self.det_spin.setValue(settings.get("det", 0.5))
            self.prs_spin.setValue(settings.get("prs", 0.7))
            self.trk_spin.setValue(settings.get("trk", 0.7))
            self.ema_spin.setValue(settings.get("ema", 0.25))
            model_value = settings.get("model", "full")
            idx = self.model_cb.findText(model_value)
            if idx < 0:
                idx = 0
            self.model_cb.setCurrentIndex(idx)
            self.seg_check.setChecked(settings.get("seg", False))
            self.preprocess_check.setChecked(settings.get("preprocess", True))
            self.pre_height_spin.setValue(int(settings.get("pre_height", 720)))
            self.pre_fps_spin.setValue(float(settings.get("pre_fps", 15.0)))
        finally:
            self.det_spin.blockSignals(False)
            self.prs_spin.blockSignals(False)
            self.trk_spin.blockSignals(False)
            self.ema_spin.blockSignals(False)
            self.model_cb.blockSignals(False)
            self.seg_check.blockSignals(False)
            self.preprocess_check.blockSignals(False)
            self.pre_height_spin.blockSignals(False)
            self.pre_fps_spin.blockSignals(False)
            self._loading_pose_settings = False
        parts = set(settings.get("body_parts") or BODY_PART_OPTIONS)
        for chk in self.body_part_checks:
            chk.setChecked(chk.text() in parts)
        self._refresh_all_frames()

    def __del__(self):
        try:
            if self._pose_executor:
                self._pose_executor.shutdown(wait=False)
        except Exception:
            pass
        self._release_entry_caps(clear_labels=False)
