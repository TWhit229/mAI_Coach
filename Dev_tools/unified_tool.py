#!/usr/bin/env python3
"""Unified PySide6 workspace for all Dev_tools utilities."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from label_config import ensure_config_file, load_label_config, save_label_config


# --- Shared helpers copied from the legacy Tk labeler -----------------------


def lowpass_ema(
    prev: Optional[np.ndarray], curr: np.ndarray, alpha: float
) -> np.ndarray:
    if prev is None or prev.shape != curr.shape:
        return curr.copy()
    return alpha * curr + (1.0 - alpha) * prev


def draw_upper_body_overlay(
    frame_bgr: np.ndarray,
    landmarks: List[Dict[str, float]],
    allowed_parts: Optional[List[str]] = None,
):
    pts = np.array([[p["x"], p["y"]] for p in landmarks], dtype=np.float32)
    H, W = frame_bgr.shape[:2]

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(x * W), int(y * H)

    UPPER_IDS = [11, 12, 13, 14, 15, 16, 23, 24]
    UPPER_LINES = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
    ]

    BODY_PART_GROUPS = {
        "hands": {"points": {15, 16}, "lines": []},
        "wrists": {"points": {15, 16}, "lines": []},
        "forearms": {"points": {13, 14, 15, 16}, "lines": [(13, 15), (14, 16)]},
        "upper_arms": {"points": {11, 12, 13, 14}, "lines": [(11, 13), (12, 14)]},
        "shoulders": {"points": {11, 12}, "lines": [(11, 12)]},
        "torso": {"points": {11, 12, 23, 24}, "lines": [(11, 23), (12, 24), (23, 24)]},
        "hips": {"points": {23, 24}, "lines": []},
        "legs": {"points": set(), "lines": []},
        "full_body": {"points": set(UPPER_IDS), "lines": list(UPPER_LINES)},
    }

    if not allowed_parts:
        allowed_points = set(UPPER_IDS)
        allowed_lines = list(UPPER_LINES)
    else:
        allowed_points = set()
        allowed_lines: List[tuple[int, int]] = []
        expanded = set(allowed_parts)
        if "full_body" in expanded:
            allowed_points = set(UPPER_IDS)
            allowed_lines = list(UPPER_LINES)
        else:
            for name in expanded:
                group = BODY_PART_GROUPS.get(name)
                if not group:
                    continue
                allowed_points.update(group["points"])
                allowed_lines.extend(group["lines"])
            if not allowed_points:
                allowed_points = set(UPPER_IDS)
            if not allowed_lines:
                allowed_lines = list(UPPER_LINES)

    for a, b in allowed_lines:
        pa = to_px(*pts[a])
        pb = to_px(*pts[b])
        cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    for idx in allowed_points:
        cx, cy = to_px(*pts[idx])
        cv2.circle(frame_bgr, (cx, cy), 4, (255, 255, 0), -1)


ensure_config_file()


def load_label_options():
    cfg = load_label_config()
    return (
        cfg.get("movements") or [],
        cfg.get("issues") or [],
        cfg.get("movement_settings") or {},
    )


QUALITY_OPTIONS = ["1", "2", "3", "4", "5"]
RPE_OPTIONS = [f"{x / 2:.1f}" for x in range(2, 21)]
CAMERA_ANGLE_OPTIONS = [
    "front",
    "front_45",
    "side",
    "rear_45",
    "rear",
    "overhead",
    "unknown",
]
LENS_OPTIONS = ["0.5", "1.0", "2.0", "3.0", "5.0", "other", "unknown"]
MODEL_VARIANTS = ["lite", "full", "heavy"]
BODY_PART_OPTIONS = [
    "hands",
    "wrists",
    "forearms",
    "upper_arms",
    "shoulders",
    "torso",
    "hips",
    "legs",
    "full_body",
]
BENCH_DEFAULT_PARTS = ["hands", "forearms", "upper_arms", "shoulders"]


def default_movement_settings(name: str = "") -> Dict:
    lower = name.lower()
    body_parts = (
        BENCH_DEFAULT_PARTS.copy() if "bench" in lower else BODY_PART_OPTIONS.copy()
    )
    return {
        "model": "full",
        "det": 0.5,
        "prs": 0.7,
        "trk": 0.7,
        "ema": 0.25,
        "seg": False,
        "body_parts": body_parts,
    }


# --- Video session ----------------------------------------------------------


class VideoSession(QtCore.QObject):
    dataset_loaded = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.video_paths: List[Path] = []
        self.dataset_dir: Optional[Path] = None
        self.frames_bgr: List[np.ndarray] = []
        self.current_dataset: Optional[Dict] = None
        self.current_index = -1
        self.fps = 30.0

    @property
    def has_video(self) -> bool:
        return bool(self.frames_bgr)

    @property
    def total_frames(self) -> int:
        return len(self.frames_bgr)

    def set_video_list(self, paths: List[Path]):
        self.video_paths = paths
        self.current_index = -1

    def set_dataset_dir(self, path: Path):
        self.dataset_dir = path

    def load_index(self, index: int) -> bool:
        if not self.video_paths:
            return False
        index = max(0, min(index, len(self.video_paths) - 1))
        self.current_index = index
        vpath = self.video_paths[index]

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {vpath}")

        fps_val = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_val if fps_val and fps_val > 0 else 30.0
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames found in video: {vpath}")
        self.frames_bgr = frames

        dataset = None
        if self.dataset_dir:
            dataset_path = self.dataset_dir / f"{vpath.stem}.json"
            if dataset_path.exists():
                dataset = json.loads(dataset_path.read_text())

        if not dataset:
            dataset = self._blank_dataset(vpath)
        else:
            fps_meta = dataset.get("fps")
            if isinstance(fps_meta, (int, float)) and fps_meta > 0:
                self.fps = fps_meta

        frames_meta = dataset.get("frames") or []
        if len(frames_meta) != len(self.frames_bgr):
            frames_meta = [
                self._blank_frame_meta(i, self.fps) for i in range(len(self.frames_bgr))
            ]
            dataset["frames"] = frames_meta

        dataset.setdefault("issue_events", [])
        self.current_dataset = dataset
        self.dataset_loaded.emit()
        return True

    def _blank_dataset(self, vpath: Path) -> Dict:
        movements, _, _ = load_label_options()
        movement = movements[0] if movements else ""
        return {
            "rep_id": vpath.stem,
            "video_path": str(vpath),
            "movement": movement,
            "overall_quality": None,
            "issues": [],
            "load_lbs": None,
            "rpe": 1.0,
            "camera_angle": "front",
            "lens": "0.5",
            "issue_events": [],
            "fps": self.fps,
            "frames": [
                self._blank_frame_meta(i, self.fps) for i in range(len(self.frames_bgr))
            ],
        }

    @staticmethod
    def _blank_frame_meta(fi: int, fps: float) -> Dict:
        fps = fps if fps > 0 else 30.0
        return {
            "frame_index": fi,
            "time_ms": int(round(fi * 1000.0 / fps)),
            "pose_present": False,
            "landmarks": None,
        }

    def save_current_dataset(self):
        if not (self.dataset_dir and self.current_dataset and self.current_index >= 0):
            return
        target = self.dataset_dir / f"{self.video_paths[self.current_index].stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.current_dataset, indent=2))


# --- Labeler view ----------------------------------------------------------


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

        self.det_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, singleStep=0.05, value=cfg.get("det", 0.5)
        )
        self.prs_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, singleStep=0.05, value=cfg.get("prs", 0.7)
        )
        self.trk_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, singleStep=0.05, value=cfg.get("trk", 0.7)
        )
        self.ema_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.0, maximum=1.0, singleStep=0.05, value=cfg.get("ema", 0.25)
        )
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setChecked(bool(cfg.get("seg", False)))

        layout.addRow("det threshold", self.det_spin)
        layout.addRow("prs threshold", self.prs_spin)
        layout.addRow("trk threshold", self.trk_spin)
        layout.addRow("EMA alpha", self.ema_spin)
        layout.addRow(self.seg_check)

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
            "det": self.det_spin.value(),
            "prs": self.prs_spin.value(),
            "trk": self.trk_spin.value(),
            "ema": self.ema_spin.value(),
            "seg": self.seg_check.isChecked(),
            "body_parts": body_parts,
        }


class AdminPanel(QtWidgets.QWidget):
    config_saved = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        main_layout = QtWidgets.QVBoxLayout(self)

        header_row = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel(
            "Manage lifts and issue tags. Saving updates label_config.json."
        )
        header_row.addWidget(header_label)
        header_row.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        header_row.addWidget(self.home_button)
        main_layout.addLayout(header_row)

        splitter = QtWidgets.QHBoxLayout()
        main_layout.addLayout(splitter)

        # Movements section
        move_group = QtWidgets.QGroupBox("Lifts / Exercises")
        splitter.addWidget(move_group, stretch=2)
        move_layout = QtWidgets.QVBoxLayout(move_group)
        self.movement_list = QtWidgets.QListWidget()
        self.movement_list.currentTextChanged.connect(self._show_movement_settings)
        move_layout.addWidget(self.movement_list)

        move_btn_row = QtWidgets.QHBoxLayout()
        move_layout.addLayout(move_btn_row)
        add_move = QtWidgets.QPushButton("Add")
        add_move.clicked.connect(self._add_movement)
        edit_move = QtWidgets.QPushButton("Edit")
        edit_move.clicked.connect(self._edit_movement)
        del_move = QtWidgets.QPushButton("Remove")
        del_move.clicked.connect(self._remove_movement)
        move_btn_row.addWidget(add_move)
        move_btn_row.addWidget(edit_move)
        move_btn_row.addWidget(del_move)

        self.movement_info = QtWidgets.QTextEdit()
        self.movement_info.setReadOnly(True)
        move_layout.addWidget(self.movement_info)

        # Issues section
        issue_group = QtWidgets.QGroupBox("Issue tags")
        splitter.addWidget(issue_group, stretch=1)
        issue_layout = QtWidgets.QVBoxLayout(issue_group)
        self.issue_list = QtWidgets.QListWidget()
        issue_layout.addWidget(self.issue_list)

        issue_btn_row = QtWidgets.QHBoxLayout()
        issue_layout.addLayout(issue_btn_row)
        add_issue = QtWidgets.QPushButton("Add")
        add_issue.clicked.connect(self._add_issue)
        edit_issue = QtWidgets.QPushButton("Edit")
        edit_issue.clicked.connect(self._edit_issue)
        del_issue = QtWidgets.QPushButton("Remove")
        del_issue.clicked.connect(self._remove_issue)
        issue_btn_row.addWidget(add_issue)
        issue_btn_row.addWidget(edit_issue)
        issue_btn_row.addWidget(del_issue)

        # Bottom actions
        btn_row = QtWidgets.QHBoxLayout()
        main_layout.addLayout(btn_row)
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self._load_from_file)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_to_file)
        btn_row.addWidget(reload_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)

        self.movements: List[str] = []
        self.issues: List[str] = []
        self.movement_settings: Dict[str, Dict] = {}
        self._load_from_file()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def _load_from_file(self):
        cfg = load_label_config()
        self.movements = cfg.get("movements") or []
        self.issues = cfg.get("issues") or []
        self.movement_settings = cfg.get("movement_settings") or {}

        self.movement_list.clear()
        self.movement_list.addItems(self.movements)
        self.issue_list.clear()
        self.issue_list.addItems(self.issues)
        self.movement_info.clear()

    def _save_to_file(self):
        movements = [
            self.movement_list.item(i).text() for i in range(self.movement_list.count())
        ]
        issues = [
            self.issue_list.item(i).text() for i in range(self.issue_list.count())
        ]
        settings = {
            name: self.movement_settings.get(name, default_movement_settings(name))
            for name in movements
        }
        save_label_config(
            {
                "movements": movements,
                "issues": issues,
                "movement_settings": settings,
            }
        )
        QtWidgets.QMessageBox.information(self, "Saved", "Configuration updated.")
        self.config_saved.emit()

    def _add_movement(self):
        dialog = MovementDialog(self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            name, settings = dialog.values()
            if name in self.movements:
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate", f"{name} already exists."
                )
                return
            self.movements.append(name)
            self.movement_settings[name] = settings
            self.movement_list.addItem(name)

    def _edit_movement(self):
        item = self.movement_list.currentItem()
        if not item:
            return
        old_name = item.text()
        settings = self.movement_settings.get(
            old_name, default_movement_settings(old_name)
        )
        dialog = MovementDialog(self, name=old_name, settings=settings)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_name, new_settings = dialog.values()
            if new_name != old_name and new_name in self.movements:
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate", f"{new_name} already exists."
                )
                return
            idx = self.movements.index(old_name)
            self.movements[idx] = new_name
            del self.movement_settings[old_name]
            self.movement_settings[new_name] = new_settings
            item.setText(new_name)

    def _remove_movement(self):
        item = self.movement_list.currentItem()
        if not item:
            return
        name = item.text()
        self.movements.remove(name)
        self.movement_settings.pop(name, None)
        self.movement_list.takeItem(self.movement_list.row(item))
        self.movement_info.clear()

    def _show_movement_settings(self, name: str):
        if not name:
            self.movement_info.clear()
            return
        settings = self.movement_settings.get(name, default_movement_settings(name))
        pretty = json.dumps(settings, indent=2)
        self.movement_info.setPlainText(pretty)

    def _add_issue(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add issue tag", "Tag name")
        if ok and text.strip():
            value = text.strip()
            self.issues.append(value)
            self.issue_list.addItem(value)

    def _edit_issue(self):
        item = self.issue_list.currentItem()
        if not item:
            return
        current = item.text()
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Edit issue tag", "Tag name", text=current
        )
        if ok and text.strip():
            value = text.strip()
            index = self.issue_list.row(item)
            self.issues[index] = value
            item.setText(value)

    def _remove_issue(self):
        item = self.issue_list.currentItem()
        if not item:
            return
        index = self.issue_list.row(item)
        self.issue_list.takeItem(index)
        if 0 <= index < len(self.issues):
            self.issues.pop(index)


class HomePage(QtWidgets.QWidget):
    requested_admin = QtCore.Signal()
    requested_cutting = QtCore.Signal()
    requested_labeling = QtCore.Signal()
    requested_pose = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "Welcome to the MAI Coach tool suite.\nChoose an option to get started.",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        button_grid = QtWidgets.QGridLayout()
        layout.addLayout(button_grid)

        def add_button(row, col, text, signal):
            btn = QtWidgets.QPushButton(text)
            btn.setMinimumHeight(80)
            button_grid.addWidget(btn, row, col)
            btn.clicked.connect(signal)

        add_button(0, 0, "Admin Controls", self.requested_admin.emit)
        add_button(0, 1, "Video Cutting", self.requested_cutting.emit)
        add_button(1, 0, "Video Labeling", self.requested_labeling.emit)
        add_button(1, 1, "Pose Tuning", self.requested_pose.emit)
        layout.addStretch(1)


class WorkflowPlaceholderPage(QtWidgets.QWidget):
    def __init__(self, title: str, description: str):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        layout = QtWidgets.QVBoxLayout(self)
        header_row = QtWidgets.QHBoxLayout()
        header = QtWidgets.QLabel(title)
        header.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch(1)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._go_home)
        header_row.addWidget(self.home_button)
        layout.addLayout(header_row)

        self.description_label = QtWidgets.QLabel(
            description, alignment=QtCore.Qt.AlignCenter
        )
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        layout.addWidget(QtWidgets.QLabel("Selected files:"))
        self.file_list = QtWidgets.QListWidget()
        layout.addWidget(self.file_list, stretch=1)

    def set_files(self, files: List[Path]):
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(str(f))

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()


class LabelerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.issues, self.movement_settings = load_label_options()
        self.current_body_parts = BODY_PART_OPTIONS.copy()
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

        self._build_ui()
        self._update_timer_interval()

    # UI ------------------------------------------------------------------
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
        self.video_label.setMinimumSize(640, 360)
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

        # right: form
        right = QtWidgets.QVBoxLayout()
        content.addLayout(right, stretch=2)

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
        self.quality_cb.addItems(QUALITY_OPTIONS)
        self.quality_cb.setCurrentText("3")
        form.addRow("overall_quality", self.quality_cb)

        self.load_spin = QtWidgets.QSpinBox()
        self.load_spin.setRange(0, 2000)
        self.load_spin.setSuffix(" lbs")
        self.load_spin.setSingleStep(5)
        form.addRow("load_lbs", self.load_spin)

        self.rpe_cb = QtWidgets.QComboBox()
        self.rpe_cb.addItems(RPE_OPTIONS)
        self.rpe_cb.setCurrentText("1.0")
        form.addRow("RPE", self.rpe_cb)

        self.camera_cb = QtWidgets.QComboBox()
        self.camera_cb.addItems(CAMERA_ANGLE_OPTIONS)
        form.addRow("camera_angle", self.camera_cb)

        self.lens_cb = QtWidgets.QComboBox()
        self.lens_cb.addItems(LENS_OPTIONS)
        form.addRow("lens", self.lens_cb)

        form.addRow(QtWidgets.QLabel("Issues (hold Cmd or Shift to multi-select):"))
        self.issue_list = QtWidgets.QListWidget()
        self.issue_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.issue_list.addItems(self.issues)
        form.addRow(self.issue_list)

        issue_tag_row = QtWidgets.QHBoxLayout()
        self.issue_picker = QtWidgets.QComboBox()
        self.issue_picker.addItems(self.issues)
        issue_tag_row.addWidget(self.issue_picker)
        add_tag = QtWidgets.QPushButton("Add tag @ frame")
        add_tag.clicked.connect(self._add_issue_tag)
        issue_tag_row.addWidget(add_tag)
        right.addLayout(issue_tag_row)

        self.issue_events = QtWidgets.QListWidget()
        right.addWidget(self.issue_events, stretch=1)

        remove_tag = QtWidgets.QPushButton("Remove selected tag")
        remove_tag.clicked.connect(self._remove_issue_tag)
        right.addWidget(remove_tag)

        nav = QtWidgets.QHBoxLayout()
        right.addLayout(nav)
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda: self._load_relative(-1))
        nav.addWidget(self.prev_btn)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save_dataset)
        nav.addWidget(self.save_btn)
        self.next_btn = QtWidgets.QPushButton("Save + Next")
        self.next_btn.clicked.connect(lambda: self._load_relative(+1, save=True))
        nav.addWidget(self.next_btn)
        self._update_body_part_preview()
        self._update_nav_buttons()
        self._update_nav_buttons()
        self._update_nav_buttons()

    def refresh_label_options(self):
        self.movements, self.issues, self.movement_settings = load_label_options()
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

        selected = (
            set(self.session.current_dataset.get("issues", []))
            if self.session.current_dataset
            else set()
        )
        self.issue_list.clear()
        self.issue_list.addItems(self.issues)
        for i in range(self.issue_list.count()):
            item = self.issue_list.item(i)
            item.setSelected(item.text() in selected)

        current_issue = self.issue_picker.currentText()
        self.issue_picker.blockSignals(True)
        self.issue_picker.clear()
        self.issue_picker.addItems(self.issues)
        idx = self.issue_picker.findText(current_issue)
        if idx >= 0:
            self.issue_picker.setCurrentIndex(idx)
        self.issue_picker.blockSignals(False)
        self._update_body_part_preview()

    def _on_movement_changed(self, name: str):
        if self.session.current_dataset is not None:
            self.session.current_dataset["movement"] = name
        self._update_body_part_preview()

    def _update_body_part_preview(self):
        movement = self.movement_cb.currentText().strip()
        settings = self.movement_settings.get(movement)
        if not settings:
            settings = default_movement_settings(movement)
        self.current_body_parts = settings.get("body_parts") or BODY_PART_OPTIONS.copy()

    def _add_btn(self, layout: QtWidgets.QHBoxLayout, text: str, slot):
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(slot)
        layout.addWidget(btn)

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
        ll = d.get("load_lbs") or 0
        self.load_spin.setValue(float(ll))
        self.rpe_cb.setCurrentText(str(d.get("rpe", "1.0")))
        self.camera_cb.setCurrentText(d.get("camera_angle") or "front")
        self.lens_cb.setCurrentText(d.get("lens") or "0.5")
        self._update_body_part_preview()

        issues = set(d.get("issues") or [])
        for i in range(self.issue_list.count()):
            item = self.issue_list.item(i)
            item.setSelected(item.text() in issues)

        self._refresh_issue_events()

    def _refresh_issue_events(self):
        self.issue_events.clear()
        if not self.session.current_dataset:
            return
        for evt in self.session.current_dataset.get("issue_events", []):
            txt = f"frame={evt.get('frame_index', '?')} time={evt.get('time_ms', '?')}ms  {evt.get('issue')}"
            self.issue_events.addItem(txt)

    def _update_nav_buttons(self):
        total = len(self.session.video_paths)
        idx = self.session.current_index
        has_video = total > 0 and idx >= 0
        finish = self._auto_finish and has_video and idx >= total - 1
        self.prev_btn.setEnabled(has_video and idx > 0)
        self.save_btn.setEnabled(has_video)
        self.next_btn.setEnabled(has_video)
        self.next_btn.setText("Save & Finish" if finish else "Save + Next")

    # Playback --------------------------------------------------------------
    def _render_frame(self, index: int):
        if not self.session.has_video:
            return
        index = max(0, min(index, self.session.total_frames - 1))
        self.current_frame = index
        frame = self.session.frames_bgr[index].copy()
        dataset = self.session.current_dataset
        if dataset and 0 <= index < len(dataset.get("frames", [])):
            frec = dataset["frames"][index]
            if frec.get("pose_present") and frec.get("landmarks"):
                draw_upper_body_overlay(
                    frame, frec["landmarks"], self.current_body_parts
                )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        scaled = qt_image.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(index)
        self.scrubber.blockSignals(False)

    def _advance_frame(self):
        if not (self.playing and self.session.has_video):
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
        next_index = self.current_frame + step
        if next_index >= self.session.total_frames:
            next_index = self.session.total_frames - 1
            self.playing = False
            self.play_btn.setText("Play")
        self._render_frame(next_index)

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._update_timer_interval()

    def _replay(self):
        self.playing = True
        self.play_btn.setText("Pause")
        self._update_timer_interval()
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if not self.session.has_video:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self._render_frame(self.current_frame + delta)

    def _step_seconds(self, seconds: float):
        fps = self.session.fps if self.session.fps > 0 else 30.0
        delta_frames = int(round(seconds * fps))
        if delta_frames == 0:
            delta_frames = 1 if seconds > 0 else -1
        self._step_frames(delta_frames)

    def _change_speed(self, text: str):
        try:
            self.playback_speed = max(0.1, float(text.replace("x", "")))
        except ValueError:
            self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._update_timer_interval()

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _scrubbed(self, value: int):
        if self.session.has_video:
            self._render_frame(value)

    def _update_timer_interval(self):
        fps = self.session.fps if self.session.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)

    # Issue tagging ---------------------------------------------------------
    def _add_issue_tag(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        frames = dataset.get("frames") or []
        if not frames:
            return
        fi = max(0, min(self.current_frame, len(frames) - 1))
        tag = self.issue_picker.currentText()
        time_ms = frames[fi].get(
            "time_ms", int(round(fi * 1000.0 / (dataset.get("fps") or 30.0)))
        )
        dataset.setdefault("issue_events", []).append(
            {"issue": tag, "frame_index": int(fi), "time_ms": int(time_ms)}
        )
        self._refresh_issue_events()

    def _remove_issue_tag(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        sel = self.issue_events.selectedIndexes()
        if not sel:
            return
        idx = sel[0].row()
        events = dataset.get("issue_events", [])
        if 0 <= idx < len(events):
            events.pop(idx)
            self._refresh_issue_events()

    # Saving ---------------------------------------------------------
    def _save_dataset(self):
        dataset = self.session.current_dataset
        if not dataset:
            return
        dataset["rep_id"] = self.rep_id.text()
        dataset["movement"] = self.movement_cb.currentText().strip()
        dataset["overall_quality"] = int(self.quality_cb.currentText())
        dataset["load_lbs"] = int(self.load_spin.value())
        dataset["rpe"] = float(self.rpe_cb.currentText())
        dataset["camera_angle"] = self.camera_cb.currentText()
        dataset["lens"] = self.lens_cb.currentText()
        selected = [item.text() for item in self.issue_list.selectedItems()]
        dataset["issues"] = selected
        self.session.save_current_dataset()
        QtWidgets.QMessageBox.information(self, "Saved", "Dataset JSON saved.")


class VideoCutView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        self.videos: List[Path] = []
        self.current_index = -1
        self.frames: List[np.ndarray] = []
        self.fps = 30.0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._speed_residual = 0.0
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        self.play_timer.start(30)
        self.pending_start_ms: Optional[int] = None
        self.cuts: Dict[Path, List[tuple[int, int]]] = {}
        self.output_dir: Optional[Path] = None

        self._build_ui()
        self._change_speed("1.0x")

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load Videos")
        self.load_button.clicked.connect(self._pick_videos)
        self.output_button = QtWidgets.QPushButton("Choose Output Folder")
        self.output_button.clicked.connect(self._pick_output)
        top.addWidget(self.load_button)
        top.addWidget(self.output_button)
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
        self.mark_start_btn = QtWidgets.QPushButton("Mark In")
        self.mark_start_btn.clicked.connect(self._mark_start)
        self.mark_end_btn = QtWidgets.QPushButton("Mark Out")
        self.mark_end_btn.clicked.connect(self._mark_end)
        mark_row.addWidget(self.mark_start_btn)
        mark_row.addWidget(self.mark_end_btn)
        mark_row.addStretch(1)

        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._pause_for_scrub)
        self.scrubber.valueChanged.connect(self._scrubbed)
        left.addWidget(self.scrubber)

        self.status_label = QtWidgets.QLabel("")
        left.addWidget(self.status_label)

        # Right side: cuts list and options
        right = QtWidgets.QVBoxLayout()
        content.addLayout(right, stretch=2)

        self.cut_list = QtWidgets.QListWidget()
        right.addWidget(QtWidgets.QLabel("Marked clips"))
        right.addWidget(self.cut_list, stretch=1)

        pad_row = QtWidgets.QHBoxLayout()
        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 2000)
        self.pad_spin.setValue(120)
        self.pad_spin.setSuffix(" ms pad")
        pad_row.addWidget(QtWidgets.QLabel("Padding:"))
        pad_row.addWidget(self.pad_spin)
        right.addLayout(pad_row)

        cut_buttons = QtWidgets.QHBoxLayout()
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_cut)
        clear_btn = QtWidgets.QPushButton("Clear clips")
        clear_btn.clicked.connect(self._clear_cuts)
        cut_buttons.addWidget(remove_btn)
        cut_buttons.addWidget(clear_btn)
        right.addLayout(cut_buttons)

        export_btn = QtWidgets.QPushButton("Export clips")
        export_btn.clicked.connect(self._export_clips)
        right.addWidget(export_btn)

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_cutting_inputs(self, videos: List[Path]):
        if videos:
            self.videos = videos
            self.current_index = -1
            self._load_video(0)

    def _pick_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        self.videos = [Path(f) for f in files]
        self.current_index = -1
        self._load_video(0)

    def _pick_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path.home())
        )
        if folder:
            self.output_dir = Path(folder)
            self.status_label.setText(f"Output: {self.output_dir}")

    def _load_video(self, index: int):
        if not self.videos:
            return
        index = max(0, min(index, len(self.videos) - 1))
        if index == self.current_index:
            return
        path = self.videos[index]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open {path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            QtWidgets.QMessageBox.critical(self, "Error", f"No frames in {path}")
            return
        self.current_index = index
        self.frames = frames
        self.fps = fps
        self.current_frame = 0
        self._speed_residual = 0.0
        self.scrubber.setRange(0, max(0, len(self.frames) - 1))
        self.pending_start_ms = None
        self.playing = False
        self.play_btn.setText("Play")
        self._update_timer_interval()
        self._render_frame(0)
        self._refresh_cut_list()

    def _render_frame(self, idx: int):
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        self.current_frame = idx
        frame = self.frames[idx]
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

    def _mark_start(self):
        self.pending_start_ms = self._current_time_ms()
        self.status_label.setText(f"Start marked at {self.pending_start_ms:.0f} ms")

    def _mark_end(self):
        if self.pending_start_ms is None:
            self.status_label.setText("Set a start point first.")
            return
        end_ms = self._current_time_ms()
        start_ms = self.pending_start_ms
        if end_ms <= start_ms:
            self.status_label.setText("End must be after start.")
            return
        path = self.videos[self.current_index]
        self.cuts.setdefault(path, []).append((int(start_ms), int(end_ms)))
        self.pending_start_ms = None
        self._refresh_cut_list()

    def _refresh_cut_list(self):
        self.cut_list.clear()
        if not self.videos or self.current_index < 0:
            return
        path = self.videos[self.current_index]
        clips = self.cuts.get(path, [])
        for idx, (start, end) in enumerate(clips, 1):
            self.cut_list.addItem(
                f"{idx}. {start / 1000:.2f}s -> {end / 1000:.2f}s (len {(end - start) / 1000:.2f}s)"
            )

    def _remove_selected_cut(self):
        if not self.videos:
            return
        row = self.cut_list.currentRow()
        if row < 0:
            return
        path = self.videos[self.current_index]
        clips = self.cuts.get(path, [])
        if 0 <= row < len(clips):
            clips.pop(row)
            self._refresh_cut_list()

    def _clear_cuts(self):
        if not self.videos:
            return
        path = self.videos[self.current_index]
        self.cuts[path] = []
        self._refresh_cut_list()

    def _export_clips(self):
        if not self.videos:
            return
        out_dir = self.output_dir or QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path.home())
        )
        if not out_dir:
            return
        out_dir = Path(out_dir)
        pad = self.pad_spin.value()
        for video in self.videos:
            clips = self.cuts.get(video, [])
            if not clips:
                continue
            for idx, (start, end) in enumerate(clips, 1):
                s = max(0, start - pad)
                e = end + pad
                stem = video.stem
                out_path = out_dir / f"{stem}_clip{idx:02d}.mp4"
                self._run_ffmpeg(video, out_path, s, e)
        QtWidgets.QMessageBox.information(self, "Done", "Export completed.")

    def _run_ffmpeg(self, src: Path, dst: Path, start_ms: int, end_ms: int):
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-to",
                f"{end_ms / 1000:.3f}",
                "-i",
                str(src),
                "-c",
                "copy",
                str(dst),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _current_time_ms(self) -> float:
        if not self.frames:
            return 0.0
        return (self.current_frame / (self.fps if self.fps > 0 else 30.0)) * 1000.0

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frame(self):
        if not (self.playing and self.frames):
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
        if next_idx >= len(self.frames):
            next_idx = len(self.frames) - 1
            self.playing = False
            self.play_btn.setText("Play")
        self._render_frame(next_idx)

    def _replay(self):
        if not self.frames:
            return
        self.playing = True
        self.play_btn.setText("Pause")
        self._render_frame(0)

    def _step_frames(self, delta: int):
        if not self.frames:
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
            self.playback_speed = max(0.1, float(text.replace("x", "")))
        except ValueError:
            self.playback_speed = 1.0
        self._speed_residual = 0.0
        self._update_timer_interval()

    def _pause_for_scrub(self):
        self.playing = False
        self.play_btn.setText("Play")

    def _scrubbed(self, value: int):
        if self.frames:
            self._render_frame(value)

    def _update_timer_interval(self):
        fps = self.fps if self.fps > 0 else 30.0
        effective = self.playback_speed if self.playback_speed < 1.0 else 1.0
        interval = max(10, int(1000 / (fps * max(effective, 0.1))))
        self.play_timer.setInterval(interval)


class PoseTunerView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.movements, self.issues, self.movement_settings = load_label_options()
        self.dataset_dir: Optional[Path] = None
        self.video_entries: List[Dict] = []
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frames)
        self.play_timer.start(30)
        self.playing = False
        self.playback_speed = 1.0
        self._home_cb: Optional[Callable[[], None]] = None

        self._build_ui()

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
            hint.setStyleSheet("color: #888; font-size: 11px;")
            hint.setWordWrap(True)
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

        self.det_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, value=0.5, singleStep=0.05
        )
        self.det_spin.setToolTip(
            "Detection confidence threshold (higher = fewer detections)."
        )
        self.prs_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, value=0.7, singleStep=0.05
        )
        self.prs_spin.setToolTip(
            "Pose score threshold used when filtering landmark quality."
        )
        self.trk_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=1.0, value=0.7, singleStep=0.05
        )
        self.trk_spin.setToolTip(
            "Tracker confidence threshold before dropping a track."
        )
        self.ema_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.0, maximum=1.0, value=0.25, singleStep=0.05
        )
        self.ema_spin.setToolTip(
            "Exponential moving average amount for smoothing landmarks."
        )
        self.seg_check = QtWidgets.QCheckBox("Enable segmentation masks")
        self.seg_check.setToolTip("Overlay segmentation masks when available.")
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

        # Right: video grid and controls
        video_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(video_panel, stretch=2)

        self.video_grid = QtWidgets.QGridLayout()
        video_panel.addLayout(self.video_grid)
        self.video_labels: List[QtWidgets.QLabel] = []
        for idx in range(4):
            label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            label.setStyleSheet("background: #111; border: 1px solid #333;")
            label.setMinimumSize(420, 260)
            self.video_labels.append(label)
            row, col = divmod(idx, 2)
            self.video_grid.addWidget(label, row, col)

        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
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
        self.refresh_movements()

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def load_pose_inputs(self, videos: List[Path], dataset_dir: Optional[Path]):
        if dataset_dir:
            self.dataset_dir = dataset_dir
        if videos:
            self._load_videos(videos[:4])
        else:
            self.refresh_movements()

    def _pick_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select up to 6 videos",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        if not files:
            return
        self._load_videos([Path(f) for f in files[:4]])

    def _pick_dataset_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if folder:
            self.dataset_dir = Path(folder)
            self.status_label.setText(f"Dataset dir: {self.dataset_dir}")
            self.refresh_movements()

    def refresh_movements(self):
        self.movements, self.issues, self.movement_settings = load_label_options()
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
        if not self.dataset_dir:
            QtWidgets.QMessageBox.warning(
                self, "Missing dataset", "Select a dataset folder first."
            )
            return
        self.video_entries.clear()
        for label in self.video_labels:
            label.clear()
        for idx, path in enumerate(videos):
            entry = self._load_single_video(path)
            if entry:
                entry["label"] = self.video_labels[idx]
                self.video_entries.append(entry)
        if self.video_entries:
            self.scrubber.setEnabled(True)
            self.scrubber.setValue(0)
            self.playing = False
            self.play_btn.setText("Play")
            self._refresh_all_frames()

    def _load_single_video(self, path: Path) -> Optional[Dict]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Video error", f"Failed to open {path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return None
        json_path = self.dataset_dir / f"{path.stem}.json"
        pose_frames = None
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                pose_frames = data.get("frames", [])
            except Exception:
                pose_frames = None
        return {
            "path": path,
            "frames": frames,
            "pose": pose_frames,
            "fps": fps,
            "current_frame": 0,
            "next_time": time.perf_counter(),
        }

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _advance_frames(self):
        if not (self.playing and self.video_entries):
            return
        for entry in self.video_entries:
            fps = entry["fps"]
            interval = 1.0 / (fps * max(self.playback_speed, 0.1))
            now = time.perf_counter()
            if now >= entry.get("next_time", 0):
                entry["current_frame"] = min(
                    len(entry["frames"]) - 1, entry["current_frame"] + 1
                )
                entry["next_time"] = now + interval
                self._render_entry(entry)

    def _render_entry(self, entry: Dict):
        frames = entry["frames"]
        idx = max(0, min(entry["current_frame"], len(frames) - 1))
        frame = frames[idx].copy()
        pose_frames = entry.get("pose")
        allowed = self._selected_body_parts()
        if pose_frames and idx < len(pose_frames):
            frec = pose_frames[idx]
            if frec and frec.get("pose_present") and frec.get("landmarks"):
                draw_upper_body_overlay(frame, frec["landmarks"], allowed)
        else:
            if allowed and "full_body" not in allowed:
                # Dim frame if limiting body parts without pose
                frame[:] = frame * 0.8
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        label: QtWidgets.QLabel = entry["label"]
        scaled = image.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(QtGui.QPixmap.fromImage(scaled))

    def _refresh_all_frames(self):
        for entry in self.video_entries:
            self._render_entry(entry)

    def _selected_body_parts(self) -> List[str]:
        selected = [chk.text() for chk in self.body_part_checks if chk.isChecked()]
        return selected or BODY_PART_OPTIONS.copy()

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
            frames = entry["frames"]
            if frames:
                entry["current_frame"] = int(ratio * (len(frames) - 1))
                entry["next_time"] = time.perf_counter()
                self._render_entry(entry)

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
            "model": "full",
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
                "issues": self.issues,
                "movement_settings": self.movement_settings,
            }
        )

    def _load_settings_for_movement(self, name: str):
        settings = self.movement_settings.get(name)
        if not settings:
            settings = default_movement_settings(name)
        self.det_spin.setValue(settings.get("det", 0.5))
        self.prs_spin.setValue(settings.get("prs", 0.7))
        self.trk_spin.setValue(settings.get("trk", 0.7))
        self.ema_spin.setValue(settings.get("ema", 0.25))
        self.seg_check.setChecked(settings.get("seg", False))
        parts = set(settings.get("body_parts") or BODY_PART_OPTIONS)
        for chk in self.body_part_checks:
            chk.setChecked(chk.text() in parts)
        self._refresh_all_frames()


# --- Main window ------------------------------------------------------------


class UnifiedToolWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAI Coach Tools")
        self.resize(1600, 900)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.admin_page = AdminPanel()
        self.labeler_page = LabelerView()
        self.cutting_page = VideoCutView()
        self.pose_page = PoseTunerView()
        self.admin_page.set_home_callback(self.show_home)
        self.labeler_page.set_home_callback(self.show_home)
        self.cutting_page.set_home_callback(self.show_home)
        self.pose_page.set_home_callback(self.show_home)

        for page in [
            self.home_page,
            self.admin_page,
            self.labeler_page,
            self.cutting_page,
            self.pose_page,
        ]:
            self.stack.addWidget(page)

        toolbar = self.addToolBar("Navigation")
        home_action = QtGui.QAction("Home", self)
        home_action.triggered.connect(self.show_home)
        toolbar.addAction(home_action)

        self.home_page.requested_admin.connect(lambda: self.show_page(self.admin_page))
        self.home_page.requested_labeling.connect(self.start_labeling_workflow)
        self.home_page.requested_cutting.connect(self.start_cutting_workflow)
        self.home_page.requested_pose.connect(self.start_pose_workflow)

        self.admin_page.config_saved.connect(self.labeler_page.refresh_label_options)
        self.admin_page.config_saved.connect(self._reload_pose_settings)

        self.show_home()

    def show_home(self):
        self.stack.setCurrentWidget(self.home_page)

    def show_page(self, widget: QtWidgets.QWidget):
        self.stack.setCurrentWidget(widget)

    def _select_videos(self) -> List[Path]:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi)",
        )
        return [Path(f) for f in files] if files else []

    def start_labeling_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        dataset_dir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if not dataset_dir_str:
            return
        dataset_dir = Path(dataset_dir_str)
        self.labeler_page.load_labeler_inputs(videos, dataset_dir)
        self.show_page(self.labeler_page)

    def start_cutting_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        self.cutting_page.load_cutting_inputs(videos)
        self.show_page(self.cutting_page)

    def start_pose_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        dataset_dir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if not dataset_dir_str:
            return
        dataset_dir = Path(dataset_dir_str)
        self.pose_page.load_pose_inputs(videos, dataset_dir)
        self.show_page(self.pose_page)

    def _reload_pose_settings(self):
        self.pose_page.refresh_movements()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = UnifiedToolWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
