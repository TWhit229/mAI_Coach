"""Admin and Training UI components."""

import concurrent.futures
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets

from core.config import DATA_DIR, MODEL_DIR
from core.training import (
    preprocess_dataset,
    train_dataset_model,
)
from label_config import (
    DEFAULT_ML_PRESETS,
    DEFAULT_TAGS,
    load_label_config,
    save_label_config,
    default_movement_settings,
    load_ml_presets,
    save_ml_presets,
    BODY_PART_OPTIONS,
    MODEL_VARIANTS,
)

# Shared with labeler, but defined here if labeler doesn't export it well, 
# or we move it to widgets later.
GRIP_WIDE_THRESHOLD = 2.0
GRIP_NARROW_THRESHOLD = 0.5
GRIP_UNEVEN_THRESHOLD = 0.15
BAR_TILT_THRESHOLD_DEG = 5.0

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

class AdminPanel(QtWidgets.QWidget):
    config_saved = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        main_layout = QtWidgets.QVBoxLayout(self)

        header_row = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel(
            "Manage lifts and form tags. Saving updates label_config.json."
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

        # Tags section
        tag_group = QtWidgets.QGroupBox("Form tags")
        splitter.addWidget(tag_group, stretch=1)
        tag_layout = QtWidgets.QVBoxLayout(tag_group)
        self.tag_list = QtWidgets.QListWidget()
        tag_layout.addWidget(self.tag_list)

        tag_btn_row = QtWidgets.QHBoxLayout()
        tag_layout.addLayout(tag_btn_row)
        add_tag = QtWidgets.QPushButton("Add")
        add_tag.clicked.connect(self._add_tag)
        edit_tag = QtWidgets.QPushButton("Edit")
        edit_tag.clicked.connect(self._edit_tag)
        del_tag = QtWidgets.QPushButton("Remove")
        del_tag.clicked.connect(self._remove_tag)
        tag_btn_row.addWidget(add_tag)
        tag_btn_row.addWidget(edit_tag)
        tag_btn_row.addWidget(del_tag)

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
        self.tags: List[str] = []
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
        self.tags = cfg.get("tags") or []
        raw_settings = cfg.get("movement_settings") or {}
        self.movement_settings = {}
        for name in self.movements:
            merged = default_movement_settings(name)
            merged.update(raw_settings.get(name) or {})
            self.movement_settings[name] = merged

        self.movement_list.clear()
        self.movement_list.addItems(self.movements)
        self.tag_list.clear()
        self.tag_list.addItems(self.tags)
        self.movement_info.clear()

    def _save_to_file(self):
        movements = [
            self.movement_list.item(i).text() for i in range(self.movement_list.count())
        ]
        tags = [self.tag_list.item(i).text() for i in range(self.tag_list.count())]
        settings = {
            name: self.movement_settings.get(name, default_movement_settings(name))
            for name in movements
        }
        save_label_config(
            {
                "movements": movements,
                "tags": tags,
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

    def _add_tag(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add form tag", "Tag name")
        if ok and text.strip():
            value = text.strip()
            self.tags.append(value)
            self.tag_list.addItem(value)

    def _edit_tag(self):
        item = self.tag_list.currentItem()
        if not item:
            return
        current = item.text()
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Edit form tag", "Tag name", text=current
        )
        if ok and text.strip():
            value = text.strip()
            index = self.tag_list.row(item)
            self.tags[index] = value
            item.setText(value)

    def _remove_tag(self):
        item = self.tag_list.currentItem()
        if not item:
            return
        index = self.tag_list.row(item)
        self.tag_list.takeItem(index)
        if 0 <= index < len(self.tags):
            self.tags.pop(index)

class DatasetTrainerView(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._home_cb: Optional[Callable[[], None]] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._future: Optional[concurrent.futures.Future] = None
        self.ml_presets: Dict[str, Dict] = {}
        self.default_tags: List[str] = []
        self.movements: List[str] = []

        self._build_ui()
        self.log_signal.connect(self._append_log)
        self.reload_presets()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Dataset + Model Trainer")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.status_label = QtWidgets.QLabel("")
        header_row.addWidget(self.status_label)
        home_btn = QtWidgets.QPushButton("Home")
        home_btn.clicked.connect(self._go_home)
        header_row.addWidget(home_btn)
        layout.addLayout(header_row)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset"))
        self.preset_cb = QtWidgets.QComboBox()
        self.preset_cb.currentTextChanged.connect(self._on_preset_change)
        preset_row.addWidget(self.preset_cb, stretch=1)
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.clicked.connect(self.reload_presets)
        preset_row.addWidget(self.reload_btn)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save_preset)
        preset_row.addWidget(self.save_btn)
        self.save_as_btn = QtWidgets.QPushButton("Save As")
        self.save_as_btn.clicked.connect(self._save_preset_as)
        preset_row.addWidget(self.save_as_btn)
        layout.addLayout(preset_row)

        self.tag_edit = QtWidgets.QPlainTextEdit()
        self.tag_edit.setPlaceholderText("Tag order (one per line or comma-separated)")
        tag_layout = QtWidgets.QVBoxLayout()
        tag_layout.addWidget(QtWidgets.QLabel("Tags (order for multi-hot labels)"))
        tag_layout.addWidget(self.tag_edit)

        pre_group = QtWidgets.QGroupBox("Preprocess JSON -> tensors")
        pre_form = QtWidgets.QFormLayout(pre_group)
        self.dataset_dir_edit = QtWidgets.QLineEdit()
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(self.dataset_dir_edit)
        dataset_btn = QtWidgets.QPushButton("Browse")
        dataset_btn.clicked.connect(self._browse_dataset_dir)
        dataset_row.addWidget(dataset_btn)
        pre_form.addRow("Dataset folder", dataset_row)

        self.pre_output_edit = QtWidgets.QLineEdit()
        pre_form.addRow("Output prefix", self.pre_output_edit)
        self.preprocess_btn = QtWidgets.QPushButton("Run preprocess")
        self.preprocess_btn.clicked.connect(self._start_preprocess)
        pre_form.addRow(self.preprocess_btn)

        train_group = QtWidgets.QGroupBox("Train MLP classifier")
        train_form = QtWidgets.QFormLayout(train_group)
        self.train_data_prefix_edit = QtWidgets.QLineEdit()
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(self.train_data_prefix_edit)
        data_btn = QtWidgets.QPushButton("Pick _X.npy")
        data_btn.clicked.connect(self._browse_data_prefix)
        data_row.addWidget(data_btn)
        train_form.addRow("Data prefix", data_row)

        self.train_output_prefix_edit = QtWidgets.QLineEdit()
        train_form.addRow("Output prefix", self.train_output_prefix_edit)

        self.epoch_spin = QtWidgets.QSpinBox()
        self.epoch_spin.setRange(1, 10000)
        self.epoch_spin.setValue(200)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(32)
        self.dev_spin = QtWidgets.QDoubleSpinBox()
        self.dev_spin.setRange(0.05, 0.9)
        self.dev_spin.setSingleStep(0.05)
        self.dev_spin.setValue(0.2)
        self.dev_spin.setDecimals(3)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 10_000)
        self.seed_spin.setValue(42)

        train_form.addRow("Epochs", self.epoch_spin)
        train_form.addRow("Batch size", self.batch_spin)
        train_form.addRow("Dev fraction", self.dev_spin)
        train_form.addRow("Seed", self.seed_spin)

        self.train_btn = QtWidgets.QPushButton("Run training")
        self.train_btn.clicked.connect(self._start_training)
        train_form.addRow(self.train_btn)

        top_split = QtWidgets.QHBoxLayout()
        top_split.addWidget(pre_group, stretch=1)
        top_split.addWidget(train_group, stretch=1)
        top_split.addLayout(tag_layout, stretch=1)
        layout.addLayout(top_split)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Logs"))
        layout.addWidget(self.log_box, stretch=1)

    def set_home_callback(self, cb: Callable[[], None]):
        self._home_cb = cb

    def _go_home(self):
        if self._home_cb:
            self._home_cb()

    def reload_presets(self):
        presets, tags, movements = load_ml_presets()
        self.default_tags = tags or self.default_tags
        self.movements = movements or self.movements
        if not presets:
            presets = self.ml_presets or {}
        if not presets:
            base = dict(DEFAULT_ML_PRESETS["bench"])
            base["tags"] = self.default_tags or base.get("tags") or DEFAULT_TAGS
            presets = {"bench": base}
        self.ml_presets = presets
        current = self.preset_cb.currentText()
        self.preset_cb.blockSignals(True)
        self.preset_cb.clear()
        names = list(presets.keys())
        extra = [m for m in self.movements if m not in names]
        names += extra
        if not names:
            names = ["bench"]
        for name in names:
            self.preset_cb.addItem(name)
        if current and self.preset_cb.findText(current) >= 0:
            self.preset_cb.setCurrentText(current)
        else:
            self.preset_cb.setCurrentIndex(0)
        self.preset_cb.blockSignals(False)
        self._load_preset(self.preset_cb.currentText())

    def _on_preset_change(self, name: str):
        if name:
            self._load_preset(name)

    def _load_preset(self, name: str):
        preset = self.ml_presets.get(name)
        if not preset:
            dataset_dir = DATA_DIR / "JSON"
            if "side" in name:
                dataset_dir = dataset_dir / "side"
            preset = {
                "preprocess": {
                    "dataset_dir": str(dataset_dir),
                    "output_prefix": str(DATA_DIR / f"{name}_v1"),
                },
                "train": {
                    "data_prefix": str(DATA_DIR / f"{name}_v1"),
                    "output_prefix": str(MODEL_DIR / f"{name}_mlp_v1"),
                    "epochs": 200,
                    "batch_size": 32,
                    "dev_fraction": 0.2,
                    "seed": 42,
                },
                "tags": self.default_tags,
            }
            self.ml_presets[name] = preset
        pre_cfg = preset.get("preprocess") or {}
        train_cfg = preset.get("train") or {}
        tags = preset.get("tags") or self.default_tags
        self.dataset_dir_edit.setText(str(pre_cfg.get("dataset_dir", "")))
        self.pre_output_edit.setText(str(pre_cfg.get("output_prefix", "")))
        self.train_data_prefix_edit.setText(str(train_cfg.get("data_prefix", "")))
        self.train_output_prefix_edit.setText(str(train_cfg.get("output_prefix", "")))
        self.epoch_spin.setValue(int(train_cfg.get("epochs", 200)))
        self.batch_spin.setValue(int(train_cfg.get("batch_size", 32)))
        self.dev_spin.setValue(float(train_cfg.get("dev_fraction", 0.2)))
        self.seed_spin.setValue(int(train_cfg.get("seed", 42)))
        self._set_tags_text(tags)

    def _set_tags_text(self, tags: Sequence[str]):
        cleaned = "\n".join(tags)
        self.tag_edit.blockSignals(True)
        self.tag_edit.setPlainText(cleaned)
        self.tag_edit.blockSignals(False)

    def _active_tags(self) -> List[str]:
        raw = self.tag_edit.toPlainText().replace(",", "\n").splitlines()
        tags = [t.strip() for t in raw if t.strip()]
        if not tags:
            tags = self.default_tags or DEFAULT_TAGS
        return tags

    def _save_preset(self):
        name = self.preset_cb.currentText().strip() or "traditional_bench"
        self._persist_preset(name)
        QtWidgets.QMessageBox.information(self, "Saved", f"Preset saved for {name}.")

    def _save_preset_as(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Save preset as", "Name")
        if not ok or not text.strip():
            return
        name = text.strip()
        if self.preset_cb.findText(name) < 0:
            self.preset_cb.addItem(name)
        self.preset_cb.setCurrentText(name)
        self._persist_preset(name)
        QtWidgets.QMessageBox.information(self, "Saved", f"Preset saved for {name}.")

    def _persist_preset(self, name: str):
        preset = {
            "preprocess": {
                "dataset_dir": self.dataset_dir_edit.text().strip(),
                "output_prefix": self.pre_output_edit.text().strip(),
            },
            "train": {
                "data_prefix": self.train_data_prefix_edit.text().strip(),
                "output_prefix": self.train_output_prefix_edit.text().strip(),
                "epochs": int(self.epoch_spin.value()),
                "batch_size": int(self.batch_spin.value()),
                "dev_fraction": float(self.dev_spin.value()),
                "seed": int(self.seed_spin.value()),
            },
            "tags": self._active_tags(),
        }
        self.ml_presets[name] = preset
        save_ml_presets(self.ml_presets)

    def _browse_dataset_dir(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(Path.home())
        )
        if selected:
            self.dataset_dir_edit.setText(selected)

    def _browse_data_prefix(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select features file (_X.npy)",
            str(Path.home()),
            "NumPy files (*.npy);;All files (*.*)",
        )
        if not fname:
            return
        path = Path(fname)
        if path.name.endswith("_X.npy"):
            prefix = str(path.with_name(path.name[:-6]))
        else:
            prefix = str(path.with_suffix(""))
        self.train_data_prefix_edit.setText(prefix)
        if not self.train_output_prefix_edit.text().strip():
            base = Path(prefix).name
            self.train_output_prefix_edit.setText(str(MODEL_DIR / f"{base}_mlp_v1"))

    def _start_preprocess(self):
        dataset_dir = self.dataset_dir_edit.text().strip()
        output_prefix = self.pre_output_edit.text().strip()
        tags = self._active_tags()
        name = self.preset_cb.currentText().strip() or "traditional_bench"
        preset = self.ml_presets.get(name, {})
        preset_pre = preset.get("preprocess") or {}
        if not dataset_dir:
            dataset_dir = preset_pre.get("dataset_dir") or str(
                (DATA_DIR / "JSON" / "side") if "side" in name else (DATA_DIR / "JSON")
            )
            self.dataset_dir_edit.setText(dataset_dir)
        if not output_prefix:
            output_prefix = preset_pre.get("output_prefix") or str(DATA_DIR / f"{name}_v1")
            self.pre_output_edit.setText(output_prefix)
        self._persist_preset(name)
        self.log_box.clear()
        self._run_job(
            lambda: self._run_preprocess_job(dataset_dir, output_prefix, tags),
            "Preprocessing...",
        )

    def _start_training(self):
        data_prefix = self.train_data_prefix_edit.text().strip()
        output_prefix = self.train_output_prefix_edit.text().strip()
        tags = self._active_tags()
        name = self.preset_cb.currentText().strip() or "traditional_bench"
        preset = self.ml_presets.get(name, {})
        preset_train = preset.get("train") or {}
        if not data_prefix:
            data_prefix = preset_train.get("data_prefix") or str(DATA_DIR / f"{name}_v1")
            self.train_data_prefix_edit.setText(data_prefix)
        if not output_prefix:
            output_prefix = preset_train.get("output_prefix") or str(MODEL_DIR / f"{name}_mlp_v1")
            self.train_output_prefix_edit.setText(output_prefix)
        self._persist_preset(name)
        self.log_box.clear()
        self._run_job(
            lambda: self._run_training_job(data_prefix, output_prefix, tags),
            "Training...",
        )

    def _run_preprocess_job(self, dataset_dir: str, output_prefix: str, tags: Sequence[str]):
        try:
            preprocess_dataset(Path(dataset_dir), Path(output_prefix), tags, self.log_signal.emit)
            return True, "Preprocess completed."
        except Exception as exc:
            return False, str(exc)

    def _run_training_job(self, data_prefix: str, output_prefix: str, tags: Sequence[str]):
        try:
            train_dataset_model(
                Path(data_prefix),
                Path(output_prefix),
                tags,
                int(self.epoch_spin.value()),
                int(self.batch_spin.value()),
                float(self.dev_spin.value()),
                int(self.seed_spin.value()),
                self.log_signal.emit,
            )
            return True, "Training completed."
        except Exception as exc:
            return False, str(exc)

    def _run_job(self, fn: Callable[[], Tuple[bool, str]], label: str):
        if self._future:
            return
        self._set_buttons_enabled(False)
        self.status_label.setText(label)
        if not self._executor:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = self._executor.submit(fn)
        self._future = future

        def _finish(fut):
            QtCore.QTimer.singleShot(0, lambda: self._on_job_finished(fut))

        future.add_done_callback(_finish)

    def _on_job_finished(self, future: concurrent.futures.Future):
        if future is not self._future:
            return
        self._future = None
        self._set_buttons_enabled(True)
        try:
            ok, message = future.result()
        except Exception as exc:
            ok = False
            message = str(exc)
        self.status_label.setText(message if ok else "Failed")
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Task failed", message)

    def _set_buttons_enabled(self, enabled: bool):
        self.preprocess_btn.setEnabled(enabled)
        self.train_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.save_as_btn.setEnabled(enabled)
        self.reload_btn.setEnabled(enabled)

    def _append_log(self, text: str):
        self.log_box.append(text)

    def closeEvent(self, event):
        if self._executor:
             self._executor.shutdown(wait=False)
        super().closeEvent(event)
