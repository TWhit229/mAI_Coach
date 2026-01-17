"""Reusable Qt widgets for the Dev_tools UI."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PySide6 import QtCore, QtGui, QtWidgets

from core.utils import draw_upper_body_overlay
from core.video import (
    video_rotation_degrees,
    rotate_frame_if_needed,
)

import json


class VideoSession(QtCore.QObject):
    """Manages video frames in memory for playback/labeling.
    
    Also manages a list of videos and their associated JSON datasets.
    """
    dataset_loaded = QtCore.Signal()

    def __init__(self, path: Optional[Path] = None, max_height: int = 720, load: bool = True):
        super().__init__()
        self.path = path
        self.max_height = max_height
        self.frames: List[np.ndarray] = []
        self.fps = 30.0
        self.rotation = 0
        self.current_dataset: Optional[Dict] = None
        # Multi-video management
        self.video_paths: List[Path] = []
        self.current_index: int = -1
        self.dataset_dir: Optional[Path] = None
        if load and path is not None:
            self.load()

    def set_video_list(self, paths) -> None:
        """Set the list of video paths to manage."""
        self.video_paths = [Path(p) for p in paths]
        self.current_index = -1
        self.current_dataset = None
        self.frames = []

    def set_dataset_dir(self, directory: Path) -> None:
        """Set the directory where JSON datasets are stored."""
        self.dataset_dir = directory

    def _json_path_for_video(self, video_path: Path) -> Optional[Path]:
        """Return the JSON dataset path for a given video."""
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / (video_path.stem + ".json")

    def load_index(self, index: int) -> bool:
        """Load a video by index from video_paths."""
        if index < 0 or index >= len(self.video_paths):
            return False
        self.current_index = index
        video_path = self.video_paths[index]
        self.load(video_path)
        # Try to load JSON dataset
        json_path = self._json_path_for_video(video_path)
        if json_path and json_path.exists():
            try:
                self.current_dataset = json.loads(json_path.read_text())
            except Exception:
                self.current_dataset = {}
        else:
            self.current_dataset = {}
        self.dataset_loaded.emit()
        return True

    def save_current_dataset(self) -> None:
        """Save current dataset to JSON with labeling data first."""
        if self.current_index < 0 or self.current_index >= len(self.video_paths):
            return
        if self.current_dataset is None:
            return
        video_path = self.video_paths[self.current_index]
        json_path = self._json_path_for_video(video_path)
        if json_path:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            # Reorder keys: labeling metadata first, frames last
            label_keys = [
                "rep_id", "movement", "overall_quality", "load_lbs", "rpe",
                "camera_angle", "lens", "tags", "auto_tags", "tag_events",
                "metrics", "tracking_unreliable", "tracking_auto_recommended",
                "tracking_manual_override", "rotation_override_degrees", "fps"
            ]
            ordered = {}
            # Add label keys first (if present)
            for key in label_keys:
                if key in self.current_dataset:
                    ordered[key] = self.current_dataset[key]
            # Add any other keys except 'frames'
            for key in self.current_dataset:
                if key not in ordered and key != "frames":
                    ordered[key] = self.current_dataset[key]
            # Add frames last (largest data)
            if "frames" in self.current_dataset:
                ordered["frames"] = self.current_dataset["frames"]
            json_path.write_text(json.dumps(ordered, indent=2))

    def load(self, path: Optional[Path] = None):
        if path is not None:
            self.path = path
        if self.path is None:
            return
        self.rotation = video_rotation_degrees(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                if h > self.max_height:
                    scale = self.max_height / float(h)
                    nh, nw = int(h * scale), int(w * scale)
                    if nw % 2:
                        nw -= 1
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                self.frames.append(frame)
        finally:
            cap.release()

    def get_frame(
        self,
        idx: int,
        rotation_override: Optional[int] = None,
        overlays: Optional[List[Dict]] = None,
        draw_opts: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        if idx < 0 or idx >= len(self.frames):
            return None
        frame = self.frames[idx]
        if rotation_override is not None:
            rot = rotation_override
        else:
            rot = self.rotation
        
        if rot or overlays:
            frame = frame.copy()
            if rot:
                frame = rotate_frame_if_needed(frame, rot)

        if overlays:
            for rec in overlays:
                lms = rec.get("landmarks")
                if lms:
                    allowed = (draw_opts or {}).get("parts")
                    draw_upper_body_overlay(frame, lms, allowed_parts=allowed)

        return frame

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return len(self.frames) / self.fps

    @property
    def current_video_path(self) -> Optional[Path]:
        """Return the path of the currently loaded video."""
        if self.current_index < 0 or self.current_index >= len(self.video_paths):
            return self.path
        return self.video_paths[self.current_index]

    @property
    def current_rotation(self) -> int:
        """Return the rotation of the currently loaded video."""
        return self.rotation

    @property
    def frame_count(self) -> int:
        """Alias for total_frames."""
        return len(self.frames)


class RangeSlider(QtWidgets.QWidget):
    """Two-handle slider for selecting a start/end range."""
    rangeChanged = QtCore.Signal(int, int)

    def __init__(self, orientation=QtCore.Qt.Horizontal):
        super().__init__()
        self.orientation = orientation
        self._min = 0
        self._max = 100
        self._low = 20
        self._high = 80
        self.setMinimumSize(200, 30)

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self.update()

    def setValues(self, low, high):
        self._low = max(self._min, min(high, low))
        self._high = min(self._max, max(high, low))
        self.update()

    def values(self):
        return self._low, self._high

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        # Draw groove
        mid_y = h // 2
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(200, 200, 200))
        painter.drawRoundedRect(5, mid_y - 2, w - 10, 4, 2, 2)

        # Draw range
        span = self._max - self._min
        if span <= 0:
            return
        
        # Mapping function
        def to_x(val):
            return 5 + (val - self._min) / span * (w - 10)

        x1 = to_x(self._low)
        x2 = to_x(self._high)

        painter.setBrush(QtGui.QColor(100, 150, 255))
        painter.drawRoundedRect(int(x1), mid_y - 2, int(x2 - x1), 4, 2, 2)

        # Draw handles
        painter.setBrush(QtGui.QColor(50, 50, 50))
        for x in [x1, x2]:
            painter.drawEllipse(QtCore.QPoint(int(x), mid_y), 8, 8)

    def mousePressEvent(self, event):
        x = event.x()
        w = self.width()
        span = self._max - self._min
        if span <= 0:
            return
        val = self._min + (x - 5) / (w - 10) * span
        val = max(self._min, min(self._max, val))
        
        # Determine which handle is closer
        d1 = abs(val - self._low)
        d2 = abs(val - self._high)
        if d1 < d2:
            self._low = int(val)
        else:
            self._high = int(val)
        
        # Enforce order
        if self._low > self._high:
            self._low, self._high = self._high, self._low
            
        self.rangeChanged.emit(self._low, self._high)
        self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.mousePressEvent(event)
