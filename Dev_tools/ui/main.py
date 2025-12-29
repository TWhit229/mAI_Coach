"""Main entry point for the Dev Tools UI."""

import sys
from pathlib import Path
from typing import List
from PySide6 import QtCore, QtWidgets

from ui.admin import AdminPanel, DatasetTrainerView
from ui.labeler import LabelerView
from ui.video_cut import VideoCutView
from ui.pose_tuner import PoseTunerView

class HomePage(QtWidgets.QWidget):
    requested_admin = QtCore.Signal()
    requested_cutting = QtCore.Signal()
    requested_labeling = QtCore.Signal()
    requested_pose = QtCore.Signal()
    requested_training = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "Welcome to the mAI Coach tool suite.\nChoose an option to get started.",
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
        add_button(2, 0, "Dataset + Model", self.requested_training.emit)
        layout.addStretch(1)


class UnifiedToolWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mAI Coach Tools")
        self.resize(1600, 900)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.admin_page = AdminPanel()
        self.labeler_page = LabelerView()
        self.cutting_page = VideoCutView()
        self.pose_page = PoseTunerView()
        self.training_page = DatasetTrainerView()
        self.admin_page.set_home_callback(self.show_home)
        self.labeler_page.set_home_callback(self.show_home)
        self.cutting_page.set_home_callback(self.show_home)
        self.pose_page.set_home_callback(self.show_home)
        self.training_page.set_home_callback(self.show_home)

        for page in [
            self.home_page,
            self.admin_page,
            self.labeler_page,
            self.cutting_page,
            self.pose_page,
            self.training_page,
        ]:
            self.stack.addWidget(page)

        self.home_page.requested_admin.connect(lambda: self.show_page(self.admin_page))
        self.home_page.requested_labeling.connect(self.start_labeling_workflow)
        self.home_page.requested_cutting.connect(self.start_cutting_workflow)
        self.home_page.requested_pose.connect(self.start_pose_workflow)
        self.home_page.requested_training.connect(self.start_training_workflow)

        self.admin_page.config_saved.connect(self.labeler_page.refresh_label_options)
        self.admin_page.config_saved.connect(self._reload_pose_settings)
        self.admin_page.config_saved.connect(self.training_page.reload_presets)

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
        self.show_page(self.cutting_page)
        if not self.cutting_page.start_new_session():
            self.show_home()

    def start_pose_workflow(self):
        videos = self._select_videos()
        if not videos:
            return
        self.pose_page.load_pose_inputs(videos, None)
        self.show_page(self.pose_page)

    def start_training_workflow(self):
        self.training_page.reload_presets()
        self.show_page(self.training_page)

    def _reload_pose_settings(self):
        self.pose_page.refresh_movements()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = UnifiedToolWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
