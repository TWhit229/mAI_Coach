#!/usr/bin/env python3
"""
Tool Suite launcher for the bench workflow tools.

Provides a simple Tk menu that can launch:
  - Data Labeler (bench_labeler.py)
  - Multi Video Pose Tuner
  - Pose Tasks Overlay Tuner
  - Auto Cut Video
and exposes an admin panel to edit movement names + issue tags consumed
by bench_labeler via label_config.json.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

from label_config import (
    LABEL_CONFIG_PATH,
    load_label_config,
    save_label_config,
    ensure_config_file,
)


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

TOOLS = [
    ("Data Labeler", ROOT_DIR / "bench_labeler.py"),
    ("Multi Video Pose Tuner", ROOT_DIR / "multi_video_pose_tuner.py"),
    ("Pose Tasks Overlay Tuner", ROOT_DIR / "pose_tasks_overlay_tuner.py"),
    ("Auto Cut Video", ROOT_DIR / "auto_cut_video.py"),
]


class LabelAdminPanel(tk.Toplevel):
    """Simple editor for movement + issue lists."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.title("Bench Labels Admin")
        self.resizable(False, False)
        self.config(padx=12, pady=12)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.cfg = load_label_config()
        self._build()

    def _build(self):
        info = ttk.Label(
            self,
            text=(
                "These lists feed the Data Labeler drop-downs. Changes are saved\n"
                "into label_config.json and will be picked up the next time the\n"
                "labeler launches."
            ),
            justify="left",
        )
        info.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self._sections = {}
        self._make_section("Lift / Movement Names", "movements", row=1)
        self._make_section("Issue Tags", "issues", row=2)

        save_btn = ttk.Button(self, text="Save Changes", command=self.save)
        save_btn.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        close_btn = ttk.Button(self, text="Close", command=self.close)
        close_btn.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))

    def _make_section(self, title: str, key: str, row: int):
        frame = ttk.LabelFrame(self, text=title, padding=10)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)

        listbox = tk.Listbox(frame, height=8, width=40)
        listbox.grid(row=0, column=0, columnspan=2, sticky="ew")

        for item in self.cfg.get(key, []):
            listbox.insert(tk.END, item)

        entry = ttk.Entry(frame, width=30)
        entry.grid(row=1, column=0, sticky="ew", pady=5)

        def add_item():
            text = entry.get().strip()
            if not text:
                return
            listbox.insert(tk.END, text)
            entry.delete(0, tk.END)

        def remove_selected():
            sel = list(listbox.curselection())
            for idx in reversed(sel):
                listbox.delete(idx)

        add_btn = ttk.Button(frame, text="Add", command=add_item)
        add_btn.grid(row=1, column=1, padx=(5, 0), sticky="w")

        rm_btn = ttk.Button(frame, text="Remove Selected", command=remove_selected)
        rm_btn.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._sections[key] = listbox

    def save(self):
        for key, listbox in self._sections.items():
            items = listbox.get(0, tk.END)
            self.cfg[key] = list(items)
        save_label_config(self.cfg)
        messagebox.showinfo("Saved", f"Updated {LABEL_CONFIG_PATH}")

    def close(self):
        self.destroy()


class ToolSuiteApp:
    def __init__(self):
        ensure_config_file()

        self.root = tk.Tk()
        self.root.title("Bench Tool Suite")
        self.root.geometry("420x320")
        self.root.resizable(False, False)

        ttk.Label(
            self.root,
            text="Select a tool to launch. Each one opens in its own window/process.",
            wraplength=380,
            justify="left",
        ).pack(padx=15, pady=(15, 10), anchor="w")

        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        for name, script in TOOLS:
            ttk.Button(
                btn_frame,
                text=name,
                command=lambda s=script: self.launch_tool(s),
            ).pack(fill=tk.X, pady=5)

        ttk.Separator(self.root).pack(fill=tk.X, padx=15, pady=5)

        ttk.Button(
            self.root,
            text="Open Admin Panel (labels & tags)",
            command=self.open_admin_panel,
        ).pack(fill=tk.X, padx=20, pady=(5, 10))

        ttk.Label(
            self.root,
            text=f"Config file: {LABEL_CONFIG_PATH}",
            wraplength=380,
            justify="left",
            font=("TkDefaultFont", 9),
        ).pack(padx=15, pady=(0, 10), anchor="w")

        self.admin_window: Optional[LabelAdminPanel] = None

    def launch_tool(self, script_path: Path):
        script_path = Path(script_path)
        if not script_path.exists():
            messagebox.showerror("Missing script", f"Cannot find {script_path}")
            return
        try:
            subprocess.Popen([sys.executable, str(script_path)])
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))

    def open_admin_panel(self):
        if self.admin_window and tk.Toplevel.winfo_exists(self.admin_window):
            self.admin_window.focus_set()
            return
        self.admin_window = LabelAdminPanel(self.root)

    def run(self):
        self.root.mainloop()


def main():
    app = ToolSuiteApp()
    app.run()


if __name__ == "__main__":
    main()
