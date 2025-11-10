#!/usr/bin/env python3
"""Shared configuration for bench labeling options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

LABEL_CONFIG_PATH = Path(__file__).resolve().parent / "label_config.json"

DEFAULT_CONFIG = {
    "movements": [
        "traditional_bench",
    ],
    "issues": [
        "01_angle_issues",
        "02_tracking_unreliable",
        "03_no_major_issues",
        "04_hands_too_wide",
        "05_hands_too_narrow",
        "06_grip_uneven",
        "07_body_placement_too_forward",
        "08_body_placement_too_backward",
        "09_bar_path_too_forward",
        "10_bar_path_too_backward",
        "11_elbows_flared",
        "12_elbows_flared_bottom",
        "13_elbows_flared_lockout",
        "14_elbows_tucked",
        "15_elbows_tucked_excessive",
        "16_pause_too_long",
        "17_pause_too_short",
        "18_no_pause_on_chest",
        "19_bar_depth_insufficient",
        "20_bar_depth_excessive",
        "21_barbell_tilt_right",
        "22_barbell_tilt_left",
        "23_uneven_lockout",
        "24_hips_off_bench",
        "25_descent_too_fast",
        "26_bounce_off_chest",
        "27_incomplete_lockout",
        "28_wrists_tilted_forward",
        "29_wrists_tilted_backward",
        "30_right_elbow_in",
        "31_left_elbow_in",
        "32_right_elbow_out",
        "33_left_elbow_out",
    ],
}


def _write_default() -> None:
    LABEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))


def ensure_config_file() -> None:
    """Create the config file with defaults if missing."""
    if not LABEL_CONFIG_PATH.exists():
        _write_default()


def _sanitize_list(values, fallback: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(fallback)
    cleaned = []
    for val in values:
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                cleaned.append(stripped)
    return cleaned or list(fallback)


def load_label_config() -> Dict[str, List[str]]:
    """Return dict with 'movements' and 'issues' lists."""
    ensure_config_file()
    data = json.loads(LABEL_CONFIG_PATH.read_text())
    cfg = {}
    for key, default in DEFAULT_CONFIG.items():
        cfg[key] = _sanitize_list(data.get(key), default)
    return cfg


def save_label_config(cfg: Dict[str, List[str]]) -> None:
    """Persist config (after sanitizing)."""
    clean = {}
    for key, default in DEFAULT_CONFIG.items():
        clean[key] = _sanitize_list(cfg.get(key), default)
    LABEL_CONFIG_PATH.write_text(json.dumps(clean, indent=2))


__all__ = [
    "LABEL_CONFIG_PATH",
    "DEFAULT_CONFIG",
    "ensure_config_file",
    "load_label_config",
    "save_label_config",
]
