#!/usr/bin/env python3
"""Shared configuration for bench labeling options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

LABEL_CONFIG_PATH = Path(__file__).resolve().parent / "label_config.json"

DEFAULT_TAGS = [
    "no_major_issues",
    "hands_too_wide",
    "hands_too_narrow",
    "grip_uneven",
    "barbell_tilted",
    "lockout_incomplete",
    "bar_depth_insufficient",
]

EMPTY_CONFIG = {
    "movements": [],
    "tags": DEFAULT_TAGS,
    "movement_settings": {},
}


def ensure_config_file() -> None:
    """Create an empty config file if missing."""
    if LABEL_CONFIG_PATH.exists():
        return
    LABEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_CONFIG_PATH.write_text(json.dumps(EMPTY_CONFIG, indent=2))


def _sanitize_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for val in values:
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                cleaned.append(stripped)
    return cleaned


def _sanitize_movement_settings(values) -> Dict[str, Dict]:
    if not isinstance(values, dict):
        return {}
    clean = {}
    for name, settings in values.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(settings, dict):
            continue
        clean[name.strip()] = {
            "model": str(settings.get("model", "full")),
            "det": float(settings.get("det", 0.5)),
            "prs": float(settings.get("prs", 0.7)),
            "trk": float(settings.get("trk", 0.7)),
            "ema": float(settings.get("ema", 0.25)),
            "seg": bool(settings.get("seg", False)),
            "grip_wide_threshold": float(settings.get("grip_wide_threshold", 2.1)),
            "grip_narrow_threshold": float(settings.get("grip_narrow_threshold", 1.2)),
            "grip_uneven_threshold": float(settings.get("grip_uneven_threshold", 0.10)),
            "bar_tilt_threshold": float(settings.get("bar_tilt_threshold", 5.0)),
            "body_parts": _sanitize_list(settings.get("body_parts")),
        }
    return clean


def load_label_config() -> Dict[str, List[str]]:
    """Return dict with config stored entirely in JSON."""
    ensure_config_file()
    data = json.loads(LABEL_CONFIG_PATH.read_text())
    tags = data.get("tags")
    if not tags:
        tags = data.get("issues") or DEFAULT_TAGS
    tags = _sanitize_list(tags)
    if not tags:
        tags = DEFAULT_TAGS.copy()
    return {
        "movements": _sanitize_list(data.get("movements")),
        "tags": tags,
        "movement_settings": _sanitize_movement_settings(data.get("movement_settings")),
    }


def save_label_config(cfg: Dict[str, List[str]]) -> None:
    """Persist config (after sanitizing)."""
    clean = {
        "movements": _sanitize_list(cfg.get("movements")),
        "tags": _sanitize_list(cfg.get("tags") or cfg.get("issues")),
        "movement_settings": _sanitize_movement_settings(cfg.get("movement_settings")),
    }
    LABEL_CONFIG_PATH.write_text(json.dumps(clean, indent=2))


__all__ = [
    "LABEL_CONFIG_PATH",
    "ensure_config_file",
    "load_label_config",
    "save_label_config",
]
