#!/usr/bin/env python3
"""Shared configuration for bench labeling options."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if getattr(sys, "frozen", False):
    DEV_ROOT = Path.home() / "Documents" / "mAI_Coach" / "Dev_tools"
else:
    DEV_ROOT = Path(__file__).resolve().parent

LABEL_CONFIG_PATH = DEV_ROOT / "label_config.json"
DEFAULT_DATA_DIR = DEV_ROOT / "data"
DEFAULT_MODEL_DIR = DEV_ROOT / "models"

DEFAULT_TAGS = [
    "no_major_issues",
    "hands_too_wide",
    "hands_too_narrow",
    "grip_uneven",
    "barbell_tilted",
    "bar_depth_insufficient",
    "incomplete_lockout",
]

DEFAULT_OK_TAG = "no_major_issues"

DEFAULT_ML_PRESETS = {
    "traditional_bench": {
        "preprocess": {
            "dataset_dir": str(DEFAULT_DATA_DIR / "JSON"),
            "output_prefix": str(DEFAULT_DATA_DIR / "traditional_bench_v1"),
        },
        "train": {
            "data_prefix": str(DEFAULT_DATA_DIR / "traditional_bench_v1"),
            "output_prefix": str(DEFAULT_MODEL_DIR / "traditional_bench_mlp_v1"),
            "epochs": 200,
            "batch_size": 32,
            "dev_fraction": 0.2,
            "seed": 42,
        },
        "tags": DEFAULT_TAGS,
    },
    "traditional_bench_side": {
        "preprocess": {
            "dataset_dir": str(DEFAULT_DATA_DIR / "JSON" / "side"),
            "output_prefix": str(DEFAULT_DATA_DIR / "traditional_bench_side_v1"),
        },
        "train": {
            "data_prefix": str(DEFAULT_DATA_DIR / "traditional_bench_side_v1"),
            "output_prefix": str(DEFAULT_MODEL_DIR / "traditional_bench_side_mlp_v1"),
            "epochs": 200,
            "batch_size": 32,
            "dev_fraction": 0.2,
            "seed": 42,
        },
        "tags": DEFAULT_TAGS,
    },
}

EMPTY_CONFIG = {
    "movements": [],
    "tags": DEFAULT_TAGS,
    "movement_settings": {},
    "ml_presets": DEFAULT_ML_PRESETS,
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


def _sanitize_ml_presets(values, default_tags) -> Dict[str, Dict]:
    if not isinstance(values, dict):
        values = {}
    clean: Dict[str, Dict] = {}
    for name, preset in values.items():
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if not isinstance(preset, dict):
            continue
        preprocess = preset.get("preprocess") or {}
        train = preset.get("train") or {}
        tags = preset.get("tags")
        clean[name] = {
            "preprocess": {
                "dataset_dir": str(preprocess.get("dataset_dir", "")),
                "output_prefix": str(preprocess.get("output_prefix", "")),
            },
            "train": {
                "data_prefix": str(train.get("data_prefix", "")),
                "output_prefix": str(train.get("output_prefix", "")),
                "epochs": int(train.get("epochs", 200)),
                "batch_size": int(train.get("batch_size", 32)),
                "dev_fraction": float(train.get("dev_fraction", 0.2)),
                "seed": int(train.get("seed", 42)),
            },
            "tags": _sanitize_list(tags) or default_tags,
        }
    if not clean:
        fallback = json.loads(json.dumps(DEFAULT_ML_PRESETS))
        for key in fallback:
            fallback[key]["tags"] = default_tags
        clean = fallback
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
        "ml_presets": _sanitize_ml_presets(data.get("ml_presets"), tags),
    }


def save_label_config(cfg: Dict[str, List[str]]) -> None:
    """Persist config (after sanitizing)."""
    ensure_config_file()
    current = load_label_config()
    clean = {
        "movements": _sanitize_list(cfg.get("movements", current.get("movements"))),
        "tags": _sanitize_list(cfg.get("tags") or cfg.get("issues") or current.get("tags")),
        "movement_settings": _sanitize_movement_settings(
            cfg.get("movement_settings", current.get("movement_settings"))
        ),
        "ml_presets": _sanitize_ml_presets(
            cfg.get("ml_presets", current.get("ml_presets")), current.get("tags")
        ),
    }
    LABEL_CONFIG_PATH.write_text(json.dumps(clean, indent=2))


__all__ = [
    "LABEL_CONFIG_PATH",
    "ensure_config_file",
    "load_label_config",
    "save_label_config",
    "load_label_options",
    "load_ml_presets",
    "save_ml_presets",
    "default_movement_settings",
    "DEFAULT_TAGS",
    "DEFAULT_OK_TAG",
    "MODEL_VARIANTS",
    "BODY_PART_OPTIONS",
    "BENCH_DEFAULT_PARTS",
    "GRIP_WIDE_THRESHOLD",
    "GRIP_NARROW_THRESHOLD",
    "GRIP_UNEVEN_THRESHOLD",
    "BAR_TILT_THRESHOLD_DEG",
]


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

# Metric thresholds
GRIP_WIDE_THRESHOLD = 2.1
GRIP_NARROW_THRESHOLD = 1.2
GRIP_UNEVEN_THRESHOLD = 0.10
BAR_TILT_THRESHOLD_DEG = 5.0


def default_movement_settings(name: str = "") -> Dict:
    lower = name.lower()
    body_parts = (
        BENCH_DEFAULT_PARTS.copy() if "bench" in lower else BODY_PART_OPTIONS.copy()
    )
    return {
        "model": "full",
        "preprocess": True,
        "pre_height": 720,
        "pre_fps": 15.0,
        "det": 0.5,
        "prs": 0.7,
        "trk": 0.7,
        "ema": 0.25,
        "seg": False,
        "grip_wide_threshold": GRIP_WIDE_THRESHOLD,
        "grip_narrow_threshold": GRIP_NARROW_THRESHOLD,
        "grip_uneven_threshold": GRIP_UNEVEN_THRESHOLD,
        "bar_tilt_threshold": BAR_TILT_THRESHOLD_DEG,
        "body_parts": body_parts,
    }


def load_label_options():
    """Load movements, tags, and per-movement settings from config."""
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


def load_ml_presets() -> Tuple[Dict[str, Dict], List[str], List[str]]:
    """Load ML presets, tags, and movements from config."""
    cfg = load_label_config()
    return (
        cfg.get("ml_presets") or {},
        cfg.get("tags") or [],
        cfg.get("movements") or [],
    )


def save_ml_presets(presets: Dict[str, Dict]) -> None:
    """Update only the ML presets in the config."""
    ensure_config_file()
    cfg = load_label_config()
    clean_presets = _sanitize_ml_presets(presets, cfg.get("tags") or DEFAULT_TAGS)
    save_label_config({"ml_presets": clean_presets})
