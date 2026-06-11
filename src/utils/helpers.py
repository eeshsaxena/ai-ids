"""Utility helpers shared across the project."""

from __future__ import annotations

import os
import json
import numpy as np
from pathlib import Path


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_results(path: str = "results/training_summary.json") -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def model_exists(key: str, task: str) -> bool:
    ext = ".pt" if key == "nn" else ".pkl"
    return Path(f"models/saved/{key}_{task}{ext}").exists()


def dataset_exists(config_path: str = "config/config.yaml") -> bool:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return (
        Path(cfg["data"]["train_path"]).exists()
        and Path(cfg["data"]["test_path"]).exists()
    )
