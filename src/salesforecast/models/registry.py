"""Minimal file-based model registry.

Each training run gets a directory ``artifacts/models/<run_id>/`` holding the
LightGBM model and a ``metadata.json``. ``artifacts/models/latest.json`` points
at the run that downstream steps (prediction, API, dashboard) should use.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb

from salesforecast.config import Config
from salesforecast.utils import read_json, write_json

LATEST = "latest.json"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run(cfg: Config, run_id: str, booster: lgb.Booster, metadata: dict[str, Any]) -> Path:
    run_dir = cfg.models_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "model.txt"), num_iteration=booster.best_iteration or None)
    write_json(metadata, run_dir / "metadata.json")
    write_json(
        {"run_id": run_id, "path": str(run_dir.relative_to(cfg.models_dir))},
        cfg.models_dir / LATEST,
    )
    return run_dir


def latest_run_dir(cfg: Config) -> Path:
    pointer = cfg.models_dir / LATEST
    if not pointer.exists():
        raise FileNotFoundError(
            f"No trained model found ({pointer} missing). Run `salesforecast train`."
        )
    return cfg.models_dir / read_json(pointer)["path"]


def load_latest(cfg: Config) -> tuple[lgb.Booster, dict[str, Any]]:
    run_dir = latest_run_dir(cfg)
    booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
    return booster, read_json(run_dir / "metadata.json")
