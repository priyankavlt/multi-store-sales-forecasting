"""Typed access to the YAML configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = field(default=PROJECT_ROOT)

    # ---- paths -------------------------------------------------------------
    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def raw_dir(self) -> Path:
        return self._path("raw_dir")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self._path("artifacts_dir")

    @property
    def reports_dir(self) -> Path:
        return self._path("reports_dir")

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.artifacts_dir / "predictions"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def forecast_store_path(self) -> Path:
        return self.processed_dir / "forecast_features.parquet"

    # ---- sections ----------------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def cleaning(self) -> dict[str, Any]:
        return self.raw["cleaning"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def monitoring(self) -> dict[str, Any]:
        return self.raw["monitoring"]

    def raw_file(self, name: str) -> Path:
        return self.raw_dir / self.data["files"][name]

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.predictions_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def override(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with selected keys of top-level sections replaced."""
        raw = copy.deepcopy(self.raw)
        for section, values in sections.items():
            raw.setdefault(section, {}).update(values)
        return Config(raw=raw, root=self.root)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=Path(root) if root else PROJECT_ROOT)
