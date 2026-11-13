"""Persistence for the feature table and lookup tables (Parquet)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from salesforecast.config import Config
from salesforecast.features.build import FeatureSet
from salesforecast.utils import read_json, write_json

META_FILE = "features_meta.json"


def save_features(cfg: Config, fs: FeatureSet, lookups: dict[str, pd.DataFrame]) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    fs.frame.to_parquet(cfg.features_path, index=False, row_group_size=250_000)
    for name, table in lookups.items():
        table.to_parquet(cfg.processed_dir / f"{name}.parquet", index=False)
    write_json(
        {
            "feature_names": fs.feature_names,
            "categorical": fs.categorical,
            "forecast_month": fs.forecast_month,
            "last_observed_month": fs.last_observed_month,
            "rows": len(fs.frame),
        },
        cfg.processed_dir / META_FILE,
    )


def load_meta(cfg: Config) -> dict[str, Any]:
    path = cfg.processed_dir / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `salesforecast features` first.")
    return read_json(path)


def load_features(
    cfg: Config,
    min_month: int | None = None,
    max_month: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    filters = []
    if min_month is not None:
        filters.append(("date_block_num", ">=", min_month))
    if max_month is not None:
        filters.append(("date_block_num", "<=", max_month))
    table = pq.read_table(cfg.features_path, columns=columns, filters=filters or None)
    return table.to_pandas()


def load_lookup(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.processed_dir / f"{name}.parquet")
