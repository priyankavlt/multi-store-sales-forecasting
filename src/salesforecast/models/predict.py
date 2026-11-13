"""Batch scoring of the forecast horizon and an in-memory forecaster for serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from salesforecast.config import Config
from salesforecast.features.store import load_features, load_meta
from salesforecast.models.registry import load_latest
from salesforecast.utils import get_logger, write_json

log = get_logger(__name__)

PREDICTION = "item_cnt_month_pred"


def _forecast_ids(cfg: Config) -> pd.DataFrame:
    test = pd.read_csv(
        cfg.raw_file("test"), dtype={"ID": "int32", "shop_id": "int16", "item_id": "int32"}
    )
    shop_map = {int(k): int(v) for k, v in cfg.cleaning.get("shop_id_map", {}).items()}
    test["shop_id"] = test["shop_id"].replace(shop_map).astype("int16")
    return test


def predict_batch(cfg: Config) -> pd.DataFrame:
    cfg.ensure_dirs()
    booster, model_meta = load_latest(cfg)
    fmeta = load_meta(cfg)
    month = fmeta["forecast_month"]
    feats = load_features(cfg, min_month=month)
    feats[PREDICTION] = np.clip(
        booster.predict(feats[model_meta["features"]]), *model_meta["target_clip"]
    ).astype(np.float32)

    ids = _forecast_ids(cfg)
    out = ids.merge(
        feats[["shop_id", "item_id", PREDICTION]], on=["shop_id", "item_id"], how="left"
    )
    missing = int(out[PREDICTION].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} forecast pairs have no features")

    forecast = out.rename(columns={PREDICTION: "item_cnt_month"})
    forecast.to_csv(cfg.predictions_dir / "forecast.csv", index=False)
    forecast[["ID", "item_cnt_month"]].to_csv(cfg.predictions_dir / "submission.csv", index=False)

    store = feats.drop(columns=["item_cnt_month"], errors="ignore")
    store.to_parquet(cfg.forecast_store_path, index=False)

    summary = {
        "run_id": model_meta["run_id"],
        "forecast_month": month,
        "pairs": len(forecast),
        "total_units": float(forecast["item_cnt_month"].sum()),
        "mean_units_per_pair": float(forecast["item_cnt_month"].mean()),
        "share_predicted_zero": float((forecast["item_cnt_month"] < 0.05).mean()),
    }
    write_json(summary, cfg.reports_dir / "forecast_summary.json")
    log.info(
        "Forecast for month %d: %s pairs, %.0f total units",
        month,
        f"{len(forecast):,}",
        summary["total_units"],
    )
    return forecast


@dataclass
class Forecaster:
    """Loads the production model and the forecast-month feature store once; scores on request."""

    booster: lgb.Booster
    metadata: dict[str, Any]
    store: pd.DataFrame  # indexed by (shop_id, item_id)

    @classmethod
    def from_config(cls, cfg: Config) -> Forecaster:
        booster, metadata = load_latest(cfg)
        store = (
            pd.read_parquet(cfg.forecast_store_path).set_index(["shop_id", "item_id"]).sort_index()
        )
        return cls(booster=booster, metadata=metadata, store=store)

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]

    def _score(self, rows: pd.DataFrame) -> np.ndarray:
        X = rows.reset_index()[self.features]
        return np.clip(self.booster.predict(X), *self.metadata["target_clip"])

    def predict_pairs(self, pairs: list[tuple[int, int]]) -> list[float | None]:
        if not pairs:
            return []
        idx = pd.MultiIndex.from_tuples(pairs, names=["shop_id", "item_id"])
        known = idx.isin(self.store.index)
        result: list[float | None] = [None] * len(pairs)
        if known.any():
            scores = self._score(self.store.loc[idx[known]])
            for pos, score in zip(np.flatnonzero(known), scores, strict=True):
                result[pos] = float(score)
        return result

    def top_items(self, shop_id: int, n: int) -> pd.DataFrame:
        if shop_id not in self.store.index.get_level_values(0):
            raise KeyError(shop_id)
        rows = self.store.xs(shop_id, level="shop_id", drop_level=False)
        scored = rows.reset_index()[["shop_id", "item_id"]].assign(prediction=self._score(rows))
        return scored.nlargest(n, "prediction")
