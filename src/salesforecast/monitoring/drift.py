"""Feature drift between the last observed month and the forecast month (PSI)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from salesforecast.config import Config
from salesforecast.features.build import AGGREGATES
from salesforecast.features.store import load_features, load_meta
from salesforecast.utils import get_logger, write_json

log = get_logger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins from the reference; NaN is its own bin."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    ref_ok, cur_ok = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = (
        np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1))) if len(ref_ok) else np.array([])
    )
    if len(edges) > 1:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(ref_ok, edges)[0]
        cur_counts = np.histogram(cur_ok, edges)[0]
    else:  # constant (or empty) feature: compare equality to the single value
        value = edges[0] if len(edges) else 0.0
        ref_counts = np.array([np.sum(ref_ok == value), np.sum(ref_ok != value)])
        cur_counts = np.array([np.sum(cur_ok == value), np.sum(cur_ok != value)])
    ref_counts = np.append(ref_counts, np.isnan(reference).sum())
    cur_counts = np.append(cur_counts, np.isnan(current).sum())
    ref_p = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_p = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _group_level_features(feature_names: list[str]) -> set[str]:
    """Aggregates keyed on coarse groups (month, shop, category, ...) rather than on the item.

    They take only a handful of distinct values per month, so quantile-binned PSI
    degenerates; they are monitored by the relative change of their mean instead.
    """
    prefixes = [name for name, keys in AGGREGATES if "item_id" not in keys]
    group = {f for f in feature_names if any(f.startswith(f"{p}_lag_") for p in prefixes)}
    return group | {"shop_revenue_trend"}


def drift_report(cfg: Config) -> dict[str, Any]:
    meta = load_meta(cfg)
    ref_month, cur_month = meta["last_observed_month"], meta["forecast_month"]
    df = load_features(cfg, min_month=ref_month)
    ref = df[df["date_block_num"] == ref_month]
    cur = df[df["date_block_num"] == cur_month]
    mon = cfg.monitoring
    # Calendar features change by construction between months; skip them.
    skip = {"month_of_year", "days_in_month"}
    group_level = _group_level_features(meta["feature_names"])

    def status(value: float, warn: float, alert: float) -> str:
        return "alert" if value >= alert else "warning" if value >= warn else "ok"

    rows, group_rows = [], []
    for f in meta["feature_names"]:
        if f in skip:
            continue
        r, c = ref[f].to_numpy(np.float64), cur[f].to_numpy(np.float64)
        missing_delta = round(float(np.isnan(c).mean() - np.isnan(r).mean()), 4)
        if f in group_level:
            ref_mean, cur_mean = np.nanmean(r), np.nanmean(c)
            change = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9)
            group_rows.append(
                {
                    "feature": f,
                    "reference_mean": round(float(ref_mean), 4),
                    "current_mean": round(float(cur_mean), 4),
                    "relative_change": round(float(change), 4),
                    "status": status(change, mon["mean_shift_warning"], mon["mean_shift_alert"]),
                }
            )
        else:
            value = psi(r, c)
            rows.append(
                {
                    "feature": f,
                    "psi": round(value, 4),
                    "missing_rate_change": missing_delta,
                    "status": status(value, mon["psi_warning"], mon["psi_alert"]),
                }
            )
    table = pd.DataFrame(rows).sort_values("psi", ascending=False)
    group_table = pd.DataFrame(group_rows).sort_values("relative_change", ascending=False)
    statuses = pd.concat([table["status"], group_table["status"]])
    report = {
        "reference_month": ref_month,
        "current_month": cur_month,
        "thresholds": {
            "psi_warning": mon["psi_warning"],
            "psi_alert": mon["psi_alert"],
            "mean_shift_warning": mon["mean_shift_warning"],
            "mean_shift_alert": mon["mean_shift_alert"],
        },
        "n_alert": int((statuses == "alert").sum()),
        "n_warning": int((statuses == "warning").sum()),
        "features": table.to_dict(orient="records"),
        "group_level_features": group_table.to_dict(orient="records"),
    }
    write_json(report, cfg.reports_dir / "drift.json")
    log.info(
        "Drift month %d vs %d: %d alert, %d warning features",
        cur_month,
        ref_month,
        report["n_alert"],
        report["n_warning"],
    )
    return report
