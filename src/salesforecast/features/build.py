"""Monthly shop x item feature table.

The training table is a grid: for every month, the cartesian product of shops
that sold something and items that were sold (mirroring how the forecast set is
built), plus the forecast month's (shop, item) pairs. Every feature for month
``t`` uses information from months ``< t`` only, so the same code produces
features for training, validation and the forecast horizon without leakage.

Aggregates and lags are computed on dense ``month x key`` numpy arrays, which
makes lag lookups a fancy-indexing operation instead of repeated DataFrame merges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from salesforecast.data.ingest import RawData
from salesforecast.features.text import category_features, item_features, shop_features
from salesforecast.utils import get_logger

log = get_logger(__name__)

KEYS = ["date_block_num", "shop_id", "item_id"]
TARGET = "item_cnt_month"

CATEGORICAL_FEATURES = [
    "shop_id",
    "item_category_id",
    "shop_city_code",
    "shop_type_code",
    "cat_family_code",
    "cat_subtype_code",
    "item_name_group_code",
]

# (feature prefix, grouping keys) for mean-target aggregates that get lagged.
AGGREGATES: list[tuple[str, list[str]]] = [
    ("month_avg_cnt", []),
    ("item_avg_cnt", ["item_id"]),
    ("shop_avg_cnt", ["shop_id"]),
    ("cat_avg_cnt", ["item_category_id"]),
    ("shop_cat_avg_cnt", ["shop_id", "item_category_id"]),
    ("shop_family_avg_cnt", ["shop_id", "cat_family_code"]),
    ("city_avg_cnt", ["shop_city_code"]),
    ("city_item_avg_cnt", ["shop_city_code", "item_id"]),
    ("family_avg_cnt", ["cat_family_code"]),
    ("name_group_avg_cnt", ["item_name_group_code"]),
]


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    feature_names: list[str]
    categorical: list[str]
    forecast_month: int
    last_observed_month: int


# --------------------------------------------------------------------------- helpers
def _dense(
    months: np.ndarray,
    keys: list[np.ndarray],
    values: np.ndarray,
    shape: tuple[int, ...],
    how: str = "mean",
) -> np.ndarray:
    """Mean (or sum) of ``values`` per (month, *keys) cell as a dense float32 array.

    Empty cells are NaN for ``mean`` and 0 for ``sum``.
    """
    flat = np.ravel_multi_index((months, *keys), shape)
    size = int(np.prod(shape))
    sums = np.bincount(flat, weights=values, minlength=size)
    if how == "sum":
        return sums.astype(np.float32).reshape(shape)
    counts = np.bincount(flat, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (sums / counts).astype(np.float32)
    return out.reshape(shape)


def _lag(arr: np.ndarray, months: np.ndarray, keys: list[np.ndarray], lag: int) -> np.ndarray:
    src = months - lag
    valid = src >= 0
    res = np.full(len(months), np.nan, dtype=np.float32)
    res[valid] = arr[(src[valid], *(k[valid] for k in keys))]
    return res


def _last_true_before(flags: np.ndarray) -> np.ndarray:
    """For a (month, ...) boolean array return, per cell, the last month < m with a True (or -1)."""
    n_months = flags.shape[0]
    month_idx = np.arange(n_months, dtype=np.int16).reshape((-1,) + (1,) * (flags.ndim - 1))
    stamped = np.where(flags, month_idx, np.int16(-1))
    running = np.maximum.accumulate(stamped, axis=0)
    shifted = np.full_like(running, -1)
    shifted[1:] = running[:-1]
    return shifted


# --------------------------------------------------------------------------- grid
def build_grid(sales: pd.DataFrame, test: pd.DataFrame, forecast_month: int) -> pd.DataFrame:
    parts = []
    for month, g in sales.groupby("date_block_num", sort=True):
        shops = g["shop_id"].unique()
        items = g["item_id"].unique()
        parts.append(
            pd.DataFrame(
                {
                    "date_block_num": np.int16(month),
                    "shop_id": np.repeat(shops, len(items)).astype(np.int16),
                    "item_id": np.tile(items, len(shops)).astype(np.int32),
                }
            )
        )
    parts.append(
        pd.DataFrame(
            {
                "date_block_num": np.int16(forecast_month),
                "shop_id": test["shop_id"].to_numpy(np.int16),
                "item_id": test["item_id"].to_numpy(np.int32),
            }
        )
    )
    grid = pd.concat(parts, ignore_index=True)
    return grid.sort_values(KEYS, kind="stable").reset_index(drop=True)


def monthly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    m = sales.assign(revenue=sales["item_price"] * sales["item_cnt_day"])
    return (
        m.groupby(KEYS, sort=True)
        .agg(
            item_cnt=("item_cnt_day", "sum"),
            revenue=("revenue", "sum"),
            price=("item_price", "mean"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------- main
def build_features(raw: RawData, cfg_features: dict[str, Any]) -> FeatureSet:
    lo, hi = cfg_features["target_clip"]
    last_month = int(raw.sales["date_block_num"].max())
    forecast_month = last_month + 1
    n_months = forecast_month + 1

    shops = shop_features(raw.shops)
    cats = category_features(raw.categories)
    items = item_features(raw.items, cfg_features["min_item_name_group"])
    items = items.merge(
        cats[["item_category_id", "cat_family_code", "cat_subtype_code", "cat_is_digital"]],
        on="item_category_id",
        how="left",
    )

    log.info("Building month x shop x item grid")
    grid = build_grid(raw.sales, raw.test, forecast_month)
    msales = monthly_sales(raw.sales)
    log.info("Grid has %s rows", f"{len(grid):,}")

    n_shops = int(max(grid["shop_id"].max(), raw.shops["shop_id"].max())) + 1
    n_items = int(max(grid["item_id"].max(), raw.items["item_id"].max())) + 1

    # ---- target on the grid
    cnt_dense = np.full((n_months, n_shops, n_items), np.nan, dtype=np.float32)
    cnt_dense[
        msales["date_block_num"].to_numpy(),
        msales["shop_id"].to_numpy(),
        msales["item_id"].to_numpy(),
    ] = msales["item_cnt"].to_numpy(np.float32)

    g_month = grid["date_block_num"].to_numpy(np.int64)
    g_shop = grid["shop_id"].to_numpy(np.int64)
    g_item = grid["item_id"].to_numpy(np.int64)
    raw_cnt = cnt_dense[g_month, g_shop, g_item]
    target = np.clip(np.nan_to_num(raw_cnt, nan=0.0), lo, hi).astype(np.float32)
    target[g_month == forecast_month] = np.nan

    # ---- static attributes looked up by id
    shop_lut = shops.set_index("shop_id").reindex(range(n_shops))
    item_lut = items.set_index("item_id").reindex(range(n_items))

    if shop_lut.loc[np.unique(g_shop), "shop_city_code"].isna().any():
        raise ValueError("Grid references shops missing from the shop table")
    if item_lut.loc[np.unique(g_item), "item_category_id"].isna().any():
        raise ValueError("Grid references items missing from the item table")

    def shop_attr(col: str) -> np.ndarray:
        return shop_lut[col].fillna(-1).to_numpy(np.int16)[g_shop]

    def item_attr(col: str) -> np.ndarray:
        return item_lut[col].fillna(-1).to_numpy(np.int16)[g_item]

    cols: dict[str, np.ndarray] = {
        "date_block_num": g_month.astype(np.int16),
        "shop_id": g_shop.astype(np.int16),
        "item_id": g_item.astype(np.int32),
        TARGET: target,
        "item_category_id": item_attr("item_category_id"),
        "shop_city_code": shop_attr("shop_city_code"),
        "shop_type_code": shop_attr("shop_type_code"),
        "shop_is_online": shop_attr("shop_is_online").astype(np.int8),
        "cat_family_code": item_attr("cat_family_code"),
        "cat_subtype_code": item_attr("cat_subtype_code"),
        "cat_is_digital": item_attr("cat_is_digital").astype(np.int8),
        "item_name_group_code": item_attr("item_name_group_code"),
        "item_is_digital": item_attr("item_is_digital").astype(np.int8),
        "item_is_preorder": item_attr("item_is_preorder").astype(np.int8),
    }

    # ---- calendar
    month_of_year = (g_month % 12).astype(np.int8)
    cols["month_of_year"] = month_of_year
    cols["days_in_month"] = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], np.int8)[
        month_of_year
    ]

    # ---- shop x item target lags (clipped history)
    clipped_hist = np.clip(cnt_dense, lo, hi)
    # Pairs present in the grid but with no sales are true zeros, not missing.
    obs = g_month < forecast_month
    zero_fill = np.isnan(clipped_hist[g_month[obs], g_shop[obs], g_item[obs]])
    clipped_hist[g_month[obs][zero_fill], g_shop[obs][zero_fill], g_item[obs][zero_fill]] = 0.0
    for lag in cfg_features["target_lags"]:
        cols[f"shop_item_cnt_lag_{lag}"] = _lag(clipped_hist, g_month, [g_shop, g_item], lag)
    recent = np.stack([cols[f"shop_item_cnt_lag_{lag}"] for lag in (1, 2, 3)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        cols["shop_item_cnt_mean_3"] = np.nanmean(recent, axis=1).astype(np.float32)
    cols["shop_item_cnt_trend"] = cols["shop_item_cnt_lag_1"] - cols["shop_item_cnt_mean_3"]

    # ---- mean-target aggregates over the grid, lagged
    hist_mask = g_month < forecast_month
    hist_target = target[hist_mask].astype(np.float64)
    key_dims = {
        "item_id": n_items,
        "shop_id": n_shops,
        "item_category_id": int(cols["item_category_id"].max()) + 1,
        "shop_city_code": int(cols["shop_city_code"].max()) + 1,
        "cat_family_code": int(cols["cat_family_code"].max()) + 1,
        "item_name_group_code": int(cols["item_name_group_code"].max()) + 1,
    }
    for name, keys in AGGREGATES:
        key_arrays = [cols[k].astype(np.int64) for k in keys]
        shape = (n_months, *(key_dims[k] for k in keys))
        dense = _dense(g_month[hist_mask], [k[hist_mask] for k in key_arrays], hist_target, shape)
        for lag in cfg_features["aggregate_lags"]:
            cols[f"{name}_lag_{lag}"] = _lag(dense, g_month, key_arrays, lag)

    # ---- price features (item level, history only)
    price_dense = np.full((n_months, n_items), np.nan, dtype=np.float32)
    item_month_price = msales.groupby(["date_block_num", "item_id"])["price"].mean()
    pm = item_month_price.index.get_level_values(0).to_numpy(np.int64)
    pi = item_month_price.index.get_level_values(1).to_numpy(np.int64)
    price_dense[pm, pi] = item_month_price.to_numpy()
    observed = ~np.isnan(price_dense)
    last_seen = _last_true_before(observed)  # last month < m with a price
    item_cols = np.broadcast_to(np.arange(n_items), last_seen.shape)
    last_price = np.where(
        last_seen >= 0, price_dense[np.clip(last_seen, 0, None), item_cols], np.nan
    ).astype(np.float32)
    cum_sum = np.cumsum(np.nan_to_num(price_dense), axis=0)
    cum_cnt = np.cumsum(observed, axis=0)
    hist_mean = np.full_like(price_dense, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        hist_mean[1:] = cum_sum[:-1] / np.where(cum_cnt[:-1] > 0, cum_cnt[:-1], np.nan)
    cols["item_price_last"] = last_price[g_month, g_item]
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["item_price_delta"] = (
            cols["item_price_last"] / hist_mean[g_month, g_item] - 1
        ).astype(np.float32)

    cat_of_item = item_lut["item_category_id"].fillna(0).to_numpy(np.int64)
    n_cats = key_dims["item_category_id"]
    cat_price = _dense(
        pm,
        [cat_of_item[pi]],
        item_month_price.to_numpy(np.float64),
        (n_months, n_cats),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["item_price_rel_cat"] = (
            cols["item_price_last"]
            / _lag(cat_price, g_month, [cols["item_category_id"].astype(np.int64)], 1)
        ).astype(np.float32)

    # ---- shop revenue trend
    shop_rev = _dense(
        msales["date_block_num"].to_numpy(np.int64),
        [msales["shop_id"].to_numpy(np.int64)],
        msales["revenue"].to_numpy(np.float64),
        (n_months, n_shops),
        how="sum",
    )
    rev_l1 = _lag(shop_rev, g_month, [g_shop], 1)
    rev_avg = np.mean([_lag(shop_rev, g_month, [g_shop], lag) for lag in (1, 2, 3)], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["shop_revenue_trend"] = (rev_l1 / rev_avg - 1).astype(np.float32)

    # ---- recency / novelty
    sold = np.nan_to_num(cnt_dense) > 0  # (month, shop, item)
    item_sold = sold.any(axis=1)  # (month, item)
    item_first = np.where(item_sold.any(axis=0), item_sold.argmax(axis=0), n_months)
    pair_first = np.where(sold.any(axis=0), sold.argmax(axis=0), n_months)  # (shop, item)
    first_item = item_first[g_item]
    first_pair = pair_first[g_shop, g_item]
    cols["item_age"] = np.where(first_item < g_month, g_month - first_item, 0).astype(np.int16)
    cols["item_is_new"] = (first_item >= g_month).astype(np.int8)
    cols["shop_item_age"] = np.where(first_pair < g_month, g_month - first_pair, -1).astype(
        np.int16
    )
    cols["shop_item_is_new"] = (first_pair >= g_month).astype(np.int8)

    item_last = _last_true_before(item_sold)[g_month, g_item]
    cols["item_months_since_sale"] = np.where(item_last >= 0, g_month - item_last, -1).astype(
        np.int16
    )
    pair_last = _last_true_before(sold)[g_month, g_shop, g_item]
    cols["shop_item_months_since_sale"] = np.where(pair_last >= 0, g_month - pair_last, -1).astype(
        np.int16
    )
    del sold, cnt_dense, clipped_hist

    # ---- assemble (drop early months without enough history)
    keep = g_month >= cfg_features["min_month"]
    frame = pd.DataFrame({k: v[keep] for k, v in cols.items()})
    float_cols = frame.select_dtypes("float64").columns
    frame[float_cols] = frame[float_cols].astype(np.float32)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    feature_names = [c for c in frame.columns if c not in ("date_block_num", "item_id", TARGET)]
    log.info(
        "Feature table: %s rows x %d features (months %d-%d, forecast month %d)",
        f"{len(frame):,}",
        len(feature_names),
        cfg_features["min_month"],
        last_month,
        forecast_month,
    )
    return FeatureSet(
        frame=frame,
        feature_names=feature_names,
        categorical=[c for c in CATEGORICAL_FEATURES if c in feature_names],
        forecast_month=forecast_month,
        last_observed_month=last_month,
    )


def lookup_tables(raw: RawData, cfg_features: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Human-readable dimension tables and monthly history, used by the API and dashboard."""
    shops = shop_features(raw.shops)[["shop_id", "shop_name", "shop_city", "shop_type"]]
    cats = category_features(raw.categories)[
        [
            "item_category_id",
            "item_category_name",
            "cat_family",
            "cat_family_en",
            "cat_family_code",
            "cat_subtype",
        ]
    ]
    items = item_features(raw.items, cfg_features["min_item_name_group"])[
        ["item_id", "item_name", "item_category_id", "item_name_group"]
    ]
    history = monthly_sales(raw.sales)
    history["date_block_num"] = history["date_block_num"].astype(np.int16)
    history["item_cnt"] = history["item_cnt"].astype(np.float32)
    history["revenue"] = history["revenue"].astype(np.float32)
    history = history.drop(columns="price")
    return {"shops": shops, "categories": cats, "items": items, "history": history}
