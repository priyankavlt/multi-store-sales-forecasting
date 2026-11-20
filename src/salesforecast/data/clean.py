"""Cleaning of transactional sales data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from salesforecast.data.ingest import RawData
from salesforecast.utils import get_logger

log = get_logger(__name__)


def clean(raw: RawData, cfg_cleaning: dict[str, Any]) -> RawData:
    s = raw.sales.drop_duplicates()
    n0 = len(raw.sales)

    # Remove data-entry outliers.
    s = s[(s["item_cnt_day"] <= cfg_cleaning["max_item_cnt_day"])]
    s = s[(s["item_price"] <= cfg_cleaning["max_item_price"])]

    # Non-positive prices: impute with the median price of the same item in the same shop-month,
    # falling back to the item's overall median.
    bad = s["item_price"] <= 0
    if bad.any():
        good = s.loc[~bad]
        by_ctx = good.groupby(["date_block_num", "shop_id", "item_id"])["item_price"].median()
        by_item = good.groupby("item_id")["item_price"].median()
        keys = pd.MultiIndex.from_frame(s.loc[bad, ["date_block_num", "shop_id", "item_id"]])
        fill = pd.Series(by_ctx.reindex(keys).to_numpy(), index=s.index[bad])
        fill = fill.fillna(s.loc[bad, "item_id"].map(by_item))
        s.loc[bad, "item_price"] = fill
        s = s[s["item_price"] > 0]

    # Merge shops that are the same physical store under two ids.
    shop_map = {int(k): int(v) for k, v in cfg_cleaning.get("shop_id_map", {}).items()}
    s = s.assign(shop_id=s["shop_id"].replace(shop_map).astype("int16"))
    test = raw.test.assign(shop_id=raw.test["shop_id"].replace(shop_map).astype("int16"))
    shops = raw.shops[~raw.shops["shop_id"].isin(shop_map.keys())].reset_index(drop=True)

    log.info("Cleaning kept %s of %s sales rows", f"{len(s):,}", f"{n0:,}")
    return RawData(
        sales=s.reset_index(drop=True),
        items=raw.items,
        categories=raw.categories,
        shops=shops,
        test=test,
    )
