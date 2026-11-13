import numpy as np
import pandas as pd

from salesforecast.data.clean import clean
from salesforecast.features.build import KEYS, TARGET, build_features
from salesforecast.features.text import category_features, shop_features

from .conftest import UNSOLD_ITEM, make_config


def _build(raw, tmp_path):
    cfg = make_config(tmp_path)
    return build_features(clean(raw, cfg.cleaning), cfg.features), cfg


def test_shop_and_category_parsing():
    shops = pd.DataFrame(
        {
            "shop_name": ['Москва ТЦ "Семеновский"', "Интернет-магазин ЧС", "!Якутск ТРК"],
            "shop_id": [0, 1, 2],
        }
    )
    out = shop_features(shops)
    assert list(out["shop_city"]) == ["Москва", "online", "Якутск"]
    assert list(out["shop_type"]) == ["ТЦ", "online", "ТРК"]

    cats = pd.DataFrame(
        {
            "item_category_name": ["Игры PC - Цифра", "Кино - DVD", "Билеты (Цифра)"],
            "item_category_id": [0, 1, 2],
        }
    )
    out = category_features(cats)
    assert list(out["cat_family"]) == ["Игры", "Кино", "Билеты"]
    assert list(out["cat_subtype"]) == ["Цифра", "DVD", "Билеты (Цифра)"]
    assert list(out["cat_is_digital"]) == [1, 0, 1]


def test_grid_shape_target_and_forecast_rows(raw, tmp_path):
    fs, cfg = _build(raw, tmp_path)
    f = fs.frame
    assert fs.forecast_month == 18 and fs.last_observed_month == 17
    assert f["date_block_num"].min() == cfg.features["min_month"]
    assert not f.duplicated(KEYS).any()

    fc = f[f.date_block_num == fs.forecast_month]
    assert len(fc) == len(raw.test)
    assert fc[TARGET].isna().all()

    hist = f[f.date_block_num < fs.forecast_month]
    lo, hi = cfg.features["target_clip"]
    assert hist[TARGET].between(lo, hi).all()

    # the target equals the clipped monthly sum of daily sales
    m = 10
    expected = (
        raw.sales[raw.sales.date_block_num == m]
        .groupby(["shop_id", "item_id"])["item_cnt_day"]
        .sum()
        .clip(lo, hi)
    )
    got = hist[hist.date_block_num == m].set_index(["shop_id", "item_id"])[TARGET]
    pd.testing.assert_series_equal(
        got.loc[expected.index].astype(float), expected.astype(float), check_names=False
    )


def test_lag_feature_matches_previous_month_target(raw, tmp_path):
    fs, _ = _build(raw, tmp_path)
    f = fs.frame.set_index(KEYS)
    cur = f.xs(12, level="date_block_num")
    prev = f.xs(11, level="date_block_num")[TARGET]
    joined = cur[["shop_item_cnt_lag_1"]].join(prev.rename("prev"), how="inner")
    np.testing.assert_allclose(joined["shop_item_cnt_lag_1"], joined["prev"])


def test_features_do_not_leak_current_month(raw, tmp_path):
    """Changing sales in month t must not change any feature of month t."""
    fs_a, _ = _build(raw, tmp_path / "a")

    sales = raw.sales.copy()
    last = sales.date_block_num == 17
    sales.loc[last, "item_cnt_day"] *= 3
    sales.loc[last, "item_price"] *= 2
    raw_b = type(raw)(
        sales=sales, items=raw.items, categories=raw.categories, shops=raw.shops, test=raw.test
    )
    fs_b, _ = _build(raw_b, tmp_path / "b")

    a = fs_a.frame[fs_a.frame.date_block_num == 17].reset_index(drop=True)
    b = fs_b.frame[fs_b.frame.date_block_num == 17].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[fs_a.feature_names], b[fs_b.feature_names])
    assert not np.allclose(a[TARGET], b[TARGET])


def test_unsold_item_is_flagged_new(raw, tmp_path):
    fs, _ = _build(raw, tmp_path)
    row = fs.frame[
        (fs.frame.date_block_num == fs.forecast_month) & (fs.frame.item_id == UNSOLD_ITEM)
    ]
    assert len(row) == 1
    assert row["item_is_new"].iloc[0] == 1
    assert np.isnan(row["shop_item_cnt_lag_1"].iloc[0])
