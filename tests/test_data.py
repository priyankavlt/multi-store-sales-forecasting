import numpy as np
import pandas as pd
import pytest

from salesforecast.data.clean import clean
from salesforecast.data.ingest import RawData
from salesforecast.data.validate import DataValidationError, assert_valid, validate_raw

CLEANING = {"max_item_cnt_day": 1000, "max_item_price": 100000, "shop_id_map": {}}


def _with_sales(raw: RawData, sales: pd.DataFrame) -> RawData:
    return RawData(
        sales=sales, items=raw.items, categories=raw.categories, shops=raw.shops, test=raw.test
    )


def test_synthetic_data_passes_validation(raw):
    rep = validate_raw(raw, CLEANING)
    assert rep.passed, rep.errors
    assert rep.stats["forecast_items_never_sold"] == 1


def test_validation_rejects_unknown_item(raw):
    sales = raw.sales.copy()
    sales.loc[0, "item_id"] = 99_999
    rep = validate_raw(_with_sales(raw, sales), CLEANING)
    assert not rep.passed
    assert any("unknown item_id" in e for e in rep.errors)
    with pytest.raises(DataValidationError):
        assert_valid(rep)


def test_validation_rejects_month_index_mismatch(raw):
    sales = raw.sales.copy()
    sales.loc[sales.index[-1], "date_block_num"] = 0
    rep = validate_raw(_with_sales(raw, sales), CLEANING)
    assert any("date_block_num disagrees" in e for e in rep.errors)


def test_validation_rejects_missing_column(raw):
    rep = validate_raw(_with_sales(raw, raw.sales.drop(columns="item_price")), CLEANING)
    assert not rep.passed


def test_clean_handles_duplicates_outliers_prices_and_shop_merge(raw):
    sales = raw.sales.copy()
    first = sales.iloc[[0]]
    outlier = first.assign(item_cnt_day=5000.0)
    bad_price = first.assign(item_price=-1.0, item_cnt_day=2.0)
    shop5 = first.assign(shop_id=np.int16(5))
    sales = pd.concat([sales, first, outlier, bad_price, shop5], ignore_index=True)

    cfg = {**CLEANING, "shop_id_map": {5: 3}}
    out = clean(_with_sales(raw, sales), cfg)

    assert not out.sales.duplicated().any()
    assert out.sales["item_cnt_day"].max() <= 1000
    assert (out.sales["item_price"] > 0).all()
    assert 5 not in set(out.sales["shop_id"]) and 5 not in set(out.shops["shop_id"])
    assert 5 not in set(out.test["shop_id"])
