"""Data-quality checks run before any feature engineering.

Hard failures (schema, referential integrity) raise; soft findings (outliers,
duplicates) are reported so the cleaning step can handle them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from salesforecast.data.ingest import RawData

REQUIRED_COLUMNS = {
    "sales": ["date", "date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"],
    "items": ["item_name", "item_id", "item_category_id"],
    "categories": ["item_category_name", "item_category_id"],
    "shops": ["shop_name", "shop_id"],
    "test": ["ID", "shop_id", "item_id"],
}


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_raw(raw: RawData, cfg_cleaning: dict[str, Any]) -> ValidationReport:
    rep = ValidationReport()

    # 1. schema
    for name, cols in REQUIRED_COLUMNS.items():
        missing = set(cols) - set(getattr(raw, name).columns)
        if missing:
            rep.error(f"{name}: missing columns {sorted(missing)}")
    if not rep.passed:
        return rep

    s = raw.sales

    # 2. nulls
    for name in REQUIRED_COLUMNS:
        df = getattr(raw, name)
        n_null = int(df[REQUIRED_COLUMNS[name]].isna().sum().sum())
        if n_null:
            rep.error(f"{name}: {n_null} null values in required columns")

    # 3. keys are unique in dimension tables
    for name, key in (
        ("items", "item_id"),
        ("categories", "item_category_id"),
        ("shops", "shop_id"),
    ):
        if getattr(raw, name)[key].duplicated().any():
            rep.error(f"{name}: duplicate {key}")
    if raw.test[["shop_id", "item_id"]].duplicated().any():
        rep.error("test: duplicate (shop_id, item_id) pairs")

    # 4. referential integrity
    for df_name, df in (("sales", s), ("test", raw.test)):
        unknown_items = ~df["item_id"].isin(raw.items["item_id"])
        unknown_shops = ~df["shop_id"].isin(raw.shops["shop_id"])
        if unknown_items.any():
            rep.error(f"{df_name}: {int(unknown_items.sum())} rows reference unknown item_id")
        if unknown_shops.any():
            rep.error(f"{df_name}: {int(unknown_shops.sum())} rows reference unknown shop_id")
    if (~raw.items["item_category_id"].isin(raw.categories["item_category_id"])).any():
        rep.error("items: unknown item_category_id")

    # 5. month index consistent with calendar dates
    months_from_date = (s["date"].dt.year - s["date"].dt.year.min()) * 12 + s["date"].dt.month - 1
    months_from_date -= months_from_date.min()
    mismatched = int((months_from_date.values != s["date_block_num"].values).sum())
    if mismatched:
        rep.error(f"sales: {mismatched} rows where date_block_num disagrees with date")
    blocks = sorted(s["date_block_num"].unique())
    if blocks != list(range(blocks[0], blocks[-1] + 1)):
        rep.error("sales: date_block_num has gaps")

    # 6. soft findings handled by cleaning
    n_dup = int(s.duplicated().sum())
    n_neg_price = int((s["item_price"] <= 0).sum())
    n_returns = int((s["item_cnt_day"] < 0).sum())
    n_qty_outliers = int((s["item_cnt_day"] > cfg_cleaning["max_item_cnt_day"]).sum())
    n_price_outliers = int((s["item_price"] > cfg_cleaning["max_item_price"]).sum())
    for count, msg in (
        (n_dup, "exact duplicate sales rows"),
        (n_neg_price, "rows with non-positive price"),
        (n_qty_outliers, "rows with extreme daily quantity"),
        (n_price_outliers, "rows with extreme price"),
    ):
        if count:
            rep.warnings.append(f"sales: {count} {msg}")

    test_shops = set(raw.test["shop_id"])
    test_items = set(raw.test["item_id"])
    seen_items = set(s["item_id"])
    rep.stats = {
        "sales_rows": len(s),
        "months": len(blocks),
        "first_date": str(s["date"].min().date()),
        "last_date": str(s["date"].max().date()),
        "shops": int(s["shop_id"].nunique()),
        "items": int(s["item_id"].nunique()),
        "duplicate_rows": n_dup,
        "non_positive_price_rows": n_neg_price,
        "return_rows": n_returns,
        "forecast_pairs": len(raw.test),
        "forecast_shops": len(test_shops),
        "forecast_items": len(test_items),
        "forecast_items_never_sold": len(test_items - seen_items),
    }
    return rep


def assert_valid(rep: ValidationReport) -> None:
    if not rep.passed:
        raise DataValidationError("; ".join(rep.errors))
