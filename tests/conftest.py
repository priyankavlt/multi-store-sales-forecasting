"""Shared fixtures: a small synthetic dataset with the same schema as the real one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from salesforecast.config import Config, load_config
from salesforecast.data.ingest import RawData

N_MONTHS = 18
SHOPS = [
    (0, 'Москва ТЦ "Семеновский"'),
    (1, 'Москва ТРК "Атриум"'),
    (2, "Воронеж (Плехановская, 13)"),
    (3, 'Казань ТЦ "Бехетле"'),
    (4, "Интернет-магазин ЧС"),
    (5, 'Якутск ТЦ "Центральный"'),
]
CATEGORIES = [
    (0, "Игры - PS3"),
    (1, "Игры PC - Стандартные издания"),
    (2, "Кино - DVD"),
    (3, "Музыка - CD локального производства"),
    (4, "Подарки - Мягкие игрушки"),
]
N_ITEMS = 40
UNSOLD_ITEM = 39


def make_raw(seed: int = 0) -> RawData:
    rng = np.random.default_rng(seed)
    items = pd.DataFrame(
        {
            "item_name": [
                f"Товар {i} [PC, Цифровая версия]" if i % 7 == 0 else f"Игра {i} (BD)"
                for i in range(N_ITEMS)
            ],
            "item_id": np.arange(N_ITEMS, dtype="int32"),
            "item_category_id": (np.arange(N_ITEMS) % len(CATEGORIES)).astype("int16"),
        }
    )
    popularity = rng.gamma(1.5, 1.0, N_ITEMS)
    rows = []
    for m in range(N_MONTHS):
        month_start = pd.Timestamp("2013-01-01") + pd.DateOffset(months=m)
        for shop_id, _ in SHOPS:
            if shop_id == 5 and m < 4:  # a shop that opens later
                continue
            for item_id in range(N_ITEMS - 1):  # UNSOLD_ITEM never sells
                n_days = rng.poisson(popularity[item_id] * (1 + 0.3 * np.sin(m / 2)))
                for _ in range(min(n_days, 6)):
                    rows.append(
                        (
                            (month_start + pd.Timedelta(days=int(rng.integers(0, 28)))).strftime(
                                "%d.%m.%Y"
                            ),
                            m,
                            shop_id,
                            item_id,
                            float(100 + 10 * item_id + rng.normal(0, 5)),
                            float(
                                rng.choice([1, 1, 1, 2, 3, -1], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
                            ),
                        )
                    )
    sales = pd.DataFrame(
        rows, columns=["date", "date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"]
    )
    sales = sales.astype(
        {
            "date_block_num": "int16",
            "shop_id": "int16",
            "item_id": "int32",
            "item_cnt_day": "float32",
        }
    )
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    test_pairs = [(s, i) for s in (0, 1, 2, 3, 5) for i in range(30)] + [(0, UNSOLD_ITEM)]
    test = pd.DataFrame(test_pairs, columns=["shop_id", "item_id"]).astype(
        {"shop_id": "int16", "item_id": "int32"}
    )
    test.insert(0, "ID", np.arange(len(test), dtype="int32"))
    return RawData(
        sales=sales,
        items=items,
        categories=pd.DataFrame(
            CATEGORIES, columns=["item_category_id", "item_category_name"]
        ).astype({"item_category_id": "int16"})[["item_category_name", "item_category_id"]],
        shops=pd.DataFrame(SHOPS, columns=["shop_id", "shop_name"]).astype({"shop_id": "int16"})[
            ["shop_name", "shop_id"]
        ],
        test=test,
    )


def write_raw(raw: RawData, cfg: Config) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    sales = raw.sales.assign(date=raw.sales["date"].dt.strftime("%d.%m.%Y"))
    sales.to_csv(cfg.raw_file("sales"), index=False)
    raw.items.to_csv(cfg.raw_file("items"), index=False)
    raw.categories.to_csv(cfg.raw_file("categories"), index=False)
    raw.shops.to_csv(cfg.raw_file("shops"), index=False)
    raw.test.to_csv(cfg.raw_file("test"), index=False)


def make_config(root: Path) -> Config:
    base = load_config()
    cfg = Config(raw=base.raw, root=root).override(
        cleaning={"shop_id_map": {}},
        features={
            "min_month": 6,
            "target_lags": [1, 2, 3, 6],
            "aggregate_lags": [1, 2],
            "min_item_name_group": 2,
        },
        training={
            "backtest_months": [16, 17],
            "num_boost_round": 40,
            "early_stopping_rounds": 10,
            "sample_for_linear_baseline": 5000,
            "lgbm_params": {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_data_in_leaf": 5,
                "verbosity": -1,
                "num_threads": 2,
            },
        },
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def raw() -> RawData:
    return make_raw()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Runs the full pipeline once on synthetic data; shared by end-to-end tests."""
    from salesforecast import pipeline

    cfg = make_config(tmp_path_factory.mktemp("project"))
    write_raw(make_raw(), cfg)
    pipeline.stage_validate(cfg)
    pipeline.stage_features(cfg)
    pipeline.stage_train(cfg)
    pipeline.stage_backtest(cfg, num_rounds=10)
    pipeline.stage_predict(cfg)
    pipeline.stage_monitor(cfg)
    return cfg
