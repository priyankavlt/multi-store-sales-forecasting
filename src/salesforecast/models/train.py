"""Model training, hold-out evaluation and rolling-origin backtesting."""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from salesforecast.config import Config
from salesforecast.features.build import TARGET
from salesforecast.features.store import load_features, load_lookup, load_meta
from salesforecast.models import baselines, evaluate
from salesforecast.models.registry import new_run_id, save_run
from salesforecast.utils import get_logger, git_sha, read_json, timer, write_json

log = get_logger(__name__)


def _params(cfg: Config) -> dict[str, Any]:
    params = dict(cfg.training["lgbm_params"])
    params.setdefault("seed", cfg.training["random_state"])
    params.setdefault("num_threads", 0)
    return params


def _fit(
    cfg: Config,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    num_rounds: int,
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    dtrain = lgb.Dataset(X, y, categorical_feature=categorical, free_raw_data=True)
    history: dict[str, Any] = {}
    callbacks = [lgb.log_evaluation(100), lgb.record_evaluation(history)]
    valid_sets, valid_names = [], []
    if valid is not None:
        dvalid = lgb.Dataset(valid[0], valid[1], categorical_feature=categorical, reference=dtrain)
        valid_sets, valid_names = [dtrain, dvalid], ["train", "valid"]
        callbacks.append(lgb.early_stopping(cfg.training["early_stopping_rounds"], verbose=True))
    booster = lgb.train(
        _params(cfg),
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster, history


def _predict(booster: lgb.Booster, X: pd.DataFrame, clip: tuple[float, float]) -> np.ndarray:
    return np.clip(booster.predict(X, num_iteration=booster.best_iteration or None), *clip)


def train(cfg: Config) -> dict[str, Any]:
    cfg.ensure_dirs()
    meta = load_meta(cfg)
    features: list[str] = meta["feature_names"]
    categorical: list[str] = meta["categorical"]
    last = meta["last_observed_month"]
    clip = tuple(cfg.features["target_clip"])
    run_id = new_run_id()

    with timer(log, "Loading feature table"):
        df = load_features(cfg, max_month=last)
    is_valid = df["date_block_num"].to_numpy() == last
    train_df, valid_df = df.loc[~is_valid], df.loc[is_valid]
    log.info(
        "Train rows %s (months < %d) | validation rows %s (month %d)",
        f"{len(train_df):,}",
        last,
        f"{len(valid_df):,}",
        last,
    )
    y_valid = valid_df[TARGET].to_numpy()

    # ---- baselines
    preds: dict[str, np.ndarray] = {
        "Zero": baselines.zero(valid_df),
        "Last month": baselines.last_month(valid_df),
        "3-month average": baselines.moving_average_3(valid_df),
    }
    with timer(log, "Fitting ridge baseline"):
        ridge = baselines.fit_ridge(
            train_df,
            train_df[TARGET],
            features,
            cfg.training["sample_for_linear_baseline"],
            cfg.training["random_state"],
        )
        preds["Ridge regression"] = np.clip(
            ridge.predict(valid_df[features].fillna(0).to_numpy(np.float32)), *clip
        )
    del ridge

    # ---- gradient boosting with early stopping on the hold-out month
    with timer(log, "Training LightGBM (early stopping on hold-out month)"):
        booster, history = _fit(
            cfg,
            train_df[features],
            train_df[TARGET],
            categorical,
            cfg.training["num_boost_round"],
            valid=(valid_df[features], valid_df[TARGET]),
        )
    best_iter = booster.best_iteration or booster.current_iteration()
    preds["LightGBM"] = _predict(booster, valid_df[features], clip)

    metrics = {name: evaluate.regression_metrics(y_valid, p) for name, p in preds.items()}
    for name, m in metrics.items():
        log.info("%-18s RMSE %.4f  MAE %.4f", name, m["rmse"], m["mae"])

    # ---- report artifacts
    fig_dir = cfg.figures_dir
    lgb_pred = preds["LightGBM"]
    evaluate.plot_model_comparison(
        {k: v["rmse"] for k, v in metrics.items()}, "LightGBM", fig_dir / "model_comparison.png"
    )
    importance = evaluate.plot_feature_importance(booster, fig_dir / "feature_importance.png")
    evaluate.plot_learning_curve(history, best_iter, fig_dir / "learning_curve.png")
    evaluate.plot_shop_totals(
        valid_df["shop_id"].to_numpy(), y_valid, lgb_pred, fig_dir / "shop_totals.png"
    )

    cats = load_lookup(cfg, "categories")
    family_names = cats.drop_duplicates("cat_family_code").set_index("cat_family_code")[
        "cat_family_en"
    ]
    by_family = evaluate.segment_metrics(
        valid_df, y_valid, lgb_pred, valid_df["cat_family_code"].map(family_names)
    )
    evaluate.plot_segment_rmse(
        by_family, "Hold-out RMSE by category family", fig_dir / "rmse_by_category.png"
    )
    novelty = np.select(
        [valid_df["item_is_new"].to_numpy() == 1, valid_df["shop_item_is_new"].to_numpy() == 1],
        ["New item", "Known item, new to shop"],
        "Previously sold in shop",
    )
    by_novelty = evaluate.segment_metrics(valid_df, y_valid, lgb_pred, pd.Series(novelty))
    evaluate.plot_segment_rmse(
        by_novelty, "Hold-out RMSE by item history", fig_dir / "rmse_by_history.png"
    )

    del train_df, valid_df, booster
    gc.collect()

    # ---- refit on all observed months for the production model
    if cfg.training["refit_on_full_history"]:
        with timer(log, f"Refitting on months <= {last} with {best_iter} rounds"):
            final, _ = _fit(cfg, df[features], df[TARGET], categorical, best_iter)
    else:
        final, _ = _fit(
            cfg, df.loc[~is_valid, features], df.loc[~is_valid, TARGET], categorical, best_iter
        )
    del df
    gc.collect()

    manifest_path = cfg.raw_dir / "MANIFEST.json"
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model_type": "lightgbm",
        "lightgbm_version": lgb.__version__,
        "params": _params(cfg),
        "best_iteration": best_iter,
        "features": features,
        "categorical": categorical,
        "target_clip": list(clip),
        "train_months": [cfg.features["min_month"], last - 1],
        "validation_month": last,
        "refit_months": [cfg.features["min_month"], last]
        if cfg.training["refit_on_full_history"]
        else None,
        "forecast_month": meta["forecast_month"],
        "validation_metrics": metrics,
        "data_manifest": read_json(manifest_path) if manifest_path.exists() else None,
    }
    run_dir = save_run(cfg, run_id, final, metadata)

    report = {
        "run_id": run_id,
        "validation_month": last,
        "best_iteration": best_iter,
        "metrics": metrics,
        "lift_vs_best_baseline": 1
        - metrics["LightGBM"]["rmse"]
        / min(v["rmse"] for k, v in metrics.items() if k != "LightGBM"),
        "rmse_by_category_family": by_family.round(4).reset_index().to_dict(orient="records"),
        "rmse_by_item_history": by_novelty.round(4).reset_index().to_dict(orient="records"),
        "top_features": importance.head(20)[["feature", "gain_share"]]
        .round(4)
        .to_dict(orient="records"),
    }
    write_json(report, cfg.reports_dir / "metrics.json")
    log.info("Saved model run %s to %s", run_id, run_dir)
    return report


def backtest(cfg: Config, num_rounds: int | None = None) -> pd.DataFrame:
    """Rolling-origin evaluation: for each month m, train on months < m and score month m."""
    meta = load_meta(cfg)
    features, categorical = meta["feature_names"], meta["categorical"]
    clip = tuple(cfg.features["target_clip"])
    if num_rounds is None:
        try:
            num_rounds = read_json(cfg.reports_dir / "metrics.json")["best_iteration"]
        except FileNotFoundError:
            num_rounds = 300
    months = cfg.training["backtest_months"]
    df = load_features(cfg, max_month=max(months))
    rows = []
    for m in months:
        tr = df["date_block_num"].to_numpy() < m
        te = df["date_block_num"].to_numpy() == m
        test_df = df.loc[te]
        y = test_df[TARGET].to_numpy()
        with timer(log, f"Backtest month {m}: training on {int(tr.sum()):,} rows"):
            booster, _ = _fit(
                cfg, df.loc[tr, features], df.loc[tr, TARGET], categorical, num_rounds
            )
        rows.append(
            {
                "month": m,
                "LightGBM": evaluate.regression_metrics(
                    y, _predict(booster, test_df[features], clip)
                )["rmse"],
                "3-month average": evaluate.regression_metrics(
                    y, baselines.moving_average_3(test_df)
                )["rmse"],
                "Last month": evaluate.regression_metrics(y, baselines.last_month(test_df))["rmse"],
            }
        )
        del booster
        gc.collect()
    results = pd.DataFrame(rows).set_index("month")
    evaluate.plot_backtest(results, cfg.figures_dir / "backtest.png")
    write_json(
        {
            "num_rounds": num_rounds,
            "months": results.reset_index().to_dict(orient="records"),
            "mean": results.mean().to_dict(),
        },
        cfg.reports_dir / "backtest.json",
    )
    log.info("Backtest RMSE by month:\n%s", results.round(4).to_string())
    return results
