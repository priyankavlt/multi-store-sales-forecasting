"""Metrics and report figures for the hold-out month."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from salesforecast import viz
from salesforecast.utils import rmse


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "total_actual": float(y_true.sum()),
        "total_predicted": float(y_pred.sum()),
    }


def segment_metrics(
    frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, by: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame({"segment": by.to_numpy(), "y": y_true, "p": y_pred})
    out = df.groupby("segment").apply(
        lambda g: pd.Series(
            {
                "rows": len(g),
                "rmse": rmse(g["y"], g["p"]),
                "actual_units": g["y"].sum(),
                "predicted_units": g["p"].sum(),
            }
        ),
        include_groups=False,
    )
    return out.sort_values("rows", ascending=False)


# --------------------------------------------------------------------------- figures
def plot_model_comparison(scores: dict[str, float], best: str, path: Path) -> None:
    viz.apply_style()
    names = sorted(scores, key=scores.get, reverse=True)
    values = [scores[n] for n in names]
    colors = [viz.SERIES[0] if n == best else viz.NEUTRAL_BAR for n in names]
    fig, ax = viz.plt.subplots(figsize=(7, 0.55 * len(names) + 1.2))
    bars = ax.barh(names, values, color=colors, height=0.6)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v,
            bar.get_y() + bar.get_height() / 2,
            f"  {v:.3f}",
            va="center",
            color=viz.INK_SECONDARY,
        )
    ax.set_xlabel("RMSE on hold-out month (lower is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.15)
    viz.save(fig, path)


def plot_feature_importance(booster: lgb.Booster, path: Path, top: int = 20) -> pd.DataFrame:
    imp = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "splits": booster.feature_importance("split"),
        }
    )
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    viz.apply_style()
    head = imp.head(top).iloc[::-1]
    fig, ax = viz.plt.subplots(figsize=(7, 0.32 * len(head) + 1.2))
    ax.barh(head["feature"], head["gain_share"] * 100, color=viz.SERIES[0], height=0.6)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title(f"Top {top} features")
    ax.grid(axis="y", visible=False)
    viz.save(fig, path)
    return imp


def plot_segment_rmse(seg: pd.DataFrame, title: str, path: Path) -> None:
    viz.apply_style()
    seg = seg.sort_values("rmse")
    fig, ax = viz.plt.subplots(figsize=(7, 0.35 * len(seg) + 1.2))
    ax.barh([str(s) for s in seg.index], seg["rmse"], color=viz.SERIES[0], height=0.6)
    for y, (v, n) in enumerate(zip(seg["rmse"], seg["rows"], strict=True)):
        ax.text(
            v, y, f"  {v:.2f}  (n={int(n):,})", va="center", color=viz.INK_SECONDARY, fontsize=8
        )
    ax.set_xlabel("RMSE")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, seg["rmse"].max() * 1.35)
    viz.save(fig, path)


def plot_shop_totals(
    shop_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path
) -> None:
    df = pd.DataFrame({"shop": shop_ids, "y": y_true, "p": y_pred}).groupby("shop").sum()
    viz.apply_style()
    fig, ax = viz.plt.subplots(figsize=(6, 6))
    lim = max(df["y"].max(), df["p"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=viz.BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(
        df["y"], df["p"], s=40, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.5, zorder=2
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual units (clipped target)")
    ax.set_ylabel("Predicted units")
    ax.set_title("Hold-out month: units per shop")
    viz.save(fig, path)


def plot_learning_curve(history: dict[str, Any], best_iteration: int, path: Path) -> None:
    curves = history.get("valid", {}).get("rmse") or []
    train_curve = history.get("train", {}).get("rmse") or []
    if not curves:
        return
    viz.apply_style()
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    if train_curve:
        ax.plot(train_curve, color=viz.SERIES[1], label="Train")
    ax.plot(curves, color=viz.SERIES[0], label="Validation")
    ax.axvline(best_iteration - 1, color=viz.MUTED, linewidth=1, linestyle=":")
    ax.text(
        best_iteration - 1,
        ax.get_ylim()[1],
        f" best iteration {best_iteration}",
        va="top",
        color=viz.INK_SECONDARY,
        fontsize=8,
    )
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE")
    ax.set_title("LightGBM learning curve")
    ax.legend(loc="upper right")
    viz.save(fig, path)


def plot_backtest(results: pd.DataFrame, path: Path) -> None:
    """results: index = month, columns = model names, values = RMSE."""
    viz.apply_style()
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    for i, col in enumerate(results.columns):
        ax.plot(
            results.index,
            results[col],
            marker="o",
            markersize=7,
            color=viz.SERIES[i % len(viz.SERIES)],
            label=col,
        )
    ax.set_xticks(results.index)
    ax.set_xlabel("Forecast month (date_block_num)")
    ax.set_ylabel("RMSE")
    ax.set_title("Rolling-origin backtest")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", ncols=len(results.columns))
    viz.save(fig, path)
