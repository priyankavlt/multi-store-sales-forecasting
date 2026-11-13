"""Reference models the gradient-boosted model has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def zero(frame: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(frame), dtype=np.float32)


def last_month(frame: pd.DataFrame) -> np.ndarray:
    """Seasonal-naive with period 1: next month equals last month."""
    return frame["shop_item_cnt_lag_1"].fillna(0).to_numpy(np.float32)


def moving_average_3(frame: pd.DataFrame) -> np.ndarray:
    return frame["shop_item_cnt_mean_3"].fillna(0).to_numpy(np.float32)


def fit_ridge(
    train: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    sample: int,
    random_state: int,
):
    """Linear model on a row sample (the full table does not need to fit in a dense matrix)."""
    if len(train) > sample:
        idx = np.random.default_rng(random_state).choice(len(train), sample, replace=False)
        train, target = train.iloc[idx], target.iloc[idx]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[features].fillna(0).to_numpy(np.float32), target.to_numpy(np.float32))
    return model
