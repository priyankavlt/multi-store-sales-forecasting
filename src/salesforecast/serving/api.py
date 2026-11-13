"""FastAPI service exposing the production forecast.

Run with ``uvicorn salesforecast.serving.api:app``. The config path can be set
with the ``SALESFORECAST_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from salesforecast import __version__
from salesforecast.config import load_config
from salesforecast.features.store import load_lookup
from salesforecast.models.predict import Forecaster
from salesforecast.serving.schemas import (
    HealthResponse,
    PairPrediction,
    PredictRequest,
    PredictResponse,
    ShopForecastItem,
    ShopForecastResponse,
)
from salesforecast.utils import get_logger

log = get_logger(__name__)
STATE: dict[str, Any] = {}


def _load_state() -> None:
    cfg = load_config(os.environ.get("SALESFORECAST_CONFIG"))
    STATE["forecaster"] = Forecaster.from_config(cfg)
    items = load_lookup(cfg, "items")
    cats = load_lookup(cfg, "categories")
    shops = load_lookup(cfg, "shops")
    STATE["item_names"] = items.set_index("item_id")["item_name"].to_dict()
    STATE["item_category"] = (
        items.merge(cats, on="item_category_id", how="left")
        .set_index("item_id")["item_category_name"]
        .to_dict()
    )
    STATE["shop_names"] = shops.set_index("shop_id")["shop_name"].to_dict()
    log.info("Loaded model run %s", STATE["forecaster"].metadata["run_id"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_state()
    except FileNotFoundError as exc:  # keep serving /health so orchestration can report the problem
        log.error("Model not available: %s", exc)
    yield
    STATE.clear()


app = FastAPI(
    title="Multi-store Sales Forecast API",
    version=__version__,
    description="Next-month unit sales forecasts for shop/item pairs.",
    lifespan=lifespan,
)


def _forecaster() -> Forecaster:
    fc = STATE.get("forecaster")
    if fc is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model and run `salesforecast predict`.",
        )
    return fc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    fc = STATE.get("forecaster")
    return HealthResponse(
        status="ok" if fc else "degraded",
        model_loaded=fc is not None,
        model_run_id=fc.metadata["run_id"] if fc else None,
    )


@app.get("/model")
def model_info() -> dict[str, Any]:
    meta = _forecaster().metadata
    keys = (
        "run_id",
        "created_at",
        "git_sha",
        "model_type",
        "best_iteration",
        "validation_month",
        "forecast_month",
        "train_months",
        "refit_months",
        "validation_metrics",
        "features",
    )
    return {k: meta.get(k) for k in keys}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    fc = _forecaster()
    pairs = [(p.shop_id, p.item_id) for p in req.pairs]
    scores = fc.predict_pairs(pairs)
    names = STATE["item_names"]
    return PredictResponse(
        model_run_id=fc.metadata["run_id"],
        forecast_month=fc.metadata["forecast_month"],
        predictions=[
            PairPrediction(
                shop_id=s,
                item_id=i,
                item_name=names.get(i),
                predicted_units=None if score is None else round(score, 4),
                in_forecast_set=score is not None,
            )
            for (s, i), score in zip(pairs, scores, strict=True)
        ],
    )


@app.get("/forecast/shops/{shop_id}", response_model=ShopForecastResponse)
def shop_forecast(shop_id: int, top: int = Query(20, ge=1, le=500)) -> ShopForecastResponse:
    fc = _forecaster()
    try:
        best: pd.DataFrame = fc.top_items(shop_id, top)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Shop {shop_id} is not in the forecast set"
        ) from None
    return ShopForecastResponse(
        model_run_id=fc.metadata["run_id"],
        forecast_month=fc.metadata["forecast_month"],
        shop_id=shop_id,
        shop_name=STATE["shop_names"].get(shop_id),
        items=[
            ShopForecastItem(
                item_id=int(r.item_id),
                item_name=STATE["item_names"].get(int(r.item_id)),
                category=STATE["item_category"].get(int(r.item_id)),
                predicted_units=round(float(r.prediction), 4),
            )
            for r in best.itertuples()
        ],
    )
