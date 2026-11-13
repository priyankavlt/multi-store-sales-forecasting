"""Request / response models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pair(BaseModel):
    shop_id: int = Field(..., ge=0, examples=[31])
    item_id: int = Field(..., ge=0, examples=[20949])


class PredictRequest(BaseModel):
    pairs: list[Pair] = Field(..., min_length=1, max_length=10_000)


class PairPrediction(BaseModel):
    shop_id: int
    item_id: int
    item_name: str | None = None
    predicted_units: float | None = Field(
        None,
        description="Forecast units for the month, clipped to the target range. "
        "Null when the pair is outside the forecast set.",
    )
    in_forecast_set: bool


class PredictResponse(BaseModel):
    model_run_id: str
    forecast_month: int
    predictions: list[PairPrediction]


class ShopForecastItem(BaseModel):
    item_id: int
    item_name: str | None
    category: str | None
    predicted_units: float


class ShopForecastResponse(BaseModel):
    model_run_id: str
    forecast_month: int
    shop_id: int
    shop_name: str | None
    items: list[ShopForecastItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_run_id: str | None = None
