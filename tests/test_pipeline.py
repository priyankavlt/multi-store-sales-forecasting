"""End-to-end: validate -> features -> train -> backtest -> predict -> monitor, then serve."""

import numpy as np
import pandas as pd
import pytest

from salesforecast.models.registry import latest_run_dir
from salesforecast.monitoring.drift import psi
from salesforecast.utils import read_json


def test_training_produces_registered_model_and_reports(trained_project):
    cfg = trained_project
    run_dir = latest_run_dir(cfg)
    assert (run_dir / "model.txt").exists()
    meta = read_json(run_dir / "metadata.json")
    assert meta["forecast_month"] == 18 and meta["validation_month"] == 17
    assert meta["best_iteration"] >= 1

    report = read_json(cfg.reports_dir / "metrics.json")
    assert {"Zero", "Last month", "3-month average", "Ridge regression", "LightGBM"} <= set(
        report["metrics"]
    )
    # the model should beat predicting zero on this synthetic signal
    assert report["metrics"]["LightGBM"]["rmse"] < report["metrics"]["Zero"]["rmse"]
    for fig in (
        "model_comparison",
        "feature_importance",
        "learning_curve",
        "rmse_by_category",
        "backtest",
    ):
        assert (cfg.figures_dir / f"{fig}.png").exists()
    assert len(read_json(cfg.reports_dir / "backtest.json")["months"]) == 2


def test_batch_forecast_covers_every_requested_pair(trained_project):
    cfg = trained_project
    fc = pd.read_csv(cfg.predictions_dir / "forecast.csv")
    test = pd.read_csv(cfg.raw_file("test"))
    assert len(fc) == len(test)
    assert set(fc["ID"]) == set(test["ID"])
    assert fc["item_cnt_month"].between(0, 20).all()
    sub = pd.read_csv(cfg.predictions_dir / "submission.csv")
    assert list(sub.columns) == ["ID", "item_cnt_month"]


def test_drift_report_written(trained_project):
    drift = read_json(trained_project.reports_dir / "drift.json")
    assert drift["current_month"] == 18
    assert all(r["status"] in {"ok", "warning", "alert"} for r in drift["features"])


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert psi(a, rng.normal(size=20_000)) < 0.01
    assert psi(a, rng.normal(loc=1.5, size=20_000)) > 0.25


@pytest.fixture
def client(trained_project, monkeypatch):
    from fastapi.testclient import TestClient

    from salesforecast.serving import api

    monkeypatch.setattr(api, "load_config", lambda _path=None: trained_project)
    with TestClient(api.app) as c:
        yield c


def test_api_health_and_model(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "model_run_id": client.get("/model").json()["run_id"],
    }


def test_api_predict_matches_batch_forecast(client, trained_project):
    fc = pd.read_csv(trained_project.predictions_dir / "forecast.csv")
    row = fc.iloc[3]
    resp = client.post(
        "/predict",
        json={
            "pairs": [
                {"shop_id": int(row.shop_id), "item_id": int(row.item_id)},
                {"shop_id": 4, "item_id": 1},
            ]
        },
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert preds[0]["in_forecast_set"] is True
    assert preds[0]["predicted_units"] == pytest.approx(row.item_cnt_month, abs=1e-3)
    assert preds[1] == {
        "shop_id": 4,
        "item_id": 1,
        "item_name": preds[1]["item_name"],
        "predicted_units": None,
        "in_forecast_set": False,
    }


def test_api_shop_forecast(client):
    resp = client.get("/forecast/shops/0", params={"top": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 5
    units = [i["predicted_units"] for i in body["items"]]
    assert units == sorted(units, reverse=True)
    assert client.get("/forecast/shops/4").status_code == 404


def test_api_rejects_empty_request(client):
    assert client.post("/predict", json={"pairs": []}).status_code == 422
