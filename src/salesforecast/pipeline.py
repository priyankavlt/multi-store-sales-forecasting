"""Pipeline stages, each runnable on its own or chained end to end."""

from __future__ import annotations

from typing import Any

from salesforecast.config import Config
from salesforecast.data.clean import clean
from salesforecast.data.ingest import download, load_raw
from salesforecast.data.validate import assert_valid, validate_raw
from salesforecast.features.build import build_features, lookup_tables
from salesforecast.features.store import save_features
from salesforecast.utils import get_logger, timer, write_json

log = get_logger(__name__)


def stage_download(cfg: Config, force: bool = False) -> dict[str, str]:
    return download(cfg, force=force)


def stage_validate(cfg: Config) -> dict[str, Any]:
    raw = load_raw(cfg)
    report = validate_raw(raw, cfg.cleaning)
    write_json(report.to_dict(), cfg.reports_dir / "data_validation.json")
    for w in report.warnings:
        log.warning(w)
    assert_valid(report)
    log.info("Data validation passed")
    return report.to_dict()


def stage_features(cfg: Config) -> None:
    raw = load_raw(cfg)
    assert_valid(validate_raw(raw, cfg.cleaning))
    raw = clean(raw, cfg.cleaning)
    with timer(log, "Feature engineering"):
        fs = build_features(raw, cfg.features)
    with timer(log, "Writing feature store"):
        save_features(cfg, fs, lookup_tables(raw, cfg.features))


def stage_train(cfg: Config) -> dict[str, Any]:
    from salesforecast.models.train import train

    return train(cfg)


def stage_backtest(cfg: Config, num_rounds: int | None = None):
    from salesforecast.models.train import backtest

    return backtest(cfg, num_rounds)


def stage_predict(cfg: Config):
    from salesforecast.models.predict import predict_batch

    return predict_batch(cfg)


def stage_monitor(cfg: Config) -> dict[str, Any]:
    from salesforecast.monitoring.drift import drift_report

    return drift_report(cfg)


def run_all(cfg: Config, with_backtest: bool = False) -> None:
    cfg.ensure_dirs()
    stage_download(cfg)
    stage_validate(cfg)
    stage_features(cfg)
    stage_train(cfg)
    if with_backtest:
        stage_backtest(cfg)
    stage_predict(cfg)
    stage_monitor(cfg)
    log.info("Pipeline complete")
