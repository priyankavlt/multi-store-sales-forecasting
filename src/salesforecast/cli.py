"""Command-line entry point: ``salesforecast <stage>`` or ``python -m salesforecast <stage>``."""

from __future__ import annotations

import argparse

from salesforecast import pipeline
from salesforecast.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="salesforecast", description=__doc__)
    parser.add_argument(
        "--config", default=None, help="Path to a YAML config (default: configs/config.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Fetch the raw dataset")
    p.add_argument("--force", action="store_true", help="Re-download files that already exist")
    sub.add_parser("validate", help="Run data-quality checks on the raw data")
    sub.add_parser("features", help="Clean data and build the feature store")
    sub.add_parser("train", help="Train, evaluate and register a model")
    p = sub.add_parser("backtest", help="Rolling-origin backtest over configured months")
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Boosting rounds (default: best iteration of last training)",
    )
    sub.add_parser("predict", help="Score the forecast month with the latest model")
    sub.add_parser("monitor", help="Feature drift report for the forecast month")
    p = sub.add_parser("all", help="Run every stage end to end")
    p.add_argument("--backtest", action="store_true", help="Include the rolling-origin backtest")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.ensure_dirs()

    match args.command:
        case "download":
            pipeline.stage_download(cfg, force=args.force)
        case "validate":
            pipeline.stage_validate(cfg)
        case "features":
            pipeline.stage_features(cfg)
        case "train":
            pipeline.stage_train(cfg)
        case "backtest":
            pipeline.stage_backtest(cfg, args.rounds)
        case "predict":
            pipeline.stage_predict(cfg)
        case "monitor":
            pipeline.stage_monitor(cfg)
        case "all":
            pipeline.run_all(cfg, with_backtest=args.backtest)


if __name__ == "__main__":
    main()
