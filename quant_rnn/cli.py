from __future__ import annotations

import argparse

from .data import run_data_stage
from .evaluation import run_evaluate_stage
from .training import run_train_stage
from .walk_forward import run_walk_forward_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant_rnn", description="Three-stage quant sequence-model pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="Download stock data and create leakage-safe features/splits.")
    data.add_argument("--output-dir", default="data")
    data.add_argument("--years", type=int, default=10)
    data.add_argument("--top-n", type=int, default=100)
    data.add_argument("--train-years", type=int, default=6)
    data.add_argument("--val-years", type=int, default=2)
    data.add_argument("--test-years", type=int, default=2)
    data.add_argument("--vol-window", type=int, default=20)
    data.add_argument("--target-horizon", type=int, default=1)
    data.add_argument("--end-date", default=None, help="YYYY-MM-DD. Defaults to today.")
    data.add_argument("--tickers", default=None, help="Optional comma-separated ticker override.")
    data.add_argument("--ticker-file", default=None, help="Optional newline/comma separated ticker file.")
    data.set_defaults(func=run_data_stage)

    train = subparsers.add_parser("train", help="Train selected PyTorch sequence models.")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--run-dir", default=None)
    train.add_argument("--models", default="rnn,lstm,tcn")
    train.add_argument("--sequence-length", type=int, default=60)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--tcn-kernel-size", type=int, default=3)
    train.add_argument("--early-stopping-patience", type=int, default=5)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=run_train_stage)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained models on the test split.")
    evaluate.add_argument("--data-dir", default=None)
    evaluate.add_argument("--run-dir", default=None)
    evaluate.add_argument("--models", default=None)
    evaluate.add_argument("--strategies", default="rank_long_short,zscore")
    evaluate.add_argument("--baselines", default="cash_zero,equal_weight,momentum_12_1")
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--tc-bps", type=float, default=5.0)
    evaluate.add_argument("--low-quantile", type=float, default=0.2)
    evaluate.add_argument("--high-quantile", type=float, default=0.8)
    evaluate.add_argument("--gross-exposure", type=float, default=1.0)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=run_evaluate_stage)

    walk = subparsers.add_parser("walk-forward", help="Run expanding-window train/validate/test research folds.")
    walk.add_argument("--data-dir", default="data")
    walk.add_argument("--run-dir", default=None)
    walk.add_argument("--models", default="rnn,lstm,tcn")
    walk.add_argument("--baselines", default="cash_zero,equal_weight,momentum_12_1")
    walk.add_argument("--strategies", default="rank_long_short,zscore")
    walk.add_argument("--initial-train-years", type=int, default=6)
    walk.add_argument("--val-years", type=int, default=1)
    walk.add_argument("--test-years", type=int, default=1)
    walk.add_argument("--step-years", type=int, default=1)
    walk.add_argument("--sequence-length", type=int, default=60)
    walk.add_argument("--epochs", type=int, default=50)
    walk.add_argument("--batch-size", type=int, default=256)
    walk.add_argument("--learning-rate", type=float, default=1e-3)
    walk.add_argument("--hidden-size", type=int, default=64)
    walk.add_argument("--num-layers", type=int, default=1)
    walk.add_argument("--dropout", type=float, default=0.0)
    walk.add_argument("--tcn-kernel-size", type=int, default=3)
    walk.add_argument("--early-stopping-patience", type=int, default=5)
    walk.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    walk.add_argument("--tc-bps", type=float, default=5.0)
    walk.add_argument("--low-quantile", type=float, default=0.2)
    walk.add_argument("--high-quantile", type=float, default=0.8)
    walk.add_argument("--gross-exposure", type=float, default=1.0)
    walk.add_argument("--device", default="auto")
    walk.add_argument("--seed", type=int, default=42)
    walk.set_defaults(func=run_walk_forward_stage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
