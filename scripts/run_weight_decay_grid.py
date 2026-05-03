from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_PYTHON = r"C:\ProgramData\anaconda3\envs\torch_env2\python.exe"
DEFAULT_OUTPUT_DIR = Path("runs") / "grid_lr1e5_wd"


@dataclass(frozen=True)
class GridConfig:
    model: str = "lstm"
    learning_rate: float = 1e-5
    sequence_length: int = 60
    batch_size: int = 128
    dropout: float = 0.2
    epochs: int = 50
    early_stopping_patience: int = 8
    selection_metric: str = "val_daily_ic"
    strategies: str = "rank_long_short,zscore"
    baselines: str = ""
    monotonicity_buckets: int = 5
    hidden_sizes: tuple[int, ...] = (64, 128)
    num_layers: tuple[int, ...] = (1, 2)
    weight_decays: tuple[float, ...] = (0.0, 1e-5, 1e-4)


@dataclass(frozen=True)
class GridRun:
    hidden_size: int
    num_layers: int
    weight_decay: float


def weight_decay_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")


def run_name(config: GridConfig, run: GridRun) -> str:
    lr = f"{config.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return f"grid_lr{lr.replace('-', '')}_bs{config.batch_size:03d}_hs{run.hidden_size:03d}_l{run.num_layers}_wd{weight_decay_label(run.weight_decay)}"


def expand_grid(config: GridConfig) -> list[GridRun]:
    return [
        GridRun(hidden_size=hidden_size, num_layers=num_layers, weight_decay=weight_decay)
        for hidden_size in config.hidden_sizes
        for num_layers in config.num_layers
        for weight_decay in config.weight_decays
    ]


def train_command(python_exe: str, config: GridConfig, run: GridRun, run_dir: Path) -> list[str]:
    return [
        python_exe,
        "-m",
        "quant_rnn",
        "train",
        "--run-dir",
        str(run_dir),
        "--models",
        config.model,
        "--sequence-length",
        str(config.sequence_length),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--hidden-size",
        str(run.hidden_size),
        "--num-layers",
        str(run.num_layers),
        "--dropout",
        str(config.dropout),
        "--weight-decay",
        str(run.weight_decay),
        "--selection-metric",
        config.selection_metric,
        "--early-stopping-patience",
        str(config.early_stopping_patience),
    ]


def evaluate_command(python_exe: str, config: GridConfig, run_dir: Path, split: str = "val") -> list[str]:
    return [
        python_exe,
        "-m",
        "quant_rnn",
        "evaluate",
        "--split",
        split,
        "--run-dir",
        str(run_dir),
        "--models",
        config.model,
        "--strategies",
        config.strategies,
        "--baselines",
        config.baselines,
        "--monotonicity-buckets",
        str(config.monotonicity_buckets),
    ]


def command_line(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def validation_summary_exists(run_dir: Path) -> bool:
    return (run_dir / "validation" / "evaluation_summary.csv").exists()


def write_grid_config(output_dir: Path, config: GridConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["hidden_sizes"] = list(config.hidden_sizes)
    payload["num_layers"] = list(config.num_layers)
    payload["weight_decays"] = list(config.weight_decays)
    (output_dir / "grid_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_command_log(output_dir: Path, command: list[str], dry_run: bool, skipped: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DRY_RUN" if dry_run else "SKIP" if skipped else "RUN"
    with (output_dir / "commands.log").open("a", encoding="utf-8") as f:
        f.write(f"[{prefix}] {command_line(command)}\n")


def run_command(command: list[str], dry_run: bool, output_dir: Path) -> None:
    append_command_log(output_dir, command, dry_run=dry_run)
    if dry_run:
        print(command_line(command))
        return
    subprocess.run(command, check=True)


def aggregate_validation_summary(output_dir: Path, config: GridConfig) -> pd.DataFrame:
    rows = []
    for run in expand_grid(config):
        child_dir = output_dir / run_name(config, run)
        summary_path = child_dir / "validation" / "evaluation_summary.csv"
        training_path = child_dir / "training_summary.json"
        if not summary_path.exists() or not training_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        training = json.loads(training_path.read_text(encoding="utf-8"))
        model_meta = training.get(config.model, {})
        for _, row in summary.iterrows():
            record = {
                "run_dir": str(child_dir),
                "model": row.get("model"),
                "strategy": row.get("strategy"),
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "hidden_size": run.hidden_size,
                "num_layers": run.num_layers,
                "dropout": config.dropout,
                "weight_decay": run.weight_decay,
                "best_epoch": model_meta.get("best_epoch"),
                "stopped_epoch": model_meta.get("stopped_epoch"),
                "early_stopped": model_meta.get("early_stopped"),
                "selection_metric": model_meta.get("selection_metric"),
                "best_selection_value": model_meta.get("best_selection_value"),
                "best_val_mse": model_meta.get("best_val_mse"),
                "best_val_daily_ic": model_meta.get("best_val_daily_ic"),
                "best_val_rank_long_short_sharpe": model_meta.get("best_val_rank_long_short_sharpe"),
                "best_val_zscore_sharpe": model_meta.get("best_val_zscore_sharpe"),
            }
            for column in [
                "mse",
                "mae",
                "directional_accuracy",
                "correlation",
                "daily_ic",
                "monotonic_spread",
                "ann_return",
                "sharpe",
                "max_drawdown",
                "total_turnover",
                "total_transaction_cost",
            ]:
                record[column] = row.get(column)
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "grid_validation_summary.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LSTM 1e-5 learning-rate weight-decay grid.")
    parser.add_argument("--python-exe", default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GridConfig()
    output_dir = Path(args.output_dir)
    write_grid_config(output_dir, config)

    for run in expand_grid(config):
        child_dir = output_dir / run_name(config, run)
        if args.resume and validation_summary_exists(child_dir):
            append_command_log(output_dir, train_command(args.python_exe, config, run, child_dir), dry_run=False, skipped=True)
            continue
        run_command(train_command(args.python_exe, config, run, child_dir), dry_run=args.dry_run, output_dir=output_dir)
        run_command(evaluate_command(args.python_exe, config, child_dir, split="val"), dry_run=args.dry_run, output_dir=output_dir)
        if args.include_test:
            run_command(evaluate_command(args.python_exe, config, child_dir, split="test"), dry_run=args.dry_run, output_dir=output_dir)

    if not args.dry_run:
        aggregate_validation_summary(output_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
