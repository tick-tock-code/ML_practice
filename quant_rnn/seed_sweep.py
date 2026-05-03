from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .evaluation import run_evaluate_stage
from .io import ensure_dir, parse_csv_list, read_json, timestamp_slug, write_json
from .training import run_train_stage


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(seed) for seed in parse_csv_list(raw)]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def seed_run_dir(parent_dir: Path, seed: int) -> Path:
    return parent_dir / f"seed_{seed:03d}"


def collect_split_summary(child_run_dir: Path, split: str, seed: int) -> pd.DataFrame:
    output_dir = child_run_dir / ("validation" if split == "val" else "evaluation")
    summary_path = output_dir / "evaluation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {split} summary for seed {seed}: {summary_path}")
    frame = pd.read_csv(summary_path)
    frame.insert(0, "seed", seed)
    frame.insert(1, "split", split)
    frame.insert(2, "run_dir", str(child_run_dir))

    training_summary = read_json(child_run_dir / "training_summary.json")
    for column in ["best_epoch", "stopped_epoch", "early_stopped", "best_selection_value"]:
        frame[column] = pd.NA
    for model_name, metadata in training_summary.items():
        mask = frame["model"] == model_name
        frame.loc[mask, "best_epoch"] = metadata.get("best_epoch")
        frame.loc[mask, "stopped_epoch"] = metadata.get("stopped_epoch")
        frame.loc[mask, "early_stopped"] = metadata.get("early_stopped")
        frame.loc[mask, "best_selection_value"] = metadata.get("best_selection_value")
    return frame


def run_seed_sweep_stage(args) -> int:
    seeds = parse_seed_list(args.seeds)
    parent_dir = ensure_dir(Path(args.run_dir) if args.run_dir else Path("runs") / f"seed_sweep_{timestamp_slug()}")
    write_json(
        parent_dir / "seed_sweep_config.json",
        {
            "data_dir": args.data_dir,
            "models": parse_csv_list(args.models),
            "seeds": seeds,
            "strategies": parse_csv_list(args.strategies),
            "baselines": parse_csv_list(args.baselines),
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "tcn_kernel_size": args.tcn_kernel_size,
            "selection_metric": args.selection_metric,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "tc_bps": args.tc_bps,
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "gross_exposure": args.gross_exposure,
            "monotonicity": args.monotonicity,
            "monotonicity_buckets": args.monotonicity_buckets,
            "device": args.device,
        },
    )

    summary_frames = []
    for seed in seeds:
        child_run_dir = ensure_dir(seed_run_dir(parent_dir, seed))
        train_args = SimpleNamespace(
            data_dir=args.data_dir,
            run_dir=str(child_run_dir),
            models=args.models,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tcn_kernel_size=args.tcn_kernel_size,
            selection_metric=args.selection_metric,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            device=args.device,
            seed=seed,
        )
        run_train_stage(train_args)

        for split in ["val", "test"]:
            eval_args = SimpleNamespace(
                run_dir=str(child_run_dir),
                data_dir=args.data_dir,
                split=split,
                models=args.models,
                strategies=args.strategies,
                baselines=args.baselines,
                batch_size=args.batch_size,
                tc_bps=args.tc_bps,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
                gross_exposure=args.gross_exposure,
                monotonicity=args.monotonicity,
                monotonicity_buckets=args.monotonicity_buckets,
                device=args.device,
            )
            run_evaluate_stage(eval_args)
            summary_frames.append(collect_split_summary(child_run_dir, split, seed))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary.to_csv(parent_dir / "seed_sweep_summary.csv", index=False)
    print(f"Seed-sweep directory: {parent_dir}")
    return 0
