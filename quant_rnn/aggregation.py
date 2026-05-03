from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch

from .constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN
from .evaluation import evaluate_prediction_frame, load_model_from_checkpoint, predict_dataset
from .io import ensure_dir, parse_csv_list
from .sequences import EquitySequenceDataset
from .training import choose_device


GRID_NAME_RE = re.compile(r"grid_lr1e4_bs(?P<batch>\d{3})_hs(?P<hidden>\d{3})_l(?P<layers>\d+)")


def parse_grid_run_name(run_dir: Path) -> dict[str, int] | None:
    match = GRID_NAME_RE.fullmatch(run_dir.name)
    if not match:
        return None
    return {
        "batch_size": int(match.group("batch")),
        "hidden_size": int(match.group("hidden")),
        "num_layers": int(match.group("layers")),
    }


def completed_grid_runs(runs_dir: Path, pattern: str, max_hidden_size: int | None) -> list[Path]:
    candidates = []
    for run_dir in sorted(runs_dir.glob(pattern)):
        if not run_dir.is_dir():
            continue
        parsed = parse_grid_run_name(run_dir)
        if parsed is None:
            continue
        if max_hidden_size is not None and parsed["hidden_size"] > max_hidden_size:
            continue
        if not (run_dir / "training_summary.json").exists():
            continue
        candidates.append(run_dir)
    return candidates


def validation_summary_for_run(
    run_dir: Path,
    frame: pd.DataFrame,
    models: list[str],
    strategies: list[str],
    batch_size: int,
    device: torch.device,
    tc_bps: float,
) -> pd.DataFrame:
    parsed = parse_grid_run_name(run_dir) or {}
    config = pd.read_json(run_dir / "run_config.json", typ="series")
    sequence_length = int(config["sequence_length"])
    dataset = EquitySequenceDataset(frame, "val", sequence_length, DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)
    output_dir = ensure_dir(run_dir / "validation")
    rows = []
    for model_name in models:
        model, metadata = load_model_from_checkpoint(model_name, run_dir, device)
        predictions = predict_dataset(model, dataset, batch_size, device)
        model_rows = evaluate_prediction_frame(
            predictions,
            model_name,
            strategies,
            output_dir,
            tc_bps=tc_bps,
            low_quantile=0.2,
            high_quantile=0.8,
            gross_exposure=1.0,
        )
        for row in model_rows:
            row.update(
                {
                    "run_dir": str(run_dir),
                    **parsed,
                    "best_epoch": metadata.get("best_epoch"),
                    "stopped_epoch": metadata.get("stopped_epoch"),
                    "early_stopped": metadata.get("early_stopped"),
                    "checkpoint_selection_metric": metadata.get("selection_metric", "val_mse"),
                    "checkpoint_best_selection_value": metadata.get("best_selection_value"),
                    "checkpoint_best_val_mse": metadata.get("best_val_mse"),
                    "checkpoint_best_val_daily_ic": metadata.get("best_val_daily_ic"),
                    "checkpoint_best_val_rank_long_short_sharpe": metadata.get("best_val_rank_long_short_sharpe"),
                    "checkpoint_best_val_zscore_sharpe": metadata.get("best_val_zscore_sharpe"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def load_test_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "evaluation" / "evaluation_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    parsed = parse_grid_run_name(run_dir) or {}
    df = pd.read_csv(path)
    df.insert(0, "run_dir", str(run_dir))
    for key, value in parsed.items():
        df[key] = value
    return df


def build_selection_comparison(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if validation.empty or test.empty:
        return pd.DataFrame()
    neural_models = {"rnn", "lstm", "tcn"}
    val_neural = validation[validation["model"].isin(neural_models)].copy()
    test_neural = test[test["model"].isin(neural_models)].copy()
    merged = val_neural.merge(
        test_neural,
        on=["run_dir", "batch_size", "hidden_size", "num_layers", "model", "strategy"],
        suffixes=("_val", "_test"),
    )
    selectors = [
        ("selected_by_val_daily_ic", ["daily_ic_val", "sharpe_val"], [False, False]),
        ("selected_by_val_rank_long_short_sharpe", ["sharpe_val"], [False]),
        ("selected_by_val_zscore_sharpe", ["sharpe_val"], [False]),
        ("selected_by_val_mse", ["mse_val", "daily_ic_val"], [True, False]),
    ]
    rows = []
    for label, metrics, ascending in selectors:
        pool = merged.copy()
        if label == "selected_by_val_rank_long_short_sharpe":
            pool = pool[pool["strategy"] == "rank_long_short"]
        elif label == "selected_by_val_zscore_sharpe":
            pool = pool[pool["strategy"] == "zscore"]
        pool = pool.dropna(subset=[metrics[0]])
        if pool.empty:
            continue
        selected = pool.sort_values(metrics, ascending=ascending).iloc[0]
        rows.append(
            {
                "selection_rule": label,
                "run_dir": selected["run_dir"],
                "batch_size": selected["batch_size"],
                "hidden_size": selected["hidden_size"],
                "num_layers": selected["num_layers"],
                "model": selected["model"],
                "strategy": selected["strategy"],
                "val_mse": selected.get("mse_val"),
                "val_daily_ic": selected.get("daily_ic_val"),
                "val_sharpe": selected.get("sharpe_val"),
                "test_mse": selected.get("mse_test"),
                "test_daily_ic": selected.get("daily_ic_test"),
                "test_sharpe": selected.get("sharpe_test"),
                "test_ann_return": selected.get("ann_return_test"),
                "test_max_drawdown": selected.get("max_drawdown_test"),
            }
        )
    return pd.DataFrame(rows)


def run_aggregate_grid_stage(args) -> int:
    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)
    frame = pd.read_parquet(data_dir / "processed_scaled.parquet")
    models = parse_csv_list(args.models)
    strategies = parse_csv_list(args.strategies)
    run_dirs = completed_grid_runs(runs_dir, args.run_pattern, args.max_hidden_size)
    if not run_dirs:
        raise ValueError("No completed grid runs found.")
    device = choose_device(args.device)

    validation_frames = []
    test_frames = []
    for run_dir in run_dirs:
        validation_frames.append(
            validation_summary_for_run(
                run_dir,
                frame,
                models=models,
                strategies=strategies,
                batch_size=args.batch_size,
                device=device,
                tc_bps=args.tc_bps,
            )
        )
        test = load_test_summary(run_dir)
        if not test.empty:
            test_frames.append(test)

    validation_summary = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    test_summary = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    validation_summary.to_csv(args.validation_output, index=False)
    comparison = build_selection_comparison(validation_summary, test_summary)
    comparison.to_csv(args.comparison_output, index=False)
    print(f"Wrote validation summary: {args.validation_output}")
    print(f"Wrote selection comparison: {args.comparison_output}")
    return 0
