from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .backtest import make_weights, strategy_returns
from .baselines import baseline_predictions, equal_weight_weights
from .constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN
from .io import ensure_dir, parse_csv_list, read_json, write_json
from .metrics import daily_information_coefficient, regression_metrics
from .models import create_model
from .sequences import EquitySequenceDataset
from .training import choose_device


def latest_run_dir(base: str | Path = "runs") -> Path:
    root = Path(base)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError("No run directories found. Pass --run-dir explicitly.")
    return sorted(candidates)[-1]


def predict_dataset(model, dataset: EquitySequenceDataset, batch_size: int, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, indices in loader:
            output = model(x.to(device)).detach().cpu().numpy()
            for pred, target, sample_index in zip(output, y.numpy(), indices.numpy()):
                metadata = dataset.sample_metadata(int(sample_index))
                rows.append(
                    {
                        "ticker": metadata.ticker,
                        "date": metadata.date,
                        "target_date": metadata.target_date,
                        "target": float(target),
                        "prediction": float(pred),
                    }
                )
    return pd.DataFrame(rows)


def load_model_from_checkpoint(model_name: str, run_dir: Path, device: torch.device):
    metadata_path = run_dir / "checkpoints" / f"{model_name}_checkpoint.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata for {model_name}: {metadata_path}")
    metadata = read_json(metadata_path)
    config = metadata["model_config"]
    model = create_model(
        model_name,
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        tcn_kernel_size=int(config.get("tcn_kernel_size", 3)),
    )
    state_path = Path(metadata["state_dict_path"])
    if not state_path.is_absolute():
        state_path = run_dir / state_path
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model, metadata


def plot_cumulative_return(daily: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(daily["target_date"]), daily["cum_index"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative index")
    plt.grid(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    label: str,
    strategies: list[str],
    evaluation_dir: Path,
    tc_bps: float,
    low_quantile: float,
    high_quantile: float,
    gross_exposure: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    predictions_path = evaluation_dir / f"{label}_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    model_metrics = regression_metrics(predictions["target"], predictions["prediction"])
    model_metrics["daily_ic"] = daily_information_coefficient(predictions)
    write_json(evaluation_dir / f"{label}_metrics.json", model_metrics)

    for strategy in strategies:
        weighted = make_weights(
            predictions,
            strategy,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            gross_exposure=gross_exposure,
        )
        weighted.to_parquet(evaluation_dir / f"{label}_{strategy}_weights.parquet", index=False)
        daily, stats = strategy_returns(weighted, tc_bps=tc_bps)
        daily.to_csv(evaluation_dir / f"{label}_{strategy}_daily_returns.csv", index=False)
        write_json(evaluation_dir / f"{label}_{strategy}_stats.json", stats)
        plot_cumulative_return(daily, evaluation_dir / f"{label}_{strategy}_cum_return.png", title=f"{label} {strategy}")
        rows.append({"model": label, "strategy": strategy, **model_metrics, **stats})
        print(f"Evaluated {label}/{strategy}: Sharpe={stats['sharpe']:.4f}")
    return rows


def evaluate_equal_weight_baseline(
    predictions: pd.DataFrame,
    evaluation_dir: Path,
    tc_bps: float,
    gross_exposure: float,
) -> dict[str, float | str]:
    weighted = equal_weight_weights(predictions, gross_exposure=gross_exposure)
    weighted.to_parquet(evaluation_dir / "equal_weight_weights.parquet", index=False)
    daily, stats = strategy_returns(weighted, tc_bps=tc_bps)
    daily.to_csv(evaluation_dir / "equal_weight_daily_returns.csv", index=False)
    write_json(evaluation_dir / "equal_weight_stats.json", stats)
    plot_cumulative_return(daily, evaluation_dir / "equal_weight_cum_return.png", title="equal_weight")
    print(f"Evaluated equal_weight: Sharpe={stats['sharpe']:.4f}")
    return {"model": "equal_weight", "strategy": "equal_weight", **stats}


def run_evaluate_stage(args) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir("runs")
    run_config = read_json(run_dir / "run_config.json")
    data_dir = Path(args.data_dir) if args.data_dir else Path(run_config["data_dir"])
    frame = pd.read_parquet(data_dir / "processed_scaled.parquet")
    unscaled_path = data_dir / "processed_returns.parquet"
    unscaled_frame = pd.read_parquet(unscaled_path) if unscaled_path.exists() else None
    feature_columns = run_config.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    sequence_length = int(run_config["sequence_length"])

    test_dataset = EquitySequenceDataset(frame, "test", sequence_length, feature_columns, TARGET_COLUMN)
    if len(test_dataset) == 0:
        raise ValueError("Test dataset contains no sequences.")

    models = parse_csv_list(args.models) or list(run_config["models"])
    strategies = parse_csv_list(args.strategies)
    baselines = parse_csv_list(args.baselines)
    device = choose_device(args.device)
    evaluation_dir = ensure_dir(run_dir / "evaluation")

    summary_rows = []
    for model_name in models:
        model, metadata = load_model_from_checkpoint(model_name, run_dir, device)
        predictions = predict_dataset(model, test_dataset, args.batch_size, device)
        summary_rows.extend(
            evaluate_prediction_frame(
                predictions,
                model_name,
                strategies,
                evaluation_dir,
                args.tc_bps,
                args.low_quantile,
                args.high_quantile,
                args.gross_exposure,
            )
        )

    for baseline in baselines:
        predictions = baseline_predictions(baseline, test_dataset, unscaled_frame)
        if baseline == "equal_weight":
            summary_rows.append(evaluate_equal_weight_baseline(predictions, evaluation_dir, args.tc_bps, args.gross_exposure))
        else:
            summary_rows.extend(
                evaluate_prediction_frame(
                    predictions,
                    baseline,
                    strategies,
                    evaluation_dir,
                    args.tc_bps,
                    args.low_quantile,
                    args.high_quantile,
                    args.gross_exposure,
                )
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(evaluation_dir / "evaluation_summary.csv", index=False)
    print(f"Evaluation directory: {evaluation_dir}")
    return 0
