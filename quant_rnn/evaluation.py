from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .backtest import make_weights, strategy_returns
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


def run_evaluate_stage(args) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir("runs")
    run_config = read_json(run_dir / "run_config.json")
    data_dir = Path(args.data_dir) if args.data_dir else Path(run_config["data_dir"])
    frame = pd.read_parquet(data_dir / "processed_scaled.parquet")
    feature_columns = run_config.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    sequence_length = int(run_config["sequence_length"])

    test_dataset = EquitySequenceDataset(frame, "test", sequence_length, feature_columns, TARGET_COLUMN)
    if len(test_dataset) == 0:
        raise ValueError("Test dataset contains no sequences.")

    models = parse_csv_list(args.models) or list(run_config["models"])
    strategies = parse_csv_list(args.strategies)
    device = choose_device(args.device)
    evaluation_dir = ensure_dir(run_dir / "evaluation")

    summary_rows = []
    for model_name in models:
        model, metadata = load_model_from_checkpoint(model_name, run_dir, device)
        predictions = predict_dataset(model, test_dataset, args.batch_size, device)
        predictions_path = evaluation_dir / f"{model_name}_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)

        model_metrics = regression_metrics(predictions["target"], predictions["prediction"])
        model_metrics["daily_ic"] = daily_information_coefficient(predictions)
        write_json(evaluation_dir / f"{model_name}_metrics.json", model_metrics)

        for strategy in strategies:
            weighted = make_weights(
                predictions,
                strategy,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
                gross_exposure=args.gross_exposure,
            )
            weighted.to_parquet(evaluation_dir / f"{model_name}_{strategy}_weights.parquet", index=False)
            daily, stats = strategy_returns(weighted, tc_bps=args.tc_bps)
            daily.to_csv(evaluation_dir / f"{model_name}_{strategy}_daily_returns.csv", index=False)
            write_json(evaluation_dir / f"{model_name}_{strategy}_stats.json", stats)
            plot_cumulative_return(
                daily,
                evaluation_dir / f"{model_name}_{strategy}_cum_return.png",
                title=f"{model_name} {strategy}",
            )
            row = {"model": model_name, "strategy": strategy, **model_metrics, **stats}
            summary_rows.append(row)
            print(f"Evaluated {model_name}/{strategy}: Sharpe={stats['sharpe']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(evaluation_dir / "evaluation_summary.csv", index=False)
    print(f"Evaluation directory: {evaluation_dir}")
    return 0
