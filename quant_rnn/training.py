from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .backtest import make_weights, strategy_returns
from .constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN
from .io import ensure_dir, parse_csv_list, read_json, timestamp_slug, write_json
from .metrics import daily_information_coefficient, regression_metrics
from .models import create_model
from .sequences import EquitySequenceDataset

SUPPORTED_SELECTION_METRICS = {
    "val_mse": "min",
    "val_daily_ic": "max",
    "val_rank_long_short_sharpe": "max",
    "val_zscore_sharpe": "max",
}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            output = model(x).detach().cpu().numpy()
            preds.extend(output.tolist())
            targets.extend(y.numpy().tolist())
    return regression_metrics(targets, preds)


def validation_prediction_frame(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    dataset = loader.dataset
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


def validation_selection_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    predictions = validation_prediction_frame(model, loader, device)
    metrics = regression_metrics(predictions["target"], predictions["prediction"])
    out = {f"val_{key}": value for key, value in metrics.items()}
    out["val_daily_ic"] = daily_information_coefficient(predictions)
    for strategy in ["rank_long_short", "zscore"]:
        weighted = make_weights(predictions, strategy)
        _, stats = strategy_returns(weighted, tc_bps=5.0)
        out[f"val_{strategy}_sharpe"] = stats["sharpe"]
    return out


def metric_improved(metric: str, value: float, best_value: float, min_delta: float) -> bool:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric '{metric}'. Expected one of {sorted(SUPPORTED_SELECTION_METRICS)}")
    if not np.isfinite(value):
        return False
    if not np.isfinite(best_value):
        return True
    if SUPPORTED_SELECTION_METRICS[metric] == "min":
        return value < best_value - min_delta
    return value > best_value + min_delta


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    selection_metric: str = "val_daily_ic",
    weight_decay: float = 0.0,
) -> list[dict[str, float]]:
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric '{selection_metric}'. Expected one of {sorted(SUPPORTED_SELECTION_METRICS)}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_selection_value = float("inf") if SUPPORTED_SELECTION_METRICS[selection_metric] == "min" else -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = validation_selection_metrics(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **val_metrics,
            "selection_metric": selection_metric,
            "best_selection_value": best_selection_value,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stop_triggered": False,
        }
        if metric_improved(selection_metric, float(row[selection_metric]), best_selection_value, early_stopping_min_delta):
            best_selection_value = float(row[selection_metric])
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        row["best_selection_value"] = best_selection_value
        row["epochs_without_improvement"] = epochs_without_improvement
        history.append(row)
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            row["early_stop_triggered"] = True
            break
    return history


def run_train_stage(args) -> int:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    scaled_path = data_dir / "processed_scaled.parquet"
    if not scaled_path.exists():
        raise FileNotFoundError(f"Missing scaled data: {scaled_path}")

    frame = pd.read_parquet(scaled_path)
    feature_columns = DEFAULT_FEATURE_COLUMNS
    train_dataset = EquitySequenceDataset(frame, "train", args.sequence_length, feature_columns, TARGET_COLUMN)
    val_dataset = EquitySequenceDataset(frame, "val", args.sequence_length, feature_columns, TARGET_COLUMN)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation datasets must both contain at least one sequence.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / timestamp_slug()
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir)

    models = parse_csv_list(args.models)
    if not models:
        raise ValueError("At least one model must be selected.")

    device = choose_device(args.device)
    config = {
        "data_dir": str(data_dir),
        "sequence_length": args.sequence_length,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "models": models,
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
        "device": str(device),
        "seed": args.seed,
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
    }
    write_json(run_dir / "run_config.json", config)
    for source_name in ["split_metadata.json", "scaler.json", "data_config.json"]:
        source = data_dir / source_name
        if source.exists():
            shutil.copy2(source, run_dir / source_name)

    summary = {}
    for model_name in models:
        model = create_model(
            model_name,
            input_size=len(feature_columns),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tcn_kernel_size=args.tcn_kernel_size,
        )
        checkpoint_path = checkpoints_dir / f"{model_name}_best.pt"
        history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_path=checkpoint_path,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            selection_metric=args.selection_metric,
            weight_decay=args.weight_decay,
        )
        pd.DataFrame(history).to_csv(metrics_dir / f"{model_name}_history.csv", index=False)
        if SUPPORTED_SELECTION_METRICS[args.selection_metric] == "min":
            best = min(history, key=lambda row: row[args.selection_metric])
        else:
            best = max(history, key=lambda row: row[args.selection_metric])
        stopped_epoch = int(history[-1]["epoch"])
        early_stopped = bool(history[-1].get("early_stop_triggered", False))
        try:
            state_dict_path = checkpoint_path.relative_to(run_dir)
        except ValueError:
            state_dict_path = checkpoint_path
        checkpoint_payload = {
            "model_name": model_name,
            "model_config": {
                "input_size": len(feature_columns),
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "tcn_kernel_size": args.tcn_kernel_size,
            },
            "feature_columns": feature_columns,
            "sequence_length": args.sequence_length,
            "target_column": TARGET_COLUMN,
            "state_dict_path": str(state_dict_path),
            "selection_metric": args.selection_metric,
            "best_selection_value": float(best[args.selection_metric]),
            "best_epoch": int(best["epoch"]),
            "best_val_mse": float(best["val_mse"]),
            "best_val_daily_ic": float(best["val_daily_ic"]),
            "best_val_rank_long_short_sharpe": float(best["val_rank_long_short_sharpe"]),
            "best_val_zscore_sharpe": float(best["val_zscore_sharpe"]),
            "stopped_epoch": stopped_epoch,
            "early_stopped": early_stopped,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "weight_decay": args.weight_decay,
        }
        write_json(checkpoints_dir / f"{model_name}_checkpoint.json", checkpoint_payload)
        summary[model_name] = checkpoint_payload
        print(f"Trained {model_name}: best val MSE={best['val_mse']:.8f} at epoch {int(best['epoch'])}")

    write_json(run_dir / "training_summary.json", summary)
    print(f"Run directory: {run_dir}")
    return 0
