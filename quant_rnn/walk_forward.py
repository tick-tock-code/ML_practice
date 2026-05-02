from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import baseline_predictions
from .constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN
from .data import apply_standard_scaler, fit_standard_scaler
from .evaluation import evaluate_equal_weight_baseline, evaluate_prediction_frame, predict_dataset
from .io import ensure_dir, parse_csv_list, timestamp_slug, write_json
from .models import create_model
from .sequences import EquitySequenceDataset
from .training import choose_device, set_seed, train_model


def make_expanding_folds(
    frame: pd.DataFrame,
    initial_train_years: int,
    val_years: int,
    test_years: int,
    step_years: int,
) -> list[dict[str, object]]:
    df = frame.copy()
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.tz_localize(None)
    min_target = df["target_date"].min()
    max_target = df["target_date"].max()
    train_end = min_target + pd.DateOffset(years=initial_train_years)
    folds: list[dict[str, object]] = []
    fold_index = 1

    while True:
        val_end = train_end + pd.DateOffset(years=val_years)
        test_end = val_end + pd.DateOffset(years=test_years)
        if test_end > max_target:
            break
        fold = df.copy()
        fold["split"] = "unused"
        fold.loc[fold["target_date"] <= train_end, "split"] = "train"
        fold.loc[(fold["target_date"] > train_end) & (fold["target_date"] <= val_end), "split"] = "val"
        fold.loc[(fold["target_date"] > val_end) & (fold["target_date"] <= test_end), "split"] = "test"
        counts = fold["split"].value_counts().to_dict()
        if counts.get("train", 0) and counts.get("val", 0) and counts.get("test", 0):
            folds.append(
                {
                    "fold_index": fold_index,
                    "train_end": train_end.date().isoformat(),
                    "val_end": val_end.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "split_counts": counts,
                    "frame": fold,
                }
            )
            fold_index += 1
        train_end = train_end + pd.DateOffset(years=step_years)
    return folds


def run_walk_forward_stage(args) -> int:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    processed_path = data_dir / "processed_returns.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing unscaled processed data: {processed_path}")

    raw_frame = pd.read_parquet(processed_path)
    folds = make_expanding_folds(
        raw_frame,
        initial_train_years=args.initial_train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        step_years=args.step_years,
    )
    if not folds:
        raise ValueError("No walk-forward folds could be created from the available data.")

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / timestamp_slug()
    walk_dir = ensure_dir(run_dir / "walk_forward")
    models = parse_csv_list(args.models)
    baselines = parse_csv_list(args.baselines)
    strategies = parse_csv_list(args.strategies)
    device = choose_device(args.device)

    write_json(
        run_dir / "walk_forward_config.json",
        {
            "data_dir": str(data_dir),
            "models": models,
            "baselines": baselines,
            "strategies": strategies,
            "initial_train_years": args.initial_train_years,
            "val_years": args.val_years,
            "test_years": args.test_years,
            "step_years": args.step_years,
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "feature_columns": DEFAULT_FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "device": str(device),
            "seed": args.seed,
        },
    )

    all_summary_rows: list[dict[str, object]] = []
    fold_manifest = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        fold_dir = ensure_dir(walk_dir / f"fold_{fold_index:03d}")
        checkpoints_dir = ensure_dir(fold_dir / "checkpoints")
        metrics_dir = ensure_dir(fold_dir / "metrics")
        evaluation_dir = ensure_dir(fold_dir / "evaluation")
        fold_frame = fold["frame"].copy()

        scaler = fit_standard_scaler(fold_frame, DEFAULT_FEATURE_COLUMNS)
        scaled_fold = apply_standard_scaler(fold_frame, scaler)
        write_json(fold_dir / "scaler.json", scaler)
        fold_meta = {key: value for key, value in fold.items() if key != "frame"}
        write_json(fold_dir / "fold_metadata.json", fold_meta)
        fold_manifest.append(fold_meta)

        train_dataset = EquitySequenceDataset(scaled_fold, "train", args.sequence_length, DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)
        val_dataset = EquitySequenceDataset(scaled_fold, "val", args.sequence_length, DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)
        test_dataset = EquitySequenceDataset(scaled_fold, "test", args.sequence_length, DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping fold {fold_index}: at least one split has no valid sequences.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        for model_name in models:
            model = create_model(
                model_name,
                input_size=len(DEFAULT_FEATURE_COLUMNS),
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
            )
            pd.DataFrame(history).to_csv(metrics_dir / f"{model_name}_history.csv", index=False)
            best = min(history, key=lambda row: row["val_mse"])
            checkpoint_payload = {
                "model_name": model_name,
                "model_config": {
                    "input_size": len(DEFAULT_FEATURE_COLUMNS),
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "tcn_kernel_size": args.tcn_kernel_size,
                },
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "sequence_length": args.sequence_length,
                "target_column": TARGET_COLUMN,
                "state_dict_path": str(checkpoint_path.relative_to(fold_dir)),
                "best_epoch": int(best["epoch"]),
                "best_val_mse": float(best["val_mse"]),
                "stopped_epoch": int(history[-1]["epoch"]),
                "early_stopped": bool(history[-1].get("early_stop_triggered", False)),
            }
            write_json(checkpoints_dir / f"{model_name}_checkpoint.json", checkpoint_payload)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            predictions = predict_dataset(model, test_dataset, args.batch_size, device)
            rows = evaluate_prediction_frame(
                predictions,
                model_name,
                strategies,
                evaluation_dir,
                args.tc_bps,
                args.low_quantile,
                args.high_quantile,
                args.gross_exposure,
            )
            for row in rows:
                row["fold"] = fold_index
                all_summary_rows.append(row)

        for baseline in baselines:
            predictions = baseline_predictions(baseline, test_dataset, fold_frame)
            if baseline == "equal_weight":
                row = evaluate_equal_weight_baseline(predictions, evaluation_dir, args.tc_bps, args.gross_exposure)
                row["fold"] = fold_index
                all_summary_rows.append(row)
            else:
                rows = evaluate_prediction_frame(
                    predictions,
                    baseline,
                    strategies,
                    evaluation_dir,
                    args.tc_bps,
                    args.low_quantile,
                    args.high_quantile,
                    args.gross_exposure,
                )
                for row in rows:
                    row["fold"] = fold_index
                    all_summary_rows.append(row)

    write_json(walk_dir / "fold_manifest.json", {"folds": fold_manifest})
    pd.DataFrame(all_summary_rows).to_csv(walk_dir / "walk_forward_summary.csv", index=False)
    print(f"Walk-forward directory: {walk_dir}")
    return 0
