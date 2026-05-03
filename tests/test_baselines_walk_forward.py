import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_rnn.backtest import make_weights
from quant_rnn.baselines import (
    equal_weight_weights,
    momentum_12_1_predictions,
    predictions_from_constant,
)
from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.data import apply_standard_scaler, fit_standard_scaler
from quant_rnn.sequences import EquitySequenceDataset
from quant_rnn.walk_forward import make_expanding_folds, run_walk_forward_stage


def synthetic_processed(start="2014-01-01", periods=2600, tickers=("AAA", "BBB", "CCC")):
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for ticker_id, ticker in enumerate(tickers):
        for i, date in enumerate(dates):
            row = {
                "date": date,
                "target_date": date + pd.tseries.offsets.BDay(1),
                "ticker": ticker,
                "forward_return": 0.001 * ((i + ticker_id) % 5 - 2),
            }
            for j, feature in enumerate(DEFAULT_FEATURE_COLUMNS):
                row[feature] = float(i + j + ticker_id)
            row["r_cc"] = 0.001 * (i + 1 + ticker_id)
            rows.append(row)
    return pd.DataFrame(rows)


class BaselineAndWalkForwardTests(unittest.TestCase):
    def test_expanding_folds_are_chronological(self):
        frame = synthetic_processed()
        folds = make_expanding_folds(frame, initial_train_years=6, val_years=1, test_years=1, step_years=1)
        self.assertGreaterEqual(len(folds), 2)
        first = folds[0]
        self.assertLess(first["train_end"], first["val_end"])
        self.assertLess(first["val_end"], first["test_end"])

    def test_fold_scaler_uses_fold_train_rows_only(self):
        frame = synthetic_processed(periods=800, tickers=("AAA",))
        folds = make_expanding_folds(frame, initial_train_years=1, val_years=1, test_years=1, step_years=1)
        fold_frame = folds[0]["frame"]
        scaler = fit_standard_scaler(fold_frame, DEFAULT_FEATURE_COLUMNS)
        train_mean = fold_frame.loc[fold_frame["split"] == "train", "r_cc"].mean()
        self.assertAlmostEqual(scaler["mean"]["r_cc"], train_mean)
        scaled = apply_standard_scaler(fold_frame, scaler)
        self.assertAlmostEqual(scaled.loc[scaled["split"] == "train", "r_cc"].mean(), 0.0, places=6)

    def test_cash_zero_has_no_rank_or_zscore_exposure(self):
        frame = synthetic_processed(periods=8)
        frame["split"] = "test"
        dataset = EquitySequenceDataset(frame, "test", sequence_length=3)
        predictions = predictions_from_constant(dataset, 0.0)
        for strategy in ["rank_long_short", "zscore"]:
            weighted = make_weights(predictions, strategy)
            self.assertAlmostEqual(weighted["weight"].abs().sum(), 0.0)

    def test_equal_weight_is_long_only_and_sums_to_one(self):
        frame = synthetic_processed(periods=8)
        frame["split"] = "test"
        dataset = EquitySequenceDataset(frame, "test", sequence_length=3)
        predictions = predictions_from_constant(dataset, 0.0)
        weighted = equal_weight_weights(predictions)
        self.assertTrue((weighted["weight"] >= 0).all())
        for _, group in weighted.groupby("date"):
            self.assertAlmostEqual(group["weight"].sum(), 1.0)

    def test_momentum_12_1_uses_only_skipped_history(self):
        frame = synthetic_processed(periods=10, tickers=("AAA",))
        frame["split"] = "test"
        dataset = EquitySequenceDataset(frame, "test", sequence_length=3)
        predictions = momentum_12_1_predictions(dataset, frame, lookback_days=3, skip_days=1, min_periods=3)
        row = predictions[predictions["date"] == pd.Timestamp(frame.iloc[4]["date"]).date().isoformat()].iloc[0]
        expected = frame.loc[1:3, "r_cc"].sum()
        self.assertAlmostEqual(row["prediction"], expected)

    def test_walk_forward_smoke_runs_one_fold(self):
        Path("runs").mkdir(exist_ok=True)
        frame = synthetic_processed(start="2020-01-01", periods=900, tickers=("AAA", "BBB"))
        with tempfile.TemporaryDirectory(dir="runs") as tmp_run, tempfile.TemporaryDirectory(dir="runs") as tmp_data:
            data_dir = Path(tmp_data)
            frame.to_parquet(data_dir / "processed_returns.parquet", index=False)
            args = SimpleNamespace(
                data_dir=str(data_dir),
                run_dir=tmp_run,
                models="rnn",
                baselines="cash_zero",
                strategies="rank_long_short",
                initial_train_years=1,
                val_years=1,
                test_years=1,
                step_years=1,
                sequence_length=5,
                epochs=1,
                batch_size=64,
                learning_rate=1e-3,
                weight_decay=1e-4,
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
                tcn_kernel_size=3,
                selection_metric="val_mse",
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                tc_bps=5.0,
                low_quantile=0.2,
                high_quantile=0.8,
                gross_exposure=1.0,
                device="cpu",
                seed=42,
            )
            run_walk_forward_stage(args)
            summary = Path(tmp_run) / "walk_forward" / "walk_forward_summary.csv"
            self.assertTrue(summary.exists())
            self.assertFalse(pd.read_csv(summary).empty)
            fold_meta = pd.read_json(Path(tmp_run) / "walk_forward" / "fold_001" / "fold_metadata.json", typ="series")
            self.assertEqual(float(fold_meta["weight_decay"]), 1e-4)


if __name__ == "__main__":
    unittest.main()
