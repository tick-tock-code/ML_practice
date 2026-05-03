import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.io import read_json
from quant_rnn.models import create_model
from quant_rnn.sequences import EquitySequenceDataset
from quant_rnn.training import run_train_stage, train_model


class TrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_training_writes_checkpoint(self):
        rows = []
        dates = pd.bdate_range("2020-01-01", periods=20)
        for ticker in ["AAA", "BBB"]:
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "target_date": date + pd.tseries.offsets.BDay(1),
                    "ticker": ticker,
                    "forward_return": float(i % 3) / 100.0,
                    "split": "train" if i < 14 else "val",
                }
                for j, feature in enumerate(DEFAULT_FEATURE_COLUMNS):
                    row[feature] = float(i + j) / 10.0
                rows.append(row)
        frame = pd.DataFrame(rows)
        train_dataset = EquitySequenceDataset(frame, "train", sequence_length=4)
        val_dataset = EquitySequenceDataset(frame, "val", sequence_length=4)
        model = create_model("rnn", input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=4)
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp:
            checkpoint = Path(tmp) / "best.pt"
            history = train_model(
                model,
                DataLoader(train_dataset, batch_size=8, shuffle=True),
                DataLoader(val_dataset, batch_size=8, shuffle=False),
                torch.device("cpu"),
                epochs=1,
                learning_rate=1e-3,
                checkpoint_path=checkpoint,
                selection_metric="val_mse",
            )
            self.assertEqual(len(history), 1)
            self.assertTrue(checkpoint.exists())

    def test_early_stopping_stops_before_max_epochs(self):
        rows = []
        dates = pd.bdate_range("2020-01-01", periods=16)
        for ticker in ["AAA", "BBB"]:
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "target_date": date + pd.tseries.offsets.BDay(1),
                    "ticker": ticker,
                    "forward_return": 0.0,
                    "split": "train" if i < 10 else "val",
                }
                for feature in DEFAULT_FEATURE_COLUMNS:
                    row[feature] = 0.0
                rows.append(row)
        frame = pd.DataFrame(rows)
        train_dataset = EquitySequenceDataset(frame, "train", sequence_length=3)
        val_dataset = EquitySequenceDataset(frame, "val", sequence_length=3)
        model = create_model("rnn", input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=4)
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp:
            checkpoint = Path(tmp) / "best.pt"
            history = train_model(
                model,
                DataLoader(train_dataset, batch_size=8, shuffle=False),
                DataLoader(val_dataset, batch_size=8, shuffle=False),
                torch.device("cpu"),
                epochs=10,
                learning_rate=0.0,
                checkpoint_path=checkpoint,
                early_stopping_patience=1,
                early_stopping_min_delta=0.0,
                selection_metric="val_mse",
            )
            self.assertLess(len(history), 10)
            self.assertTrue(history[-1]["early_stop_triggered"])

    def test_selection_direction_for_mse_and_ic(self):
        from quant_rnn.training import metric_improved

        self.assertTrue(metric_improved("val_mse", 0.1, 0.2, 0.0))
        self.assertFalse(metric_improved("val_mse", 0.3, 0.2, 0.0))
        self.assertTrue(metric_improved("val_daily_ic", 0.2, 0.1, 0.0))
        self.assertFalse(metric_improved("val_daily_ic", 0.0, 0.1, 0.0))

    def test_train_model_passes_weight_decay_to_adam(self):
        rows = []
        dates = pd.bdate_range("2020-01-01", periods=12)
        for ticker in ["AAA", "BBB"]:
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "target_date": date + pd.tseries.offsets.BDay(1),
                    "ticker": ticker,
                    "forward_return": float(i) / 100.0,
                    "split": "train" if i < 8 else "val",
                }
                for feature in DEFAULT_FEATURE_COLUMNS:
                    row[feature] = float(i)
                rows.append(row)
        frame = pd.DataFrame(rows)
        train_dataset = EquitySequenceDataset(frame, "train", sequence_length=3)
        val_dataset = EquitySequenceDataset(frame, "val", sequence_length=3)
        model = create_model("rnn", input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=4)
        captured = {}
        original_adam = torch.optim.Adam

        def adam_factory(params, lr, weight_decay):
            captured["weight_decay"] = weight_decay
            return original_adam(params, lr=lr, weight_decay=weight_decay)

        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp, patch("quant_rnn.training.torch.optim.Adam", side_effect=adam_factory):
            train_model(
                model,
                DataLoader(train_dataset, batch_size=8, shuffle=True),
                DataLoader(val_dataset, batch_size=8, shuffle=False),
                torch.device("cpu"),
                epochs=1,
                learning_rate=1e-3,
                checkpoint_path=Path(tmp) / "best.pt",
                selection_metric="val_mse",
                weight_decay=1e-4,
            )
        self.assertEqual(captured["weight_decay"], 1e-4)

    def test_run_train_stage_records_weight_decay(self):
        Path("runs").mkdir(exist_ok=True)
        rows = []
        dates = pd.bdate_range("2020-01-01", periods=12)
        for ticker in ["AAA", "BBB"]:
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "target_date": date + pd.tseries.offsets.BDay(1),
                    "ticker": ticker,
                    "forward_return": float(i) / 100.0,
                    "split": "train" if i < 8 else "val",
                }
                for feature in DEFAULT_FEATURE_COLUMNS:
                    row[feature] = float(i)
                rows.append(row)
        with tempfile.TemporaryDirectory(dir="runs") as tmp_run, tempfile.TemporaryDirectory(dir="runs") as tmp_data:
            data_dir = Path(tmp_data)
            pd.DataFrame(rows).to_parquet(data_dir / "processed_scaled.parquet", index=False)
            args = SimpleNamespace(
                data_dir=str(data_dir),
                run_dir=tmp_run,
                models="rnn",
                sequence_length=3,
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                weight_decay=1e-4,
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
                tcn_kernel_size=3,
                selection_metric="val_mse",
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                device="cpu",
                seed=42,
            )
            run_train_stage(args)
            self.assertEqual(read_json(Path(tmp_run) / "run_config.json")["weight_decay"], 1e-4)
            self.assertEqual(read_json(Path(tmp_run) / "checkpoints" / "rnn_checkpoint.json")["weight_decay"], 1e-4)


if __name__ == "__main__":
    unittest.main()
