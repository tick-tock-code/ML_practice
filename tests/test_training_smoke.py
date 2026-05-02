import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.models import create_model
from quant_rnn.sequences import EquitySequenceDataset
from quant_rnn.training import train_model


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
            )
            self.assertLess(len(history), 10)
            self.assertTrue(history[-1]["early_stop_triggered"])


if __name__ == "__main__":
    unittest.main()
