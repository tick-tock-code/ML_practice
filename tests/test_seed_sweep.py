import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.io import read_json
from quant_rnn.seed_sweep import run_seed_sweep_stage


class SeedSweepTests(unittest.TestCase):
    def test_seed_sweep_creates_child_runs_and_summary(self):
        Path("runs").mkdir(exist_ok=True)
        rows = []
        dates = pd.bdate_range("2020-01-01", periods=14)
        for ticker in ["AAA", "BBB"]:
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "target_date": date + pd.tseries.offsets.BDay(1),
                    "ticker": ticker,
                    "forward_return": float(i % 3) / 100.0,
                    "split": "train" if i < 8 else "val" if i < 11 else "test",
                }
                for feature in DEFAULT_FEATURE_COLUMNS:
                    row[feature] = float(i)
                rows.append(row)
        with tempfile.TemporaryDirectory(dir="runs") as tmp_run, tempfile.TemporaryDirectory(dir="runs") as tmp_data:
            data_dir = Path(tmp_data)
            frame = pd.DataFrame(rows)
            frame.to_parquet(data_dir / "processed_scaled.parquet", index=False)
            frame.to_parquet(data_dir / "processed_returns.parquet", index=False)
            args = SimpleNamespace(
                data_dir=str(data_dir),
                run_dir=tmp_run,
                models="rnn",
                seeds="1,2",
                strategies="rank_long_short",
                baselines="",
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
                tc_bps=5.0,
                low_quantile=0.2,
                high_quantile=0.8,
                gross_exposure=1.0,
                monotonicity=False,
                monotonicity_buckets=5,
                device="cpu",
            )
            run_seed_sweep_stage(args)

            parent = Path(tmp_run)
            self.assertTrue((parent / "seed_001" / "training_summary.json").exists())
            self.assertTrue((parent / "seed_002" / "training_summary.json").exists())
            summary = pd.read_csv(parent / "seed_sweep_summary.csv")
            self.assertEqual(set(summary["seed"]), {1, 2})
            self.assertEqual(set(summary["split"]), {"val", "test"})

            config_1 = read_json(parent / "seed_001" / "run_config.json")
            config_2 = read_json(parent / "seed_002" / "run_config.json")
            self.assertEqual(config_1["weight_decay"], 1e-4)
            self.assertEqual(config_2["weight_decay"], 1e-4)
            comparable_1 = {key: value for key, value in config_1.items() if key not in {"seed"}}
            comparable_2 = {key: value for key, value in config_2.items() if key not in {"seed"}}
            self.assertEqual(comparable_1, comparable_2)
            self.assertNotEqual(config_1["seed"], config_2["seed"])


if __name__ == "__main__":
    unittest.main()
