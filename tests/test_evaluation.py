import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.evaluation import assign_monotonicity_buckets, load_model_from_checkpoint, monotonicity_analysis, run_evaluate_stage
from quant_rnn.io import write_json
from quant_rnn.models import create_model


class EvaluationTests(unittest.TestCase):
    def test_load_model_uses_checkpoint_path_relative_to_run_dir(self):
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp:
            run_dir = Path(tmp)
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            model = create_model("rnn", input_size=2, hidden_size=4)
            torch.save(model.state_dict(), checkpoints / "rnn_best.pt")
            write_json(
                checkpoints / "rnn_checkpoint.json",
                {
                    "model_name": "rnn",
                    "model_config": {
                        "input_size": 2,
                        "hidden_size": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "tcn_kernel_size": 3,
                    },
                    "state_dict_path": "checkpoints/rnn_best.pt",
                },
            )
            loaded, metadata = load_model_from_checkpoint("rnn", run_dir, torch.device("cpu"))
            self.assertEqual(metadata["state_dict_path"], "checkpoints/rnn_best.pt")
            x = torch.randn(1, 3, 2)
            self.assertEqual(tuple(loaded(x).shape), (1,))

    def test_evaluate_val_split_writes_validation_directory(self):
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp_run, tempfile.TemporaryDirectory(dir="runs") as tmp_data:
            run_dir = Path(tmp_run)
            data_dir = Path(tmp_data)
            rows = []
            dates = pd.bdate_range("2020-01-01", periods=10)
            for ticker in ["AAA", "BBB"]:
                for i, date in enumerate(dates):
                    row = {
                        "date": date,
                        "target_date": date + pd.tseries.offsets.BDay(1),
                        "ticker": ticker,
                        "forward_return": float(i) / 100.0,
                        "split": "val" if i >= 4 else "train",
                    }
                    for feature in DEFAULT_FEATURE_COLUMNS:
                        row[feature] = float(i)
                    rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_parquet(data_dir / "processed_scaled.parquet", index=False)
            frame.to_parquet(data_dir / "processed_returns.parquet", index=False)

            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            model = create_model("rnn", input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=4)
            torch.save(model.state_dict(), checkpoints / "rnn_best.pt")
            write_json(
                run_dir / "run_config.json",
                {
                    "data_dir": str(data_dir),
                    "sequence_length": 3,
                    "feature_columns": DEFAULT_FEATURE_COLUMNS,
                    "models": ["rnn"],
                },
            )
            write_json(
                checkpoints / "rnn_checkpoint.json",
                {
                    "model_name": "rnn",
                    "model_config": {
                        "input_size": len(DEFAULT_FEATURE_COLUMNS),
                        "hidden_size": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "tcn_kernel_size": 3,
                    },
                    "state_dict_path": "checkpoints/rnn_best.pt",
                },
            )
            args = SimpleNamespace(
                run_dir=str(run_dir),
                data_dir=str(data_dir),
                split="val",
                models="rnn",
                strategies="rank_long_short",
                baselines="",
                batch_size=8,
                tc_bps=5.0,
                low_quantile=0.2,
                high_quantile=0.8,
                gross_exposure=1.0,
                monotonicity=True,
                monotonicity_buckets=5,
                device="cpu",
            )
            run_evaluate_stage(args)
            self.assertTrue((run_dir / "validation" / "evaluation_summary.csv").exists())
            self.assertFalse((run_dir / "evaluation" / "evaluation_summary.csv").exists())
            self.assertTrue((run_dir / "validation" / "rnn_monotonicity.csv").exists())

    def test_monotonicity_buckets_are_assigned_within_target_date(self):
        predictions = pd.DataFrame(
            [
                {"target_date": "2020-01-02", "ticker": "A", "prediction": 0.1, "target": 0.01},
                {"target_date": "2020-01-02", "ticker": "B", "prediction": 0.2, "target": 0.02},
                {"target_date": "2020-01-03", "ticker": "A", "prediction": -0.2, "target": -0.02},
                {"target_date": "2020-01-03", "ticker": "B", "prediction": -0.1, "target": -0.01},
            ]
        )
        bucketed = assign_monotonicity_buckets(predictions, buckets=2)
        for _, group in bucketed.groupby("target_date"):
            self.assertEqual(set(group["bucket"]), {1, 2})

    def test_monotonicity_spread_uses_top_minus_bottom_target(self):
        predictions = pd.DataFrame(
            [
                {"target_date": "2020-01-02", "ticker": "A", "prediction": 0.1, "target": -0.01},
                {"target_date": "2020-01-02", "ticker": "B", "prediction": 0.2, "target": 0.03},
                {"target_date": "2020-01-03", "ticker": "A", "prediction": 0.1, "target": -0.02},
                {"target_date": "2020-01-03", "ticker": "B", "prediction": 0.2, "target": 0.04},
            ]
        )
        summary, spread = monotonicity_analysis(predictions, buckets=2)
        low = summary.loc[summary["bucket"] == 1, "avg_target"].iloc[0]
        high = summary.loc[summary["bucket"] == 2, "avg_target"].iloc[0]
        self.assertAlmostEqual(spread, high - low)
        self.assertAlmostEqual(spread, 0.05)

    def test_evaluate_no_monotonicity_skips_files(self):
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp_run, tempfile.TemporaryDirectory(dir="runs") as tmp_data:
            run_dir = Path(tmp_run)
            data_dir = Path(tmp_data)
            rows = []
            dates = pd.bdate_range("2020-01-01", periods=10)
            for ticker in ["AAA", "BBB"]:
                for i, date in enumerate(dates):
                    row = {
                        "date": date,
                        "target_date": date + pd.tseries.offsets.BDay(1),
                        "ticker": ticker,
                        "forward_return": float(i) / 100.0,
                        "split": "test" if i >= 4 else "train",
                    }
                    for feature in DEFAULT_FEATURE_COLUMNS:
                        row[feature] = float(i)
                    rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_parquet(data_dir / "processed_scaled.parquet", index=False)
            frame.to_parquet(data_dir / "processed_returns.parquet", index=False)
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            model = create_model("rnn", input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=4)
            torch.save(model.state_dict(), checkpoints / "rnn_best.pt")
            write_json(run_dir / "run_config.json", {"data_dir": str(data_dir), "sequence_length": 3, "feature_columns": DEFAULT_FEATURE_COLUMNS, "models": ["rnn"]})
            write_json(
                checkpoints / "rnn_checkpoint.json",
                {
                    "model_name": "rnn",
                    "model_config": {"input_size": len(DEFAULT_FEATURE_COLUMNS), "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "tcn_kernel_size": 3},
                    "state_dict_path": "checkpoints/rnn_best.pt",
                },
            )
            args = SimpleNamespace(
                run_dir=str(run_dir),
                data_dir=str(data_dir),
                split="test",
                models="rnn",
                strategies="rank_long_short",
                baselines="",
                batch_size=8,
                tc_bps=5.0,
                low_quantile=0.2,
                high_quantile=0.8,
                gross_exposure=1.0,
                monotonicity=False,
                monotonicity_buckets=5,
                device="cpu",
            )
            run_evaluate_stage(args)
            self.assertFalse((run_dir / "evaluation" / "rnn_monotonicity.csv").exists())


if __name__ == "__main__":
    unittest.main()
