import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from quant_rnn.aggregation import run_aggregate_grid_stage
from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.io import write_json
from quant_rnn.models import create_model


class AggregationTests(unittest.TestCase):
    def test_aggregate_grid_writes_validation_and_comparison_outputs(self):
        Path("runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runs") as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            runs_dir = root / "runs"
            data_dir.mkdir()
            runs_dir.mkdir()
            rows = []
            dates = pd.bdate_range("2020-01-01", periods=12)
            for ticker in ["AAA", "BBB"]:
                for i, date in enumerate(dates):
                    row = {
                        "date": date,
                        "target_date": date + pd.tseries.offsets.BDay(1),
                        "ticker": ticker,
                        "forward_return": float(i % 4) / 100.0,
                        "split": "val" if i >= 5 else "train",
                    }
                    for feature in DEFAULT_FEATURE_COLUMNS:
                        row[feature] = float(i)
                    rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_parquet(data_dir / "processed_scaled.parquet", index=False)

            run_dir = runs_dir / "grid_lr1e4_bs064_hs064_l1"
            checkpoints = run_dir / "checkpoints"
            evaluation = run_dir / "evaluation"
            checkpoints.mkdir(parents=True)
            evaluation.mkdir()
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
                run_dir / "training_summary.json",
                {
                    "rnn": {
                        "best_epoch": 1,
                        "stopped_epoch": 1,
                        "early_stopped": False,
                        "best_val_mse": 0.1,
                    }
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
                    "selection_metric": "val_daily_ic",
                    "best_selection_value": 0.01,
                    "best_epoch": 1,
                    "stopped_epoch": 1,
                    "early_stopped": False,
                    "best_val_mse": 0.1,
                },
            )
            pd.DataFrame(
                [
                    {
                        "model": "rnn",
                        "strategy": "zscore",
                        "mse": 0.1,
                        "daily_ic": 0.01,
                        "sharpe": 0.2,
                        "ann_return": 0.03,
                        "max_drawdown": 0.04,
                    }
                ]
            ).to_csv(evaluation / "evaluation_summary.csv", index=False)

            validation_output = root / "validation.csv"
            comparison_output = root / "comparison.csv"
            args = SimpleNamespace(
                runs_dir=str(runs_dir),
                data_dir=str(data_dir),
                run_pattern="grid_lr1e4_*",
                max_hidden_size=128,
                models="rnn",
                strategies="zscore",
                batch_size=8,
                tc_bps=5.0,
                validation_output=str(validation_output),
                comparison_output=str(comparison_output),
                device="cpu",
            )
            run_aggregate_grid_stage(args)
            self.assertTrue(validation_output.exists())
            self.assertTrue(comparison_output.exists())
            self.assertFalse(pd.read_csv(validation_output).empty)


if __name__ == "__main__":
    unittest.main()
