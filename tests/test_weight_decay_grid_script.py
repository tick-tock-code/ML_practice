import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_rnn.io import write_json


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_weight_decay_grid.py"
SPEC = importlib.util.spec_from_file_location("run_weight_decay_grid", SCRIPT_PATH)
grid_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid_script
SPEC.loader.exec_module(grid_script)


class WeightDecayGridScriptTests(unittest.TestCase):
    def test_grid_spec_expands_to_twelve_configs(self):
        config = grid_script.GridConfig()
        runs = grid_script.expand_grid(config)
        self.assertEqual(len(runs), 12)
        self.assertEqual({run.hidden_size for run in runs}, {64, 128})
        self.assertEqual({run.num_layers for run in runs}, {1, 2})
        self.assertEqual({run.weight_decay for run in runs}, {0.0, 1e-5, 1e-4})

    def test_run_name_includes_key_hyperparameters(self):
        config = grid_script.GridConfig()
        name = grid_script.run_name(config, grid_script.GridRun(hidden_size=64, num_layers=2, weight_decay=1e-4))
        self.assertIn("lr1e5", name)
        self.assertIn("bs128", name)
        self.assertIn("hs064", name)
        self.assertIn("l2", name)
        self.assertIn("wd1e-4", name)

    def test_resume_detection_uses_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertFalse(grid_script.validation_summary_exists(run_dir))
            validation = run_dir / "validation"
            validation.mkdir()
            (validation / "evaluation_summary.csv").write_text("model,strategy\n", encoding="utf-8")
            self.assertTrue(grid_script.validation_summary_exists(run_dir))

    def test_dry_run_logs_command_without_child_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            command = ["python", "-m", "quant_rnn", "train", "--run-dir", str(output_dir / "child")]
            grid_script.run_command(command, dry_run=True, output_dir=output_dir)
            self.assertTrue((output_dir / "commands.log").exists())
            self.assertFalse((output_dir / "child").exists())

    def test_aggregate_validation_summary_joins_training_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = grid_script.GridConfig(hidden_sizes=(64,), num_layers=(1,), weight_decays=(1e-4,))
            child = output_dir / grid_script.run_name(config, grid_script.GridRun(hidden_size=64, num_layers=1, weight_decay=1e-4))
            validation = child / "validation"
            validation.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "model": "lstm",
                        "strategy": "zscore",
                        "mse": 0.1,
                        "daily_ic": 0.2,
                        "monotonic_spread": 0.03,
                        "sharpe": 0.4,
                    }
                ]
            ).to_csv(validation / "evaluation_summary.csv", index=False)
            write_json(
                child / "training_summary.json",
                {
                    "lstm": {
                        "best_epoch": 3,
                        "stopped_epoch": 8,
                        "early_stopped": True,
                        "selection_metric": "val_daily_ic",
                        "best_selection_value": 0.2,
                        "best_val_mse": 0.1,
                        "best_val_daily_ic": 0.2,
                        "best_val_rank_long_short_sharpe": 0.1,
                        "best_val_zscore_sharpe": 0.4,
                    }
                },
            )
            summary = grid_script.aggregate_validation_summary(output_dir, config)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.iloc[0]["best_epoch"], 3)
            self.assertEqual(summary.iloc[0]["weight_decay"], 1e-4)
            self.assertTrue((output_dir / "grid_validation_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
