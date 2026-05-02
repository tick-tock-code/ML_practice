import tempfile
import unittest
from pathlib import Path

import torch

from quant_rnn.evaluation import load_model_from_checkpoint
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


if __name__ == "__main__":
    unittest.main()
