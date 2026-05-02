import unittest

import numpy as np
import pandas as pd
import torch

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.models import create_model
from quant_rnn.sequences import EquitySequenceDataset


def synthetic_frame(n_days=12, tickers=("AAA", "BBB")):
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    for ticker in tickers:
        for i, date in enumerate(dates):
            row = {
                "date": date,
                "target_date": date + pd.tseries.offsets.BDay(1),
                "ticker": ticker,
                "forward_return": float(i) / 100.0,
                "split": "train" if i < 8 else "val" if i < 10 else "test",
            }
            for j, feature in enumerate(DEFAULT_FEATURE_COLUMNS):
                row[feature] = float(i + j)
            rows.append(row)
    return pd.DataFrame(rows)


class SequenceAndModelTests(unittest.TestCase):
    def test_sequence_dataset_uses_trailing_context(self):
        frame = synthetic_frame(n_days=6, tickers=("AAA",))
        frame["split"] = ["train", "train", "train", "val", "test", "test"]
        dataset = EquitySequenceDataset(frame, "test", sequence_length=3)
        self.assertEqual(len(dataset), 2)
        x, y, index = dataset[0]
        self.assertEqual(tuple(x.shape), (3, len(DEFAULT_FEATURE_COLUMNS)))
        np.testing.assert_allclose(x[:, 0].numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(y), 0.04)
        metadata = dataset.sample_metadata(int(index))
        self.assertEqual(metadata.ticker, "AAA")

    def test_model_registry_outputs_scalar_per_sample(self):
        x = torch.randn(2, 5, len(DEFAULT_FEATURE_COLUMNS))
        for name in ["rnn", "lstm", "tcn"]:
            model = create_model(name, input_size=len(DEFAULT_FEATURE_COLUMNS), hidden_size=8, num_layers=1)
            y = model(x)
            self.assertEqual(tuple(y.shape), (2,))


if __name__ == "__main__":
    unittest.main()
