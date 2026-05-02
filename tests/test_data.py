import math
import unittest

import numpy as np
import pandas as pd

from quant_rnn.constants import DEFAULT_FEATURE_COLUMNS
from quant_rnn.data import apply_standard_scaler, assign_splits, compute_return_features, fit_standard_scaler


class DataTests(unittest.TestCase):
    def test_compute_return_features_matches_hand_calculation(self):
        raw = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="B").tolist() * 2,
                "ticker": ["AAA"] * 4 + ["BBB"] * 4,
                "open": [10, 11, 12, 13, 20, 21, 22, 23],
                "high": [11, 12, 13, 14, 21, 22, 23, 24],
                "low": [9, 10, 11, 12, 19, 20, 21, 22],
                "close": [10, 12, 11, 13, 20, 19, 21, 22],
                "volume": [100, 110, 120, 130, 200, 210, 220, 230],
            }
        )
        processed = compute_return_features(raw, vol_window=2, target_horizon=1)
        row = processed[(processed["ticker"] == "AAA") & (processed["date"] == pd.Timestamp("2020-01-02"))].iloc[0]
        self.assertAlmostEqual(row["r_cc"], math.log(12 / 10))
        self.assertAlmostEqual(row["r_on"], math.log(11 / 10))
        self.assertAlmostEqual(row["r_oc"], math.log(12 / 11))
        self.assertAlmostEqual(row["forward_return"], 11 / 12 - 1)
        self.assertEqual(row["target_date"], pd.Timestamp("2020-01-03"))

    def test_splits_are_assigned_by_target_date_and_scaler_uses_train_only(self):
        dates = pd.bdate_range("2014-01-01", "2024-12-31")
        frame = pd.DataFrame(
            {
                "date": dates,
                "target_date": dates,
                "ticker": "AAA",
                "forward_return": 0.01,
                **{feature: np.arange(len(dates), dtype=float) for feature in DEFAULT_FEATURE_COLUMNS},
            }
        )
        split, metadata = assign_splits(frame, train_years=6, val_years=2, test_years=2)
        self.assertLessEqual(split.loc[split["split"] == "train", "target_date"].max(), split.loc[split["split"] == "val", "target_date"].min())
        self.assertLessEqual(split.loc[split["split"] == "val", "target_date"].max(), split.loc[split["split"] == "test", "target_date"].min())
        self.assertIn("split_counts", metadata)

        scaler = fit_standard_scaler(split, DEFAULT_FEATURE_COLUMNS)
        train_mean = split.loc[split["split"] == "train", "r_cc"].mean()
        self.assertAlmostEqual(scaler["mean"]["r_cc"], train_mean)
        scaled = apply_standard_scaler(split, scaler)
        self.assertAlmostEqual(scaled.loc[scaled["split"] == "train", "r_cc"].mean(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
