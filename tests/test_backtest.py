import unittest

import numpy as np
import pandas as pd

from quant_rnn.backtest import make_weights, strategy_returns


class BacktestTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for day in pd.bdate_range("2020-01-01", periods=3):
            for i, ticker in enumerate(["A", "B", "C", "D"]):
                rows.append(
                    {
                        "date": day.date().isoformat(),
                        "target_date": (day + pd.tseries.offsets.BDay(1)).date().isoformat(),
                        "ticker": ticker,
                        "prediction": float(i),
                        "target": 0.01 * (i - 1),
                    }
                )
        self.predictions = pd.DataFrame(rows)

    def test_rank_long_short_weights_are_neutral_and_normalized(self):
        weighted = make_weights(self.predictions, "rank_long_short", low_quantile=0.25, high_quantile=0.75)
        for _, group in weighted.groupby("date"):
            self.assertAlmostEqual(group["weight"].sum(), 0.0)
            self.assertAlmostEqual(group["weight"].abs().sum(), 1.0)

    def test_zscore_weights_are_neutral_and_normalized(self):
        weighted = make_weights(self.predictions, "zscore")
        for _, group in weighted.groupby("date"):
            self.assertAlmostEqual(group["weight"].sum(), 0.0, places=7)
            self.assertAlmostEqual(group["weight"].abs().sum(), 1.0)

    def test_strategy_returns_uses_target_returns_after_signal_date(self):
        weighted = make_weights(self.predictions, "rank_long_short", low_quantile=0.25, high_quantile=0.75)
        daily, stats = strategy_returns(weighted, tc_bps=5)
        self.assertEqual(len(daily), 3)
        self.assertTrue(np.isfinite(daily["cum_index"]).all())
        self.assertIn("sharpe", stats)


if __name__ == "__main__":
    unittest.main()
