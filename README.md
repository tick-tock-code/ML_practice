# Quant RNN Research Pipeline

Three-stage research pipeline for training sequence models on daily equity return features and evaluating model-driven long-short trading rules.

This branch (`2026_05_RNN_exp`) starts from the quant backtester ideas in the older branch and reorganizes them into a leakage-aware machine-learning workflow:

1. Create stock data and return features.
2. Train selected sequence models.
3. Evaluate selected models on the untouched test set with backtest-style trading logic.

## Runtime

Use the local PyTorch environment:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn --help
```

The project uses Parquet files via `pyarrow`.

## Quick Start

Create data:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn data --years 10 --top-n 100 --train-years 6 --val-years 2 --test-years 2
```

Train models:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn,lstm,tcn --sequence-length 60 --epochs 50 --early-stopping-patience 5
```

Evaluate on the test set:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --models rnn,lstm,tcn --strategies rank_long_short,zscore --baselines cash_zero,equal_weight,momentum_12_1
```

Run expanding-window walk-forward research:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn walk-forward --models rnn,lstm,tcn --initial-train-years 6 --val-years 1 --test-years 1 --step-years 1
```

## Defaults

- Universe: top-N current S&P 500 symbols from Wikipedia.
- Data window: 10 years.
- Split: 6 years train, 2 years validation, 2 years test.
- Target: next trading day simple return.
- Features: log return features from OHLCV plus volume surprise.
- Models: RNN, LSTM, TCN.
- Test trading rules: rank long-short and z-score weights.
- Transaction cost: 5 bps per unit turnover.
- Baselines: zero prediction, equal-weight long-only, and fixed 12-1 momentum.
- Early stopping: validation MSE patience of 5 epochs by default.

## Important Leakage Rules

- Splits are assigned by `target_date`, not by feature date.
- Validation is used for checkpoint selection.
- Test rows are used only in the evaluation stage.
- Feature scaling is fit on train rows only, then applied to train/val/test.
- Validation/test sequences may use earlier historical features as context, but their labels stay in their assigned split.
- Cross-sectional ranking/z-scoring is done within each signal date only.
- Walk-forward mode fits a fresh scaler on each fold's train rows only.

## Repository Map

```text
quant_rnn/
|-- cli.py           # argparse CLI with data/train/evaluate subcommands
|-- data.py          # S&P loader, Yahoo download, return features, splits, scaling
|-- sequences.py     # pooled cross-ticker sequence dataset
|-- models.py        # RNN, LSTM, and TCN model registry
|-- training.py      # PyTorch training loop and checkpoint writing
|-- evaluation.py    # test-set predictions and evaluation orchestration
|-- backtest.py      # rank and z-score strategy logic
|-- baselines.py     # zero, equal-weight, and momentum baseline providers
|-- walk_forward.py  # expanding-window retraining/evaluation
|-- metrics.py       # regression/directional metrics
`-- io.py            # JSON and directory helpers

tests/               # unittest-based tests; no pytest dependency required
```

## Survivorship Bias

The default universe uses the current S&P 500 list, not historical membership. That is acceptable for v1 research infrastructure but introduces survivorship bias. Historical constituent membership should be added before treating results as production-grade research.

## Verification

Run the test suite:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m unittest discover -s tests
```

## Momentum References

- Jegadeesh and Titman, 1993, "Returns to Buying Winners and Selling Losers": [EconPapers](https://econpapers.repec.org/RePEc%3Abla%3Ajfinan%3Av%3A48%3Ay%3A1993%3Ai%3A1%3Ap%3A65-91)
- Moskowitz, Ooi, and Pedersen, 2012, "Time Series Momentum": [EconPapers](https://econpapers.repec.org/RePEc%3Aeee%3Ajfinec%3Av%3A104%3Ay%3A2012%3Ai%3A2%3Ap%3A228-250)
- Asness, Moskowitz, and Pedersen, 2013, "Value and Momentum Everywhere": [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1363476)
- Hurst, Ooi, and Pedersen, 2017, "A Century of Evidence on Trend-Following Investing": [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026)
