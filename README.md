# Quant RNN Research Pipeline

Three-stage research pipeline for training sequence models on daily equity return features and evaluating model-driven long-short trading rules.

This branch (`2026_05_RNN_exp`) starts from the quant backtester ideas in the older branch and reorganizes them into a leakage-aware machine-learning workflow:

1. Create stock data and return features.
2. Train selected sequence models using validation-based checkpoint selection.
3. Evaluate selected models on validation or the untouched test set with backtest-style trading logic.

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
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn,lstm,tcn --sequence-length 60 --epochs 50 --selection-metric val_daily_ic --early-stopping-patience 5 --weight-decay 1e-4
```

Evaluate on the test set:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --models rnn,lstm,tcn --strategies rank_long_short,zscore --baselines cash_zero,equal_weight,momentum_12_1
```

Evaluation writes prediction monotonicity diagnostics by default. To disable them:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --no-monotonicity
```

Evaluate the validation split:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --split val --run-dir runs\YYYYMMDD_HHMMSS --models rnn,lstm,tcn --strategies rank_long_short,zscore
```

Aggregate stored grid checkpoints:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn aggregate-grid --run-pattern grid_lr1e4_* --max-hidden-size 128
```

Repeat a promising config across seeds:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn seed-sweep --models lstm --seeds 1,2,3,4,5 --sequence-length 60 --epochs 30 --batch-size 128 --learning-rate 1e-4 --hidden-size 64 --num-layers 2 --dropout 0.2 --weight-decay 1e-4 --selection-metric val_daily_ic
```

Run the LSTM weight-decay grid at `learning_rate=1e-5`:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe scripts\run_weight_decay_grid.py --dry-run
C:\ProgramData\anaconda3\envs\torch_env2\python.exe scripts\run_weight_decay_grid.py --resume
```

The grid evaluates validation only by default. After choosing one or two configs from `runs\grid_lr1e5_wd\grid_validation_summary.csv`, run test evaluation explicitly on the selected run.

Run expanding-window walk-forward research:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn walk-forward --models rnn,lstm,tcn --initial-train-years 6 --val-years 1 --test-years 1 --step-years 1 --weight-decay 1e-4
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
- Early stopping: patience of 5 epochs on the configured validation selection metric.
- Checkpoint selection: validation daily IC by default.
- Regularization: optional Adam weight decay; `1e-4` is the current recommended next experiment.
- Diagnostics: evaluation writes quintile monotonicity tables/plots by default.

## Important Leakage Rules

- Splits are assigned by `target_date`, not by feature date.
- Validation is used for checkpoint and hyperparameter selection.
- Test is not used for selection.
- Test rows are used only in the evaluation stage.
- Feature scaling is fit on train rows only, then applied to train/val/test.
- Validation/test sequences may use earlier historical features as context, but their labels stay in their assigned split.
- Cross-sectional ranking/z-scoring is done within each signal date only.
- Walk-forward mode fits a fresh scaler on each fold's train rows only.

## Repository Map

```text
quant_rnn/
|-- cli.py           # argparse CLI with data/train/evaluate/walk-forward/aggregate-grid/seed-sweep subcommands
|-- data.py          # S&P loader, Yahoo download, return features, splits, scaling
|-- sequences.py     # pooled cross-ticker sequence dataset
|-- models.py        # RNN, LSTM, and TCN model registry
|-- training.py      # PyTorch training loop, validation metrics, checkpoint writing
|-- evaluation.py    # validation/test predictions and evaluation orchestration
|-- backtest.py      # rank and z-score strategy logic
|-- baselines.py     # zero, equal-weight, and momentum baseline providers
|-- walk_forward.py  # expanding-window retraining/evaluation
|-- aggregation.py   # stored-grid validation summaries and selection comparisons
|-- seed_sweep.py    # repeated fixed-config training across random seeds
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
