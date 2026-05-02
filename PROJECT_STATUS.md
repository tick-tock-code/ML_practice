# Quant RNN Pipeline Status

Living project note for the `2026_05_RNN_exp` branch.

## Current State

The three-stage quant ML pipeline is implemented in `quant_rnn/` but has not yet been committed.

Implemented CLI stages:

```text
data      Download adjusted OHLCV data, compute return features, create train/val/test splits.
train     Train selected pooled PyTorch sequence models.
evaluate  Evaluate trained models on the untouched test set using trading-style rules.
```

Implemented models:

```text
rnn   Plain recurrent neural network baseline.
lstm  Gated recurrent model for longer sequence memory.
tcn   Temporal convolutional network using causal/dilated convolutions.
```

Implemented safeguards:

- Splits are assigned by `target_date`, not feature date.
- Default split is 6 years train, 2 years validation, 2 years test inside a 10-year window.
- Scalers are fit only on train rows.
- Validation is used for checkpoint selection.
- Test is reserved for final evaluation.
- Test evaluation supports both rank long-short and z-score trading rules.
- Transaction costs are included using turnover and default `tc_bps = 5`.

Verification completed:

```text
Ran 9 tests
OK
```

Not yet completed:

- No live Yahoo Finance data generation run yet.
- No full 10-year training run yet.
- No real test-set model comparison yet.
- No committed baseline results or research write-up yet.

## Repository Status

Current expected git state after implementation:

```text
 M README.md
?? .gitignore
?? AGENTS.md
?? ARCHITECTURE.md
?? PROJECT_STATUS.md
?? quant_rnn/
?? requirements.txt
?? tests/
```

Generated artifacts are ignored:

```text
data/
runs/
*.parquet
*.pt
```

## Key Commands

Use the local PyTorch environment:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe
```

Show CLI help:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn --help
```

Run tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m unittest discover -s tests
```

Create default data artifacts:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn data --years 10 --top-n 100 --train-years 6 --val-years 2 --test-years 2
```

Create a smaller smoke-test dataset:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn data --years 3 --top-n 10 --train-years 1 --val-years 1 --test-years 1
```

Train all default models:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn,lstm,tcn --sequence-length 60 --epochs 50
```

Train a quick smoke model:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn --sequence-length 20 --epochs 2 --batch-size 128
```

Evaluate latest run:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --models rnn,lstm,tcn --strategies rank_long_short,zscore
```

Evaluate a specific run:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --run-dir runs\YYYYMMDD_HHMMSS --models lstm,tcn --strategies rank_long_short,zscore
```

## Model Summaries

### RNN

Plain recurrent neural network baseline. It reads a sequence of daily return features and predicts next-day return from the final hidden state.

Use it as a sanity-check benchmark. If complex models cannot beat it out of sample, the added complexity is not justified.

### LSTM

Recurrent model with gating mechanisms that help preserve or discard information over longer sequences.

This is the main sequence-model benchmark. It is usually more stable than a plain RNN when `sequence_length` is large.

### TCN

Temporal convolutional network using causal/dilated 1D convolutions.

This is a strong non-recurrent sequence baseline. It can train faster than recurrent models and can capture multi-horizon temporal patterns.

## Hyperparameters To Tune

Core modeling:

- `sequence_length`: try `20`, `60`, `120`, `252`.
- `learning_rate`: try `1e-4`, `3e-4`, `1e-3`.
- `batch_size`: try `128`, `256`, `512`.
- `hidden_size`: try `32`, `64`, `128`.
- `num_layers`: try `1`, `2`, `3`.
- `dropout`: try `0.0`, `0.1`, `0.2`.
- `epochs`: start small for smoke tests, then use early stopping in future work.

TCN-specific:

- `tcn_kernel_size`: try `2`, `3`, `5`.
- dilation depth via `num_layers`.

Data and target:

- feature subsets: all return features vs close-to-close only vs intraday/overnight groups.
- prediction horizon: start with next-day return, later compare 5-day and 20-day targets.
- universe size: `top_n = 50`, `100`, `250`.

Trading evaluation:

- strategy: `rank_long_short`, `zscore`.
- rank quantiles: 10%, 20%, 30%.
- `tc_bps`: 0, 5, 10.
- gross exposure.
- rebalance frequency, to be added later.

## Intended Research Direction

The goal is not just to maximize one backtest metric. The stronger CV-worthy project is a reproducible research pipeline with clear leakage controls and honest out-of-sample evaluation.

Recommended next steps:

1. Run a small smoke data/train/evaluate cycle to confirm end-to-end behavior with live Yahoo data.
2. Run the default 10-year, top-100 dataset.
3. Add simple baselines:
   - zero prediction
   - previous return persistence
   - momentum/reversal signal
   - linear/ridge regression
4. Compare RNN, LSTM, and TCN against those baselines.
5. Add experiment tracking tables under `runs/`.
6. Run ablations:
   - feature groups
   - sequence lengths
   - model families
   - target horizons
   - trading rules
7. Write a research note:
   - hypothesis
   - data protocol
   - leakage controls
   - model setup
   - experiment table
   - equity curves
   - failure analysis

## CV Angle

Strong positioning:

```text
Built a leakage-safe PyTorch research pipeline for cross-sectional equity return prediction, with reproducible Parquet data artifacts, RNN/LSTM/TCN sequence models, validation-selected checkpoints, and transaction-cost-aware long-short backtest evaluation.
```

What makes it credible:

- strict train/val/test chronology
- no scaler leakage
- validation-selected checkpoints
- untouched test evaluation
- transaction-cost and turnover accounting
- comparison against simple baselines
- clear discussion of survivorship bias

## Known Limitations

- Current S&P 500 membership introduces survivorship bias.
- Yahoo Finance data is convenient but not institutional-grade.
- No rolling walk-forward retraining yet.
- No early stopping yet.
- No benchmark factor library yet.
- No full real-data results have been generated yet.
