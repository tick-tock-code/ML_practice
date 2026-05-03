# Quant RNN Pipeline Status

Living project note for the `2026_05_RNN_exp` branch.

## Current State

The three-stage quant ML pipeline is implemented in `quant_rnn/` but has not yet been committed.

Implemented CLI stages:

```text
data      Download adjusted OHLCV data, compute return features, create train/val/test splits.
train     Train selected pooled PyTorch sequence models.
evaluate  Evaluate trained models on the untouched test set using trading-style rules.
walk-forward  Run expanding-window retraining, validation, and test evaluation.
seed-sweep  Repeat one fixed config across random seeds and aggregate validation/test results.
```

Implemented models:

```text
rnn   Plain recurrent neural network baseline.
lstm  Gated recurrent model for longer sequence memory.
tcn   Temporal convolutional network using causal/dilated convolutions.
```

Implemented baselines:

```text
cash_zero      Zero prediction no-skill baseline.
equal_weight   Passive long-only equal-weight benchmark.
momentum_12_1  Fixed skipped-history cross-sectional momentum signal.
```

Implemented safeguards:

- Splits are assigned by `target_date`, not feature date.
- Default split is 6 years train, 2 years validation, 2 years test inside a 10-year window.
- Scalers are fit only on train rows.
- Validation is used for checkpoint and hyperparameter selection.
- Early stopping uses the configured validation selection metric to save compute.
- Default checkpoint selection metric is `val_daily_ic`, because MSE is objective-misaligned for cross-sectional trading.
- Test is reserved for final evaluation.
- Test evaluation supports both rank long-short and z-score trading rules.
- Transaction costs are included using turnover and default `tc_bps = 5`.
- Walk-forward mode fits a separate scaler on each fold's train rows only.
- Adam weight decay is supported for train, walk-forward, and seed-sweep.
- Evaluation writes quintile monotonicity diagnostics by default.

Verification completed:

```text
Ran 30 tests
OK
```

Not yet completed:

- No full 10-year training run yet.
- No walk-forward research run yet.
- No committed final research write-up yet.

## Pilot v1 Results - 2026-05-02

Run directory:

```text
runs/20260502_170658
```

Pilot configuration:

```text
top_n=50
years=5
split=3y train / 1y validation / 1y test
sequence_length=60
models=rnn,lstm,tcn
learning_rate=1e-3
batch_size=128
hidden_size=256
num_layers=1
dropout=0.2
epochs=10
early_stopping_patience=3
```

Training summary:

```text
rnn   best val MSE 0.000506 at epoch 5, stopped epoch 8
lstm  best val MSE 0.000493 at epoch 2, stopped epoch 5
tcn   best val MSE 0.000570 at epoch 2, stopped epoch 5
```

Evaluation summary:

```text
model         strategy          test MSE   dir acc   daily IC   ann return   Sharpe
rnn           rank_long_short   0.000472   0.4945   -0.0079    -0.2508     -2.5575
rnn           zscore            0.000472   0.4945   -0.0079    -0.2683     -2.7571
lstm          rank_long_short   0.000456   0.5134    0.0253    -0.0108     -0.1214
lstm          zscore            0.000456   0.5134    0.0253     0.0364      0.4039
tcn           rank_long_short   0.000538   0.4882   -0.0253    -0.2564     -2.9437
tcn           zscore            0.000538   0.4882   -0.0253    -0.2977     -3.3097
cash_zero     active rules      0.000448   0.5314       n/a      0.0000        n/a
equal_weight  equal_weight          n/a      n/a        n/a      0.2617      2.1488
momentum_12_1 rank_long_short   0.103555   0.5140    0.0378     0.1601      1.2240
momentum_12_1 zscore            0.103555   0.5140    0.0378     0.1620      1.1936
```

Interpretation:

- The LSTM is the only neural model with positive test correlation, positive daily IC, and positive z-score strategy Sharpe.
- The neural models did not beat the simple equal-weight or fixed momentum baselines in this pilot.
- `cash_zero` has lower MSE than all neural models because next-day returns are noisy and close to zero; MSE alone is not enough for judging ranking signal.
- Equal-weight performed very strongly during this specific test window, so it is a useful but market-regime-sensitive benchmark.
- Next research step should be feature/regularization ablation, not larger model size.

## 1e-4 Grid Results - 2026-05-02

Summary artifact:

```text
runs/grid_lr1e4_summary.csv
```

Completed scope:

```text
learning_rate=1e-4
dropout=0.2
sequence_length=60
early_stopping_patience=5
epochs=30
models=rnn,lstm,tcn
batch_size=[64,128,256]
hidden_size=[64,128]
num_layers=[1,2]
```

The planned `hidden_size=256` group was stopped because runtime was too long. One partial `hs256` run completed and one partial run was interrupted, but the official summary excludes `hidden_size=256`.

Best neural configuration by daily IC and Sharpe:

```text
run_dir: runs/grid_lr1e4_bs128_hs064_l2
model: lstm
num_layers: 2
hidden_size: 64
batch_size: 128
best_epoch: 2
stopped_epoch: 7
best_val_mse: 0.000495
test_mse: 0.000448
directional_accuracy: 0.5292
daily_ic: 0.0316
rank_long_short ann_return: 0.1674
rank_long_short Sharpe: 1.4743
zscore ann_return: 0.1422
zscore Sharpe: 1.3103
```

Baseline comparison:

```text
equal_weight Sharpe: 2.1488
momentum_12_1 rank_long_short Sharpe: 1.2240
momentum_12_1 zscore Sharpe: 1.1936
```

Interpretation:

- Lowering learning rate helped produce smoother and more useful LSTM results.
- The best LSTM configuration beat the fixed momentum baseline on Sharpe, but still did not beat equal-weight in this test window.
- Smaller hidden size worked better than the original `hidden_size=256`; this argues against increasing model size next.
- Two LSTM layers looked useful in the best run, but this needs confirmation with repeat seeds and walk-forward evaluation.
- Next best step is to repeat the best config across seeds and run feature ablations before adding larger models.

## Validation Selection Review - 2026-05-02

Artifacts:

```text
runs/grid_lr1e4_validation_summary.csv
runs/grid_lr1e4_selection_comparison.csv
```

Validation split metrics were computed from the 12 stored grid checkpoints. These checkpoints were originally selected by validation MSE, so this is a post-hoc review, not a replacement for retraining with validation IC selection.

Top validation daily IC:

```text
run_dir: runs/grid_lr1e4_bs064_hs128_l2
model: tcn
strategy: zscore
val_mse: 0.000492
val_directional_accuracy: 0.5069
val_daily_ic: 0.0150
val_zscore_sharpe: 0.3894
test_daily_ic: -0.0128
test_zscore_sharpe: -1.8752
```

Top validation Sharpe:

```text
run_dir: runs/grid_lr1e4_bs256_hs064_l1
model: lstm
strategy: zscore
val_mse: 0.000492
val_directional_accuracy: 0.5231
val_daily_ic: 0.0140
val_zscore_sharpe: 0.5574
test_daily_ic: -0.0297
test_zscore_sharpe: -2.7774
```

Interpretation:

- The existing MSE-selected checkpoints do not generalize well when ranked by validation IC or validation Sharpe.
- This reinforces that the next grid should train and early-stop directly on `val_daily_ic`, not only review stored MSE-selected checkpoints.
- Test results should be treated as final reporting only; research decisions should be frozen from validation metrics first.

Selection comparison:

```text
selection_rule                         run_dir                          model strategy          val_metric  test_daily_ic  test_sharpe
selected_by_val_daily_ic               runs/grid_lr1e4_bs064_hs128_l2   tcn   zscore            0.0150     -0.0128       -1.8752
selected_by_val_rank_long_short_sharpe runs/grid_lr1e4_bs256_hs064_l1   lstm  rank_long_short   0.2264     -0.0297       -3.2697
selected_by_val_zscore_sharpe          runs/grid_lr1e4_bs256_hs064_l1   lstm  zscore            0.5574     -0.0297       -2.7774
selected_by_val_mse                    runs/grid_lr1e4_bs064_hs128_l1   lstm  rank_long_short   0.000486   -0.0323       -3.1907
```

## Repository Status

Current expected git state after implementation:

```text
 M AGENTS.md
 M ARCHITECTURE.md
 M PROJECT_STATUS.md
 M README.md
 M quant_rnn/cli.py
 M quant_rnn/evaluation.py
 M quant_rnn/training.py
 M quant_rnn/walk_forward.py
 M tests/test_baselines_walk_forward.py
 M tests/test_evaluation.py
 M tests/test_training_smoke.py
?? quant_rnn/aggregation.py
?? quant_rnn/seed_sweep.py
?? scripts/
?? tests/test_aggregation.py
?? tests/test_seed_sweep.py
?? tests/test_weight_decay_grid_script.py
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
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn,lstm,tcn --sequence-length 60 --epochs 50 --selection-metric val_daily_ic --early-stopping-patience 5 --weight-decay 1e-4
```

Train a quick smoke model:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn train --models rnn --sequence-length 20 --epochs 2 --batch-size 128
```

Evaluate latest run:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --models rnn,lstm,tcn --strategies rank_long_short,zscore
```

Evaluate validation split:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --split val --run-dir runs\YYYYMMDD_HHMMSS --models rnn,lstm,tcn --strategies rank_long_short,zscore
```

Aggregate grid validation metrics:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn aggregate-grid --run-pattern grid_lr1e4_* --max-hidden-size 128
```

Repeat best candidate across seeds:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn seed-sweep --models lstm --seeds 1,2,3,4,5 --sequence-length 60 --epochs 30 --batch-size 128 --learning-rate 1e-4 --hidden-size 64 --num-layers 2 --dropout 0.2 --weight-decay 1e-4 --selection-metric val_daily_ic
```

Run LSTM weight-decay grid at `learning_rate=1e-5`:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe scripts\run_weight_decay_grid.py --dry-run
C:\ProgramData\anaconda3\envs\torch_env2\python.exe scripts\run_weight_decay_grid.py --resume
```

Grid output:

```text
runs/grid_lr1e5_wd/grid_validation_summary.csv
```

This grid is validation-only by default to avoid repeated test-set peeking. `learning_rate=1e-5` may train slowly, so the script uses `epochs=50` and `early_stopping_patience=8`.

Evaluate with baselines:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn evaluate --models rnn,lstm,tcn --strategies rank_long_short,zscore --baselines cash_zero,equal_weight,momentum_12_1
```

Run expanding-window walk-forward research:

```powershell
C:\ProgramData\anaconda3\envs\torch_env2\python.exe -m quant_rnn walk-forward --models rnn,lstm,tcn --initial-train-years 6 --val-years 1 --test-years 1 --step-years 1 --weight-decay 1e-4
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
- `weight_decay`: try `0.0`, `1e-5`, `1e-4`.
- `epochs`: start small for smoke tests, then use early stopping in future work.

TCN-specific:

- `tcn_kernel_size`: try `2`, `3`, `5`.
- dilation depth via `num_layers`.

Data and target:

- feature subsets: all return features vs close-to-close only vs intraday/overnight groups.
- prediction horizon: keep next-day return for the current phase; 5-day and 20-day targets are explicitly deferred.
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
   - equal-weight long-only
   - fixed 12-1 momentum
4. Use walk-forward retraining as the main research path once the simple static run is working.
5. Compare RNN, LSTM, and TCN against those baselines.
6. Add experiment tracking tables under `runs/`.
7. Use seed-sweep to test whether the best LSTM config is stable across initializations.
8. Add monotonicity review before trusting backtest results:
   - prediction quintile table
   - top-minus-bottom realized-return spread
   - validation vs test monotonicity comparison
9. Run ablations:
   - feature groups
   - sequence lengths
   - model families
   - trading rules
10. Write a research note:
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

- Current S&P 500 membership introduces survivorship bias. This is accepted for v1 infrastructure, but it must be disclosed in any results.
- Yahoo Finance data is convenient but not institutional-grade. This is acceptable for prototyping and CV research, not production trading claims.
- Walk-forward retraining is now the intended serious research path; static train/val/test remains useful for quick iteration.
- Early stopping is implemented to reduce wasted compute, but patience/min-delta still need empirical tuning.
- Repeat seeds are robustness checks for initialization/minibatch randomness, not a leakage prevention method.
- Five-day forward-return targets are deferred; the current research target remains next-day return.
- Monotonicity diagnostics now check whether higher prediction buckets actually realize higher average returns.
- Baselines are now included, but they are deliberately minimal: cash/zero prediction, equal-weight, and fixed 12-1 momentum.
- Limited trading realism is deferred. Short borrow, slippage, market impact, liquidity filters, and sector/beta constraints should come after the pipeline produces credible first results.
- One universe and one daily frequency are acceptable initially. Once the pipeline works, robustness across universes, periods, and frequencies should become an experiment track.
- No full real-data results have been generated yet.

## Momentum References

- Jegadeesh and Titman, 1993, "Returns to Buying Winners and Selling Losers": [EconPapers](https://econpapers.repec.org/RePEc%3Abla%3Ajfinan%3Av%3A48%3Ay%3A1993%3Ai%3A1%3Ap%3A65-91)
- Moskowitz, Ooi, and Pedersen, 2012, "Time Series Momentum": [EconPapers](https://econpapers.repec.org/RePEc%3Aeee%3Ajfinec%3Av%3A104%3Ay%3A2012%3Ai%3A2%3Ap%3A228-250)
- Asness, Moskowitz, and Pedersen, 2013, "Value and Momentum Everywhere": [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1363476)
- Hurst, Ooi, and Pedersen, 2017, "A Century of Evidence on Trend-Following Investing": [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026)
