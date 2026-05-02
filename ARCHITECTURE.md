# Architecture

## Pipeline

```text
data stage
  current S&P symbols + Yahoo Finance adjusted OHLCV
  -> raw_ohlcv.parquet
  -> processed_returns.parquet
  -> processed_scaled.parquet
  -> split_metadata.json + scaler.json

train stage
  processed_scaled.parquet
  -> pooled ticker/date sequence samples
  -> RNN/LSTM/TCN checkpoints selected by validation loss

evaluate stage
  test split + saved checkpoints
  -> predictions
  -> rank and z-score weights
  -> daily returns, cumulative index, stats, plots
```

## Data Contract

Raw data is long format:

```text
date, ticker, open, high, low, close, volume
```

Processed data adds:

```text
r_cc, r_on, r_oo, r_oc, r_ho, r_hc, r_lo, r_lc, r_hl, vol_norm,
forward_return, target_date, split
```

`forward_return` is the simple return from feature date `t` to `target_date`.

## Modeling Contract

The dataset is pooled across tickers. Each sample is:

```text
X: [sequence_length, num_features]
y: scalar next-day forward_return
metadata: ticker, date, target_date
```

The models all expose the same interface: `forward(x)` returns one scalar prediction per sample.

## Leakage Controls

- Split boundaries are based on `target_date`.
- Scalers are fit only on train rows.
- Validation selects checkpoints.
- Test metrics and backtests run only in `evaluate`.
- Sequences for validation/test may include earlier context rows, but never future feature rows.

## Output Layout

```text
data/
|-- raw_ohlcv.parquet
|-- processed_returns.parquet
|-- processed_scaled.parquet
|-- scaler.json
`-- split_metadata.json

runs/<timestamp>/
|-- run_config.json
|-- checkpoints/
|-- metrics/
`-- evaluation/
```
