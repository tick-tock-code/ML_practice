from __future__ import annotations

import numpy as np
import pandas as pd

from .sequences import EquitySequenceDataset


def predictions_from_constant(dataset: EquitySequenceDataset, value: float = 0.0) -> pd.DataFrame:
    rows = []
    for index in range(len(dataset)):
        metadata = dataset.sample_metadata(index)
        rows.append(
            {
                "ticker": metadata.ticker,
                "date": metadata.date,
                "target_date": metadata.target_date,
                "target": metadata.target,
                "prediction": float(value),
            }
        )
    return pd.DataFrame(rows)


def add_momentum_12_1_signal(
    frame: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
    min_periods: int = 126,
) -> pd.DataFrame:
    if "r_cc" not in frame.columns:
        raise ValueError("momentum_12_1 requires an r_cc column.")
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["momentum_12_1"] = (
        df.groupby("ticker")["r_cc"]
        .transform(lambda values: values.shift(skip_days).rolling(lookback_days, min_periods=min_periods).sum())
        .replace([np.inf, -np.inf], np.nan)
    )
    return df


def momentum_12_1_predictions(
    dataset: EquitySequenceDataset,
    unscaled_frame: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
    min_periods: int = 126,
) -> pd.DataFrame:
    signal_frame = add_momentum_12_1_signal(unscaled_frame, lookback_days, skip_days, min_periods)
    signal_map = {
        (str(row.ticker), pd.Timestamp(row.date).date().isoformat()): row.momentum_12_1
        for row in signal_frame[["ticker", "date", "momentum_12_1"]].itertuples(index=False)
    }
    rows = []
    for index in range(len(dataset)):
        metadata = dataset.sample_metadata(index)
        prediction = signal_map.get((metadata.ticker, metadata.date), np.nan)
        rows.append(
            {
                "ticker": metadata.ticker,
                "date": metadata.date,
                "target_date": metadata.target_date,
                "target": metadata.target,
                "prediction": 0.0 if pd.isna(prediction) else float(prediction),
            }
        )
    return pd.DataFrame(rows)


def equal_weight_predictions(dataset: EquitySequenceDataset) -> pd.DataFrame:
    return predictions_from_constant(dataset, 0.0)


def equal_weight_weights(predictions: pd.DataFrame, gross_exposure: float = 1.0) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby("date", sort=True):
        group = group.copy()
        n_assets = len(group)
        group["weight"] = 0.0 if n_assets == 0 else gross_exposure / n_assets
        rows.append(group)
    if not rows:
        return predictions.assign(weight=0.0)
    return pd.concat(rows, ignore_index=True)


def baseline_predictions(
    name: str,
    dataset: EquitySequenceDataset,
    unscaled_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if name == "cash_zero":
        return predictions_from_constant(dataset, 0.0)
    if name == "equal_weight":
        return equal_weight_predictions(dataset)
    if name == "momentum_12_1":
        if unscaled_frame is None:
            raise ValueError("momentum_12_1 requires unscaled processed data.")
        return momentum_12_1_predictions(dataset, unscaled_frame)
    raise ValueError("Unknown baseline. Expected cash_zero, equal_weight, or momentum_12_1.")
