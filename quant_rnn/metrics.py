from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "directional_accuracy": float("nan"), "correlation": float("nan")}
    truth = truth[mask]
    pred = pred[mask]
    mse = np.mean((pred - truth) ** 2)
    mae = np.mean(np.abs(pred - truth))
    directional = np.mean((pred >= 0.0) == (truth >= 0.0))
    corr = np.corrcoef(pred, truth)[0, 1] if len(truth) > 1 and np.std(pred) > 0 and np.std(truth) > 0 else np.nan
    return {
        "mse": float(mse),
        "mae": float(mae),
        "directional_accuracy": float(directional),
        "correlation": float(corr),
    }


def daily_information_coefficient(predictions: pd.DataFrame) -> float:
    values = []
    for _, group in predictions.groupby("date"):
        if len(group) < 2:
            continue
        if group["prediction"].std() == 0 or group["target"].std() == 0:
            continue
        corr = group["prediction"].corr(group["target"])
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")
