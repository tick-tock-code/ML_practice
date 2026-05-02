from __future__ import annotations

import numpy as np
import pandas as pd


def rank_long_short_weights(
    predictions: pd.DataFrame,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for signal_date, group in predictions.groupby("date", sort=True):
        group = group.copy()
        ranks = group["prediction"].rank(method="average", pct=True)
        long_mask = ranks >= high_quantile
        short_mask = ranks <= low_quantile
        weights = pd.Series(0.0, index=group.index)
        if long_mask.any() and short_mask.any():
            weights.loc[long_mask] = gross_exposure * 0.5 / long_mask.sum()
            weights.loc[short_mask] = -gross_exposure * 0.5 / short_mask.sum()
        group["weight"] = weights
        rows.append(group)
    if not rows:
        return predictions.assign(weight=0.0)
    return pd.concat(rows, ignore_index=True)


def zscore_weights(predictions: pd.DataFrame, gross_exposure: float = 1.0) -> pd.DataFrame:
    rows = []
    for signal_date, group in predictions.groupby("date", sort=True):
        group = group.copy()
        std = group["prediction"].std()
        if pd.isna(std) or std == 0:
            group["weight"] = 0.0
        else:
            z = (group["prediction"] - group["prediction"].mean()) / std
            denom = z.abs().sum()
            group["weight"] = 0.0 if denom == 0 else z / denom * gross_exposure
        rows.append(group)
    if not rows:
        return predictions.assign(weight=0.0)
    return pd.concat(rows, ignore_index=True)


def make_weights(
    predictions: pd.DataFrame,
    strategy: str,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    if strategy == "rank_long_short":
        return rank_long_short_weights(predictions, low_quantile, high_quantile, gross_exposure)
    if strategy == "zscore":
        return zscore_weights(predictions, gross_exposure)
    raise ValueError("Unknown strategy. Expected 'rank_long_short' or 'zscore'.")


def strategy_returns(
    weighted_predictions: pd.DataFrame,
    tc_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = weighted_predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["weighted_return"] = df["weight"] * df["target"]

    daily = (
        df.groupby(["date", "target_date"], as_index=False)
        .agg(gross_return=("weighted_return", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    weight_matrix = (
        df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    turnover = weight_matrix.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = weight_matrix.iloc[0].abs().sum()
    turnover_map = turnover.rename("turnover").reset_index()
    daily = daily.merge(turnover_map, on="date", how="left")
    daily["turnover"] = daily["turnover"].fillna(0.0)
    daily["transaction_cost"] = daily["turnover"] * (tc_bps / 10000.0)
    daily["net_return"] = daily["gross_return"] - daily["transaction_cost"]
    daily["cum_index"] = (1.0 + daily["net_return"].fillna(0.0)).cumprod()

    stats = summarize_returns(daily)
    return daily, stats


def summarize_returns(daily: pd.DataFrame, days_per_year: int = 252) -> dict[str, float]:
    returns = daily["net_return"].astype(float)
    ann_return = returns.mean() * days_per_year if len(returns) else np.nan
    ann_vol = returns.std() * np.sqrt(days_per_year) if len(returns) else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    cumulative = daily["cum_index"].astype(float)
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else float("nan"),
        "hit_rate": float((daily["gross_return"] > 0).mean()) if len(daily) else float("nan"),
        "net_hit_rate": float((daily["net_return"] > 0).mean()) if len(daily) else float("nan"),
        "total_turnover": float(daily["turnover"].sum()) if len(daily) else 0.0,
        "total_transaction_cost": float(daily["transaction_cost"].sum()) if len(daily) else 0.0,
        "n_days": int(len(daily)),
    }
