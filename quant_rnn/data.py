from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .constants import DEFAULT_FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN
from .io import ensure_dir, parse_csv_list, write_json


@dataclass(frozen=True)
class DataStageConfig:
    output_dir: Path
    years: int = 10
    top_n: int = 100
    train_years: int = 6
    val_years: int = 2
    test_years: int = 2
    vol_window: int = 20
    target_horizon: int = 1
    end_date: str | None = None
    tickers: tuple[str, ...] = ()
    ticker_file: str | None = None


def read_html_with_ua(url: str) -> list[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return pd.read_html(response.text)


def normalize_yahoo_symbol(symbol: str) -> str:
    return symbol.strip().replace(".", "-")


def get_sp500_tickers(top_n: int) -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = read_html_with_ua(url)[0]
    return [normalize_yahoo_symbol(value) for value in table["Symbol"].head(top_n).tolist()]


def load_ticker_file(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    tickers: list[str] = []
    for line in text.splitlines():
        for item in line.split(","):
            item = item.strip()
            if item and not item.startswith("#"):
                tickers.append(normalize_yahoo_symbol(item))
    return tickers


def resolve_tickers(config: DataStageConfig) -> list[str]:
    if config.tickers:
        return [normalize_yahoo_symbol(ticker) for ticker in config.tickers]
    if config.ticker_file:
        return load_ticker_file(config.ticker_file)[: config.top_n]
    return get_sp500_tickers(config.top_n)


def download_adjusted_ohlcv(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("At least one ticker is required.")
    data = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise ValueError("Yahoo Finance returned no data.")
    return normalize_yfinance_ohlcv(data, ticker_list)


def normalize_yfinance_ohlcv(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    field_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frames: list[pd.DataFrame] = []
    data = data.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)

    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(map(str, data.columns.get_level_values(0)))
        level1 = set(map(str, data.columns.get_level_values(1)))
        if set(field_map).issubset(level0):
            field_level, ticker_level = 0, 1
        elif set(field_map).issubset(level1):
            field_level, ticker_level = 1, 0
        else:
            raise ValueError("Could not identify OHLCV fields in yfinance output.")

        available_tickers = list(dict.fromkeys(map(str, data.columns.get_level_values(ticker_level))))
        for ticker in available_tickers:
            record = pd.DataFrame({"date": data.index, "ticker": ticker})
            for source_field, output_field in field_map.items():
                try:
                    series = data.xs(source_field, axis=1, level=field_level).loc[:, ticker]
                except KeyError:
                    series = pd.Series(np.nan, index=data.index)
                record[output_field] = series.to_numpy()
            frames.append(record)
    else:
        if len(tickers) != 1:
            raise ValueError("Single-level yfinance output can only be normalized for one ticker.")
        record = pd.DataFrame({"date": data.index, "ticker": tickers[0]})
        for source_field, output_field in field_map.items():
            record[output_field] = data[source_field].to_numpy()
        frames.append(record)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[RAW_COLUMNS]
    raw = raw.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    raw["volume"] = raw["volume"].fillna(0.0)
    return raw.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_return_features(
    raw: pd.DataFrame,
    vol_window: int = 20,
    eps: float = 1e-8,
    target_horizon: int = 1,
) -> pd.DataFrame:
    required = set(RAW_COLUMNS)
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    if target_horizon < 1:
        raise ValueError("target_horizon must be >= 1.")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = df.groupby("ticker", group_keys=False)
    prev_close = grouped["close"].shift(1)
    prev_open = grouped["open"].shift(1)

    df["r_cc"] = np.log(df["close"] / prev_close)
    df["r_on"] = np.log(df["open"] / prev_close)
    df["r_oo"] = np.log(df["open"] / prev_open)
    df["r_oc"] = np.log(df["close"] / df["open"])
    df["r_ho"] = np.log(df["high"] / df["open"])
    df["r_hc"] = np.log(df["high"] / df["close"])
    df["r_lo"] = np.log(df["low"] / df["open"])
    df["r_lc"] = np.log(df["low"] / df["close"])
    df["r_hl"] = np.log(df["high"] / df["low"])

    log_volume = np.log(df["volume"].replace(0, eps))
    rolling_log_volume = log_volume.groupby(df["ticker"]).transform(
        lambda values: values.rolling(vol_window, min_periods=vol_window).mean()
    )
    df["vol_norm"] = log_volume - rolling_log_volume

    future_close = grouped["close"].shift(-target_horizon)
    future_date = grouped["date"].shift(-target_horizon)
    df[TARGET_COLUMN] = future_close / df["close"] - 1.0
    df["target_date"] = future_date
    df = df.dropna(subset=["target_date", TARGET_COLUMN]).reset_index(drop=True)
    return df


def assign_splits(
    processed: pd.DataFrame,
    train_years: int = 6,
    val_years: int = 2,
    test_years: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if min(train_years, val_years, test_years) <= 0:
        raise ValueError("train_years, val_years, and test_years must all be positive.")

    df = processed.copy()
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.tz_localize(None)
    max_target = df["target_date"].max()
    test_start = max_target - pd.DateOffset(years=test_years)
    val_start = test_start - pd.DateOffset(years=val_years)
    train_start = val_start - pd.DateOffset(years=train_years)

    df["split"] = "unused"
    df.loc[(df["target_date"] > train_start) & (df["target_date"] <= val_start), "split"] = "train"
    df.loc[(df["target_date"] > val_start) & (df["target_date"] <= test_start), "split"] = "val"
    df.loc[df["target_date"] > test_start, "split"] = "test"

    counts = df["split"].value_counts().to_dict()
    for split in ["train", "val", "test"]:
        if counts.get(split, 0) == 0:
            raise ValueError(f"Split '{split}' has no rows. Check date range and split years.")

    metadata = {
        "train_start_exclusive": train_start.date().isoformat(),
        "val_start_exclusive": val_start.date().isoformat(),
        "test_start_exclusive": test_start.date().isoformat(),
        "max_target_date": max_target.date().isoformat(),
        "split_counts": counts,
        "split_rule": "assigned by target_date",
    }
    return df, metadata


def fit_standard_scaler(
    processed: pd.DataFrame,
    feature_columns: list[str] | None = None,
    train_split_name: str = "train",
) -> dict[str, dict[str, float]]:
    features = feature_columns or DEFAULT_FEATURE_COLUMNS
    train = processed.loc[processed["split"] == train_split_name, features]
    if train.empty:
        raise ValueError("Cannot fit scaler because the train split is empty.")
    means = train.mean(skipna=True)
    stds = train.std(skipna=True).replace(0.0, 1.0).fillna(1.0)
    return {
        "feature_columns": list(features),
        "mean": {column: float(means[column]) for column in features},
        "std": {column: float(stds[column]) for column in features},
        "fit_split": train_split_name,
    }


def apply_standard_scaler(processed: pd.DataFrame, scaler: dict[str, object]) -> pd.DataFrame:
    df = processed.copy()
    features = list(scaler["feature_columns"])
    means = scaler["mean"]
    stds = scaler["std"]
    for column in features:
        df[column] = (df[column] - float(means[column])) / float(stds[column])
    return df


def build_data_artifacts(config: DataStageConfig) -> dict[str, Path]:
    output_dir = ensure_dir(config.output_dir)
    end_ts = pd.Timestamp(config.end_date).normalize() if config.end_date else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=config.years)
    yf_end = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    yf_start = start_ts.date().isoformat()

    tickers = resolve_tickers(config)
    raw = download_adjusted_ohlcv(tickers, yf_start, yf_end)
    processed = compute_return_features(
        raw,
        vol_window=config.vol_window,
        target_horizon=config.target_horizon,
    )
    processed, split_metadata = assign_splits(
        processed,
        train_years=config.train_years,
        val_years=config.val_years,
        test_years=config.test_years,
    )
    scaler = fit_standard_scaler(processed, DEFAULT_FEATURE_COLUMNS)
    scaled = apply_standard_scaler(processed, scaler)

    paths = {
        "raw": output_dir / "raw_ohlcv.parquet",
        "processed": output_dir / "processed_returns.parquet",
        "scaled": output_dir / "processed_scaled.parquet",
        "scaler": output_dir / "scaler.json",
        "split_metadata": output_dir / "split_metadata.json",
        "data_config": output_dir / "data_config.json",
    }
    raw.to_parquet(paths["raw"], index=False)
    processed.to_parquet(paths["processed"], index=False)
    scaled.to_parquet(paths["scaled"], index=False)
    write_json(paths["scaler"], scaler)
    write_json(paths["split_metadata"], split_metadata)
    write_json(
        paths["data_config"],
        {
            "years": config.years,
            "top_n": config.top_n,
            "train_years": config.train_years,
            "val_years": config.val_years,
            "test_years": config.test_years,
            "vol_window": config.vol_window,
            "target_horizon": config.target_horizon,
            "end_date": end_ts.date().isoformat(),
            "start_date": start_ts.date().isoformat(),
            "tickers": tickers,
            "feature_columns": DEFAULT_FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "survivorship_bias_note": "Universe uses current S&P 500 membership.",
        },
    )
    return paths


def config_from_args(args) -> DataStageConfig:
    tickers = tuple(parse_csv_list(args.tickers))
    return DataStageConfig(
        output_dir=Path(args.output_dir),
        years=args.years,
        top_n=args.top_n,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        vol_window=args.vol_window,
        target_horizon=args.target_horizon,
        end_date=args.end_date,
        tickers=tickers,
        ticker_file=args.ticker_file,
    )


def run_data_stage(args) -> int:
    config = config_from_args(args)
    paths = build_data_artifacts(config)
    print("Wrote data artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0
