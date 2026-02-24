import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import pickle
import inspect
import itertools
import json
import os
from datetime import datetime
from pprint import pformat
import matplotlib.pyplot as plt
import pandas as pd

# Compute returns
def compute_returns(df, vol_window=20, eps=1e-8):

    """
    Compute a set of price- and volume-based return features.
    TRADING AT THE END OF THE DAY ASSUMED.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['Open','High','Low','Close','Volume'] for a single ticker
        or a MultiIndex columns DataFrame as returned by yfinance for multiple tickers
        (level 0 = field, level 1 = ticker).
    horizon : int (default=1)
        Horizon used for _close-to-close_ style shifts (multi-day). Overnight uses shift(1).
    vol_window : int (default=20)
        Rolling window for volume normalization.
    eps : float
        Small value to avoid log(0) issues for volume.

    Returns
    -------
    pd.DataFrame
        DataFrame of features. Column names:
        - OPEN, HIGH, LOW, CLOSE, VOLUME, (field, ticker) multiindex preserved.
        - r_cc : close-to-close log return (shift by `horizon`)
        - r_on : overnight log return (open / previous close)
        - r_oo : open-to-open log return (shift by `horizon`)
        - r_oc : open-to-close log return (intraday)
        - r_ho : high-to-open log ratio (intraday high)
        - r_hc : high-to-close log ratio (high vs prev close)
        - r_lo, r_lc, r_hl : low-based measures and high/low range
        - vol_norm : log-volume surprise (log(V) - rolling mean log(V))
    """

    # making sure df is in correct format - date going downwards
    df = df.copy()
    df.index = pd.to_datetime(df.index, dayfirst=True, errors='raise')
    df = df.sort_index(ascending=True)


    # Make local names
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float).replace(0, eps)
    
    # Close related returns
    r_cc = np.log(close/close.shift(1)) # close-to-close return
    r_on = np.log(open_/close.shift(1)) # overnight return
    r_oo = np.log(open_/open_.shift(1)) # open-to-open return
    r_oc = np.log(close/open_) # open-to-close return

    # High related returns
    r_ho = np.log(high/open_)
    r_hc = np.log(high/close)

    # Low related returns
    r_lo = np.log(low/open_)
    r_lc = np.log(low/close)

    # High-low range
    r_hl = np.log(high/low)

    # Volume changes
    vol_norm = np.log(volume) - np.log(volume).rolling(vol_window).mean()



    features = {
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
        'r_cc': r_cc,
        'r_on': r_on,
        'r_oo': r_oo,
        'r_oc': r_oc,
        'r_ho': r_ho,
        'r_hc': r_hc,
        'r_lo': r_lo,
        'r_lc': r_lc,
        'r_hl': r_hl,
        'vol_norm': vol_norm
    }


    # concat into a MultiIndex columns DataFrame: level0 = field, level1 = ticker
    pieces = []
    tuples = []
    for field, df_field in features.items():
        # df_field is T x N (tickers)
        pieces.append(df_field)
        for col in df_field.columns:
            tuples.append((field, col))
    result = pd.concat(pieces, axis=1)
    result.columns = pd.MultiIndex.from_tuples(tuples)

    #returns = pd.concat(feats, axis=1)  # keys become outer level
    #returns.columns = pd.MultiIndex.from_tuples(returns.columns) # create MultiIndex

    return result


# Cross sectional ranking - basic price reversion
def cross_sectional_signal(returns, kind = "mean_reversion", lookback=5, vol_window=20, ranking_method="percentile"):
    # making sure df is in correct format
    df = returns.copy()
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index(ascending=True)
    
    # extract close-to-close returns
    r_cc = df['r_cc']

    # require full lookback and vol_window to produce values
    cum_r_cc = r_cc.rolling(window=lookback).sum()
    vol = r_cc.rolling(window=vol_window).std()
    signal = cum_r_cc  / vol # room to clip signal if needed

    # choose sign convention
    if kind == "mean_reversion":
        signal = -signal

    # rank cross-sectionally per date (rows are dates)
    if ranking_method == "percentile":
        ranks = signal.rank(axis=1, pct=True, method='average', na_option='keep')
    elif ranking_method == "zscore":
        ranks = weights_zscore(signal)
    return ranks



# Backtesting module
def backtest_long_short(
    ranks,                  # DataFrame: index=dates, columns=tickers, values = rank/score (higher -> long)
    returns,                # DataFrame: same shape; either log returns or simple returns (see is_log_returns)
    low_q=0.1,
    high_q=0.9,
    tc_bps=5.0,             # transaction cost in basis points (e.g. 5 = 5 bps = 0.0005)
    days_per_year=252,
    rebalance_every_n_days = None,   # e.g., 5 for weekly, 21 for monthly. None => rebalance every day
    no_trade_band = 0.0,                  # do not trade tickers where |target - current| <= band
    partial_rebalance = 1.0,              # fraction of the gap to trade on rebalance (0..1)
    min_trade_size = 0.0                  # ignore trades with abs size below this (absolute weight)

    ):
    

    tc = tc_bps / 10000.0  # convert bps to decimal


    # alignment: dates must match
    assert ranks.index.equals(returns.index), "ranks and returns must share the same dates"

    # If returns is MultiIndex (field, ticker), extract the per-ticker r_cc DataFrame.
    if getattr(returns.columns, 'nlevels', 1) > 1:
        if 'r_cc' not in returns.columns.get_level_values(0):
            raise AssertionError("returns does not contain top-level field 'r_cc'")
        r_cc = returns['r_cc']   # now a T x N DataFrame with ticker columns
    else:
        # single-level returns (assume this is already the per-ticker r_cc)
        r_cc = returns.copy()

    # Ensure the tickers in ranks are present in r_cc. Reorder r_cc to match ranks' column order.
    missing_in_returns = set(ranks.columns) - set(r_cc.columns)
    if missing_in_returns:
        raise AssertionError(f"Some tickers present in ranks not found in returns['r_cc']: {sorted(missing_in_returns)}")

    # Reindex r_cc to have exactly the same columns in the same order as ranks
    r_cc = r_cc.reindex(columns=ranks.columns)




    # Getting weights from ranks
    w_target = weights_proportional_centered(ranks)



    # Build applied weights (w_used) using delayed rebalancing controls:
    # - rebalance_every_n_days: calendar batching
    # - no_trade_band: absolute threshold to avoid tiny trades
    # - partial_rebalance: move only partway to target on rebalance
    # - min_trade_size: ignore tiny trades after band/partial application
    w_used = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    prev_w = pd.Series(0.0, index=ranks.columns)
    for i, date in enumerate(ranks.index):
        target = w_target.loc[date]

        # determine if this date is a rebalance day
        is_rebalance = True
        if rebalance_every_n_days is not None and rebalance_every_n_days > 0:
            is_rebalance = (i % rebalance_every_n_days == 0)

        if not is_rebalance:
            w_today = prev_w.copy()
        else:
            raw_trade = target - prev_w
            # apply no-trade band
            trade_mask = raw_trade.abs() > no_trade_band
            trade_amount = raw_trade.where(trade_mask, 0.0)
            # partial rebalancing
            trade_amount = trade_amount * partial_rebalance
            # enforce min trade size
            trade_amount = trade_amount.where(trade_amount.abs() >= min_trade_size, 0.0)
            w_today = prev_w + trade_amount

        w_used.loc[date] = w_today
        prev_w = w_today.copy()





    # Getting c-c returns
    df = returns.copy()
    r_cc = df['r_cc'] # log returns

    # Calculating portfolio return based on weights acting on next day's returns
    #returns_next_log = r_cc.shift(-1)  # next day's log returns
    returns_next_simple = np.exp(r_cc.shift(-1)) - 1  # convert log returns to simple returns for portfolio returns calc
    portfolio_returns = (w_used * returns_next_simple).sum(axis=1)


    # contributions (for analysis)
    contributions = w_used * returns_next_simple  # DataFrame same shape as w

    # compute turnover (based on weights used each day)
    # turnover at day t = sum_i |w_t - w_{t-1}|
    turnover = w_used.diff().abs().sum(axis=1)
    turnover.iloc[0] = w_used.abs().sum(axis=1).iloc[0]  # first day full turnover

    # transaction costs 
    transaction_cost = turnover * tc # approximated as proportional to turnover
    portfolio_returns_net = portfolio_returns - transaction_cost


    # cumulative index
    cum_index = (1 + portfolio_returns_net.fillna(0)).cumprod()

    # summary stats
    ann_ret = portfolio_returns_net.mean() * days_per_year # only for sharpe calculation
    ann_vol = portfolio_returns_net.std() * np.sqrt(days_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan # subtract risk-free rate?

    # max drawdown on cumulative (simple)
    running_max = cum_index.cummax()
    drawdown = (running_max - cum_index) / running_max
    max_dd = drawdown.max()

    # hit rate (percentage of days with positive gross portfolio return)
    hit_rate = (portfolio_returns > 0).mean()
    net_hit_rate = (portfolio_returns_net > 0).mean()



    results = {
        'port_ret': portfolio_returns,               # raw daily portfolio return
        'port_ret_net': portfolio_returns_net,       # after tc
        'returns_next_simple': returns_next_simple, # next day simple returns
        'contributions': contributions,     # per-ticker contributions
        'cum_index': cum_index,             # cumulative index starting at 1
        'weights': w_used,                       # daily weights (t)
        'turnover': turnover,               # daily turnover
        'tc': transaction_cost,             # daily transaction cost
        'stats': {
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'hit_rate': hit_rate,
            'net_hit_rate': net_hit_rate
        }
    }
    return results



# Helper that runs a single combo and returns the summary row and outdir
def run_and_record(run_index, params, signals, processed_data, tc_bps, days_per_year, strategy_kind, data_dir):
    # Create a short, safe run name
    run_name = f"{strategy_kind}_run_{run_index:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n=== Run {run_index} : {run_name} ===")
    print("Params:")
    print(pformat(params))

    # Run the backtest
    backtest_results = backtest_long_short(
        signals,
        processed_data,
        tc_bps=tc_bps,
        days_per_year=days_per_year,
        **params
    )

    # Save results via existing helper
    outdir = save_backtest_results(backtest_results, base_dir=data_dir, name=run_name)

    # ensure outdir exists and write params to disk
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        # dp.save_backtest_results may already create it; ignore failure
        pass

    print(f"Saved backtest to {outdir}")

    # Persist quick CSVs for inspection (overwrite allowed)
    try:
        backtest_results['port_ret'].to_csv(os.path.join(outdir, "port_ret.csv"))
        backtest_results['cum_index'].to_csv(os.path.join(outdir, "cum_index.csv"))
    except Exception as e:
        print("Warning: failed to write quick CSVs for run:", e)

    # Prepare a single summary row
    stats = backtest_results.get('stats', {})
    total_turnover = backtest_results.get('turnover', pd.Series()).sum() if backtest_results.get('turnover') is not None else None
    total_tc = backtest_results.get('tc', pd.Series()).sum() if backtest_results.get('tc') is not None else None

    summary_row = {
        'run_name': run_name,
        'rebalance_every_n_days': params.get('rebalance_every_n_days'),
        'partial_rebalance': params.get('partial_rebalance'),
        'ann_return': stats.get('ann_return'),
        'ann_vol': stats.get('ann_vol'),
        'sharpe': stats.get('sharpe'),
        'max_drawdown': stats.get('max_drawdown'),
        'hit_rate': stats.get('hit_rate'),
        'net_hit_rate': stats.get('net_hit_rate'),
        'total_turnover': total_turnover,
        'total_tc': total_tc,
        'outdir': str(outdir)
    }
    return summary_row, outdir
















def save_backtest_results(results, base_dir="backtests", name="strategy"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(base_dir) / f"{name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    pkl_path = outdir / "results.pkl"

    # --- save full results as pickle ---
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- save stats as json ---
    with open(outdir / "stats.json", "w") as f:
        json.dump(results['stats'], f, indent=2)

    # --- metadata ---
    meta = {
        "created_at": timestamp,
        "strategy": name,
        "n_days": len(results['port_ret']),
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- save source code of backtest_long_short ---
    save_function_source(cross_sectional_signal, outdir / "cross_sectional_signal_source_code.txt")
    save_function_source(backtest_long_short, outdir / "backtest_long_short_source_code.txt")

    return outdir



def save_function_source(func, path):
    """
    Save the source code of a function to a text file.

    Parameters
    ----------
    func : callable
        The function object (not a string)
    path : str or Path
        Path to the .txt file to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    source = inspect.getsource(func)

    with open(path, "w", encoding="utf-8") as f:
        f.write(source)

    return path





"""
Functions for weighting signals
"""
def weights_proportional_centered(ranks):
    # center at 0.5 so values in [-0.5, 0.5], then normalize sum(abs(w)))=1
    w_raw = ranks.sub(0.5, axis=0)
    denom = w_raw.abs().sum(axis=1).replace(0, np.nan)
    w = w_raw.div(denom, axis=0).fillna(0.0)
    return w

def weights_threshold(ranks, low_q=0.1, high_q=0.9):
    """
    Assume ranks are percentiles in [0,1] where LARGER rank -> more long signal.
    Long = top quantile (>= high_q), short = bottom quantile (<= low_q).
    """
    long = (ranks >= high_q).astype(float)
    short = (ranks <= low_q).astype(float)
    w_raw = long - short
    denom = w_raw.abs().sum(axis=1).replace(0, np.nan)
    w = w_raw.div(denom, axis=0).fillna(0.0)
    return w

def weights_zscore(signal):
    # signal: T x N (not necessarily ranks). zscore per row then normalize
    z = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1).replace(0, np.nan), axis=0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = z.abs().sum(axis=1).replace(0, np.nan)
    w = z.div(denom, axis=0).fillna(0.0)
    return w

