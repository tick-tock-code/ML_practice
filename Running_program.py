"""
Docstring for Running_program
"""

# Importing modules
import json
import Data_Ingestion as di
import Data_processing as dp
import globals


import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import time

# Backtesting (grid run)
import itertools
from pprint import pformat


# Directories
script_dir = globals.find_script_directory()
folder_dir = os.path.join( os.path.dirname ( __file__), os.path.pardir)

# Creating folder to house data
globals.create_check_folder("data", folder_dir)
data_dir = os.path.join(folder_dir, "data")


# Defining params
ingesting_new_data = False  # Set to False to skip data ingestion
days_ago = 3*252  # Number of days of data to retrieve
top_n  = 500  # Number of top tickers to retrieve from S&P 500
failed = ['BF.B', 'BRK.B']  # List of tickers that failed ingestion

# Params for signal generation
strategy_kind = "mean_reversion"  # Type of strategy: "momentum" or "mean_reversion"
vol_window = 20  # Window for volume normalization - unused at the moment
eps = 1e-8  # Small constant to avoid division by zero
mean_reversion_lookback = 10 # Lookback period for mean reversion 

# Params for trading
rebalance_every_n_days = [1, 5, 10]   # e.g., 5 for weekly, 21 for monthly. None => rebalance every day
no_trade_band = 0.0               # do not trade tickers where |target - current| <= band
partial_rebalance = np.linspace(0.1, 1.0, 10)       # fraction of the gap to trade on rebalance (0..1)
min_trade_size = 0.0              # ignore trades with abs size below this (absolute weight)

# Params for backtesting
tc_bps = 5  # Transaction costs in basis points
days_per_year = 252  # Trading days in a year



# Creating grid values and keys for backtesting
grid_keys = ['rebalance_every_n_days', 'partial_rebalance']
grid_values = [
    globals.listify(rebalance_every_n_days),
    globals.listify(partial_rebalance),
]

def main():
    if ingesting_new_data:
        print("Ingesting new data...")

        # Start date for data
        start_date = di.get_date_n_days_ago_str(days_ago)

        # Use data ingestion pipeline to get ticker data
        raw_data = di.ingest_equity_data(start_date, top_n=top_n, features_list=None, 
                                        tickers_list=None, back_date=0)

        raw_data.to_csv(os.path.join(data_dir, "1_raw_data.csv"))

    else:
        print("Loading existing data...")

        # Load existing data
        raw_data = pd.read_csv(os.path.join(data_dir, "1_raw_data.csv"), header=[0, 1], index_col=0, parse_dates=True)
    

    # Process data to compute returns
    print("Processing data...")
    processed_data = dp.compute_returns(raw_data.dropna(axis=1, how="all"), vol_window=vol_window, eps=eps)
    processed_data.to_csv(os.path.join(data_dir, "2_processed_data.csv"))


    # Compute cross-sectional signals
    print("Computing cross-sectional signals...")
    signals = dp.cross_sectional_signal(processed_data, kind=strategy_kind, lookback=mean_reversion_lookback, vol_window=vol_window)
    signals.to_csv(os.path.join(data_dir, "3_signals.csv"))


    # Running backtest grid
    print("Preparing parameter grid for backtests...")
    param_combinations = list(itertools.product(*grid_values))
    print(f"Running {len(param_combinations)} backtests...")

    summary_rows = []
    for run_index, combo in enumerate(param_combinations, start=1):
        params = dict(zip(grid_keys, combo))
        row, outdir = dp.run_and_record(run_index, params, signals, processed_data, tc_bps, days_per_year, strategy_kind, data_dir)
        summary_rows.append(row)



    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(data_dir, f"backtest_grid_summary_{strategy_kind}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved grid summary to {summary_csv}")

    # Quick plot of best run (by sharpe) if any results exist
    if not summary_df.empty:
        best_row = summary_df.sort_values('sharpe', ascending=False).iloc[0]
        best_outdir = str(best_row['outdir'])
        print(f"Best run by sharpe: {best_row['run_name']} (sharpe={best_row['sharpe']}). Files in: {best_outdir}")

        try:
            cum_index_path = os.path.join(best_outdir, "cum_index.csv")
            if not os.path.exists(cum_index_path):
                raise FileNotFoundError(f"cum_index.csv not found at {cum_index_path}")

            ci_df = pd.read_csv(cum_index_path, index_col=0, parse_dates=True)

            # if the CSV has a single column, take it as a Series; otherwise pick first numeric column
            if isinstance(ci_df, pd.DataFrame):
                if ci_df.shape[1] == 1:
                    ci = ci_df.iloc[:, 0]
                else:
                    numeric_cols = ci_df.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        ci = ci_df[numeric_cols[0]]
                    else:
                        ci = ci_df.iloc[:, 0]
            else:
                ci = pd.Series(ci_df)

            plt.figure(figsize=(10, 6))
            ci.plot(title=f"Best run: {best_row['run_name']}")
            plt.tight_layout()
            plt.savefig(os.path.join(best_outdir, "best_run_cum_index.png"))
            plt.show()
        except FileNotFoundError as e:
            print("Could not find cum_index.csv for best run:", e)
        except Exception as e:
            print("Could not plot best run (file missing or read error):", e)


        






    """
        backtest_results = dp.backtest_long_short(signals, processed_data, tc_bps=tc_bps, days_per_year=days_per_year,
                                              rebalance_every_n_days=rebalance_every_n_days,
                                              no_trade_band=no_trade_band,
                                              partial_rebalance=partial_rebalance,
                                              min_trade_size=min_trade_size)
    

        # Saving backtest results
    print("Backtest completed, saving results...")
    backtest_results['port_ret'].to_csv(os.path.join(data_dir, "4_backtest_results_portfolio_return.csv"))
    outdir = dp.save_backtest_results(backtest_results, base_dir=data_dir, 
                name=f"{strategy_kind}_backtest_top_n_{top_n}_lookback_{mean_reversion_lookback}")


    print(f"Saved backtest to {outdir}")



    # Plotting returns from simple returns backtest
    plt.figure(figsize=(10,6))
    (1 + backtest_results['port_ret']).cumprod().plot()
    plt.savefig(os.path.join(outdir, "backtest_cumulative_return.png"))
    plt.show()

     # Plotting returns from cumulative index backtest
    plt.figure(figsize=(10,6))
    plt.plot(backtest_results['port_ret'])
    plt.savefig(os.path.join(outdir, "backtest_return.png"))
    plt.show()   

    print("Hit rate:", backtest_results['stats']['hit_rate'])
    print("Sharpe ratio:", backtest_results['stats']['sharpe'])
"""
if __name__ == "__main__":
    main()