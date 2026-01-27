"""
Docstring for Running_program
"""

# Importing modules
import Data_Ingestion as di
import Data_processing as dp
import globals


import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import time


# Directories
script_dir = globals.find_script_directory()
folder_dir = os.path.join( os.path.dirname ( __file__), os.path.pardir)

# Creating folder to house data
globals.create_check_folder("data", folder_dir)
data_dir = os.path.join(folder_dir, "data")


# Defining params
ingesting_new_data = False  # Set to False to skip data ingestion
days_ago = 252  # Number of days of data to retrieve
top_n  = 10  # Number of top tickers to retrieve from S&P 500

# Params for signal generation
strategy_kind = "momentum"  # Type of strategy: "momentum" or "mean_reversion"
vol_window = 20  # Window for volume normalization
eps = 1e-8  # Small constant to avoid division by zero
mean_reversion_lookback = 5  # Lookback period for mean reversion 

# Params for backtesting
tc_bps = 5.0  # Transaction costs in basis points
days_per_year = 252  # Trading days in a year

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
    processed_data = dp.compute_returns(raw_data, vol_window=vol_window, eps=eps)
    processed_data.to_csv(os.path.join(data_dir, "2_processed_data.csv"))


    # Compute cross-sectional signals
    print("Computing cross-sectional signals...")
    signals = dp.cross_sectional_signal(processed_data, kind=strategy_kind, lookback=mean_reversion_lookback, vol_window=vol_window)
    signals.to_csv(os.path.join(data_dir, "3_signals.csv"))


    # Backtesting
    print("Backtesting signals...")
    backtest_results = dp.backtest_long_short(signals, processed_data, tc_bps=tc_bps, days_per_year=days_per_year)
    
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

if __name__ == "__main__":
    main()