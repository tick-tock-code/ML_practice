"""
01_2026 - Data Ingestion Module
"""

import yfinance as yf
import os
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt  # Now import pyplot
import time
import pandas as pd
import requests



""" --- Functions --- """
def read_html_with_ua(url: str) -> list[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return pd.read_html(r.text)


def get_top_market_cap_tickers(n: int) -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = read_html_with_ua(url)[0]
    tickers = sp500_table["Symbol"].head(n).tolist()
    return tickers



def get_nasdaq_list(filename = "nasdaq_list"):
    # Method 1: Using relative path (most robust)
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df = pd.read_csv(filepath)
    first_column = df.iloc[:, 0].tolist()
    print(first_column)
    return first_column

def get_stock_prices(start_date, list_of_tickers, end_date_back=0):

    if end_date_back == 0:
        # Get today's date
        today = datetime.date.today()

        # Convert today's date to a string in YYYY-MM-DD format
        end_date = today.strftime("%Y-%m-%d")

    else:
        # Get today's date
        today = datetime.date.today()

        # Calculate the date 60 working days ago
        sixty_working_days_ago = np.busday_offset(today, -end_date_back).astype(datetime.date)

        # Convert to string in YYYY-MM-DD format
        end_date = sixty_working_days_ago.strftime("%Y-%m-%d")



    # Download historical data for a specific stock (e.g., Vodafone - VOD.L)
    data = yf.download(list_of_tickers, start=start_date, end=end_date)
    if data.empty:
        print(f"Warning: No data")
        
    return data

def get_date_n_days_ago_str(n):
    """Returns the date n days ago as a string in 'YYYY-MM-DD' format."""
    today = datetime.date.today()
    delta = datetime.timedelta(days=n)
    past_date = today - delta
    return past_date.strftime('%Y-%m-%d')





"""
-- Main Ingestion Function --
"""
def ingest_equity_data(start_date, tickers_list = None, features_list = None, top_n = 50, back_date = 0):
    # Get list of tickers
    if tickers_list is None:
        tickers_list = get_top_market_cap_tickers(top_n)

    if features_list is None:
        feature_list = ["Close", "Open", "High", "Low", "Volume"]
    
    return get_stock_prices(start_date, list_of_tickers=tickers_list, end_date_back=back_date)