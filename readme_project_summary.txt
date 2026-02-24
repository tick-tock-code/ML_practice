Aims:
This project investigates an alpha using momentum/reversion of returns 
from in large-cap US equities using daily OHLCV data.

Data:
 - Top 100 S&P Constituents (Wikipedia)
 - Historical prices via Yahoo Finance
 - Past 4 years of market data used


Methodology:
- Daily cross-sectional ranking by past k-day returns
- Long-short investment from linear ranking (-0.5 to +0.5)
- Daily rebalancing (with decay)
- Transaction costs: 5 bps per turnover
- Iterated parameters: 
   Strategy kind: mean reversion or momentum
   Lookback window: time window (days) for calculating mean 
   Volatility window: time window (days) for calculating volatility
   Partial rebalance: for including decay to instrument weightings


Key Results:
- Annualized Sharpe: ~0.75 for  {momentum strategy, 252 day lookback, 10 day volatility window, partial rebalance = 0.18}
- Annualised Returns: 4.79%
- Transaction costs as a percentage of traded volume restrict returns for highly fluctuating signals.



