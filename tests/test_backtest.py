import pandas as pd
import numpy as np
from trading_bot.backtest.runner import run_backtest

def test_run_backtest_returns_metrics():
    # Make dummy 100 days
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Fake gradual price increases up to 100
    close = pd.Series(np.linspace(10, 100, 100), index=dates)
    
    # Let's say we bought on day 1 (price 10), and sold on day 99 (price 99)
    signals = pd.Series(["HOLD"] * 100, index=dates)
    signals.iloc[0] = "BUY"
    signals.iloc[98] = "SELL"
    
    metrics = run_backtest(close, signals, initial_capital=100.0)
    
    # Dict needs specific keys and floats
    for key in ["total_return_pct", "sharpe_ratio", "win_rate", "max_drawdown_pct"]:
        assert key in metrics
        assert isinstance(metrics[key], float)
        
    # We bought at 10 (got 10 shares w/ $100 starting capital)
    # Sold 10 shares at 99.09 for $990.9
    # Capital increases = roughly 890%
    assert metrics['total_return_pct'] > 500.0 
    
    # 1 trade was mathematically positive
    assert metrics['win_rate'] == 1.0 
