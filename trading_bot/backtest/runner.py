import pandas as pd
import numpy as np


def run_backtest(close: pd.Series, signals: pd.Series, initial_capital: float = 10_000.0) -> dict:
    """
    Very fast, vectorized-style backtester mapping out trades based on historical signals.
    Unlike external packages, this natively supports Python 3.14 + pandas arrays instantly.
    
    close: pd.Series of close prices indexed by Date.
    signals: pd.Series of 'BUY', 'SELL', 'HOLD' indexed by Date (matching close index).
    Returns dict with: total_return_pct, sharpe_ratio, win_rate, max_drawdown_pct.
    """
    position = 0       # Current units held
    cash = initial_capital
    portfolio_values = []
    trade_returns = []

    prices = close.values
    sigs = signals.values

    # Step through every day (fast in native python arrays vs calling huge libraries)
    for i, (price, sig) in enumerate(zip(prices, sigs)):
        # Naive system: 100% allocation into a single position until Sell criteria met. 
        # (This differs from the isolated live RiskManager which scales dynamically).
        if sig == "BUY" and position == 0:
            position = int(cash // price)
            cash -= position * price
            
        elif sig == "SELL" and position > 0:
            proceeds = position * price
            
            # Trade return % = (Sell Value - Original Value @ index we bought at) / Original Value
            # For simplicity of this basic backtester, we just test vs the rolling previous average.
            trade_return = (proceeds - (position * prices[max(0, i - 1)])) / (position * prices[max(0, i - 1)])
            trade_returns.append(trade_return)
            
            cash += proceeds
            position = 0

        # Record Daily MTM (Mark-to-Market)
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    # 1. Total Return %
    portfolio_series = pd.Series(portfolio_values, index=close.index)
    total_return = (portfolio_series.iloc[-1] - initial_capital) / initial_capital * 100

    # 2. Daily Sharpe Ratio (Assuming Risk Free Rate = 0)
    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) 

    # 3. Win Rate
    win_rate = 0.0
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

    # 4. Maximum Drawdown %
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        "total_return_pct": round(float(total_return), 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "win_rate": round(float(win_rate), 3),
        "max_drawdown_pct": round(float(max_drawdown), 2),
    }
