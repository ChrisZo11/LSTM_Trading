"""
Module for fetching market data.
"""
import yfinance as yf
import pandas as pd


def fetch_ohlcv(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol using yfinance.
    Returns DataFrame with lowercase columns: open, high, low, close, volume.
    Index is a DatetimeIndex.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    # Rename columns to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]
    
    # Keep only main OHLCV cols, drop any rows missing data
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    
    # Ensure index is datetime with a proper name
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    
    return df
