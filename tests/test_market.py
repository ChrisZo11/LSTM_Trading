import pandas as pd
from trading_bot.data.market import fetch_ohlcv
from trading_bot.data.store import save_ohlcv, load_ohlcv
import sqlite3
import os

def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv("AAPL", period="5d", interval="1d")
    assert isinstance(df, pd.DataFrame)
    
    # Needs to be lowercase
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    
    # Must have returned rows
    assert len(df) > 0


def test_fetch_ohlcv_columns_lowercase():
    df = fetch_ohlcv("MSFT", period="5d", interval="1d")
    assert all(c == c.lower() for c in df.columns)
    

def test_save_and_load_ohlcv(tmp_path, monkeypatch):
    """Test standard SQLite db upserts using an override path"""
    from trading_bot.data import store

    temp_db_path = str(tmp_path / "test_trading_bot.db")
    monkeypatch.setattr(store, "DB_PATH", temp_db_path)

    # 1. Fetch real market date
    df = fetch_ohlcv("AAPL", period="5d", interval="1d")
    
    # 2. Save it
    save_ohlcv("AAPL", df)
    assert os.path.exists(temp_db_path)
    
    # 3. Load it
    loaded_df = load_ohlcv("AAPL")
    
    # We should have the same number of rows and columns and index structure
    assert len(loaded_df) == len(df)
    assert list(loaded_df.columns) == list(df.columns)
    assert sum(loaded_df["close"]) == sum(df["close"])
