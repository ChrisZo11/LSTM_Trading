"""
Module for locally storing fetched market data.
"""
import sqlite3
import pandas as pd
from config import PROJECT_ROOT

DB_PATH = str(PROJECT_ROOT / "trading_bot.db")


def save_ohlcv(symbol: str, df: pd.DataFrame) -> None:
    """Upsert OHLCV rows into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df_copy = df.copy()
    
    # Add symbol column to track which ticker properties belong to
    df_copy["symbol"] = symbol
    
    # Save to table
    df_copy.to_sql("ohlcv", conn, if_exists="append", index=True)
    conn.close()


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load all stored OHLCV rows for a given symbol."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM ohlcv WHERE symbol=? ORDER BY datetime",
        conn,
        params=(symbol,),
        index_col="datetime",
        parse_dates=["datetime"],
    )
    conn.close()
    
    # Drop symbol col before returning so it matches fetched dataframe
    return df.drop(columns=["symbol"])
