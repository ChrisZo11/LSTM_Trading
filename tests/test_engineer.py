import pandas as pd
import numpy as np
from trading_bot.data.market import fetch_ohlcv
from trading_bot.features.engineer import compute_features, create_sequences
import pytest

def test_compute_features_adds_columns():
    df = fetch_ohlcv("AAPL", period="6mo", interval="1d")
    
    # We must have at least 150 rows otherwise dropna removes them all due to indicator warmup
    if len(df) < 150: 
        pytest.skip("Not enough data fetched for features test")

    features = compute_features(df)
    
    # Assert RSI or MACD exist based on whatever ta library generated that didn't error out
    assert "momentum_rsi" in features.columns
    assert "target" in features.columns

def test_compute_features_no_nan_after_dropna():
    df = fetch_ohlcv("AAPL", period="6mo", interval="1d")
    features = compute_features(df)
    assert not features.isnull().any().any()

def test_target_is_binary():
    df = fetch_ohlcv("AAPL", period="6mo", interval="1d")
    features = compute_features(df)
    
    # target should only be 0.0 or 1.0
    assert set(features["target"].unique()).issubset({0.0, 1.0})

def test_create_sequences_dimensions():
    # Make dummy feature matrix
    df_len = 100
    sequence_len = 20
    
    feature_cols = ["open", "high", "low", "close", "volume", "momentum_rsi"]
    data = {}
    for col in feature_cols:
         data[col] = np.random.randn(df_len)
    
    data["target"] = np.random.choice([0.0, 1.0], size=df_len)
    
    dummy_df = pd.DataFrame(data)
    
    X, y = create_sequences(dummy_df, sequence_length=sequence_len)
    
    # Should yield (N - sequence_len, sequence_len, NUM_FEATURES)
    
    assert X.shape[0] == df_len - sequence_len
    assert X.shape[1] == sequence_len
    assert X.shape[2] == len(feature_cols)
    assert y.shape[0] == df_len - sequence_len 
