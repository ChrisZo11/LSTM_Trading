import pandas as pd
import numpy as np
import ta
from config import PREDICTION_HORIZON, SEQUENCE_LENGTH


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw OHLCV DataFrame, compute technical indicators and build the binary target.
    
    Target = 1 if close price is higher PREDICTION_HORIZON bars ahead, else 0.
    Drops rows with NaN values resulting from indicator lookback periods and target shifting.
    """
    # Create copy to avoid SettingWithCopy warning
    out = df.copy()

    # Apply all technical analysis features using the ta library
    # (We are using vectorized=True which is faster)
    out = ta.add_all_ta_features(
        out,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=False,       # Let na exist so we drop them explicitly
        vectorized=True    
    )
    
    # We will pick a handful of well-known features for the LSTM rather than 90+ 
    # to prevent immediate overfitting on a small dataset
    feature_cols = [
        "open", "high", "low", "close", "volume",
        # Momentum
        "momentum_rsi", "momentum_macd", "momentum_macd_signal", "momentum_stoch", 
        # Volatility
        "volatility_bbhi", "volatility_bbli", "volatility_atr",
        # Trend
        "trend_sma_fast", "trend_sma_slow", "trend_adx", "trend_cci"
    ]
    
    # Check if any column failed to compute, and drop it from feature list 
    # if it's missing (failsafe for some data edge-cases)
    missing_cols = set(feature_cols) - set(out.columns)
    valid_feature_cols = [c for c in feature_cols if c not in missing_cols]

    # Target: 1 if future price > current price
    future_close = out["close"].shift(-PREDICTION_HORIZON)
    out["target"] = (future_close > out["close"]).astype(float) # Target needs to be numeric for PyTorch
    
    # Drop rows that don't have enough history for the indicators, or target shifted out of bounds
    out = out.dropna()
    
    # Return features and the target label. Target is always the last column.
    final_cols = valid_feature_cols + ["target"]
    return out[final_cols]


def create_sequences(df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH):
    """
    Convert a feature DataFrame into 3D Numpy Arrays (sequences) for an LSTM model.
    Returns:
       X: shape (samples, sequence_length, features)
       y: shape (samples,)
    """
    X, y = [], []
    
    # Assume the last column in df is 'target'
    feature_matrix = df.drop(columns=["target"]).values
    target_vector = df["target"].values

    if len(df) <= sequence_length:
        return np.array([]), np.array([])
        
    for i in range(len(df) - sequence_length):
        X.append(feature_matrix[i:(i + sequence_length)])
        y.append(target_vector[i + sequence_length])
        
    return np.array(X), np.array(y)
