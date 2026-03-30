"""
Run a historical backtest of the LSTM model against a symbol and output metrics.
Usage: python run_backtest.py --symbol AAPL
"""
import argparse
import os
import pandas as pd

from config import DATA_INTERVAL, HISTORY_PERIOD, MODEL_DIR, SEQUENCE_LENGTH
from trading_bot.data.market import fetch_ohlcv
from trading_bot.features.engineer import compute_features, create_sequences
from trading_bot.models.trainer import train_model
from trading_bot.models.predictor import predict_signal
from trading_bot.backtest.runner import run_backtest


def execute_backtest(symbol: str):
    """
    Downloads historical DataFrame for `symbol`, generates LSTM predictions over every 
    sequence matching sequence_length, and simulates a portfolio run natively through python.
    """
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol.replace('/', '_')}.pth")

    print(f"\n[Backtest] 📊 Phase 1: Validating dataset for {symbol}...")
    raw = fetch_ohlcv(symbol, period=HISTORY_PERIOD, interval=DATA_INTERVAL)
    if raw.empty:
        print(f"[Backtest] Error: Failed to fetch any Yahoo Finance data for {symbol}.")
        return

    print(f"[Backtest] ⚙️ Phase 2: Computing Features & LSTM Sequences...")
    features = compute_features(raw)
    X, y = create_sequences(features, sequence_length=SEQUENCE_LENGTH)
    
    # We must have enough records
    if len(X) == 0:
         print(f"[Backtest] Error: Need more history. Data contains {len(features)} rows, but Sequence is {SEQUENCE_LENGTH}.")
         return

    # Train a specific test model if needed
    if not os.path.exists(model_path):
        print(f"[Backtest] 🧠 Phase 3: No pre-existing PyTorch weights found. Training new model...")
        train_model(X, y, symbol=symbol, model_path=model_path)
    else:
        print(f"[Backtest] 🧠 Phase 3: Utilizing existing PyTorch weights: {symbol}...")

    print(f"[Backtest] 🤖 Phase 4: Performing Inference over {len(X)} historical units. This can take a moment...")
    
    signals = []
    
    for i in range(len(X)):
        # Pluck single 3D frame representing what the bot "saw" identically on that specific day
        historical_frame = X[i:(i + 1)] 
        
        signal, conf = predict_signal(historical_frame, symbol=symbol, model_path=model_path)
        signals.append(signal)

    # -----------------------------------------------------
    # We generated `len(X)` signals, but remember that the first `SEQUENCE_LENGTH` 
    # days were used functionally *to* generate the first signal. Thus the signals 
    # array only maps to the back part of the `features` Dataframe indexing.
    # -----------------------------------------------------
    valid_dates = features.index[SEQUENCE_LENGTH:]
    close_series = features["close"].iloc[SEQUENCE_LENGTH:]
    signal_series = pd.Series(signals, index=valid_dates)
    
    print(f"[Backtest] 🚀 Phase 5: Executing Strategy...")
    metrics = run_backtest(close_series, signal_series)
    
    # Render Report
    print(f"\n")
    print("=" * 60)
    print(f"📊 BACKTEST RESULTS: {symbol}".center(60))
    print(f"Tested Over: {len(X)} instances ({DATA_INTERVAL})".center(60))
    print("=" * 60)
    print(f"  • Total Return:     {metrics['total_return_pct']:>8.2f}%")
    print(f"  • Max Drawdown:     {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  • Annual Sharpe:    {metrics['sharpe_ratio']:>8.3f}")
    if metrics['win_rate'] > 0:
        print(f"  • Win Rate:         {metrics['win_rate'] * 100:>8.1f}%")
    else:
        print(f"  • Win Rate:         {0.0:>8.1f}%   (No profitable closed setups)")
        
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Backtester")
    parser.add_argument("--symbol", default="AAPL", help="Stock or Crypto Symbol to evaluate.")
    args = parser.parse_args()
    
    execute_backtest(args.symbol)
