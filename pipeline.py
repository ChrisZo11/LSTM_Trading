"""
Main orchestration pipeline for live execution or daily cron jobs.
Fetches data -> Engineers features -> AI LSTM Predicts -> NewsAPI Overrides -> Alpaca Executes
"""
import os
import argparse
import time
from config import SYMBOLS, DATA_INTERVAL, HISTORY_PERIOD, MODEL_DIR, SEQUENCE_LENGTH
from trading_bot.data.market import fetch_ohlcv
from trading_bot.features.engineer import compute_features, create_sequences
from trading_bot.models.trainer import train_model
from trading_bot.models.predictor import predict_signal
from trading_bot.signals.news_sentiment import fetch_headlines, override_signal
from trading_bot.execution.risk import RiskManager
from trading_bot.execution.alpaca import AlpacaExecutor


def run_pipeline(retrain: bool = False):
    """
    Spins up the core LSTM paper trading pipeline end-to-end.
    Pass `--retrain` to rebuild the network weights from scratch before executing today's signal.
    """
    print("=" * 60)
    print("🚀 Initializing Live LSTM Trading Pipeline...")
    print("=" * 60)

    # 1. Boot up execution layers to poll liquid cash on Alpaca
    executor = AlpacaExecutor()
    portfolio_value = executor.get_portfolio_value()
    risk = RiskManager(portfolio_value=portfolio_value)

    if risk.is_halted():
        print("[Pipeline] 🛑 CRITICAL Risk Halt applied via Daily Loss limits.")
        print("          Algorithm refuses to trade further today.")
        return

    # Loop Over All Stocks in Config
    for symbol in SYMBOLS:
        print(f"\n[Pipeline] ━━━━━━━━ Processing {symbol} ━━━━━━━━")
        model_path = os.path.join(MODEL_DIR, f"lstm_{symbol.replace('/', '_')}.pth")

        # 2. Extract Data
        raw = fetch_ohlcv(symbol, period=HISTORY_PERIOD, interval=DATA_INTERVAL)
        if raw.empty:
            print(f"[Pipeline] ⚠️ No data pulled for {symbol}. Moving to next ticker.")
            continue

        # 3. Create Technical Time Series Features
        features = compute_features(raw)

        # 4. Generate sequences for PyTorch specifically
        X, y = create_sequences(features, sequence_length=SEQUENCE_LENGTH)
        
        if len(X) == 0:
            print(f"[Pipeline] ⚠️ Features DataFrame too small to create {SEQUENCE_LENGTH} length sequence for LSTM.")
            continue

        # 5. Train logic (Runs if told directly via args, or if file doesn't exist)
        if retrain or not os.path.exists(model_path):
            print(f"[Pipeline] 🧠 No existing model found. Re-training LSTM Weights for {symbol}...")
            train_model(X, y, symbol=symbol, model_path=model_path)
            
        print(f"[Pipeline] 🤖 Model successfully loaded.")

        # 6. Take the final frame from the Time Series and apply the loaded LSTM PyTorch model
        # We index the very last known period in our Dataframe:
        live_sequence = X[-1:] # Shape becomes (1, sequence_length, features)
        ml_signal, confidence = predict_signal(live_sequence, symbol=symbol, model_path=model_path)
        print(f"[Pipeline] 🎯 RAW ML Signal: {ml_signal} (Confidence: {confidence:.2%})")

        # 7. Apply NLP filtering layer to detect catastrophic headlines
        headlines = fetch_headlines(symbol)
        final_signal = override_signal(ml_signal, headlines)
        
        if final_signal != ml_signal:
             print(f"[Pipeline] ⚖️ OVERRIDE Applied: {ml_signal} -> {final_signal}. Headline sentiment strongly conflicted with AI.")
        else:
             print(f"[Pipeline] ⚖️ Signal Confirm: Headlines did not break parameters.")

        # 8. Filter via Risk and Submit
        latest_price = float(features["close"].iloc[-1])
        quantity = risk.position_size(price=latest_price)
        current_owned = executor.get_position_qty(symbol)
        
        if final_signal == "SELL" and current_owned <= 0.0001:
            print(f"[Pipeline] ⏭️ Skipping {symbol}: Signal is SELL, but no shares currently owned (Shorting aborted).")
        elif final_signal != "HOLD" and quantity > 0.0001:
            # If selling, sell EVERYTHING we currently own so we aren't short-selling the excess fraction
            if final_signal == "SELL":
                quantity = current_owned
                
            print(f"[Pipeline] 💸 Requesting: {final_signal} quantity `{quantity}` of {symbol} at estimated market price ${latest_price:.2f}")
            executor.place_order(symbol=symbol, signal=final_signal, qty=quantity)
        else:
            reason = "Signal is HOLD." if final_signal == "HOLD" else "Risk Manager permitted no shares."
            print(f"[Pipeline] ⏭️ Skipping {symbol}: {reason}")
            
    print("\n[Pipeline] ✅ Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Trading Bot Run Loop")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of LSTM models on startup")
    parser.add_argument("--loop", action="store_true", help="Run automatically in an endless loop waiting for the next day.")
    args = parser.parse_args()
    
    if args.loop:
        print("Infinite Loop Triggered. The bot will automatically run every 24 Hours.")
        while True:
            run_pipeline(retrain=args.retrain)
            print("[Loop] Sleeping until tomorrow...")
            # 86400 seconds = 24 Hours
            time.sleep(86400) 
    else:
        run_pipeline(retrain=args.retrain)
