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

    # --- PHASE 1: GLOBAL EVALUATION ---
    print("\n[Pipeline] ━━━━━━━━ Phase 1: Global Market Evaluation ━━━━━━━━")
    predictions = []
    
    for symbol in SYMBOLS:
        print(f"[Phase 1] Analyzing {symbol}...")
        model_path = os.path.join(MODEL_DIR, f"lstm_{symbol.replace('/', '_')}.pth")

        raw = fetch_ohlcv(symbol, period=HISTORY_PERIOD, interval=DATA_INTERVAL)
        if raw.empty:
            print(f"          ⚠️ No data pulled. Skipping.")
            continue

        features = compute_features(raw)
        X, y = create_sequences(features, sequence_length=SEQUENCE_LENGTH)
        
        if len(X) == 0:
            print(f"          ⚠️ Features too small for LSTM. Skipping.")
            continue

        if retrain or not os.path.exists(model_path):
            print(f"          🧠 No existing model found. Re-training LSTM Weights...")
            train_model(X, y, symbol=symbol, model_path=model_path)
            
        # Predict
        live_sequence = X[-1:] 
        ml_signal, confidence = predict_signal(live_sequence, symbol=symbol, model_path=model_path)
        
        # Pull Execution Data
        latest_price = float(features["close"].iloc[-1])
        current_owned = executor.get_position_qty(symbol)
        
        predictions.append({
            "symbol": symbol,
            "signal": ml_signal,
            "confidence": confidence,
            "price": latest_price,
            "owned": current_owned
        })
        print(f"          -> Signal: {ml_signal} | Confidence: {confidence:.2%} | Owned: {current_owned}")

    # --- PHASE 2: RANKED CAPITAL ALLOCATION ---
    print("\n[Pipeline] ━━━━━━━━ Phase 2: Portfolio Rebalancing & Execution ━━━━━━━━")
    
    if not predictions:
        print("[Pipeline] ✅ No valid predictions generated in Phase 1. Complete.")
        return
        
    # Sort predictions by confidence descending
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_asset = predictions[0]
    
    print(f"🏆 Top Ranked Strategy: {top_asset['symbol']} with {top_asset['confidence']:.2%} [{top_asset['signal']}]")

    # Rebalance: If the Top Asset is a BUY, forcefully liquidate any weak assets we own
    if "BUY" in top_asset["signal"]:
        for asset in predictions:
            if asset["symbol"] != top_asset["symbol"] and asset["owned"] > 0.0001:
                print(f"          💸 Rebalance: Liquidating {asset['owned']} shares of {asset['symbol']} to rotate capital...")
                executor.place_order(symbol=asset["symbol"], signal="SELL", qty=asset["owned"])
                
                # Recalculate portfolio size so the Top Asset can buy with the new liquidated cash
                portfolio_value = executor.get_portfolio_value()
                risk = RiskManager(portfolio_value=portfolio_value)

    # Standard Execution Loop
    for asset in predictions:
        symbol = asset["symbol"]
        signal = asset["signal"]
        qty = asset["owned"]
        price = asset["price"]
        
        # How many CAN we buy via the 10% Risk rule?
        allowed_buy_qty = risk.position_size(price=price)
        
        if signal == "SELL" and qty <= 0.0001:
            print(f"          ⏭️ {symbol}: Signal is SELL, but None Owned (Shorting aborted).")
            continue
            
        if signal != "HOLD":
            trade_qty = allowed_buy_qty
            
            # If it's a SELL, we dump everything we own.
            if signal == "SELL":
                trade_qty = qty
            # if it's a BUY, but NOT the top asset, suppress it!
            elif symbol != top_asset["symbol"]:
                print(f"          ⏭️ {symbol}: Signal is {signal}, but suppressing to concentrate capital into {top_asset['symbol']}")
                continue
                
            if trade_qty > 0.0001:
                print(f"          💸 Executing {signal} for `{trade_qty}` {symbol} at est. ${price:.2f}")
                executor.place_order(symbol=symbol, signal=signal, qty=trade_qty)
            else:
                print(f"          ⏭️ {symbol}: Risk limit is $0 or too little capital.")
        else:
            print(f"          ⏭️ {symbol}: Signal is HOLD.")
            
    print("\n[Pipeline] ✅ Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Trading Bot Run Loop")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of LSTM models on startup")
    parser.add_argument("--loop", action="store_true", help="Run automatically in an endless loop waiting for the next day.")
    args = parser.parse_args()
    
    if args.loop:
        print("Infinite Loop Triggered. The bot will automatically run every 60 Minutes.")
        while True:
            run_pipeline(retrain=args.retrain)
            print("[Loop] Sleeping until the next hour...")
            # 3600 seconds = 1 Hour
            time.sleep(3600) 
    else:
        run_pipeline(retrain=args.retrain)
