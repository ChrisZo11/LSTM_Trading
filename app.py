import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from config import SYMBOLS, DATA_INTERVAL, HISTORY_PERIOD, MODEL_DIR, SEQUENCE_LENGTH
from trading_bot.data.market import fetch_ohlcv
from trading_bot.features.engineer import compute_features, create_sequences
from trading_bot.models.trainer import train_model
from trading_bot.models.predictor import predict_signal
from trading_bot.signals.news_sentiment import fetch_headlines, analyze_sentiment, override_signal
from trading_bot.backtest.runner import run_backtest
from pipeline import run_pipeline

st.set_page_config(page_title="LSTM Trading Bot", layout="wide", page_icon="📈")

st.title("🤖 PyTorch LSTM Trading Bot")
st.markdown("Monitor AI metrics, view live charts, execute Backtests, and trigger paper trades.")

# Sidebar Controls
st.sidebar.header("Navigation")
selected_symbol = st.sidebar.selectbox("Select Market", SYMBOLS)


# --------------------------------------------------------------------
# Core Logic Execution for Dashboard
# --------------------------------------------------------------------
@st.cache_data(ttl=300) # cache data for 5 minutes
def load_and_process(symbol):
    # Fetch Data
    raw = fetch_ohlcv(symbol, period=HISTORY_PERIOD, interval=DATA_INTERVAL)
    if raw.empty:
        return None, None, None, None, "No Data found from Yahoo Finance"
        
    features = compute_features(raw)
    X, y = create_sequences(features, sequence_length=SEQUENCE_LENGTH)
    
    if len(X) == 0:
        return raw, features, None, None, "Not enough rows to build LSTM Sequence"
        
    return raw, features, X, y, None


raw_df, features_df, X_seq, y_seq, err = load_and_process(selected_symbol)

if err:
    st.error(err)
else:
    # --------------------------------------------------------------------
    # Model Status Check
    # --------------------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, f"lstm_{selected_symbol.replace('/', '_')}.pth")
    model_exists = os.path.exists(model_path)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model Status")
        if model_exists:
            st.success("✅ PyTorch Weights Found")
            
            # Place button here in main view
            st.markdown("---")
            execute_action = st.button("▶️ Execute Live Pipeline on ALL symbols", use_container_width=True)
            if execute_action:
                with st.spinner("Executing Live Pipeline. Check terminal for Alpaca receipts..."):
                    run_pipeline()
                st.success("Pipeline Run Completed.")
                
        else:
            st.warning("⚠️ No PyTorch Weight Model exists for this symbol today.")
            if st.button("🧠 Train PyTorch Model Now", use_container_width=True):
                with st.spinner(f"Training LSTM Model for {selected_symbol}. This takes ~10 seconds..."):
                    train_model(X_seq, y_seq, symbol=selected_symbol, model_path=model_path)
                st.success("Training Complete! The page will now refresh.")
                st.rerun()
            
    # --------------------------------------------------------------------
    # Live Predictions & News Sentiment
    # --------------------------------------------------------------------
    if model_exists:
        # Load the last known tensor in the sequence array
        live_frame = X_seq[-1:] 
        ml_signal, confidence = predict_signal(live_frame, symbol=selected_symbol, model_path=model_path)
        
        with col2:
            st.subheader("Deep Learning NLP Output")
            st.metric(label="PyTorch Target Bias:", value=f"{ml_signal} ({confidence:.1%})")
            
        with col3:
            st.subheader("News Sentiment Rating")
            st.info("NewsAPI logic has been disabled by user request to preserve free quota limits.")
    else:
        st.info("Train the model via `python run_backtest.py` to unlock PyTorch Intelligence tracking.")

    st.markdown("---")

    # --------------------------------------------------------------------
    # Charts
    # --------------------------------------------------------------------
    st.subheader(f"Historical Charting - {selected_symbol}")
    
    # Render interactive OHLCV Candlesticks using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=raw_df.index,
        open=raw_df['open'],
        high=raw_df['high'],
        low=raw_df['low'],
        close=raw_df['close'],
        name="Price"
    )])
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------
    # Backtester
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Historical Context")
    
    if st.button("Run Fast Backtest"):
        if not model_exists:
             st.error("You must run a python training cycle before backtesting predictions")
        else:
            with st.spinner("Generating Inference on History..."):
                signals = []
                for i in range(len(X_seq)):
                    historical_frame = X_seq[i:(i + 1)] 
                    signal, conf = predict_signal(historical_frame, symbol=selected_symbol, model_path=model_path)
                    signals.append(signal)

                valid_dates = features_df.index[SEQUENCE_LENGTH:]
                close_series = features_df["close"].iloc[SEQUENCE_LENGTH:]
                signal_series = pd.Series(signals, index=valid_dates)
                
                metrics = run_backtest(close_series, signal_series)
                
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Total Python Simulation Return", f"{metrics['total_return_pct']}%")
                r2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
                r3.metric("System Win Rate", f"{metrics['win_rate'] * 100}%")
                r4.metric("Algorithm Max Drawdown", f"{metrics['max_drawdown_pct']}%")
