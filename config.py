# config.py
"""
Central configuration for the LSTM Trading Bot.
All settings are loaded from environment variables (.env) or use sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Project root (so relative paths work everywhere)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# Markets to trade
# ──────────────────────────────────────────────
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
DATA_INTERVAL = "1h"           # 1-hour candlesticks instead of Daily
HISTORY_PERIOD = "60d"         # Yahoo Finance limits how much intraday data you can pullback to fetch
PREDICTION_HORIZON = 1         # predict N bars ahead

# ──────────────────────────────────────────────
# LSTM model hyper-parameters
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 60           # lookback window (bars fed to LSTM)
HIDDEN_SIZE = 128              # LSTM hidden units
NUM_LAYERS = 2                 # stacked LSTM layers
DROPOUT = 0.2                  # dropout between LSTM layers
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

# ──────────────────────────────────────────────
# Risk settings
# ──────────────────────────────────────────────
MAX_POSITION_PCT = 0.10        # max 10% of portfolio per position
DAILY_LOSS_LIMIT_PCT = 0.05    # halt trading if down 5% in a day

# ──────────────────────────────────────────────
# Alpaca API (paper trading)
# ──────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# ──────────────────────────────────────────────
# NewsAPI
# ──────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ──────────────────────────────────────────────
# Model storage
# ──────────────────────────────────────────────
MODEL_DIR = str(PROJECT_ROOT / "models" / "saved")
