# LSTM Trading Bot

An LSTM-based paper trading bot that uses PyTorch for price direction prediction and NewsAPI for sentiment analysis, executing trades via Alpaca's paper trading API.

## Architecture

```
Data (yfinance) → Features (pandas-ta) → LSTM Model (PyTorch) → News Sentiment → Risk Manager → Alpaca Paper Trading
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Alpaca and NewsAPI keys
```

## Usage

```bash
# Run a backtest
python run_backtest.py --symbol AAPL

# Run the live paper trading pipeline
python pipeline.py
```

## Project Structure

```
trading_bot/
├── data/          # Market data fetching & storage
├── features/      # Technical indicator engineering
├── models/        # LSTM model definition, training, prediction
├── signals/       # News-based sentiment scoring
├── execution/     # Risk management & Alpaca order execution
└── backtest/      # Vectorized backtester
```

## Requirements

- Python 3.14+
- Alpaca paper trading account
- NewsAPI key
