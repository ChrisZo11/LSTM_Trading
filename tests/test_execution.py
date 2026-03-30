import pytest
from trading_bot.execution.risk import RiskManager

def test_position_size_math():
    portfolio = 10_000.0
    price_per_share = 100.0
    
    # 10% of $10_000 is $1000 limit. 
    # $1000 / $100 price is 10.0 valid shares
    rm = RiskManager(portfolio_value=portfolio, max_position_pct=0.10)
    qty = rm.position_size(price=price_per_share)
    
    assert qty == 10.0

def test_position_floors_at_zero():
    # Attempting to buy $500 stock with $10 in portfolio
    # 1.00 dollars / 500 = 0.002
    # Because $1.00 * 0.002 = $1.00 minimum limit met? No, the math 500 * 0.002 = 1.0, wait. 
    # Attempting to buy $500 stock with $10 portfolio means max risk is $1.00.
    # $1.00 / 500 is 0.002. Then qty * price is 0.002 * 500 = 1.0. This *exactly* meets the 1.0 boundary.
    # So wait, 500 stock with $9 in portfolio means max risk is 0.90 dollars.
    # Let me change portfolio to 9.0 to explicitly fail the $1 minimum.
    rm = RiskManager(portfolio_value=9.0, max_position_pct=0.10)
    qty = rm.position_size(price=500.0)
    
    # 0.90 dollars / 500 = 0.0018 shares. 500 * 0.0018 = $0.90, under $1 threshold.
    assert qty == 0.0

def test_daily_loss_limit_halts_algorithm():
    # Down $500 total before freezing (5%)
    rm = RiskManager(portfolio_value=10_000.0, max_position_pct=0.10, daily_loss_limit_pct=0.05)
    
    assert rm.is_halted() is False
    rm.record_loss(400.0)
    assert rm.is_halted() is False
    
    rm.record_loss(101.0) # We hit $501 total loss
    assert rm.is_halted() is True
    
    
from trading_bot.signals.news_sentiment import analyze_sentiment, override_signal

def test_news_sentiment_math():
    headlines = [
         "AAPL continues historic run with massive surge over earnings record",
         "A great buy right now",
         "The stock is set to crash heavily soon after recent plunge"
    ]
    
    score = analyze_sentiment(headlines)
    # The words found are [surge, record, buy, crash, plunge]
    # Total of +3 bullish, -2 bearish
    # Final = (-2 + 3) / 5 keywords = 0.20
    assert score == (1 / 5)

def test_news_will_nullify_ml_signal():
    fake_headlines = ["Plunge crash loss down lower"]
    
    # ML says BUY! News says Terrible!
    result = override_signal("BUY", fake_headlines)
    
    # News should have forced a HOLD to prevent risk 
    assert result == "HOLD"
