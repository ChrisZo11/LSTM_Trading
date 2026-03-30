from newsapi import NewsApiClient
from config import NEWS_API_KEY
import re


def fetch_headlines(symbol: str, max_headlines: int = 5) -> list[str]:
    """
    Fetch recent news headlines mentioning the stock symbol using NewsAPI.
    """
    if not NEWS_API_KEY or "your_newsapi_key_here" in NEWS_API_KEY:
        return []
    
    try:
        client = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Crypto strings like BTC/USD should just search "BTC"
        query = symbol.split("/")[0]
        
        response = client.get_everything(
            q=query, 
            language="en", 
            sort_by="publishedAt", 
            page_size=max_headlines
        )
        return [article.get("title", "") for article in response.get("articles", [])]
    
    except Exception as e:
        print(f"[NewsAPI] Failed to fetch headlines for {symbol}: {e}")
        return []


def analyze_sentiment(headlines: list[str]) -> float:
    """
    Basic rule-based Natural Language Processing to score headlines.
    Positive words increase score, Negative decrease.
    Returns a score from -1.0 (Bearish) to 1.0 (Bullish).
    """
    if not headlines:
        return 0.0

    # Very small but functional dictionary for momentum trading context
    bullish_words = {"surge", "beat", "higher", "record", "growth", "jump", "buy", "up", "bull", "rally", "outperform", "profit"}
    bearish_words = {"plunge", "miss", "lower", "drop", "collapse", "sell", "down", "bear", "crash", "underperform", "loss", "warning"}

    total_score = 0
    words_evaluated = 0

    for headline in headlines:
        # Lowercase and strip punctuation
        words = re.sub(r'[^\w\s]', '', headline.lower()).split()
        
        for word in words:
            if word in bullish_words:
                total_score += 1
                words_evaluated += 1
            elif word in bearish_words:
                total_score -= 1
                words_evaluated += 1

    # If no recognized keywords were found, it's neutral
    if words_evaluated == 0:
        return 0.0
        
    # Scale between -1.0 and 1.0 based on how many keywords were flagged
    return max(-1.0, min(1.0, total_score / words_evaluated))


def override_signal(ml_signal: str, headlines: list[str]) -> str:
    """
    Combines PyTorch AI output with basic NewsAPI logic.
    If the news is devastatingly bad, we don't Buy.
    """
    if ml_signal == "HOLD" or not headlines:
        return ml_signal
        
    sentiment = analyze_sentiment(headlines)
    
    # 0.5 is an arbitrary limit. If sentiment is < -0.5, the headlines are mostly pessimistic.
    if ml_signal == "BUY" and sentiment < -0.5:
        print(f"[Signal Override] {ml_signal} nullified -> HOLD due to extreme bearish headlines ({sentiment:.2f})")
        return "HOLD"
        
    if ml_signal == "SELL" and sentiment > 0.5:
        print(f"[Signal Override] {ml_signal} nullified -> HOLD due to extreme bullish headlines ({sentiment:.2f})")
        return "HOLD"
        
    return ml_signal
