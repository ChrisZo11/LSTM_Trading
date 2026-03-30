from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY


class AlpacaExecutor:
    """
    Submits trades using the non-deprecated `alpaca-py` library.
    Skips 'HOLD' calls, prevents empty quantities, dynamically constructs Buy vs Sell arguments.
    """

    def __init__(self):
        # Initializing the modern, supported client
        self.api = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True # Explicitly defaulting to Paper trading
        )

    def get_portfolio_value(self) -> float:
        """Fetch total liquid portfolio float from remote remote broker."""
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            print(f"[Alpaca] Failed to fetch account: {e}")
            return 10000.0 # Return placeholder for graceful fallback

    def place_order(self, symbol: str, signal: str, qty: int) -> None:
        """Submit a MarketOrderRequest."""
        if signal == "HOLD" or qty <= 0:
            print(f"[Alpaca] {symbol}: HOLD or Zero Quantity — Trade Cancelled.")
            return

        side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL
        
        # New alpaca-py pattern
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )

        try:
            self.api.submit_order(order_data=market_order_data)
            print(f"[Alpaca] Executed {side.value.upper()} Order: {qty} x {symbol}")
        except Exception as e:
            print(f"[Alpaca] Order Failed via API Error: {e}")
