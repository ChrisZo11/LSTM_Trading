from config import MAX_POSITION_PCT, DAILY_LOSS_LIMIT_PCT

class RiskManager:
    """
    Manages position sizing and halting logic based on daily losses.
    Provides number of shares to trade given current portfolio and a stock price.
    """

    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = MAX_POSITION_PCT,
        daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT,
    ):
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.daily_loss_limit = portfolio_value * daily_loss_limit_pct
        self._daily_loss_so_far = 0.0

    def position_size(self, price: float) -> int:
        """
        Calculate the number of whole shares affordable within the max_position_pct parameter.
        """
        max_dollars = self.portfolio_value * self.max_position_pct
        return int(max_dollars // price)

    def record_loss(self, amount: float) -> None:
        """
        Accumulates realized losses so RiskManager knows when to halt all further trades.
        """
        self._daily_loss_so_far += amount

    def is_halted(self) -> bool:
        """
        True if daily loss limit breached, meaning the bot should cease executing new entries today.
        """
        return self._daily_loss_so_far >= self.daily_loss_limit
