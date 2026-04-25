import os
import shutil
import logging

from agents.api_clients import DomesticClient
from agents.strategies.maker_sniper_strategy import MakerSniperStrategy
from agents.ai_analyst import FinGPTAnalyst

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, use_ai_analyst: bool = False):
        """Initialize trader with optional FinGPT AI analyst.

        Args:
            use_ai_analyst: If True, run FinGPT analysis before executing trades.
                           Defaults to False for backward compatibility.
        """
        self.client = DomesticClient()
        self.use_ai_analyst = use_ai_analyst
        self.ai_analyst: FinGPTAnalyst | None = None
        if use_ai_analyst:
            self.ai_analyst = FinGPTAnalyst(self.client)

    def pre_trade_logic(self) -> None:
        self.clear_local_dbs()

    def clear_local_dbs(self) -> None:
        try:
            shutil.rmtree("local_db_events")
        except:
            pass
        try:
            shutil.rmtree("local_db_markets")
        except:
            pass

    def one_best_trade(self) -> None:
        """Run one cycle of the maker-sniper strategy against the domestic exchange.

        If use_ai_analyst=True, runs FinGPT-style Chain of Thought analysis before execution.
        AI analyst gates the trade: if validation fails, execution is skipped.
        """
        self.pre_trade_logic()
        symbol = os.getenv("DOMESTIC_SYMBOL", "BTC/JPY")
        size = float(os.getenv("DOMESTIC_ORDER_SIZE", "0.001"))

        # Fetch market data for both strategy and AI analysis
        orderbook = self.client.fetch_order_book(symbol, limit=5)
        ticker = self.client.fetch_ticker(symbol)
        current_price = (ticker["bid"] + ticker["ask"]) / 2.0
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe="1h", limit=50)

        # Run AI analyst if enabled
        if self.use_ai_analyst and self.ai_analyst:
            try:
                signal = self.ai_analyst.analyze_market(
                    symbol=symbol,
                    current_price=current_price,
                    orderbook=orderbook,
                    ohlcv=ohlcv,
                    macro_context="",
                )
                logger.info(
                    "AI analyst signal: action=%s price=%.0f size=%.6f confidence=%.2f intensity=%d gap=%.2f valid=%s",
                    signal.action,
                    signal.price,
                    signal.size,
                    signal.confidence,
                    signal.intensity,
                    signal.expectation_gap,
                    signal.validation.passes_validation,
                )

                # If AI analyst says NO_TRADE, skip execution
                if signal.action == "NO_TRADE":
                    logger.info(
                        "AI analyst NO_TRADE: validation failed (suff=%.2f cons=%.2f action=%.2f), skipping execution",
                        signal.validation.data_sufficiency,
                        signal.validation.consistency,
                        signal.validation.actionability,
                    )
                    return

            except Exception as e:
                logger.error("AI analyst error: %s, falling back to strategy", e, exc_info=True)

        # Run maker-sniper strategy
        strategy = MakerSniperStrategy(self.client, symbol=symbol, order_size=size)
        state = strategy.step()
        print("step:", state)

    def maintain_positions(self):
        pass

    def incentive_farm(self):
        pass


if __name__ == "__main__":
    t = Trader()
    t.one_best_trade()
