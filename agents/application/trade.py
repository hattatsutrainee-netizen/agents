import os
import shutil

from agents.api_clients import DomesticClient
from agents.strategies.maker_sniper_strategy import MakerSniperStrategy


class Trader:
    def __init__(self):
        self.client = DomesticClient()

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
        """Run one cycle of the maker-sniper strategy against the domestic exchange."""
        self.pre_trade_logic()
        symbol = os.getenv("DOMESTIC_SYMBOL", "BTC/JPY")
        size = float(os.getenv("DOMESTIC_ORDER_SIZE", "0.001"))
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
