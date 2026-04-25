"""Portfolio tracker for inventory-based strategy skewing.

Computes Base/Quote asset ratio from exchange balances and calculates
the asymmetric spread adjustment factor based on deviation from target.
Used by MakerSniperStrategy to implement inventory skewing.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.api_clients import DomesticClient

logger = logging.getLogger(__name__)


class Portfolio:
    """Minimal portfolio tracker for inventory skewing.

    Computes Base/Quote ratio from exchange balances and calculates
    the asymmetric spread adjustment factor based on deviation from target.
    """

    def __init__(
        self,
        client: DomesticClient,
        symbol: str,
        target_ratio: float = 0.5,
    ) -> None:
        """Initialize Portfolio tracker.

        Args:
            client: DomesticClient instance for fetching balances
            symbol: Trading pair (e.g., "BTC/JPY")
            target_ratio: Target Base/(Base+Quote) ratio.
                         Default 0.5 means equal value in Base and Quote assets.
        """
        self.client = client
        self.symbol = symbol
        self.target_ratio = target_ratio
        self._last_price: float | None = None
        self._last_ratio: float | None = None
        self._last_skew: float | None = None

    def calculate_ratio(
        self,
        balance: dict[str, float],
        current_price: float,
    ) -> float:
        """Calculate current Base/Quote ratio.

        Args:
            balance: Balance dict from DomesticClient.fetch_balance()['total']
            current_price: Current midpoint price (Base/Quote)

        Returns:
            Ratio in [0, 1]: base_value / (base_value + quote_value)
            Returns target_ratio if total value is zero.
        """
        # Extract base and quote asset codes from symbol (e.g., "BTC/JPY")
        parts = self.symbol.split("/")
        if len(parts) != 2:
            logger.warning(
                "unexpected symbol format: %s (expected 'BASE/QUOTE')",
                self.symbol,
            )
            return self.target_ratio

        base_code = parts[0].upper()
        quote_code = parts[1].upper()

        # Extract amounts from balance dict
        base_amount = balance.get(base_code, 0.0)
        quote_amount = balance.get(quote_code, 0.0)

        # Convert to comparable units: both in base asset value
        base_value = base_amount
        if current_price > 0:
            quote_value = quote_amount / current_price
        else:
            logger.warning("invalid price: %s", current_price)
            return self.target_ratio

        # Calculate ratio
        total_value = base_value + quote_value
        if total_value <= 0:
            logger.debug("zero total portfolio value, returning target_ratio")
            return self.target_ratio

        ratio = base_value / total_value

        # Cache for debugging
        self._last_price = current_price
        self._last_ratio = ratio

        return ratio

    def get_skew_factor(self, current_ratio: float) -> float:
        """Calculate spread skew factor from ratio deviation.

        Args:
            current_ratio: Current Base/(Base+Quote) ratio

        Returns:
            Skew factor in [-∞, ∞] (approximately):
                Positive: hold too much Base → want to sell (skew asks down)
                Negative: hold too much Quote → want to buy (skew bids up)

        Formula:
            skew = (current_ratio - target_ratio) / target_ratio

        Examples:
            - If current_ratio = 0.6, target = 0.5:
              skew = (0.6 - 0.5) / 0.5 = 0.2 (excess 20% Base)
            - If current_ratio = 0.4, target = 0.5:
              skew = (0.4 - 0.5) / 0.5 = -0.2 (excess 20% Quote)
        """
        if self.target_ratio <= 0:
            logger.warning("invalid target_ratio: %s", self.target_ratio)
            return 0.0

        skew = (current_ratio - self.target_ratio) / self.target_ratio

        # Cache for logging/debugging
        self._last_skew = skew

        return skew

    def get_inventory_info(self) -> dict[str, Any]:
        """Return cached portfolio state for logging.

        Returns:
            Dictionary with current_ratio, target_ratio, skew_factor, last_price
        """
        return {
            "current_ratio": self._last_ratio,
            "target_ratio": self.target_ratio,
            "skew_factor": self._last_skew,
            "last_price": self._last_price,
        }
