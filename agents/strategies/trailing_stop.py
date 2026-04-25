"""Trailing stop loss manager for dynamic exit control.

Monitors unrealized PnL and adjusts stop loss levels based on volatility (ATR).
Implements the Freqtrade trailing stop pattern: as unrealized gains grow, the
stop loss trailing upward to lock in profits while avoiding unnecessary exits.

Core logic:
1. Calculate ATR (Average True Range) from past 14 candles
2. Set initial stop width = ATR × initial_stop_atr_multiple (e.g., 2.0)
3. Set trailing stop width = ATR × trailing_atr_multiple (e.g., 1.5)
4. As unrealized_pnl grows, trailing stop = unrealized_pnl - trailing_width
5. When unrealized_pnl < trailing_stop, trigger exit
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.api_clients import DomesticClient

logger = logging.getLogger(__name__)


class TrailingStop:
    """Dynamic trailing stop manager based on ATR and unrealized PnL.

    Uses Average True Range (14-candle SMA of true range) to calculate
    volatility-adaptive stop loss widths. Trailing stop rises with unrealized
    gains, protecting profits while avoiding whipsaw exits.
    """

    def __init__(
        self,
        client: DomesticClient,
        symbol: str,
        initial_stop_atr_multiple: float = 2.0,
        trailing_atr_multiple: float = 1.5,
        lookback_candles: int = 14,
    ) -> None:
        """Initialize trailing stop manager.

        Args:
            client: DomesticClient for OHLCV data fetching
            symbol: Trading pair (e.g., "BTC/JPY")
            initial_stop_atr_multiple: Initial stop width = ATR × this value
            trailing_atr_multiple: Trailing stop width = ATR × this value (narrower)
            lookback_candles: Number of candles for ATR calculation
        """
        self.client = client
        self.symbol = symbol
        self.initial_stop_atr_multiple = initial_stop_atr_multiple
        self.trailing_atr_multiple = trailing_atr_multiple
        self.lookback_candles = lookback_candles

        # Cached ATR and timestamp for update frequency control
        self._cached_atr: float | None = None
        self._atr_timestamp: float | None = None
        self._atr_update_interval_sec: float = 60.0  # Recalculate every 60s

        # Position tracking (from filled_events)
        self._entry_price: float | None = None
        self._entry_quantity: float = 0.0
        self._entry_timestamp: float | None = None

    def update_with_filled_event(self, filled_event: dict[str, Any]) -> None:
        """Update position tracking from a filled order.

        Args:
            filled_event: {"side": "buy"|"sell", "price": float, "size": float, "ts": float}
        """
        try:
            side = filled_event.get("side")
            price = filled_event.get("price", 0.0)
            size = filled_event.get("size", 0.0)
            ts = filled_event.get("ts", time.time())

            if side == "buy":
                # Record entry point
                self._entry_price = price
                self._entry_quantity = size
                self._entry_timestamp = ts
            elif side == "sell":
                # Position closed, reset tracking
                self._entry_price = None
                self._entry_quantity = 0.0
                self._entry_timestamp = None
        except Exception as e:
            logger.error("failed to update position from filled event: %s", e)

    def calculate_atr(self) -> float:
        """Calculate Average True Range from recent OHLCV data.

        Returns:
            ATR value in quote currency, or 0.0 if calculation fails.

        ATR = SMA(TrueRange, 14) where TrueRange = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        """
        try:
            # Check cache: only recalculate if interval expired
            now = time.time()
            if (
                self._cached_atr is not None
                and self._atr_timestamp is not None
                and now - self._atr_timestamp < self._atr_update_interval_sec
            ):
                return self._cached_atr

            # Fetch OHLCV data (1-minute or shortest available timeframe)
            ohlcv = self.client.fetch_ohlcv(
                self.symbol, timeframe="1m", limit=self.lookback_candles
            )
            if len(ohlcv) < self.lookback_candles:
                logger.warning(
                    "insufficient OHLCV data: got %d, need %d",
                    len(ohlcv),
                    self.lookback_candles,
                )
                return 0.0

            # Calculate True Range for each candle
            true_ranges = []
            for i, (ts, open_, high, low, close, volume) in enumerate(ohlcv):
                high = float(high)
                low = float(low)
                close = float(close)

                # Previous close (or open if first candle)
                prev_close = (
                    float(ohlcv[i - 1][4]) if i > 0 else float(ohlcv[i][1])
                )

                # True Range
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close),
                )
                true_ranges.append(tr)

            # ATR = SMA(True Range)
            atr = sum(true_ranges) / len(true_ranges)

            # Cache result
            self._cached_atr = atr
            self._atr_timestamp = now

            logger.debug("ATR calculated: %.2f", atr)
            return atr

        except Exception as e:
            logger.error("failed to calculate ATR: %s", e)
            return 0.0

    def get_stop_level(self, current_unrealized_pnl: float) -> float | None:
        """Calculate stop loss level based on unrealized PnL.

        Args:
            current_unrealized_pnl: Current unrealized profit/loss in quote currency

        Returns:
            Stop loss level (if PnL is positive), or None if no position.
            When current_unrealized_pnl falls below this level, exit signal.

        Logic:
            - If unrealized_pnl <= 0: no position or in loss → return None
            - If unrealized_pnl < initial_stop_width: return 0 (protect any profit)
            - Else: return unrealized_pnl - trailing_stop_width (trail the stop)
        """
        if current_unrealized_pnl <= 0:
            return None

        atr = self.calculate_atr()
        if atr <= 0:
            return None

        initial_stop_width = atr * self.initial_stop_atr_multiple
        trailing_stop_width = atr * self.trailing_atr_multiple

        # If profit is still small, use minimal stop (0) to protect even tiny gains
        if current_unrealized_pnl < initial_stop_width:
            return 0.0

        # Otherwise, trail: as PnL grows, stop rises with it
        stop_level = current_unrealized_pnl - trailing_stop_width
        return max(0.0, stop_level)

    def get_trailing_info(self) -> dict[str, Any]:
        """Return cached trailing stop state for logging/debugging.

        Returns:
            Dictionary with entry_price, entry_quantity, atr, stop_level (if applicable)
        """
        return {
            "entry_price": self._entry_price,
            "entry_quantity": self._entry_quantity,
            "entry_timestamp": self._entry_timestamp,
            "cached_atr": self._cached_atr,
            "atr_last_update": self._atr_timestamp,
        }
