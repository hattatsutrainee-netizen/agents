"""Maker-rebate sniper strategy for Japanese crypto exchanges.

The strategy continuously quotes post-only limit orders one tick inside the
top of book on each side. When the book moves, resting orders are cancelled
and re-posted at the new desired price. When a resting order disappears from
`fetch_open_orders`, it is treated as a fill and recorded for PnL.

The economic thesis: venues such as SBI VC Trade pay a maker rebate while
charging a taker fee. Keeping both quotes strictly inside the spread as
post-only captures the spread AND the maker rebate on each round trip.

IMPORTANT — post-only flag:
  The `postOnly` params key used here follows the ccxt convention. SBI VC
  Trade's native REST API MAY use a different key (e.g. `execution_type`,
  `timeInForce=PO`). If the exchange silently ignores an unknown flag, the
  order may post as a plain limit and potentially cross the book as a taker.
  Verify the correct flag name against the official API documentation, and
  run with `dry_run=True` first to validate the quote arithmetic.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.api_clients import DomesticClient

logger = logging.getLogger(__name__)


class MakerSniperStrategy:
    def __init__(
        self,
        client: DomesticClient,
        symbol: str,
        order_size: float,
        tick_offset: int = 1,
        refresh_interval_sec: float = 1.0,
        post_only: bool = True,
        dry_run: bool = False,
        target_ratio: float | None = None,
        enable_trailing_stop: bool = False,
        trailing_stop_atr_multiple: float = 1.5,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.order_size = order_size
        self.tick_offset = tick_offset
        self.refresh_interval_sec = refresh_interval_sec
        self.post_only = post_only
        self.dry_run = dry_run
        self.target_ratio = target_ratio

        self.open_bid_id: str | None = None
        self.open_ask_id: str | None = None
        self.open_bid_price: float | None = None
        self.open_ask_price: float | None = None
        self.filled_events: list[dict[str, Any]] = []
        self.realized_pnl: float = 0.0
        self._last_buy_price: float | None = None

        # Initialize Portfolio for inventory skewing if target_ratio is set
        self.portfolio: Any | None = None
        if target_ratio is not None:
            from agents.strategies.portfolio import Portfolio
            self.portfolio = Portfolio(client, symbol, target_ratio=target_ratio)

        # Initialize TrailingStop for dynamic exit control if enabled
        self.trailing_stop: Any | None = None
        if enable_trailing_stop:
            from agents.strategies.trailing_stop import TrailingStop
            self.trailing_stop = TrailingStop(
                client, symbol, trailing_atr_multiple=trailing_stop_atr_multiple
            )

    def step(self) -> dict[str, Any]:
        book = self.client.fetch_order_book(self.symbol, limit=5)
        bids, asks = book["bids"], book["asks"]
        if not bids or not asks:
            return {"status": "empty_book"}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        tick = _infer_tick(bids, asks)
        if tick <= 0:
            return {"status": "tick_unknown", "best_bid": best_bid, "best_ask": best_ask}

        desired_bid = best_bid + tick * self.tick_offset
        desired_ask = best_ask - tick * self.tick_offset

        # Apply inventory skewing if portfolio is configured
        if self.portfolio is not None:
            desired_bid, desired_ask = self._apply_inventory_skew(
                desired_bid, desired_ask, best_bid, best_ask, tick
            )

        if desired_bid >= desired_ask:
            return {
                "status": "spread_too_tight",
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick": tick,
            }

        # Check trailing stop before posting/cancelling orders
        if self.trailing_stop is not None:
            current_unrealized_pnl = self._calculate_unrealized_pnl()
            stop_level = self.trailing_stop.get_stop_level(current_unrealized_pnl)

            if stop_level is not None and current_unrealized_pnl < stop_level:
                logger.warning(
                    "trailing_stop triggered: unrealized=%.2f stop=%.2f",
                    current_unrealized_pnl, stop_level
                )
                self._close_all_positions()
                return {
                    "status": "trailing_stop_triggered",
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "unrealized_pnl": current_unrealized_pnl,
                    "stop_level": stop_level,
                }

        open_ids = self._fetch_open_ids()
        self._reconcile_fills(open_ids)
        # Update trailing stop with most recent filled event
        if self.trailing_stop is not None and self.filled_events:
            self.trailing_stop.update_with_filled_event(self.filled_events[-1])
        self._cancel_stale(desired_bid, desired_ask, tick)
        self._repost_missing(desired_bid, desired_ask)

        return {
            "status": "ok",
            "best_bid": best_bid,
            "best_ask": best_ask,
            "desired_bid": desired_bid,
            "desired_ask": desired_ask,
            "tick": tick,
            "open_bid_id": self.open_bid_id,
            "open_ask_id": self.open_ask_id,
            "fills": len(self.filled_events),
            "realized_pnl": self.realized_pnl,
            "inventory": self.portfolio.get_inventory_info() if self.portfolio else None,
        }

    def run(self) -> None:
        while True:
            try:
                state = self.step()
                print(state)
            except Exception as e:
                print(f"[maker_sniper] step error: {e}")
            time.sleep(self.refresh_interval_sec)

    # ---------- helpers ----------

    def _fetch_open_ids(self) -> set[str]:
        if self.dry_run:
            ids: set[str] = set()
            if self.open_bid_id:
                ids.add(self.open_bid_id)
            if self.open_ask_id:
                ids.add(self.open_ask_id)
            return ids
        orders = self.client.fetch_open_orders(self.symbol)
        return {str(o.get("id")) for o in orders if o.get("id")}

    def _reconcile_fills(self, open_ids: set[str]) -> None:
        if self.open_bid_id and self.open_bid_id not in open_ids:
            self._record_fill("buy", self.open_bid_price or 0.0)
            self.open_bid_id = None
            self.open_bid_price = None
        if self.open_ask_id and self.open_ask_id not in open_ids:
            self._record_fill("sell", self.open_ask_price or 0.0)
            self.open_ask_id = None
            self.open_ask_price = None

    def _record_fill(self, side: str, price: float) -> None:
        self.filled_events.append(
            {"side": side, "price": price, "size": self.order_size, "ts": time.time()}
        )
        if side == "buy":
            self._last_buy_price = price
        elif side == "sell" and self._last_buy_price is not None:
            self.realized_pnl += (price - self._last_buy_price) * self.order_size
            self._last_buy_price = None

    def _cancel_stale(self, desired_bid: float, desired_ask: float, tick: float) -> None:
        if (
            self.open_bid_id
            and self.open_bid_price is not None
            and abs(self.open_bid_price - desired_bid) >= tick
        ):
            self._cancel(self.open_bid_id)
            self.open_bid_id = None
            self.open_bid_price = None
        if (
            self.open_ask_id
            and self.open_ask_price is not None
            and abs(self.open_ask_price - desired_ask) >= tick
        ):
            self._cancel(self.open_ask_id)
            self.open_ask_id = None
            self.open_ask_price = None

    def _repost_missing(self, desired_bid: float, desired_ask: float) -> None:
        if self.open_bid_id is None:
            oid = self._post("buy", desired_bid)
            if oid is not None:
                self.open_bid_id, self.open_bid_price = oid, desired_bid
        if self.open_ask_id is None:
            oid = self._post("sell", desired_ask)
            if oid is not None:
                self.open_ask_id, self.open_ask_price = oid, desired_ask

    def _post(self, side: str, price: float) -> str | None:
        params = {"postOnly": True} if self.post_only else None
        if self.dry_run:
            dry_id = f"dry-{side}-{int(time.time() * 1000)}"
            print(f"[dry_run] {side} {self.order_size} @ {price} params={params} -> {dry_id}")
            return dry_id
        resp = self.client.create_limit_order(
            self.symbol, side, self.order_size, price, params=params
        )
        oid = resp.get("id")
        return str(oid) if oid else None

    def _cancel(self, order_id: str) -> None:
        if self.dry_run:
            print(f"[dry_run] cancel {order_id}")
            return
        self.client.cancel_order(order_id, symbol=self.symbol)

    def _apply_inventory_skew(
        self,
        desired_bid: float,
        desired_ask: float,
        best_bid: float,
        best_ask: float,
        tick: float,
    ) -> tuple[float, float]:
        """Apply asymmetric spread adjustment based on portfolio inventory.

        Args:
            desired_bid, desired_ask: Baseline prices (without skew)
            best_bid, best_ask: Market best bid/ask (for context)
            tick: Price tick size

        Returns:
            (skewed_bid, skewed_ask): Adjusted prices, or (desired_bid, desired_ask) on error
        """
        try:
            # Fetch current balance and price
            balance = self.client.fetch_balance()["total"]
            ticker = self.client.fetch_ticker(self.symbol)
            current_price = (ticker["bid"] + ticker["ask"]) / 2.0

            # Calculate current ratio and skew
            current_ratio = self.portfolio.calculate_ratio(balance, current_price)
            skew = self.portfolio.get_skew_factor(current_ratio)

            # Apply skew: skew_adjustment = skew * tick_offset * tick
            skew_adjustment = skew * self.tick_offset * tick

            # Compute skewed prices
            skewed_bid = desired_bid - skew_adjustment
            skewed_ask = desired_ask + skew_adjustment

            # Validate: ensure spread doesn't invert
            if skewed_bid >= skewed_ask:
                logger.warning(
                    "skew resulted in inverted spread (bid=%.0f >= ask=%.0f), using baseline",
                    skewed_bid,
                    skewed_ask,
                )
                return (desired_bid, desired_ask)

            # Log adjustment
            logger.info(
                "inventory_skew: ratio=%.3f skew=%.3f bid_adj=%.0f ask_adj=%.0f",
                current_ratio,
                skew,
                skewed_bid - desired_bid,
                skewed_ask - desired_ask,
            )

            return (skewed_bid, skewed_ask)

        except Exception as e:
            logger.error("failed to apply inventory skew: %s", e)
            return (desired_bid, desired_ask)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate current unrealized profit/loss.

        Returns:
            Unrealized PnL in quote currency. Positive = profit, Negative = loss.
        """
        if not self.filled_events or self._last_buy_price is None:
            return 0.0
        try:
            ticker = self.client.fetch_ticker(self.symbol)
            current_price = (ticker["bid"] + ticker["ask"]) / 2.0
            return (current_price - self._last_buy_price) * self.order_size
        except Exception as e:
            logger.error("failed to calculate unrealized PnL: %s", e)
            return 0.0

    def _close_all_positions(self) -> None:
        """Cancel all open orders to close positions."""
        if self.open_bid_id:
            self._cancel(self.open_bid_id)
            self.open_bid_id = None
            self.open_bid_price = None
        if self.open_ask_id:
            self._cancel(self.open_ask_id)
            self.open_ask_id = None
            self.open_ask_price = None


def _infer_tick(bids: list[list[float]], asks: list[list[float]]) -> float:
    prices = [float(p) for p, _ in bids] + [float(p) for p, _ in asks]
    prices = sorted(set(prices))
    if len(prices) < 2:
        return 0.0
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    positive = [d for d in diffs if d > 0]
    return min(positive) if positive else 0.0
