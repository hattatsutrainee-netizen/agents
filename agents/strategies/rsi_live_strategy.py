"""Live RSI mean-reversion strategy for domestic JP crypto exchanges.

Accumulates close prices from ticker polling to build a rolling RSI window.
Trades only from FLAT → LONG (on oversold) → FLAT (on overbought), keeping
position management simple for the incubation phase.

State machine:
  FLAT + RSI < oversold  → place BUY limit at bid  → PENDING_BUY
  PENDING_BUY + filled   → LONG
  LONG + RSI > overbought → place SELL limit at ask → PENDING_SELL
  PENDING_SELL + filled  → FLAT  (PnL recorded)
  any state + order stale → cancel and reset to FLAT
"""
from __future__ import annotations

import logging
import time
from typing import Any

from agents.api_clients import DomesticClient

logger = logging.getLogger(__name__)

_State = str  # "flat" | "pending_buy" | "long" | "pending_sell"


class LiveRsiStrategy:
    def __init__(
        self,
        client: DomesticClient,
        symbol: str,
        order_size: float,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        stale_after: int = 10,
        dry_run: bool = True,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.order_size = order_size
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stale_after = stale_after
        self.dry_run = dry_run

        self._prices: list[float] = []
        self._state: _State = "flat"
        self._open_order_id: str | None = None
        self._open_order_price: float = 0.0
        self._open_order_age: int = 0
        self._entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0

    def step(self) -> dict[str, Any]:
        ticker = self.client.fetch_ticker(self.symbol)
        bid = ticker["bid"]
        ask = ticker["ask"]
        mid = (bid + ask) / 2

        self._prices.append(mid)
        keep = self.rsi_period * 3
        if len(self._prices) > keep:
            self._prices = self._prices[-keep:]

        rsi: float | None = None
        if len(self._prices) > self.rsi_period:
            rsi = _calc_rsi(self._prices, self.rsi_period)

        state = self._state
        result: dict[str, Any] = {
            "state": state,
            "mid": mid,
            "rsi": rsi,
            "prices_collected": len(self._prices),
            "warmup_remaining": max(0, self.rsi_period + 1 - len(self._prices)),
            "realized_pnl": self.realized_pnl,
            "trade_count": self.trade_count,
        }

        if rsi is None:
            return result

        # --- check for fill ---
        if self._open_order_id and state in ("pending_buy", "pending_sell"):
            self._open_order_age += 1
            open_ids = self._fetch_open_ids()
            if self._open_order_id not in open_ids:
                if state == "pending_buy":
                    self._entry_price = self._open_order_price
                    self._state = "long"
                    logger.info("BUY filled @ %.0f", self._entry_price)
                else:
                    pnl = (self._open_order_price - self._entry_price) * self.order_size
                    self.realized_pnl += pnl
                    self.trade_count += 1
                    logger.info("SELL filled @ %.0f  pnl=%.2f  total_pnl=%.2f",
                                self._open_order_price, pnl, self.realized_pnl)
                    self._state = "flat"
                self._open_order_id = None
            elif self._open_order_age >= self.stale_after:
                self._cancel_open()
                self._state = "flat"
                logger.info("Order stale after %d steps, cancelled", self.stale_after)

        # --- signal logic ---
        if self._state == "flat" and rsi < self.oversold:
            oid = self._post("buy", bid)
            if oid:
                self._open_order_id = oid
                self._open_order_price = bid
                self._open_order_age = 0
                self._state = "pending_buy"
                logger.info("RSI %.1f < %.1f  → BUY limit @ %.0f (id=%s)", rsi, self.oversold, bid, oid)

        elif self._state == "long" and rsi > self.overbought:
            oid = self._post("sell", ask)
            if oid:
                self._open_order_id = oid
                self._open_order_price = ask
                self._open_order_age = 0
                self._state = "pending_sell"
                logger.info("RSI %.1f > %.1f  → SELL limit @ %.0f (id=%s)", rsi, self.overbought, ask, oid)

        result["state"] = self._state
        result["open_order_id"] = self._open_order_id
        return result

    def _fetch_open_ids(self) -> set[str]:
        if self.dry_run:
            return {self._open_order_id} if self._open_order_id else set()
        orders = self.client.fetch_open_orders(self.symbol)
        return {str(o["id"]) for o in orders if o.get("id")}

    def _post(self, side: str, price: float) -> str | None:
        if self.dry_run:
            oid = f"dry-{side}-{int(time.time()*1000)}"
            logger.info("[dry_run] %s %.6f @ %.0f → %s", side, self.order_size, price, oid)
            return oid
        resp = self.client.create_limit_order(self.symbol, side, self.order_size, price)
        oid = resp.get("id")
        return str(oid) if oid else None

    def _cancel_open(self) -> None:
        if self._open_order_id is None:
            return
        if self.dry_run:
            logger.info("[dry_run] cancel %s", self._open_order_id)
        else:
            try:
                self.client.cancel_order(self._open_order_id, symbol=self.symbol)
            except Exception as e:
                logger.warning("cancel failed: %s", e)
        self._open_order_id = None


def _calc_rsi(prices: list[float], period: int) -> float:
    changes = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0.0
    avg_loss = sum(losses[-period:]) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
