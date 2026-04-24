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

import time
from typing import Any

from agents.api_clients import DomesticClient


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
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.order_size = order_size
        self.tick_offset = tick_offset
        self.refresh_interval_sec = refresh_interval_sec
        self.post_only = post_only
        self.dry_run = dry_run

        self.open_bid_id: str | None = None
        self.open_ask_id: str | None = None
        self.open_bid_price: float | None = None
        self.open_ask_price: float | None = None
        self.filled_events: list[dict[str, Any]] = []
        self.realized_pnl: float = 0.0
        self._last_buy_price: float | None = None

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
        if desired_bid >= desired_ask:
            return {
                "status": "spread_too_tight",
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick": tick,
            }

        open_ids = self._fetch_open_ids()
        self._reconcile_fills(open_ids)
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


def _infer_tick(bids: list[list[float]], asks: list[list[float]]) -> float:
    prices = [float(p) for p, _ in bids] + [float(p) for p, _ in asks]
    prices = sorted(set(prices))
    if len(prices) < 2:
        return 0.0
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    positive = [d for d in diffs if d > 0]
    return min(positive) if positive else 0.0
