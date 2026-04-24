"""Domestic (Japanese) crypto exchange REST client.

Primary target: SBI VC Trade. Interface shape follows ccxt conventions
(fetch_ticker, fetch_order_book, create_limit_order, ...) so that a future
swap to a ccxt-supported venue is a single import change at call sites.

Transport: `requests` + HMAC-SHA256 signed headers for private endpoints.
No ccxt dependency — SBI VC Trade is not yet in ccxt.exchanges.

All endpoint paths below are marked TBD-verify: confirm against the current
SBI VC Trade official API documentation before enabling order placement.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class DomesticClientError(Exception):
    """Raised for non-2xx responses or transport failures."""


class DomesticClient:
    DEFAULT_BASE_URL = "https://api.sbivc.co.jp"

    # TBD-verify: path layout is a best-effort guess following common JP REST
    # conventions. Confirm with the SBI VC Trade API specification before use.
    _paths: dict[str, str] = {
        "ticker": "/api/v1/public/ticker",
        "depth": "/api/v1/public/depth",
        "balance": "/api/v1/private/account/balance",
        "order_create": "/api/v1/private/order",
        "order_cancel": "/api/v1/private/order/cancel",
        "open_orders": "/api/v1/private/orders/open",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("DOMESTIC_EXCHANGE_API_KEY", "")
        self.api_secret = api_secret if api_secret is not None else os.getenv("DOMESTIC_EXCHANGE_API_SECRET", "")
        self.base_url = (base_url or os.getenv("DOMESTIC_EXCHANGE_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    # ---------- public endpoints ----------

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        raw = self._public_get(self._paths["ticker"], params={"symbol": self._normalize_symbol(symbol)})
        return {
            "symbol": symbol,
            "bid": _to_float(raw.get("bid") or raw.get("bestBid")),
            "ask": _to_float(raw.get("ask") or raw.get("bestAsk")),
            "last": _to_float(raw.get("last") or raw.get("lastPrice")),
            "timestamp": raw.get("timestamp") or raw.get("ts"),
            "info": raw,
        }

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        raw = self._public_get(
            self._paths["depth"],
            params={"symbol": self._normalize_symbol(symbol), "limit": limit},
        )
        return {
            "symbol": symbol,
            "bids": [[_to_float(p), _to_float(s)] for p, s in _levels(raw.get("bids", []))],
            "asks": [[_to_float(p), _to_float(s)] for p, s in _levels(raw.get("asks", []))],
            "timestamp": raw.get("timestamp") or raw.get("ts"),
            "info": raw,
        }

    # ---------- private endpoints ----------

    def fetch_balance(self, params: dict | None = None) -> dict[str, Any]:
        raw = self._private_get(self._paths["balance"], params=params or {})
        free: dict[str, float] = {}
        used: dict[str, float] = {}
        total: dict[str, float] = {}
        for row in raw.get("assets", raw.get("balances", [])) or []:
            ccy = row.get("currency") or row.get("asset")
            if not ccy:
                continue
            f = _to_float(row.get("available") or row.get("free") or 0)
            u = _to_float(row.get("locked") or row.get("used") or 0)
            free[ccy] = f
            used[ccy] = u
            total[ccy] = f + u
        return {"free": free, "used": used, "total": total, "info": raw}

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: dict | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self._normalize_symbol(symbol),
            "side": side.upper(),
            "type": "LIMIT",
            "price": str(price),
            "size": str(amount),
        }
        if params:
            # post-only flag key name is TBD-verify. `postOnly` is the ccxt
            # convention; SBI VC Trade's real key may be `execution_type` or
            # `timeInForce`. The raw params are forwarded unchanged so the
            # caller can override once the correct key is known.
            payload.update(params)
        raw = self._private_post(self._paths["order_create"], payload)
        return _normalize_order(raw)

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: dict | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self._normalize_symbol(symbol),
            "side": side.upper(),
            "type": "MARKET",
            "size": str(amount),
        }
        if params:
            payload.update(params)
        raw = self._private_post(self._paths["order_create"], payload)
        return _normalize_order(raw)

    def cancel_order(self, id: str, symbol: str | None = None, params: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"orderId": id}
        if symbol is not None:
            payload["symbol"] = self._normalize_symbol(symbol)
        if params:
            payload.update(params)
        raw = self._private_post(self._paths["order_cancel"], payload)
        return {"id": id, "info": raw}

    def fetch_open_orders(self, symbol: str | None = None, params: dict | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = dict(params or {})
        if symbol is not None:
            query["symbol"] = self._normalize_symbol(symbol)
        raw = self._private_get(self._paths["open_orders"], params=query)
        rows = raw.get("orders", raw) if isinstance(raw, dict) else raw
        return [_normalize_order(r) for r in (rows or [])]

    # ---------- transport ----------

    def _public_get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", path, params=params, signed=False)

    def _private_get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", path, params=params, signed=True)

    def _private_post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, body=body, signed=True)

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        signed: bool = False,
    ) -> Any:
        url = self.base_url + path
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = {"Content-Type": "application/json"} if body else {}
        if signed:
            headers.update(self._sign(method, path, body_str))

        attempts = 0
        while True:
            attempts += 1
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    data=body_str if body else None,
                    headers=headers,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempts >= 2:
                    raise DomesticClientError(f"transport error: {e}") from e
                time.sleep(0.5)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "1"))
                if attempts >= 2:
                    raise DomesticClientError(f"rate limited: {resp.text}")
                time.sleep(retry_after)
                continue

            if not (200 <= resp.status_code < 300):
                raise DomesticClientError(f"HTTP {resp.status_code} {method} {path}: {resp.text}")

            try:
                return resp.json()
            except ValueError:
                return resp.text

    def _sign(self, method: str, path: str, body: str) -> dict[str, str]:
        """HMAC-SHA256 signing following the common JP-exchange convention.

        TBD-verify: header names (ACCESS-KEY / ACCESS-SIGN / ACCESS-TIMESTAMP)
        and the `ts + method + path + body` message layout are a common pattern
        (bitFlyer / Coincheck style) but SBI VC Trade may use `API-KEY` /
        `API-SIGN` or place the timestamp inside the body. Cross-check with
        the official document before the first private call.
        """
        ts = str(int(time.time() * 1000))
        message = ts + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": ts,
        }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # TBD-verify: SBI VC Trade's exact symbol format. Many JP venues use
        # underscore-separated (BTC_JPY). Accept ccxt-style BTC/JPY and convert.
        return symbol.replace("/", "_").upper()


def _levels(rows: Any) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for r in rows or []:
        if isinstance(r, dict):
            out.append((r.get("price"), r.get("size") or r.get("amount")))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            out.append((r[0], r[1]))
    return out


def _to_float(x: Any) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _normalize_order(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"id": None, "info": raw}
    return {
        "id": str(raw.get("orderId") or raw.get("id") or ""),
        "symbol": raw.get("symbol"),
        "side": (raw.get("side") or "").lower() or None,
        "type": (raw.get("type") or "").lower() or None,
        "price": _to_float(raw.get("price")),
        "amount": _to_float(raw.get("size") or raw.get("amount")),
        "status": raw.get("status"),
        "info": raw,
    }
