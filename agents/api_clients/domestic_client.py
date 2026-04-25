"""bitbank REST API client for domestic JP crypto trading.

Target: bitbank (https://bitbank.cc)
Public API:  https://public.bitbank.cc
Private API: https://api.bitbank.cc/v1

Interface shape follows ccxt conventions so the strategy layer is exchange-agnostic.

Auth (ACCESS-NONCE method):
  Headers: ACCESS-KEY, ACCESS-NONCE, ACCESS-SIGNATURE
  GET  signature = HMAC-SHA256(secret, nonce + "/v1" + path + "?" + query)
  POST signature = HMAC-SHA256(secret, nonce + json_body)

Official docs: https://github.com/bitbankinc/bitbank-api-docs
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()


class DomesticClientError(Exception):
    """Raised for non-2xx responses or transport failures."""


class DomesticClient:
    PUBLIC_BASE_URL = "https://public.bitbank.cc"
    PRIVATE_BASE_URL = "https://api.bitbank.cc/v1"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("DOMESTIC_EXCHANGE_API_KEY", "")
        self.api_secret = api_secret if api_secret is not None else os.getenv("DOMESTIC_EXCHANGE_API_SECRET", "")
        self.session = session or requests.Session()
        self.timeout = timeout

    # ---------- public endpoints ----------

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        pair = self._normalize_symbol(symbol)
        raw = self._public_get(f"/{pair}/ticker")
        data = raw["data"]
        return {
            "symbol": symbol,
            # bitbank: "buy" = best bid, "sell" = best ask
            "bid": _to_float(data.get("buy")),
            "ask": _to_float(data.get("sell")),
            "last": _to_float(data.get("last")),
            "timestamp": data.get("timestamp"),
            "info": raw,
        }

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        pair = self._normalize_symbol(symbol)
        raw = self._public_get(f"/{pair}/depth")
        data = raw["data"]
        bids = [[_to_float(p), _to_float(s)] for p, s in data.get("bids", [])[:limit]]
        asks = [[_to_float(p), _to_float(s)] for p, s in data.get("asks", [])[:limit]]
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": data.get("timestamp"),
            "info": raw,
        }

    # ---------- private endpoints ----------

    def fetch_balance(self, params: dict | None = None) -> dict[str, Any]:
        raw = self._private_get("/user/assets", params=params or {})
        free: dict[str, float] = {}
        used: dict[str, float] = {}
        total: dict[str, float] = {}
        for asset in raw.get("data", {}).get("assets", []):
            ccy = asset.get("asset", "")
            f = _to_float(asset.get("free_amount", 0))
            u = _to_float(asset.get("locked_amount", 0))
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
            "pair": self._normalize_symbol(symbol),
            "side": side.lower(),          # "buy" or "sell"
            "type": "limit",
            "price": str(int(price)),       # bitbank expects integer JPY string
            "amount": str(amount),
        }
        # post_only supported as boolean (not ccxt's postOnly convention)
        if params and params.get("postOnly"):
            payload["post_only"] = True
        raw = self._private_post("/user/spot/order", payload)
        return _normalize_order(raw.get("data", raw))

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: dict | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pair": self._normalize_symbol(symbol),
            "side": side.lower(),
            "type": "market",
            "amount": str(amount),
        }
        if params:
            payload.update(params)
        raw = self._private_post("/user/spot/order", payload)
        return _normalize_order(raw.get("data", raw))

    def cancel_order(self, id: str, symbol: str | None = None, params: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"order_id": int(id)}
        if symbol is not None:
            payload["pair"] = self._normalize_symbol(symbol)
        if params:
            payload.update(params)
        raw = self._private_post("/user/spot/cancel_order", payload)
        return {"id": id, "info": raw}

    def fetch_open_orders(self, symbol: str | None = None, params: dict | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = dict(params or {})
        if symbol is not None:
            query["pair"] = self._normalize_symbol(symbol)
        raw = self._private_get("/user/spot/active_orders", params=query)
        orders = raw.get("data", {}).get("orders", [])
        return [_normalize_order(o) for o in orders]

    # ---------- transport ----------

    def _public_get(self, path: str, params: dict | None = None) -> Any:
        url = self.PUBLIC_BASE_URL + path
        return self._request("GET", url, params=params, signed=False)

    def _private_get(self, path: str, params: dict | None = None) -> Any:
        url = self.PRIVATE_BASE_URL + path
        return self._request("GET", url, params=params, signed=True, private_path=path)

    def _private_post(self, path: str, body: dict) -> Any:
        url = self.PRIVATE_BASE_URL + path
        return self._request("POST", url, body=body, signed=True, private_path=path)

    def _request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        body: dict | None = None,
        signed: bool = False,
        private_path: str = "",
    ) -> Any:
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if signed:
            query_str = urlencode(params) if params else ""
            headers.update(self._sign(method, private_path, query_str, body_str))

        attempts = 0
        while True:
            attempts += 1
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    data=body_str if body is not None else None,
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
                raise DomesticClientError(
                    f"HTTP {resp.status_code} {method} {url}: {resp.text}"
                )

            try:
                data = resp.json()
            except ValueError:
                return resp.text

            # bitbank wraps all responses in {"success": 1/0, "data": {...}}
            if isinstance(data, dict) and data.get("success") == 0:
                raise DomesticClientError(f"API error: {data}")
            return data

    def _sign(self, method: str, path: str, query_str: str, body: str) -> dict[str, str]:
        """bitbank ACCESS-NONCE signing.

        GET:  message = nonce + "/v1" + path + ("?" + query if query else "")
        POST: message = nonce + json_body
        """
        nonce = str(int(time.time() * 1000))
        if method == "GET":
            full_path = "/v1" + path
            if query_str:
                full_path += "?" + query_str
            message = nonce + full_path
        else:
            message = nonce + body

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-NONCE": nonce,
            "ACCESS-SIGNATURE": signature,
        }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # bitbank uses lowercase underscore: BTC/JPY → btc_jpy
        return symbol.replace("/", "_").lower()


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
        "id": str(raw.get("order_id") or raw.get("id") or ""),
        "symbol": raw.get("pair"),
        "side": (raw.get("side") or "").lower() or None,
        "type": (raw.get("type") or "").lower() or None,
        "price": _to_float(raw.get("price")),
        "amount": _to_float(raw.get("start_amount") or raw.get("amount")),
        "remaining": _to_float(raw.get("remaining_amount")),
        "status": raw.get("status"),
        "info": raw,
    }
