"""Download historical OHLCV data from crypto exchanges.

bitbank's public candlestick API is fetched directly (date-by-date) because
ccxt's bitbank driver returns at most one day's worth of data per call.
Other exchanges fall back to ccxt's fetch_ohlcv with simple pagination.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import ccxt


def _fetch_bitbank(symbol: str, timeframe: str, limit: int) -> list[list]:
    """Fetch OHLCV from bitbank's public candlestick API, walking backwards day by day."""
    tf_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "4h": "4hour",
        "8h": "8hour",
        "12h": "12hour",
        "1d": "1day",
        "1w": "1week",
    }
    tf = tf_map.get(timeframe)
    if tf is None:
        print(f"Unsupported timeframe for bitbank: {timeframe}", file=sys.stderr)
        sys.exit(1)

    pair = symbol.replace("/", "_").lower()
    base_url = "https://public.bitbank.cc"
    candles: list[list] = []
    current_date = datetime.now(tz=timezone.utc)

    while len(candles) < limit:
        date_str = current_date.strftime("%Y%m%d")
        url = f"{base_url}/{pair}/candlestick/{tf}/{date_str}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                current_date -= timedelta(days=1)
                time.sleep(0.3)
                continue
            data = resp.json()
            sticks = data.get("data", {}).get("candlestick", [])
            if not sticks:
                current_date -= timedelta(days=1)
                time.sleep(0.3)
                continue
            ohlcv = sticks[0].get("ohlcv", [])
            for row in ohlcv:
                # bitbank format: [open, high, low, close, volume, timestamp_ms]
                candles.append([int(row[5]), float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        except Exception as e:
            print(f"Warning: {date_str} fetch error: {e}", file=sys.stderr)

        current_date -= timedelta(days=1)
        time.sleep(0.2)

    candles.sort(key=lambda c: c[0])
    return candles[-limit:]


def _fetch_ccxt(exchange_id: str, symbol: str, timeframe: str, limit: int) -> list[list]:
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        print(f"Unknown exchange: {exchange_id}", file=sys.stderr)
        sys.exit(1)
    exchange: ccxt.Exchange = exchange_cls()
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def download(exchange_id: str, symbol: str, timeframe: str, limit: int, output: str) -> None:
    print(f"Fetching {limit} {timeframe} candles for {symbol} from {exchange_id}...")

    if exchange_id == "bitbank":
        ohlcv = _fetch_bitbank(symbol, timeframe, limit)
    else:
        ohlcv = _fetch_ccxt(exchange_id, symbol, timeframe, limit)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in ohlcv:
            ts = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            writer.writerow([ts, row[1], row[2], row[3], row[4], row[5]])

    print(f"Saved {len(ohlcv)} rows -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data from crypto exchanges")
    parser.add_argument("--exchange", default="bitbank", help="ccxt exchange id (default: bitbank)")
    parser.add_argument("--symbol", default="BTC/JPY", help="trading pair (default: BTC/JPY)")
    parser.add_argument("--timeframe", default="1h", help="candle timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=500, help="number of candles (default: 500)")
    parser.add_argument("--output", default=None, help="output CSV path (auto-named if omitted)")
    args = parser.parse_args()

    if args.output is None:
        safe_symbol = args.symbol.replace("/", "-")
        args.output = f"data/{args.exchange}_{safe_symbol}_{args.timeframe}.csv"

    download(args.exchange, args.symbol, args.timeframe, args.limit, args.output)


if __name__ == "__main__":
    main()
