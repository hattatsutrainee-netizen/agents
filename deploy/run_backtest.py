"""Run backtests across multiple strategies on OHLCV CSV data.

Strategies operate on raw JPY prices with no 0-1 price constraints.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestResult:
    strategy: str
    trades: int
    pnl: float
    win_rate: float
    max_drawdown: float
    sortino_ratio: float | None = None
    profit_factor: float | None = None
    max_consecutive_losses: int = 0
    avg_win_loss_ratio: float | None = None


def load_csv(path: str) -> list[Candle]:
    candles: list[Candle] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                Candle(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


class MomentumStrategy:
    """SMA crossover: buy when fast MA crosses above slow MA, sell on cross below."""

    name = "momentum_sma"

    def __init__(self, fast: int = 10, slow: int = 30, size: float = 0.001) -> None:
        self.fast = fast
        self.slow = slow
        self.size = size

    def run(self, candles: list[Candle]) -> BacktestResult:
        closes = [c.close for c in candles]
        trades: list[float] = []
        position: str | None = None
        entry_price = 0.0

        for i in range(self.slow, len(closes)):
            fast_ma = sum(closes[i - self.fast : i]) / self.fast
            slow_ma = sum(closes[i - self.slow : i]) / self.slow
            price = closes[i]

            if fast_ma > slow_ma and position != "long":
                if position == "short":
                    trades.append((entry_price - price) * self.size)
                position = "long"
                entry_price = price
            elif fast_ma < slow_ma and position != "short":
                if position == "long":
                    trades.append((price - entry_price) * self.size)
                position = "short"
                entry_price = price

        return _summarize(self.name, trades)


class MeanReversionStrategy:
    """RSI mean reversion: buy oversold (<30), sell overbought (>70)."""

    name = "mean_reversion_rsi"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        size: float = 0.001,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.size = size

    def run(self, candles: list[Candle]) -> BacktestResult:
        closes = [c.close for c in candles]
        trades: list[float] = []
        position: str | None = None
        entry_price = 0.0

        for i in range(self.period + 1, len(closes)):
            rsi = _calc_rsi(closes[i - self.period - 1 : i + 1], self.period)
            price = closes[i]

            if rsi < self.oversold and position != "long":
                if position == "short":
                    trades.append((entry_price - price) * self.size)
                position = "long"
                entry_price = price
            elif rsi > self.overbought and position != "short":
                if position == "long":
                    trades.append((price - entry_price) * self.size)
                position = "short"
                entry_price = price

        return _summarize(self.name, trades)


class MakerSniperBacktest:
    """Simulate maker rebate capture.

    Whenever a candle's high-low spread exceeds min_spread_pct, both sides
    of the quote are assumed to fill and the half-spread is booked as profit.
    This models the best-case scenario for the live MakerSniperStrategy.
    """

    name = "maker_sniper"

    def __init__(self, size: float = 0.001, min_spread_pct: float = 0.001) -> None:
        self.size = size
        self.min_spread_pct = min_spread_pct

    def run(self, candles: list[Candle]) -> BacktestResult:
        trades: list[float] = []
        for c in candles:
            mid = (c.high + c.low) / 2
            if mid <= 0:
                continue
            spread_pct = (c.high - c.low) / mid
            if spread_pct >= self.min_spread_pct:
                half_spread = (c.high - c.low) / 2
                trades.append(half_spread * self.size)
        return _summarize(self.name, trades)


def _calc_rsi(closes: list[float], period: int) -> float:
    changes = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0.0
    avg_loss = sum(losses[-period:]) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _summarize(name: str, pnl_list: list[float]) -> BacktestResult:
    if not pnl_list:
        return BacktestResult(strategy=name, trades=0, pnl=0.0, win_rate=0.0, max_drawdown=0.0)

    total = sum(pnl_list)
    wins = sum(1 for p in pnl_list if p > 0)
    win_rate = wins / len(pnl_list) * 100

    # Max drawdown
    peak = cumulative = max_dd = 0.0
    for p in pnl_list:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Sortino Ratio: (mean_return) / downside_volatility
    mean_return = total / len(pnl_list)
    downside_returns = [p for p in pnl_list if p < 0]
    if downside_returns:
        downside_variance = sum(r**2 for r in downside_returns) / len(pnl_list)
        downside_volatility = downside_variance**0.5
        sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else None
    else:
        sortino_ratio = float('inf') if mean_return > 0 else 0.0

    # Profit Factor: sum(positive) / abs(sum(negative))
    positive_sum = sum(p for p in pnl_list if p > 0)
    negative_sum = sum(p for p in pnl_list if p < 0)
    if negative_sum != 0:
        profit_factor = positive_sum / abs(negative_sum)
    else:
        profit_factor = float('inf') if positive_sum > 0 else 0.0

    # Max Consecutive Losses
    max_consecutive_losses = 0
    current_streak = 0
    for p in pnl_list:
        if p < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Average Win/Loss Ratio
    wins_list = [p for p in pnl_list if p > 0]
    losses_list = [p for p in pnl_list if p < 0]
    if wins_list and losses_list:
        avg_win = sum(wins_list) / len(wins_list)
        avg_loss = abs(sum(losses_list)) / len(losses_list)
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else None
    else:
        avg_win_loss_ratio = None

    return BacktestResult(
        strategy=name,
        trades=len(pnl_list),
        pnl=total,
        win_rate=win_rate,
        max_drawdown=max_dd,
        sortino_ratio=sortino_ratio,
        profit_factor=profit_factor,
        max_consecutive_losses=max_consecutive_losses,
        avg_win_loss_ratio=avg_win_loss_ratio,
    )


STRATEGIES: dict[str, type] = {
    "momentum_sma": MomentumStrategy,
    "mean_reversion_rsi": MeanReversionStrategy,
    "maker_sniper": MakerSniperBacktest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests on OHLCV CSV data")
    parser.add_argument("--csv", required=True, help="path to OHLCV CSV file")
    parser.add_argument(
        "--strategy",
        default="all",
        help=f"strategy name or 'all'. Available: {', '.join(STRATEGIES)}",
    )
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    candles = load_csv(args.csv)
    print(f"Loaded {len(candles)} candles from {args.csv}\n")

    if args.strategy == "all":
        targets = list(STRATEGIES.values())
    elif args.strategy in STRATEGIES:
        targets = [STRATEGIES[args.strategy]]
    else:
        print(
            f"Unknown strategy: {args.strategy}. Available: {', '.join(STRATEGIES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = [cls().run(candles) for cls in targets]

    # Header: Basic metrics
    print(f"{'Strategy':<25} {'Trades':>7} {'PnL (JPY)':>12} {'Win%':>8} {'MaxDD (JPY)':>12}")
    # Header: Advanced metrics
    print(f"{'':25} {'Sortino':>7} {'P.Factor':>12} {'AvgW/L':>8} {'MaxConLoss':>12}")
    print("-" * 85)
    for r in results:
        sortino_str = f"{r.sortino_ratio:.2f}" if r.sortino_ratio is not None and r.sortino_ratio != float('inf') else "∞" if r.sortino_ratio == float('inf') else "—"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor is not None and r.profit_factor != float('inf') else "∞" if r.profit_factor == float('inf') else "—"
        wl_str = f"{r.avg_win_loss_ratio:.2f}" if r.avg_win_loss_ratio is not None else "—"

        # Row 1: Basic metrics
        print(
            f"{r.strategy:<25} {r.trades:>7} {r.pnl:>12.2f} {r.win_rate:>7.1f}% {r.max_drawdown:>12.2f}"
        )
        # Row 2: Advanced metrics
        print(
            f"{'':25} {sortino_str:>7} {pf_str:>12} {wl_str:>8} {r.max_consecutive_losses:>12}"
        )
    print()


if __name__ == "__main__":
    main()
