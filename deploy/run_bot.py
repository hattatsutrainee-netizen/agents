"""Live trading bot for domestic JP crypto exchanges (SBI VC Trade).

Usage:
  # ドライラン（デフォルト・安全）
  PYTHONPATH=. python deploy/run_bot.py --strategy maker_sniper

  # 実弾モード（APIキー設定済み・確認済みのみ）
  PYTHONPATH=. python deploy/run_bot.py --strategy maker_sniper --no-dry-run

  # RSI戦略でドライラン
  PYTHONPATH=. python deploy/run_bot.py --strategy rsi --interval 60

Safety limits (環境変数で上書き可):
  MAX_DAILY_LOSS_JPY   : 日次損失上限 JPY (default: 5000)
  MAX_POSITION_BTC     : 最大ポジション BTC (default: 0.001)
  DOMESTIC_ORDER_SIZE  : 注文サイズ BTC (default: 0.0001 ≈ 1070 JPY)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- logging setup -----------------------------------------------------------

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"bot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_bot")


# --- strategy imports (after logging is set up) ------------------------------

from agents.api_clients import DomesticClient
from agents.strategies.maker_sniper_strategy import MakerSniperStrategy
from agents.strategies.rsi_live_strategy import LiveRsiStrategy


# --- safety limits -----------------------------------------------------------

MAX_DAILY_LOSS_JPY = float(os.getenv("MAX_DAILY_LOSS_JPY", "5000"))
MAX_POSITION_BTC = float(os.getenv("MAX_POSITION_BTC", "0.001"))
ORDER_SIZE = float(os.getenv("DOMESTIC_ORDER_SIZE", "0.0001"))
SYMBOL = os.getenv("DOMESTIC_SYMBOL", "BTC/JPY")


class SafetyGuard:
    """Tracks session PnL and halts the bot if limits are breached."""

    def __init__(self, max_daily_loss: float, max_position: float) -> None:
        self.max_daily_loss = max_daily_loss
        self.max_position = max_position
        self._session_pnl: float = 0.0
        self._day = date.today()

    def record_pnl(self, delta: float) -> None:
        if date.today() != self._day:
            logger.info("New trading day — resetting daily PnL counter")
            self._session_pnl = 0.0
            self._day = date.today()
        self._session_pnl += delta

    def check(self) -> None:
        if self._session_pnl < -self.max_daily_loss:
            raise RuntimeError(
                f"Daily loss limit reached: {self._session_pnl:.2f} JPY "
                f"(limit: {self.max_daily_loss:.2f} JPY). Bot stopped."
            )

    @property
    def session_pnl(self) -> float:
        return self._session_pnl


# --- main loop ---------------------------------------------------------------

_running = True


def _handle_signal(sig: int, frame: object) -> None:
    global _running
    logger.info("Shutdown signal received (%s), stopping after current step...", signal.Signals(sig).name)
    _running = False


def _warn_tbd_verify(dry_run: bool) -> None:
    """Remind the user about live-mode checklist."""
    if dry_run:
        return
    logger.warning("=" * 60)
    logger.warning("LIVE MODE — bitbank real-order checklist:")
    logger.warning("  • DOMESTIC_EXCHANGE_API_KEY / API_SECRET must be set in .env")
    logger.warning("  • APIキーの権限: 取引(注文)権限が有効になっているか確認")
    logger.warning("  • 最初の注文後 bitbank ダッシュボードで約定を確認すること")
    logger.warning("=" * 60)


def run_maker_sniper(
    client: DomesticClient,
    dry_run: bool,
    interval: float,
    guard: SafetyGuard,
    enable_trailing_stop: bool = False,
    trailing_stop_atr_multiple: float = 1.5,
) -> None:
    strategy = MakerSniperStrategy(
        client,
        symbol=SYMBOL,
        order_size=ORDER_SIZE,
        dry_run=dry_run,
        enable_trailing_stop=enable_trailing_stop,
        trailing_stop_atr_multiple=trailing_stop_atr_multiple,
    )
    logger.info("MakerSniperStrategy started | symbol=%s size=%.6f dry_run=%s", SYMBOL, ORDER_SIZE, dry_run)

    prev_pnl = 0.0
    while _running:
        try:
            state = strategy.step()
            delta = strategy.realized_pnl - prev_pnl
            if delta != 0:
                guard.record_pnl(delta)
                prev_pnl = strategy.realized_pnl
            guard.check()

            logger.info(
                "step=%s bid=%.0f ask=%.0f tick=%.0f fills=%d pnl=%.2f session_pnl=%.2f",
                state.get("status"),
                state.get("best_bid", 0),
                state.get("best_ask", 0),
                state.get("tick", 0),
                state.get("fills", 0),
                strategy.realized_pnl,
                guard.session_pnl,
            )
        except RuntimeError as e:
            logger.error(str(e))
            break
        except Exception as e:
            logger.error("step error: %s", e, exc_info=True)

        if _running:
            time.sleep(interval)


def run_rsi(client: DomesticClient, dry_run: bool, interval: float, guard: SafetyGuard) -> None:
    strategy = LiveRsiStrategy(
        client,
        symbol=SYMBOL,
        order_size=ORDER_SIZE,
        dry_run=dry_run,
    )
    logger.info(
        "LiveRsiStrategy started | symbol=%s size=%.6f dry_run=%s warmup=%d steps",
        SYMBOL, ORDER_SIZE, dry_run, strategy.rsi_period + 1,
    )

    prev_pnl = 0.0
    while _running:
        try:
            state = strategy.step()
            delta = strategy.realized_pnl - prev_pnl
            if delta != 0:
                guard.record_pnl(delta)
                prev_pnl = strategy.realized_pnl
            guard.check()

            rsi_str = f"{state['rsi']:.1f}" if state["rsi"] is not None else f"warming({state['warmup_remaining']}left)"
            logger.info(
                "state=%s mid=%.0f rsi=%s trades=%d pnl=%.2f session_pnl=%.2f",
                state["state"],
                state["mid"],
                rsi_str,
                state["trade_count"],
                strategy.realized_pnl,
                guard.session_pnl,
            )
        except RuntimeError as e:
            logger.error(str(e))
            break
        except Exception as e:
            logger.error("step error: %s", e, exc_info=True)

        if _running:
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading bot for domestic JP crypto exchange")
    parser.add_argument(
        "--strategy",
        choices=["maker_sniper", "rsi"],
        default="maker_sniper",
        help="trading strategy (default: maker_sniper)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="enable live order placement (default: dry-run only)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between steps (default: 1.0 for maker_sniper, suggest 60 for rsi)",
    )
    parser.add_argument(
        "--enable-trailing-stop",
        action="store_true",
        help="enable dynamic trailing stop (default: disabled)",
    )
    parser.add_argument(
        "--trailing-stop-atr",
        type=float,
        default=1.5,
        help="ATR multiple for trailing stop width (default: 1.5)",
    )
    args = parser.parse_args()

    dry_run = not args.no_dry_run

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("=" * 60)
    logger.info("Bot starting | strategy=%s dry_run=%s interval=%.1fs", args.strategy, dry_run, args.interval)
    logger.info("Symbol=%s  OrderSize=%.6f BTC  MaxDailyLoss=%.0f JPY", SYMBOL, ORDER_SIZE, MAX_DAILY_LOSS_JPY)
    logger.info("Log file: %s", log_file)
    logger.info("=" * 60)

    _warn_tbd_verify(dry_run)

    client = DomesticClient()
    guard = SafetyGuard(max_daily_loss=MAX_DAILY_LOSS_JPY, max_position=MAX_POSITION_BTC)

    try:
        if args.strategy == "maker_sniper":
            run_maker_sniper(
                client,
                dry_run,
                args.interval,
                guard,
                enable_trailing_stop=args.enable_trailing_stop,
                trailing_stop_atr_multiple=args.trailing_stop_atr,
            )
        else:
            interval = args.interval if args.interval != 1.0 else 60.0
            run_rsi(client, dry_run, interval, guard)
    finally:
        logger.info("Bot stopped | session_pnl=%.2f JPY", guard.session_pnl)


if __name__ == "__main__":
    main()
