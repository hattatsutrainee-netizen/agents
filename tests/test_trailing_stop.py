"""Unit tests for TrailingStop class (dynamic stop loss management)."""

import time
import unittest
from unittest.mock import Mock

from agents.strategies.trailing_stop import TrailingStop


class TestTrailingStopATRCalculation(unittest.TestCase):
    """Test TrailingStop.calculate_atr() with various market conditions."""

    def setUp(self):
        """Set up mock client and trailing stop."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.trailing_stop = TrailingStop(self.client, self.symbol)

    def test_calculate_atr_normal(self):
        """Test ATR calculation with normal volatility data."""
        # Generate 14 candles: stable closes, consistent range
        ohlcv = [
            [i * 1000, 5000000.0, 5010000.0, 4990000.0, 5005000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = ohlcv

        atr = self.trailing_stop.calculate_atr()

        # Expected: high-low = 20k, ATR ≈ 20k
        self.assertGreater(atr, 15000.0)
        self.assertLess(atr, 25000.0)

    def test_calculate_atr_high_volatility(self):
        """Test ATR with high volatility (wider ranges)."""
        # Wide ranges: high-low = 100k per candle
        ohlcv = [
            [i * 1000, 5000000.0, 5050000.0, 4950000.0, 5000000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = ohlcv

        atr = self.trailing_stop.calculate_atr()

        # Expected ATR should be ~100k (high-low = 100k)
        self.assertGreater(atr, 90000.0)
        self.assertLess(atr, 110000.0)

    def test_calculate_atr_low_volatility(self):
        """Test ATR with low volatility (narrow ranges)."""
        # Narrow ranges: high-low = 10k per candle
        ohlcv = [
            [i * 1000, 5000000.0, 5005000.0, 4995000.0, 5000000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = ohlcv

        atr = self.trailing_stop.calculate_atr()

        # Expected ATR should be ~10k (high-low = 10k)
        self.assertGreater(atr, 9000.0)
        self.assertLess(atr, 11000.0)

    def test_calculate_atr_insufficient_candles(self):
        """Test ATR when insufficient candle data is available."""
        # Only 5 candles instead of 14
        ohlcv = [
            [i * 1000, 5000000.0, 5010000.0, 4990000.0, 5005000.0, 100.0]
            for i in range(5)
        ]
        self.client.fetch_ohlcv.return_value = ohlcv

        atr = self.trailing_stop.calculate_atr()

        # Should return 0.0 when insufficient data
        self.assertEqual(atr, 0.0)

    def test_calculate_atr_caching(self):
        """Test ATR caching: same value returned within cache interval."""
        ohlcv = [
            [i * 1000, 5000000.0, 5010000.0, 4990000.0, 5005000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = ohlcv

        # First call calculates ATR
        atr1 = self.trailing_stop.calculate_atr()
        self.client.reset_mock()

        # Second call (immediately after) should use cache
        atr2 = self.trailing_stop.calculate_atr()

        # Should return same value without calling client
        self.assertEqual(atr1, atr2)
        self.client.fetch_ohlcv.assert_not_called()


class TestTrailingStopLevel(unittest.TestCase):
    """Test TrailingStop.get_stop_level() with various PnL scenarios."""

    def setUp(self):
        """Set up mock client and trailing stop."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.trailing_stop = TrailingStop(
            self.client, self.symbol,
            initial_stop_atr_multiple=2.0,
            trailing_atr_multiple=1.5
        )
        # Mock ATR = 1000 JPY
        self.ohlcv = [
            [i * 1000, 5000000.0, 5001000.0, 4999000.0, 5000000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = self.ohlcv

    def test_get_stop_level_no_position(self):
        """Test stop level when unrealized PnL is zero or negative."""
        stop_level = self.trailing_stop.get_stop_level(0.0)
        self.assertIsNone(stop_level)

        stop_level = self.trailing_stop.get_stop_level(-100.0)
        self.assertIsNone(stop_level)

    def test_get_stop_level_small_pnl(self):
        """Test stop level when PnL < initial_stop_width."""
        # ATR = 2000 (high-low = 2000 in setUp OHLCV)
        # initial_stop_width = ATR × 2.0 = 2000 × 2.0 = 4000
        # PnL = 500 (less than 4000)
        stop_level = self.trailing_stop.get_stop_level(500.0)

        # Should return 0.0 (minimal stop to protect any profit)
        self.assertEqual(stop_level, 0.0)

    def test_get_stop_level_growing_pnl(self):
        """Test stop level rises as PnL grows."""
        # ATR = 2000, trailing_width = 2000 × 1.5 = 3000
        # PnL = 5000 (above initial_stop_width = 2000 × 2.0 = 4000)
        # Expected stop = 5000 - 3000 = 2000
        stop_level = self.trailing_stop.get_stop_level(5000.0)

        self.assertAlmostEqual(stop_level, 2000.0, delta=10.0)

    def test_get_stop_level_large_pnl(self):
        """Test stop level with very large unrealized profit."""
        # PnL = 8000 (well above threshold of 4000)
        # Expected stop = 8000 - 3000 = 5000
        stop_level = self.trailing_stop.get_stop_level(8000.0)

        self.assertAlmostEqual(stop_level, 5000.0, delta=10.0)

    def test_get_stop_level_above_threshold(self):
        """Test stop level when PnL just exceeds initial threshold."""
        # initial_stop_width = 4000 (ATR 2000 × 2.0), so boundary is at 4000
        # Test PnL = 4100 (just above boundary)
        # Expected stop = 4100 - 3000 = 1100
        stop_level = self.trailing_stop.get_stop_level(4100.0)

        self.assertGreater(stop_level, 0.0)
        self.assertLess(stop_level, 2000.0)

    def test_get_stop_level_atr_multiple_scaling(self):
        """Test that ATR multiple parameters affect stop width."""
        ts_wide = TrailingStop(
            self.client, self.symbol,
            initial_stop_atr_multiple=3.0,  # Wider
            trailing_atr_multiple=2.0
        )
        self.client.fetch_ohlcv.return_value = self.ohlcv

        # ATR = 2000 from setUp OHLCV
        # PnL = 7000 (above 3.0 × 2000 = 6000 initial threshold)
        # Expected stop = 7000 - 2.0 × 2000 = 3000
        stop_level = ts_wide.get_stop_level(7000.0)

        self.assertAlmostEqual(stop_level, 3000.0, delta=10.0)

    def test_get_stop_level_zero_atr(self):
        """Test stop level when ATR calculation returns 0."""
        self.client.fetch_ohlcv.return_value = []  # Insufficient data

        stop_level = self.trailing_stop.get_stop_level(3000.0)

        # When ATR is 0, stop level should be None
        self.assertIsNone(stop_level)


class TestTrailingStopIntegration(unittest.TestCase):
    """Test TrailingStop integration: filled events, candle updates, info reporting."""

    def setUp(self):
        """Set up mock client and trailing stop."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.trailing_stop = TrailingStop(self.client, self.symbol)
        self.ohlcv = [
            [i * 1000, 5000000.0, 5001000.0, 4999000.0, 5000000.0, 100.0]
            for i in range(14)
        ]
        self.client.fetch_ohlcv.return_value = self.ohlcv

    def test_update_with_filled_event_buy(self):
        """Test position tracking after BUY fill."""
        filled_event = {
            "side": "buy",
            "price": 5000000.0,
            "size": 0.1,
            "ts": time.time(),
        }

        self.trailing_stop.update_with_filled_event(filled_event)

        info = self.trailing_stop.get_trailing_info()
        self.assertEqual(info["entry_price"], 5000000.0)
        self.assertEqual(info["entry_quantity"], 0.1)

    def test_update_with_filled_event_sell(self):
        """Test position reset after SELL fill."""
        # First BUY
        self.trailing_stop.update_with_filled_event({
            "side": "buy",
            "price": 5000000.0,
            "size": 0.1,
            "ts": time.time(),
        })

        # Then SELL
        self.trailing_stop.update_with_filled_event({
            "side": "sell",
            "price": 5001000.0,
            "size": 0.1,
            "ts": time.time(),
        })

        info = self.trailing_stop.get_trailing_info()
        self.assertIsNone(info["entry_price"])
        self.assertEqual(info["entry_quantity"], 0.0)

    def test_trailing_stop_with_candle_update(self):
        """Test that ATR is recalculated from new candle data."""
        # First calculation
        atr1 = self.trailing_stop.calculate_atr()

        # Change OHLCV data (simulate market movement)
        self.client.fetch_ohlcv.return_value = [
            [i * 1000, 5000000.0, 5005000.0, 4995000.0, 5000000.0, 100.0]
            for i in range(14)
        ]

        # Clear cache by advancing time
        self.trailing_stop._atr_timestamp = time.time() - 100

        atr2 = self.trailing_stop.calculate_atr()

        # Both should be valid (non-zero)
        self.assertGreater(atr1, 0.0)
        self.assertGreater(atr2, 0.0)

    def test_trailing_stop_info_reporting(self):
        """Test get_trailing_info() returns all required fields."""
        # Set up position
        self.trailing_stop.update_with_filled_event({
            "side": "buy",
            "price": 5000000.0,
            "size": 0.1,
            "ts": time.time(),
        })

        # Calculate ATR
        self.trailing_stop.calculate_atr()

        info = self.trailing_stop.get_trailing_info()

        # Verify all keys present
        self.assertIn("entry_price", info)
        self.assertIn("entry_quantity", info)
        self.assertIn("entry_timestamp", info)
        self.assertIn("cached_atr", info)
        self.assertIn("atr_last_update", info)

    def test_trailing_stop_error_handling(self):
        """Test graceful handling of API errors."""
        self.client.fetch_ohlcv.side_effect = Exception("API error")

        # Should return 0.0 on error, not raise
        atr = self.trailing_stop.calculate_atr()
        self.assertEqual(atr, 0.0)

        # Should return None for stop level when ATR is 0
        stop_level = self.trailing_stop.get_stop_level(3000.0)
        self.assertIsNone(stop_level)


if __name__ == "__main__":
    unittest.main()
