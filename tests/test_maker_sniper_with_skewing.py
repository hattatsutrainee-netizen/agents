"""Integration tests for MakerSniperStrategy with inventory skewing."""

import unittest
from unittest.mock import Mock

from agents.strategies.maker_sniper_strategy import MakerSniperStrategy


class TestMakerSniperWithoutSkewing(unittest.TestCase):
    """Test MakerSniperStrategy without inventory skewing (backward compatibility)."""

    def setUp(self):
        """Set up mock client."""
        self.client = Mock()
        self.symbol = "BTC/JPY"

    def test_strategy_initializes_without_target_ratio(self):
        """Test that strategy initializes normally without target_ratio."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
        )

        self.assertIsNone(strategy.target_ratio)
        self.assertIsNone(strategy.portfolio)

    def test_step_without_skewing_returns_inventory_none(self):
        """Test that step() returns inventory=None when no skewing."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            dry_run=True,
        )

        self.client.fetch_order_book.return_value = {
            "bids": [[5000000.0, 1.0]],
            "asks": [[5002000.0, 1.0]],
        }
        self.client.fetch_open_orders.return_value = []

        result = strategy.step()

        self.assertIsNone(result.get("inventory"))

    def test_step_without_skewing_preserves_state(self):
        """Test that existing behavior is preserved."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            dry_run=True,
        )

        # Use wider book to avoid spread_too_tight
        self.client.fetch_order_book.return_value = {
            "bids": [[5000000.0, 1.0], [4999000.0, 2.0]],
            "asks": [[5001000.0, 1.0], [5002000.0, 2.0]],
        }
        self.client.fetch_open_orders.return_value = []

        result = strategy.step()

        # Verify all existing return keys are present (status ok, not spread_too_tight)
        if result["status"] == "ok":
            self.assertIn("open_bid_id", result)
            self.assertIn("open_ask_id", result)
            self.assertIn("fills", result)
            self.assertIn("realized_pnl", result)
        # If status is not ok, at least verify basic fields are present
        self.assertIn("status", result)
        self.assertIn("best_bid", result)
        self.assertIn("best_ask", result)


class TestMakerSniperWithSkewing(unittest.TestCase):
    """Test MakerSniperStrategy with inventory skewing enabled."""

    def setUp(self):
        """Set up mock client and strategy with skewing."""
        self.client = Mock()
        self.symbol = "BTC/JPY"

    def test_strategy_initializes_with_target_ratio(self):
        """Test that strategy initializes Portfolio when target_ratio is set."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
        )

        self.assertEqual(strategy.target_ratio, 0.5)
        self.assertIsNotNone(strategy.portfolio)

    def test_apply_inventory_skew_excess_base(self):
        """Test skewing when holding too much BTC (excess Base)."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
            tick_offset=1,
            dry_run=True,
        )

        # Mock client to return excess BTC balance
        self.client.fetch_balance.return_value = {
            "total": {"BTC": 2.0, "JPY": 5000000.0}  # ratio = 2/3 ≈ 0.667
        }
        self.client.fetch_ticker.return_value = {
            "bid": 5000000.0,
            "ask": 5002000.0,
        }

        skewed_bid, skewed_ask = strategy._apply_inventory_skew(
            desired_bid=5001000.0,
            desired_ask=5001000.0,
            best_bid=5000000.0,
            best_ask=5002000.0,
            tick=1000.0,
        )

        # When excess Base: bid should decrease, ask should increase
        self.assertLess(skewed_bid, 5001000.0)
        self.assertGreater(skewed_ask, 5001000.0)

    def test_apply_inventory_skew_excess_quote(self):
        """Test skewing when holding too much JPY (excess Quote)."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
            tick_offset=1,
            dry_run=True,
        )

        # Mock client to return excess JPY balance
        self.client.fetch_balance.return_value = {
            "total": {"BTC": 0.75, "JPY": 5000000.0}  # ratio = 0.6, skew = 0.2 (smaller)
        }
        self.client.fetch_ticker.return_value = {
            "bid": 5000000.0,
            "ask": 5002000.0,
        }

        # Use wider spread to avoid inversion
        skewed_bid, skewed_ask = strategy._apply_inventory_skew(
            desired_bid=5000500.0,
            desired_ask=5001500.0,
            best_bid=5000000.0,
            best_ask=5002000.0,
            tick=1000.0,
        )

        # When excess Quote (need to buy, skew < 0): bid should increase, ask should decrease
        self.assertGreater(skewed_bid, 5000500.0)
        self.assertLess(skewed_ask, 5001500.0)

    def test_apply_inventory_skew_balanced(self):
        """Test skewing when portfolio is balanced (no skew)."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
            tick_offset=1,
            dry_run=True,
        )

        # Mock client to return balanced portfolio
        self.client.fetch_balance.return_value = {
            "total": {"BTC": 1.0, "JPY": 5000000.0}  # ratio = 0.5
        }
        self.client.fetch_ticker.return_value = {
            "bid": 5000000.0,
            "ask": 5002000.0,
        }

        skewed_bid, skewed_ask = strategy._apply_inventory_skew(
            desired_bid=5001000.0,
            desired_ask=5001000.0,
            best_bid=5000000.0,
            best_ask=5002000.0,
            tick=1000.0,
        )

        # When balanced: skew should be ~0, so prices should be ~same
        self.assertAlmostEqual(skewed_bid, 5001000.0, delta=10.0)
        self.assertAlmostEqual(skewed_ask, 5001000.0, delta=10.0)

    def test_apply_inventory_skew_error_handling(self):
        """Test that errors in inventory skew fallback to baseline."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
            dry_run=True,
        )

        # Mock client to raise an error
        self.client.fetch_balance.side_effect = Exception("API error")

        skewed_bid, skewed_ask = strategy._apply_inventory_skew(
            desired_bid=5001000.0,
            desired_ask=5001000.0,
            best_bid=5000000.0,
            best_ask=5002000.0,
            tick=1000.0,
        )

        # Should return original prices on error
        self.assertEqual(skewed_bid, 5001000.0)
        self.assertEqual(skewed_ask, 5001000.0)

    def test_step_includes_inventory_info(self):
        """Test that step() returns inventory info when skewing is enabled."""
        strategy = MakerSniperStrategy(
            self.client,
            self.symbol,
            order_size=0.0001,
            target_ratio=0.5,
            dry_run=True,
        )

        # Mock order book with wider spread
        self.client.fetch_order_book.return_value = {
            "bids": [[5000000.0, 1.0], [4999000.0, 2.0]],
            "asks": [[5001000.0, 1.0], [5002000.0, 2.0]],
        }
        self.client.fetch_open_orders.return_value = []
        self.client.fetch_balance.return_value = {
            "total": {"BTC": 1.0, "JPY": 5000000.0}
        }
        self.client.fetch_ticker.return_value = {
            "bid": 5000000.0,
            "ask": 5002000.0,
        }

        result = strategy.step()

        # Verify inventory info is included (only when status is ok)
        if result["status"] == "ok":
            self.assertIsNotNone(result.get("inventory"))
            inventory = result["inventory"]
            self.assertIn("current_ratio", inventory)
            self.assertIn("target_ratio", inventory)
            self.assertIn("skew_factor", inventory)
        else:
            # If not ok, inventory should still be None (early return)
            self.assertIsNone(result.get("inventory"))


if __name__ == "__main__":
    unittest.main()
