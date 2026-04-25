"""Unit tests for Portfolio class (inventory skewing)."""

import unittest
from unittest.mock import Mock

from agents.strategies.portfolio import Portfolio


class TestPortfolioRatioCalculation(unittest.TestCase):
    """Test Portfolio.calculate_ratio() with various balance scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.portfolio = Portfolio(self.client, self.symbol, target_ratio=0.5)

    def test_calculate_ratio_balanced(self):
        """Test balanced portfolio: 1.0 BTC + 5M JPY at 5M/BTC → 0.5."""
        balance = {"BTC": 1.0, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        # Base value = 1.0 BTC
        # Quote value = 5M JPY / 5M JPY/BTC = 1.0 BTC
        # Total = 2.0 BTC
        # Ratio = 1.0 / 2.0 = 0.5
        self.assertAlmostEqual(ratio, 0.5, places=5)

    def test_calculate_ratio_excess_base(self):
        """Test excess Base: 2.0 BTC + 5M JPY at 5M/BTC → 0.667."""
        balance = {"BTC": 2.0, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        # Base value = 2.0 BTC
        # Quote value = 5M JPY / 5M JPY/BTC = 1.0 BTC
        # Total = 3.0 BTC
        # Ratio = 2.0 / 3.0 ≈ 0.667
        self.assertAlmostEqual(ratio, 2.0 / 3.0, places=5)

    def test_calculate_ratio_excess_quote(self):
        """Test excess Quote: 0.5 BTC + 5M JPY at 5M/BTC → 0.333."""
        balance = {"BTC": 0.5, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        # Base value = 0.5 BTC
        # Quote value = 5M JPY / 5M JPY/BTC = 1.0 BTC
        # Total = 1.5 BTC
        # Ratio = 0.5 / 1.5 ≈ 0.333
        self.assertAlmostEqual(ratio, 0.5 / 1.5, places=5)

    def test_calculate_ratio_all_base(self):
        """Test extreme: 100% Base (1.0 BTC, 0 JPY) → 1.0."""
        balance = {"BTC": 1.0, "JPY": 0.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_calculate_ratio_all_quote(self):
        """Test extreme: 100% Quote (0 BTC, 5M JPY) → 0.0."""
        balance = {"BTC": 0.0, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        self.assertAlmostEqual(ratio, 0.0, places=5)

    def test_calculate_ratio_zero_balance(self):
        """Test zero balance: returns target_ratio."""
        balance = {"BTC": 0.0, "JPY": 0.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        self.assertEqual(ratio, 0.5)  # target_ratio

    def test_calculate_ratio_zero_price(self):
        """Test zero/invalid price: returns target_ratio."""
        balance = {"BTC": 1.0, "JPY": 5000000.0}
        current_price = 0.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        self.assertEqual(ratio, 0.5)  # target_ratio (fallback)

    def test_calculate_ratio_missing_asset_in_balance(self):
        """Test missing asset in balance: treats as zero."""
        balance = {"JPY": 5000000.0}  # BTC missing
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        # Base value = 0 BTC (default)
        # Quote value = 5M JPY / 5M JPY/BTC = 1.0 BTC
        # Ratio = 0 / 1.0 = 0.0
        self.assertAlmostEqual(ratio, 0.0, places=5)

    def test_calculate_ratio_caching(self):
        """Test that calculate_ratio caches last_ratio and last_price."""
        balance = {"BTC": 1.0, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        self.assertEqual(self.portfolio._last_ratio, ratio)
        self.assertEqual(self.portfolio._last_price, current_price)


class TestPortfolioSkewFactor(unittest.TestCase):
    """Test Portfolio.get_skew_factor() with various ratios."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.portfolio = Portfolio(self.client, self.symbol, target_ratio=0.5)

    def test_skew_factor_balanced(self):
        """Test balanced ratio: skew = 0."""
        skew = self.portfolio.get_skew_factor(0.5)

        # skew = (0.5 - 0.5) / 0.5 = 0
        self.assertAlmostEqual(skew, 0.0, places=5)

    def test_skew_factor_excess_base(self):
        """Test excess Base (ratio=0.6): skew = 0.2."""
        skew = self.portfolio.get_skew_factor(0.6)

        # skew = (0.6 - 0.5) / 0.5 = 0.2
        self.assertAlmostEqual(skew, 0.2, places=5)

    def test_skew_factor_excess_quote(self):
        """Test excess Quote (ratio=0.4): skew = -0.2."""
        skew = self.portfolio.get_skew_factor(0.4)

        # skew = (0.4 - 0.5) / 0.5 = -0.2
        self.assertAlmostEqual(skew, -0.2, places=5)

    def test_skew_factor_extreme_base(self):
        """Test extreme Base (ratio=1.0): skew = 1.0."""
        skew = self.portfolio.get_skew_factor(1.0)

        # skew = (1.0 - 0.5) / 0.5 = 1.0
        self.assertAlmostEqual(skew, 1.0, places=5)

    def test_skew_factor_extreme_quote(self):
        """Test extreme Quote (ratio=0.0): skew = -1.0."""
        skew = self.portfolio.get_skew_factor(0.0)

        # skew = (0.0 - 0.5) / 0.5 = -1.0
        self.assertAlmostEqual(skew, -1.0, places=5)

    def test_skew_factor_custom_target(self):
        """Test skew with custom target_ratio."""
        portfolio = Portfolio(self.client, self.symbol, target_ratio=0.3)
        skew = portfolio.get_skew_factor(0.6)

        # skew = (0.6 - 0.3) / 0.3 = 1.0
        self.assertAlmostEqual(skew, 1.0, places=5)

    def test_skew_factor_caching(self):
        """Test that get_skew_factor caches _last_skew."""
        skew = self.portfolio.get_skew_factor(0.6)

        self.assertEqual(self.portfolio._last_skew, skew)
        self.assertAlmostEqual(self.portfolio._last_skew, 0.2, places=5)


class TestPortfolioInventoryInfo(unittest.TestCase):
    """Test Portfolio.get_inventory_info() returns cached state."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Mock()
        self.symbol = "BTC/JPY"
        self.portfolio = Portfolio(self.client, self.symbol, target_ratio=0.5)

    def test_inventory_info_empty(self):
        """Test inventory_info before any calculations."""
        info = self.portfolio.get_inventory_info()

        self.assertIsNone(info["current_ratio"])
        self.assertIsNone(info["skew_factor"])
        self.assertIsNone(info["last_price"])
        self.assertEqual(info["target_ratio"], 0.5)

    def test_inventory_info_after_calculation(self):
        """Test inventory_info after calculate_ratio."""
        balance = {"BTC": 1.0, "JPY": 5000000.0}
        current_price = 5000000.0

        ratio = self.portfolio.calculate_ratio(balance, current_price)

        info = self.portfolio.get_inventory_info()

        self.assertAlmostEqual(info["current_ratio"], 0.5, places=5)
        self.assertEqual(info["last_price"], current_price)
        self.assertEqual(info["target_ratio"], 0.5)

    def test_inventory_info_after_skew(self):
        """Test inventory_info after get_skew_factor."""
        self.portfolio.calculate_ratio({"BTC": 2.0, "JPY": 5000000.0}, 5000000.0)
        skew = self.portfolio.get_skew_factor(2.0 / 3.0)

        info = self.portfolio.get_inventory_info()

        self.assertAlmostEqual(info["current_ratio"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(info["skew_factor"], skew, places=5)


if __name__ == "__main__":
    unittest.main()
