"""Unit tests for advanced backtest metrics (Phase 3)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deploy.run_backtest import _summarize


class TestSortinoRatio(unittest.TestCase):
    """Test Sortino Ratio calculation with various volatility scenarios."""

    def test_sortino_ratio_all_positive(self):
        """Test Sortino Ratio when all trades are profitable."""
        pnl_list = [100.0, 150.0, 200.0, 120.0, 180.0]
        result = _summarize("test", pnl_list)
        # All positive: downside_volatility = 0, sortino = inf
        self.assertEqual(result.sortino_ratio, float('inf'))

    def test_sortino_ratio_all_negative(self):
        """Test Sortino Ratio when all trades are losing."""
        pnl_list = [-50.0, -100.0, -75.0, -60.0, -80.0]
        result = _summarize("test", pnl_list)
        # All negative: mean_return < 0, downside_volatility > 0, sortino < 0
        self.assertLess(result.sortino_ratio, 0)

    def test_sortino_ratio_balanced(self):
        """Test Sortino Ratio with mixed wins and losses."""
        pnl_list = [100.0, -50.0, 150.0, -40.0, 200.0]
        result = _summarize("test", pnl_list)
        # Mean return = 360 / 5 = 72
        # Downside returns = [-50, -40]
        # Downside volatility = sqrt((2500 + 1600) / 5) = sqrt(840) ≈ 29
        # Sortino ≈ 72 / 29 ≈ 2.48
        self.assertGreater(result.sortino_ratio, 2.0)
        self.assertLess(result.sortino_ratio, 3.0)

    def test_sortino_ratio_high_volatility(self):
        """Test Sortino Ratio with high downside volatility."""
        pnl_list = [50.0, -500.0, 75.0, -450.0, 100.0]
        result = _summarize("test", pnl_list)
        # Large downside swings → low Sortino
        self.assertLess(result.sortino_ratio, 1.0)

    def test_sortino_ratio_low_volatility(self):
        """Test Sortino Ratio with low downside volatility."""
        pnl_list = [100.0, -5.0, 110.0, -3.0, 105.0]
        result = _summarize("test", pnl_list)
        # Small downside swings → high Sortino
        self.assertGreater(result.sortino_ratio, 5.0)


class TestProfitFactor(unittest.TestCase):
    """Test Profit Factor calculation with various win/loss scenarios."""

    def test_profit_factor_no_losses(self):
        """Test Profit Factor when there are no losing trades."""
        pnl_list = [100.0, 150.0, 200.0]
        result = _summarize("test", pnl_list)
        self.assertEqual(result.profit_factor, float('inf'))

    def test_profit_factor_balanced(self):
        """Test Profit Factor when wins equal losses."""
        pnl_list = [100.0, -100.0, 150.0, -150.0]
        result = _summarize("test", pnl_list)
        # Total wins = 250, Total losses = 250
        # Profit Factor = 250 / 250 = 1.0
        self.assertEqual(result.profit_factor, 1.0)

    def test_profit_factor_profitable(self):
        """Test Profit Factor with 2:1 win/loss ratio."""
        pnl_list = [100.0, 100.0, -50.0]
        result = _summarize("test", pnl_list)
        # Total wins = 200, Total losses = 50
        # Profit Factor = 200 / 50 = 4.0
        self.assertEqual(result.profit_factor, 4.0)

    def test_profit_factor_all_losses(self):
        """Test Profit Factor when all trades are losses."""
        pnl_list = [-50.0, -100.0, -75.0]
        result = _summarize("test", pnl_list)
        self.assertEqual(result.profit_factor, 0.0)


class TestMaxConsecutiveLosses(unittest.TestCase):
    """Test Max Consecutive Losses streak detection."""

    def test_max_consecutive_losses_none(self):
        """Test when all trades are profitable."""
        pnl_list = [100.0, 150.0, 200.0]
        result = _summarize("test", pnl_list)
        self.assertEqual(result.max_consecutive_losses, 0)

    def test_max_consecutive_losses_scattered(self):
        """Test with losses scattered throughout."""
        pnl_list = [100.0, -50.0, 150.0, -40.0, 200.0, -30.0]
        result = _summarize("test", pnl_list)
        # Max streak = 1 (no consecutive losses)
        self.assertEqual(result.max_consecutive_losses, 1)

    def test_max_consecutive_losses_consecutive(self):
        """Test with multiple consecutive losses."""
        pnl_list = [100.0, -50.0, -40.0, -30.0, 200.0, -10.0, -20.0]
        result = _summarize("test", pnl_list)
        # Max streak = 3 (at indices 1, 2, 3)
        self.assertEqual(result.max_consecutive_losses, 3)

    def test_max_consecutive_losses_all_negative(self):
        """Test when all trades are losses."""
        pnl_list = [-50.0, -100.0, -75.0, -60.0]
        result = _summarize("test", pnl_list)
        self.assertEqual(result.max_consecutive_losses, 4)


class TestAvgWinLossRatio(unittest.TestCase):
    """Test Average Win/Loss Ratio calculation."""

    def test_avg_win_loss_ratio_no_wins(self):
        """Test when there are no winning trades."""
        pnl_list = [-50.0, -100.0, -75.0]
        result = _summarize("test", pnl_list)
        self.assertIsNone(result.avg_win_loss_ratio)

    def test_avg_win_loss_ratio_no_losses(self):
        """Test when there are no losing trades."""
        pnl_list = [100.0, 150.0, 200.0]
        result = _summarize("test", pnl_list)
        self.assertIsNone(result.avg_win_loss_ratio)

    def test_avg_win_loss_ratio_balanced(self):
        """Test with equal average win and loss."""
        pnl_list = [100.0, 100.0, -50.0, -50.0]
        result = _summarize("test", pnl_list)
        # Avg win = 200 / 2 = 100
        # Avg loss = 100 / 2 = 50
        # Ratio = 100 / 50 = 2.0
        self.assertEqual(result.avg_win_loss_ratio, 2.0)

    def test_avg_win_loss_ratio_unequal(self):
        """Test with unequal average win and loss."""
        pnl_list = [200.0, 300.0, -50.0, -25.0]
        result = _summarize("test", pnl_list)
        # Avg win = 500 / 2 = 250
        # Avg loss = 75 / 2 = 37.5
        # Ratio = 250 / 37.5 ≈ 6.67
        self.assertAlmostEqual(result.avg_win_loss_ratio, 6.67, places=1)


class TestIntegrationSummarize(unittest.TestCase):
    """Integration tests for all metrics in _summarize()."""

    def test_summarize_empty_trades(self):
        """Test summarize with empty trade list."""
        result = _summarize("test", [])
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.pnl, 0.0)
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.max_drawdown, 0.0)

    def test_summarize_single_trade_win(self):
        """Test summarize with single winning trade."""
        result = _summarize("test", [100.0])
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.pnl, 100.0)
        self.assertEqual(result.win_rate, 100.0)
        self.assertEqual(result.max_drawdown, 0.0)
        self.assertEqual(result.sortino_ratio, float('inf'))

    def test_summarize_single_trade_loss(self):
        """Test summarize with single losing trade."""
        result = _summarize("test", [-50.0])
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.pnl, -50.0)
        self.assertEqual(result.win_rate, 0.0)
        self.assertIsNone(result.avg_win_loss_ratio)

    def test_summarize_realistic_scenario(self):
        """Test with realistic mixed trades."""
        pnl_list = [100.0, 150.0, -50.0, 200.0, -30.0, 120.0, -40.0, 180.0]
        result = _summarize("test", pnl_list)

        # Verify basic metrics
        self.assertEqual(result.trades, 8)
        self.assertEqual(result.pnl, 630.0)
        self.assertEqual(result.win_rate, 62.5)  # 5 wins out of 8

        # Verify new metrics exist and are valid
        self.assertIsNotNone(result.sortino_ratio)
        self.assertIsNotNone(result.profit_factor)
        self.assertGreater(result.profit_factor, 1.0)  # Profitable
        self.assertEqual(result.max_consecutive_losses, 1)  # Single losses scattered
        self.assertIsNotNone(result.avg_win_loss_ratio)


class TestBackwardCompatibility(unittest.TestCase):
    """Verify backward compatibility of new metrics."""

    def test_existing_fields_unchanged(self):
        """Test that existing BacktestResult fields are not affected."""
        pnl_list = [100.0, -50.0, 150.0]
        result = _summarize("test_strategy", pnl_list)

        # Existing fields must be present and correct
        self.assertEqual(result.strategy, "test_strategy")
        self.assertEqual(result.trades, 3)
        self.assertEqual(result.pnl, 200.0)
        self.assertAlmostEqual(result.win_rate, 66.67, places=1)
        self.assertGreaterEqual(result.max_drawdown, 0.0)

    def test_new_fields_optional(self):
        """Test that new fields are optional in BacktestResult."""
        # All new fields should have default values when not specified
        result = _summarize("test", [100.0, 150.0, -50.0])
        self.assertIsNotNone(result.sortino_ratio)
        self.assertIsNotNone(result.profit_factor)
        self.assertEqual(result.max_consecutive_losses, 1)  # One loss at end
        self.assertIsNotNone(result.avg_win_loss_ratio)  # Ratio is calculable


if __name__ == "__main__":
    unittest.main()
