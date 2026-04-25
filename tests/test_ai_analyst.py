"""Comprehensive test suite for FinGPT-style AI Analyst (Phase 4)."""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock

from agents.ai_analyst import FinGPTAnalyst, TradeSignal, EvaluatorScore


class TestChainOfThought(unittest.TestCase):
    """Test Chain of Thought analysis prompt generation."""

    def setUp(self):
        """Set up mock client and analyst."""
        self.mock_client = Mock()
        self.analyst = FinGPTAnalyst(self.mock_client)
        # Replace llm with a mock for testing
        self.analyst.llm = Mock()

    def test_quant_analyst_system_prompt_structure(self):
        """Test that system prompt contains all 4 analytical steps."""
        prompt = self.analyst._quant_analyst_system_prompt()
        self.assertIn("Macro Environment", prompt)
        self.assertIn("Orderbook Analysis", prompt)
        self.assertIn("Technical Indicator", prompt)
        self.assertIn("Synthesis", prompt)

    def test_quant_analyst_system_prompt_role(self):
        """Test that system prompt establishes institutional analyst role."""
        prompt = self.analyst._quant_analyst_system_prompt()
        self.assertIn("institutional", prompt.lower())
        self.assertIn("quantitative", prompt.lower())

    def test_cot_analysis_includes_symbol(self):
        """Test that CoT analysis prompt generation works with symbol."""
        self.analyst.llm.invoke = Mock(return_value=Mock(content="Macro: positive, Orderbook: bullish, Technical: BUY"))

        result = self.analyst._generate_chain_of_thought(
            symbol="BTC/JPY",
            current_price=5000000.0,
            orderbook={"bids": [[5000000, 1]], "asks": [[5001000, 1]]},
            ohlcv=[[0, 5000000, 5001000, 4999000, 5000000, 100]],
        )

        self.assertIsNotNone(result)
        self.assertIn("positive", result.lower())

    def test_cot_analysis_includes_orderbook(self):
        """Test that CoT analysis includes orderbook data."""
        self.analyst.llm.invoke = Mock(return_value=Mock(content="Analysis result"))

        orderbook = {"bids": [[5000000, 10], [4999000, 5]], "asks": [[5001000, 10], [5002000, 5]]}
        result = self.analyst._generate_chain_of_thought(
            symbol="BTC/JPY",
            current_price=5000000.0,
            orderbook=orderbook,
            ohlcv=[[0, 5000000, 5001000, 4999000, 5000000, 100]],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, "Analysis result")

    def test_cot_analysis_includes_ohlcv(self):
        """Test that CoT analysis includes OHLCV candle data."""
        self.analyst.llm.invoke = Mock(return_value=Mock(content="Analysis"))

        ohlcv = [[1000, 5000000, 5001000, 4999000, 5000500, 100]]
        result = self.analyst._generate_chain_of_thought(
            symbol="BTC/JPY",
            current_price=5000500.0,
            orderbook={"bids": [[5000000, 1]], "asks": [[5001000, 1]]},
            ohlcv=ohlcv,
        )

        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)


class TestISQSignalGeneration(unittest.TestCase):
    """Test ISQ Signal Framework extraction and validation."""

    def setUp(self):
        """Set up mock client and analyst."""
        self.mock_client = Mock()
        self.analyst = FinGPTAnalyst(self.mock_client)
        # Replace llm with a mock for testing
        self.analyst.llm = Mock()

    def test_confidence_range_valid(self):
        """Test that confidence is clamped to 0.0-1.0."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "BUY",
                        "price": 5100000,
                        "size": 0.01,
                        "confidence": 0.85,
                        "intensity": 3,
                        "expectation_gap": 0.6,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bullish"
            )

            self.assertEqual(signal.confidence, 0.85)

    def test_confidence_range_clamp_high(self):
        """Test that confidence > 1.0 is clamped to 1.0."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "BUY",
                        "price": 5100000,
                        "size": 0.01,
                        "confidence": 1.5,
                        "intensity": 3,
                        "expectation_gap": 0.5,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bullish"
            )

            self.assertEqual(signal.confidence, 1.0)

    def test_confidence_range_clamp_low(self):
        """Test that confidence < 0.0 is clamped to 0.0."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "SELL",
                        "price": 4900000,
                        "size": 0.01,
                        "confidence": -0.2,
                        "intensity": 2,
                        "expectation_gap": 0.3,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bearish"
            )

            self.assertEqual(signal.confidence, 0.0)

    def test_intensity_range_valid(self):
        """Test that intensity is integer 1-5."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "BUY",
                        "price": 5100000,
                        "size": 0.01,
                        "confidence": 0.8,
                        "intensity": 4,
                        "expectation_gap": 0.6,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bullish"
            )

            self.assertEqual(signal.intensity, 4)
            self.assertIsInstance(signal.intensity, int)

    def test_intensity_range_clamp_high(self):
        """Test that intensity > 5 is clamped to 5."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "BUY",
                        "price": 5100000,
                        "size": 0.01,
                        "confidence": 0.8,
                        "intensity": 10,
                        "expectation_gap": 0.6,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bullish"
            )

            self.assertEqual(signal.intensity, 5)

    def test_intensity_range_clamp_low(self):
        """Test that intensity < 1 is clamped to 1."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "HOLD",
                        "price": 5000000,
                        "size": 0.001,
                        "confidence": 0.3,
                        "intensity": 0,
                        "expectation_gap": 0.1,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="neutral"
            )

            self.assertEqual(signal.intensity, 1)

    def test_expectation_gap_range_valid(self):
        """Test that expectation_gap is valid 0.0-1.0."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "action": "BUY",
                        "price": 5100000,
                        "size": 0.01,
                        "confidence": 0.8,
                        "intensity": 4,
                        "expectation_gap": 0.75,
                    }
                )
            )

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="bullish"
            )

            self.assertEqual(signal.expectation_gap, 0.75)

    def test_buy_sell_hold_action_extraction(self):
        """Test that action field is correctly extracted."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            for action in ["BUY", "SELL", "HOLD"]:
                mock_invoke.return_value = Mock(
                    content=json.dumps(
                        {
                            "action": action,
                            "price": 5000000,
                            "size": 0.01,
                            "confidence": 0.7,
                            "intensity": 3,
                            "expectation_gap": 0.5,
                        }
                    )
                )

                signal = self.analyst._extract_isq_signal(
                    symbol="BTC/JPY", current_price=5000000.0, cot_analysis="test"
                )

                self.assertEqual(signal.action, action)

    def test_signal_extraction_malformed_json(self):
        """Test that malformed JSON falls back to HOLD."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(content="not valid json at all")

            signal = self.analyst._extract_isq_signal(
                symbol="BTC/JPY", current_price=5000000.0, cot_analysis="test"
            )

            self.assertEqual(signal.action, "HOLD")
            self.assertEqual(signal.size, 0.0)


class TestEvaluatorValidation(unittest.TestCase):
    """Test independent evaluator validation scores."""

    def setUp(self):
        """Set up mock client and analyst."""
        self.mock_client = Mock()
        self.analyst = FinGPTAnalyst(
            self.mock_client,
            confidence_threshold=0.6,
            consistency_threshold=0.3,
            actionability_threshold=0.5,
        )
        # Replace llm with a mock for testing
        self.analyst.llm = Mock()

    def test_data_sufficiency_sufficient(self):
        """Test data_sufficiency score when data is sufficient."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.85,
                        "consistency": 0.1,
                        "actionability": 0.8,
                        "notes": "Good data",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.85,
                intensity=4,
                expectation_gap=0.6,
                reasoning="bullish",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertGreaterEqual(score.data_sufficiency, 0.85)

    def test_consistency_aligned(self):
        """Test consistency score when analyses are aligned."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.8,
                        "consistency": 0.05,
                        "actionability": 0.75,
                        "notes": "Analyses align",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.8,
                intensity=4,
                expectation_gap=0.6,
                reasoning="bullish macro, bullish orderbook, bullish technical",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertLess(score.consistency, 0.2)

    def test_consistency_contradictory(self):
        """Test consistency score when analyses contradict."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.7,
                        "consistency": 0.85,
                        "actionability": 0.4,
                        "notes": "Macro bullish, technical bearish",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.5,
                intensity=2,
                expectation_gap=0.3,
                reasoning="conflicting signals",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertGreater(score.consistency, 0.6)

    def test_actionability_clear(self):
        """Test actionability score when signal is clear."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.85,
                        "consistency": 0.1,
                        "actionability": 0.9,
                        "notes": "Clear price and size targets",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.85,
                intensity=4,
                expectation_gap=0.6,
                reasoning="Clear target at 5100000",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertGreaterEqual(score.actionability, 0.85)

    def test_actionability_ambiguous(self):
        """Test actionability score when signal is ambiguous."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.6,
                        "consistency": 0.5,
                        "actionability": 0.3,
                        "notes": "Vague price target",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.001,
                confidence=0.4,
                intensity=1,
                expectation_gap=0.2,
                reasoning="maybe bullish",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertLess(score.actionability, 0.5)

    def test_validation_passes_all_thresholds(self):
        """Test passes_validation when all thresholds are met."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.8,
                        "consistency": 0.2,
                        "actionability": 0.75,
                        "notes": "Good signal",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.8,
                intensity=4,
                expectation_gap=0.6,
                reasoning="good analysis",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertTrue(score.passes_validation)

    def test_validation_fails_low_data_sufficiency(self):
        """Test validation fails when data_sufficiency < threshold."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.4,
                        "consistency": 0.2,
                        "actionability": 0.75,
                        "notes": "Insufficient data",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.8,
                intensity=4,
                expectation_gap=0.6,
                reasoning="limited data",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertFalse(score.passes_validation)

    def test_validation_fails_high_consistency(self):
        """Test validation fails when consistency > threshold."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.8,
                        "consistency": 0.5,
                        "actionability": 0.75,
                        "notes": "Contradictory",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.8,
                intensity=4,
                expectation_gap=0.6,
                reasoning="conflicting",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertFalse(score.passes_validation)

    def test_validation_fails_low_actionability(self):
        """Test validation fails when actionability < threshold."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = Mock(
                content=json.dumps(
                    {
                        "data_sufficiency": 0.8,
                        "consistency": 0.2,
                        "actionability": 0.3,
                        "notes": "Not actionable",
                    }
                )
            )

            signal = TradeSignal(
                action="BUY",
                price=5100000,
                size=0.01,
                confidence=0.8,
                intensity=4,
                expectation_gap=0.6,
                reasoning="unclear",
                validation=EvaluatorScore(0, 0, 0, False),
            )
            score = self.analyst._evaluate_signal("analysis", signal)

            self.assertFalse(score.passes_validation)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    def setUp(self):
        """Set up mock client and analyst."""
        self.mock_client = Mock()
        self.analyst = FinGPTAnalyst(self.mock_client)
        # Replace llm with a mock for testing
        self.analyst.llm = Mock()

    def test_full_pipeline_bullish_scenario_passes_validation(self):
        """Test full pipeline with bullish scenario that passes validation."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.side_effect = [
                Mock(content="Macro: positive, Orderbook: bullish, Technical: BUY"),
                Mock(
                    content=json.dumps(
                        {
                            "action": "BUY",
                            "price": 5100000,
                            "size": 0.01,
                            "confidence": 0.85,
                            "intensity": 4,
                            "expectation_gap": 0.6,
                        }
                    )
                ),
                Mock(
                    content=json.dumps(
                        {
                            "data_sufficiency": 0.85,
                            "consistency": 0.1,
                            "actionability": 0.9,
                            "notes": "Strong bullish signal",
                        }
                    )
                ),
            ]

            signal = self.analyst.analyze_market(
                symbol="BTC/JPY",
                current_price=5000000.0,
                orderbook={"bids": [[5000000, 100], [4999000, 50]], "asks": [[5001000, 100]]},
                ohlcv=[[0, 5000000, 5001000, 4999000, 5000500, 100]],
            )

            self.assertEqual(signal.action, "BUY")
            self.assertGreater(signal.confidence, 0.8)
            self.assertTrue(signal.validation.passes_validation)

    def test_full_pipeline_bearish_scenario_passes_validation(self):
        """Test full pipeline with bearish scenario that passes validation."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.side_effect = [
                Mock(content="Macro: negative, Orderbook: bearish, Technical: SELL"),
                Mock(
                    content=json.dumps(
                        {
                            "action": "SELL",
                            "price": 4900000,
                            "size": 0.01,
                            "confidence": 0.82,
                            "intensity": 3,
                            "expectation_gap": 0.5,
                        }
                    )
                ),
                Mock(
                    content=json.dumps(
                        {
                            "data_sufficiency": 0.8,
                            "consistency": 0.15,
                            "actionability": 0.85,
                            "notes": "Strong bearish signal",
                        }
                    )
                ),
            ]

            signal = self.analyst.analyze_market(
                symbol="BTC/JPY",
                current_price=5000000.0,
                orderbook={"bids": [[4999000, 50]], "asks": [[5001000, 100], [5002000, 200]]},
                ohlcv=[[0, 5000000, 5001000, 4999000, 4999500, 100]],
            )

            self.assertEqual(signal.action, "SELL")
            self.assertTrue(signal.validation.passes_validation)

    def test_no_trade_on_validation_failure(self):
        """Test that validation failure triggers NO_TRADE."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.side_effect = [
                Mock(content="Unclear analysis"),
                Mock(
                    content=json.dumps(
                        {
                            "action": "BUY",
                            "price": 5100000,
                            "size": 0.01,
                            "confidence": 0.3,
                            "intensity": 1,
                            "expectation_gap": 0.1,
                        }
                    )
                ),
                Mock(
                    content=json.dumps(
                        {
                            "data_sufficiency": 0.4,
                            "consistency": 0.6,
                            "actionability": 0.3,
                            "notes": "Low quality signal",
                        }
                    )
                ),
            ]

            signal = self.analyst.analyze_market(
                symbol="BTC/JPY",
                current_price=5000000.0,
                orderbook={"bids": [[5000000, 10]], "asks": [[5001000, 10]]},
                ohlcv=[[0, 5000000, 5001000, 4999000, 5000000, 100]],
            )

            self.assertEqual(signal.action, "NO_TRADE")

    def test_error_handling_fallback_to_no_trade(self):
        """Test that API errors gracefully fall back to NO_TRADE."""
        with patch.object(self.analyst.llm, "invoke") as mock_invoke:
            mock_invoke.side_effect = Exception("LLM API error")

            signal = self.analyst.analyze_market(
                symbol="BTC/JPY",
                current_price=5000000.0,
                orderbook={"bids": [[5000000, 10]], "asks": [[5001000, 10]]},
                ohlcv=[[0, 5000000, 5001000, 4999000, 5000000, 100]],
            )

            self.assertEqual(signal.action, "NO_TRADE")
            self.assertFalse(signal.validation.passes_validation)


if __name__ == "__main__":
    unittest.main()
