"""FinGPT-style institutional quantitative analyst engine.

Provides multi-dimensional signal evaluation with Chain of Thought analysis,
ISQ (Institutional Signal Quality) framework, and independent evaluator gate.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatOpenAI = None

    # Dummy message classes for testing
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

from agents.api_clients import DomesticClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorScore:
    """Self-validation scores for trade signals."""

    data_sufficiency: float  # 0.0-1.0: Do we have enough reliable data?
    consistency: float  # 0.0-1.0: Contradiction score (0=aligned, 1=contradictory)
    actionability: float  # 0.0-1.0: Can trader execute reliably?
    passes_validation: bool  # True if all thresholds met
    notes: str = ""


@dataclass
class TradeSignal:
    """Complete trade signal with ISQ scores and validation."""

    action: str  # "BUY" | "SELL" | "HOLD" | "NO_TRADE"
    price: float  # Limit price in quote currency
    size: float  # Order size in base currency
    confidence: float  # 0.0-1.0: Certainty of analysis
    intensity: int  # 1-5: Impact strength on price
    expectation_gap: float  # 0.0-1.0: Market pricing gap
    reasoning: str  # Chain of Thought analysis result
    validation: EvaluatorScore  # Self-validation scores


class FinGPTAnalyst:
    """Institutional quantitative analyst using FinGPT methodology.

    Combines:
    - Chain of Thought analysis (macro → orderbook → technical → synthesis)
    - ISQ Signal Framework (Confidence, Intensity, Expectation Gap)
    - Independent Evaluator (Data Sufficiency, Consistency, Actionability)
    """

    def __init__(
        self,
        client: DomesticClient,
        model: str = "gpt-4-turbo",
        confidence_threshold: float = 0.6,
        consistency_threshold: float = 0.3,
        actionability_threshold: float = 0.5,
    ) -> None:
        self.client = client
        self.model = model
        self.llm = None
        if HAS_LANGCHAIN:
            self.llm = ChatOpenAI(model=model, temperature=0.7)

        # Validation thresholds
        self.confidence_threshold = confidence_threshold
        self.consistency_threshold = consistency_threshold
        self.actionability_threshold = actionability_threshold

    def analyze_market(
        self,
        symbol: str,
        current_price: float,
        orderbook: dict[str, Any],
        ohlcv: list[list[float]],
        macro_context: str = "",
    ) -> TradeSignal:
        """Analyze market and generate ISQ-scored trade signal.

        Args:
            symbol: Trading pair (e.g., "BTC/JPY")
            current_price: Current mid-price in quote currency
            orderbook: Dict with 'bids' and 'asks' lists
            ohlcv: OHLCV candle data (list of lists)
            macro_context: Optional macro environment context

        Returns:
            TradeSignal with ISQ scores and validation
        """
        try:
            # Generate Chain of Thought analysis
            cot_analysis = self._generate_chain_of_thought(
                symbol=symbol,
                current_price=current_price,
                orderbook=orderbook,
                ohlcv=ohlcv,
                macro_context=macro_context,
            )

            # Extract ISQ signal from analysis
            signal = self._extract_isq_signal(
                symbol=symbol,
                current_price=current_price,
                cot_analysis=cot_analysis,
            )

            # Self-validate the signal
            validation = self._evaluate_signal(cot_analysis, signal)
            signal.validation = validation

            # Check safety gate: if validation fails, override to NO_TRADE
            if not validation.passes_validation:
                logger.warning(
                    "signal validation failed: sufficiency=%.2f consistency=%.2f actionability=%.2f",
                    validation.data_sufficiency,
                    validation.consistency,
                    validation.actionability,
                )
                signal.action = "NO_TRADE"

            return signal

        except Exception as e:
            logger.error("market analysis failed: %s", e, exc_info=True)
            # Return safe NO_TRADE signal on error
            return TradeSignal(
                action="NO_TRADE",
                price=current_price,
                size=0.0,
                confidence=0.0,
                intensity=1,
                expectation_gap=0.0,
                reasoning=f"Error during analysis: {str(e)}",
                validation=EvaluatorScore(
                    data_sufficiency=0.0,
                    consistency=1.0,
                    actionability=0.0,
                    passes_validation=False,
                    notes="Analysis error",
                ),
            )

    def _generate_chain_of_thought(
        self,
        symbol: str,
        current_price: float,
        orderbook: dict[str, Any],
        ohlcv: list[list[float]],
        macro_context: str = "",
    ) -> str:
        """Generate step-by-step Chain of Thought analysis.

        Enforces structured reasoning: Macro → Orderbook → Technical → Synthesis.

        Returns:
            Full text of Chain of Thought analysis
        """
        system_prompt = self._quant_analyst_system_prompt()

        human_prompt = f"""
Analyze {symbol} for a trade signal.

Current Price: {current_price:.0f} JPY

Orderbook (top 5):
Bids: {orderbook.get('bids', [])[:5]}
Asks: {orderbook.get('asks', [])[:5]}

Recent OHLCV (last 14 candles):
{self._format_ohlcv(ohlcv[-14:])}

Macro Context:
{macro_context if macro_context else "Normal market conditions."}

Follow the analytical process strictly:
1. Macro Environment: What is the macro-level BTC/JPY outlook?
2. Orderbook Analysis: What do the buy/sell wall imbalances tell us?
3. Technical Indicators: What do RSI, MACD, and moving averages show?
4. Synthesis: Combine all signals into a unified trading direction.

Provide explicit reasoning for each step. Be specific about price targets and size.
"""

        if not self.llm:
            logger.warning("LangChain not available; returning placeholder analysis")
            return "Placeholder analysis: LangChain not installed. Run: pip install langchain-openai langchain-core"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _extract_isq_signal(
        self,
        symbol: str,
        current_price: float,
        cot_analysis: str,
    ) -> TradeSignal:
        """Extract ISQ signal from Chain of Thought analysis.

        Uses LLM to extract:
        - action (BUY, SELL, HOLD)
        - price (limit price in JPY)
        - size (order size in BTC)
        - confidence (0.0-1.0)
        - intensity (1-5)
        - expectation_gap (0.0-1.0)

        Returns:
            TradeSignal object (without validation scores yet)
        """
        system_prompt = "You are a JSON extraction expert for trading signals."

        human_prompt = f"""
From the analysis below, extract a JSON trade signal.

Analysis:
{cot_analysis}

Extract to JSON format (ONLY return valid JSON):
{{
    "action": "BUY" | "SELL" | "HOLD",
    "price": <float>,
    "size": <float>,
    "confidence": <0.0-1.0>,
    "intensity": <1-5>,
    "expectation_gap": <0.0-1.0>
}}

Confidence: How certain is this analysis? (0.0=guessing, 1.0=certainty)
Intensity: How strongly will this signal move the price? (1=noise, 5=system-wide shift)
Expectation Gap: How much has the market priced this in? (0.0=fully priced, 1.0=undiscovered)
"""

        if not self.llm:
            logger.warning("LangChain not available; returning default HOLD signal")
            return TradeSignal(
                action="HOLD",
                price=current_price,
                size=0.0,
                confidence=0.0,
                intensity=1,
                expectation_gap=0.0,
                reasoning="LangChain not installed",
                validation=EvaluatorScore(
                    data_sufficiency=0.0,
                    consistency=0.0,
                    actionability=0.0,
                    passes_validation=False,
                    notes="LangChain not available",
                ),
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)

        try:
            signal_dict = json.loads(response.content)

            # Validate ranges
            signal_dict["confidence"] = max(0.0, min(1.0, signal_dict.get("confidence", 0.5)))
            signal_dict["intensity"] = max(1, min(5, int(signal_dict.get("intensity", 3))))
            signal_dict["expectation_gap"] = max(
                0.0, min(1.0, signal_dict.get("expectation_gap", 0.5))
            )
            signal_dict["size"] = max(0.0, signal_dict.get("size", 0.001))

            return TradeSignal(
                action=signal_dict.get("action", "HOLD"),
                price=float(signal_dict.get("price", current_price)),
                size=float(signal_dict["size"]),
                confidence=float(signal_dict["confidence"]),
                intensity=int(signal_dict["intensity"]),
                expectation_gap=float(signal_dict["expectation_gap"]),
                reasoning=cot_analysis,
                validation=EvaluatorScore(
                    data_sufficiency=0.0,
                    consistency=0.0,
                    actionability=0.0,
                    passes_validation=False,
                ),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("failed to extract ISQ signal: %s", e)
            # Return HOLD signal on extraction failure
            return TradeSignal(
                action="HOLD",
                price=current_price,
                size=0.0,
                confidence=0.0,
                intensity=1,
                expectation_gap=0.0,
                reasoning=cot_analysis,
                validation=EvaluatorScore(
                    data_sufficiency=0.0,
                    consistency=0.0,
                    actionability=0.0,
                    passes_validation=False,
                    notes="Signal extraction failed",
                ),
            )

    def _evaluate_signal(
        self,
        cot_analysis: str,
        signal: TradeSignal,
    ) -> EvaluatorScore:
        """Self-validate signal using independent evaluator.

        Scores:
        - data_sufficiency: Do we have enough data? (0.0-1.0)
        - consistency: Do analyses align or contradict? (0.0=aligned, 1.0=contradictory)
        - actionability: Can trader execute? (0.0-1.0)

        Returns:
            EvaluatorScore with passes_validation boolean
        """
        system_prompt = "You are an independent trade signal validator."

        human_prompt = f"""
Validate this trade signal:

Analysis:
{cot_analysis}

Signal:
- Action: {signal.action}
- Price: {signal.price:.0f} JPY
- Size: {signal.size:.6f} BTC
- Confidence: {signal.confidence:.2f}
- Intensity: {signal.intensity}
- Expectation Gap: {signal.expectation_gap:.2f}

Score the following dimensions (0.0-1.0):
1. data_sufficiency: Do we have enough reliable data for this call?
2. consistency: Do macro/orderbook/technical analyses align or contradict? (0=aligned, 1=contradictory)
3. actionability: Can a real trader execute this reliably?

Return ONLY valid JSON:
{{
    "data_sufficiency": <float>,
    "consistency": <float>,
    "actionability": <float>,
    "notes": "<brief explanation>"
}}
"""

        if not self.llm:
            logger.warning("LangChain not available; returning default validation failure")
            return EvaluatorScore(
                data_sufficiency=0.0,
                consistency=1.0,
                actionability=0.0,
                passes_validation=False,
                notes="LangChain not available",
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            evaluation = json.loads(response.content)

            data_suff = max(0.0, min(1.0, float(evaluation.get("data_sufficiency", 0.0))))
            consistency = max(0.0, min(1.0, float(evaluation.get("consistency", 1.0))))
            actionability = max(0.0, min(1.0, float(evaluation.get("actionability", 0.0))))
            notes = evaluation.get("notes", "")

            # Validation gate: all must pass thresholds
            passes = (
                data_suff >= self.confidence_threshold
                and consistency <= self.consistency_threshold
                and actionability >= self.actionability_threshold
            )

            return EvaluatorScore(
                data_sufficiency=data_suff,
                consistency=consistency,
                actionability=actionability,
                passes_validation=passes,
                notes=notes,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("evaluator validation failed: %s", e)
            return EvaluatorScore(
                data_sufficiency=0.0,
                consistency=1.0,
                actionability=0.0,
                passes_validation=False,
                notes=f"Evaluation error: {str(e)}",
            )

    @staticmethod
    def _quant_analyst_system_prompt() -> str:
        """System prompt for institutional quantitative analyst."""
        return """
You are an institutional-level quantitative analyst and trading strategist.

Your role:
- Analyze cryptocurrency markets with precision and transparency
- Provide multi-dimensional quantitative assessments
- Explain your reasoning step by step
- Flag uncertainty and data gaps clearly
- Generate actionable trading signals

Analytical Process (Chain of Thought):
1. Macro Environment Analysis
   - Analyze global interest rates, risk sentiment, and macro trends
   - Assess their impact on BTC/JPY demand and supply dynamics
   - Consider macroeconomic fundamentals affecting the pair

2. Orderbook Analysis
   - Evaluate buy/sell wall imbalances at top of book
   - Assess market microstructure and order flow toxicity
   - Identify market maker presence and liquidity conditions
   - Estimate probability of price moving up vs. down based on order imbalance

3. Technical Indicator Analysis
   - Calculate and interpret RSI (momentum and overbought/oversold conditions)
   - Analyze MACD (trend and momentum divergences)
   - Assess moving average positions and crossover signals
   - Identify key support/resistance levels from recent price action

4. Synthesis
   - Combine macro + orderbook + technical analyses into unified signal
   - Identify convergences (multiple signals in same direction = stronger)
   - Flag divergences (conflicting signals = lower confidence)
   - Assign direction (BUY / SELL / HOLD) with specific price target
   - Provide order size recommendation

Output all analysis steps explicitly. Be concrete with numbers.
"""

    @staticmethod
    def _format_ohlcv(ohlcv: list[list[float]]) -> str:
        """Format OHLCV data for human readability."""
        if not ohlcv:
            return "No OHLCV data available"

        lines = []
        for candle in ohlcv:
            if len(candle) >= 5:
                ts, o, h, l, c, v = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
                lines.append(f"  O:{o:.0f} H:{h:.0f} L:{l:.0f} C:{c:.0f} V:{v:.0f}")

        return "\n".join(lines)
