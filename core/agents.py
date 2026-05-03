"""Autonomous agents with inter-agent negotiation for the AutoRAG++ pipeline.

Three coordinated agents — Retriever, Reasoner, Evaluator — share session-level
memory M_t through an asynchronous negotiation protocol.  Each agent makes
autonomous decisions: the Retriever selects retrieval strategy based on query
analysis and evaluator feedback; the Reasoner conditions on session memory; the
Evaluator issues structured calibration directives.
"""

from __future__ import annotations

import abc
import math
import logging
import time
from dataclasses import dataclass

from core.memory import (
    AgentMessage,
    MessageType,
    QueryRecord,
    RetrievalStrategy,
    SessionMemory,
)
from core.config import calibration, retrieval
from core.embedder import Embedder
from core.vectorstore import VectorStore
from core.retriever import Retriever, RetrievalResult, RetrievedChunk
from core.reasoner import Reasoner as ReasonerEngine, ReasonerResult, SYSTEM_PROMPT, RECALIBRATION_PROMPT
from core.evaluator import Evaluator as EvaluatorEngine, EvaluationResult
from core.calibrator import Calibrator, CalibrationStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Abstract base for all pipeline agents."""

    name: str = "base"

    def __init__(self, memory: SessionMemory) -> None:
        self._memory = memory

    def _post(self, to: str, msg_type: MessageType, payload: dict) -> None:
        self._memory.post(AgentMessage(
            from_agent=self.name,
            to_agent=to,
            msg_type=msg_type,
            payload=payload,
        ))


# ---------------------------------------------------------------------------
# Retriever Agent
# ---------------------------------------------------------------------------

class RetrieverAgent(BaseAgent):
    """Selects retrieval strategy and executes search.

    Strategy selection is autonomous — based on query complexity analysis and
    evaluator feedback from session memory (no extra LLM call needed).
    """

    name = "retriever"

    def __init__(
        self,
        memory: SessionMemory,
        retriever: Retriever,
        embedder: Embedder,
        vectorstore: VectorStore,
    ) -> None:
        super().__init__(memory)
        self._retriever = retriever
        self._embedder = embedder
        self._vectorstore = vectorstore

    # -- Strategy selection (autonomous decision-making) ----------------------

    def analyze_and_decide(
        self,
        query: str,
        *,
        k: int | None = None,
        lambda_1: float | None = None,
        lambda_2: float | None = None,
        user_id: str | None = None,
    ) -> tuple[RetrievalStrategy, RetrievalResult]:
        """Analyze the query, choose a strategy, execute retrieval.

        Decision factors:
        1. Query complexity (length, vocabulary diversity, question type)
        2. Past evaluator feedback from session memory
        3. Current calibration cycle
        """
        strategy = self._select_strategy(query)
        k = k or retrieval.initial_k
        l1 = lambda_1 if lambda_1 is not None else retrieval.lambda_1
        l2 = lambda_2 if lambda_2 is not None else retrieval.lambda_2

        # Post analysis to shared memory
        self._post("all", MessageType.QUERY_ANALYSIS, {
            "query": query,
            "strategy": strategy.value,
            "k": k,
            "lambda_1": l1,
            "lambda_2": l2,
        })

        # Execute chosen strategy
        result = self._execute_strategy(strategy, query, k, l1, l2, user_id)

        # Post result to shared memory
        self._post("reasoner", MessageType.RETRIEVAL_RESULT, {
            "strategy": strategy.value,
            "k": k,
            "chunks_returned": len(result.chunks),
            "top_score": result.chunks[0].hybrid_score if result.chunks else 0.0,
        })

        return strategy, result

    def re_retrieve(
        self,
        query: str,
        k: int,
        lambda_1: float,
        lambda_2: float,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """Re-retrieve after calibration with adjusted parameters."""
        # During calibration always use hybrid (the calibrated parameters matter)
        return self._retriever.retrieve(
            query, k=k, lambda_1=lambda_1, lambda_2=lambda_2, user_id=user_id,
        )

    def _select_strategy(self, query: str) -> RetrievalStrategy:
        """Autonomous strategy selection based on query analysis + memory."""

        # Factor 1: Check evaluator feedback from session memory
        feedback = self._memory.last_message(MessageType.EVALUATION_FEEDBACK)
        if feedback and feedback.payload.get("needs_expansion"):
            return RetrievalStrategy.EXPANDED

        # Factor 2: Query complexity heuristics (no LLM call — fast)
        complexity = self._compute_query_complexity(query)

        if complexity < 0.3:
            # Simple factual query → dense is sufficient
            return RetrievalStrategy.DENSE
        elif complexity > 0.7:
            # Complex multi-faceted query → expanded retrieval
            return RetrievalStrategy.EXPANDED
        else:
            # Medium complexity → standard hybrid
            return RetrievalStrategy.HYBRID

    def _compute_query_complexity(self, query: str) -> float:
        """Estimate query complexity from 0.0 (simple) to 1.0 (complex).

        Uses query entropy (vocabulary diversity), length, and structural cues.
        This is a rule-based heuristic — zero latency.
        """
        words = query.lower().split()
        n = len(words)

        if n == 0:
            return 0.0

        # Length signal: longer queries tend to be more complex
        length_score = min(n / 25.0, 1.0)

        # Vocabulary diversity (type-token ratio)
        unique_words = set(words)
        ttr = len(unique_words) / n

        # Structural complexity cues
        complexity_markers = {
            "how", "why", "compare", "contrast", "difference", "between",
            "explain", "analyze", "relationship", "impact", "affect",
            "multiple", "several", "various", "comprehensive",
        }
        marker_count = sum(1 for w in unique_words if w in complexity_markers)
        marker_score = min(marker_count / 3.0, 1.0)

        # Sub-question detection (presence of conjunctions/commas)
        subq_score = 0.0
        for marker in [" and ", " or ", ", ", " also ", " additionally "]:
            if marker in query.lower():
                subq_score = min(subq_score + 0.3, 1.0)

        # Weighted combination
        complexity = (
            0.25 * length_score
            + 0.25 * ttr
            + 0.30 * marker_score
            + 0.20 * subq_score
        )
        return min(max(complexity, 0.0), 1.0)

    def _execute_strategy(
        self,
        strategy: RetrievalStrategy,
        query: str,
        k: int,
        l1: float,
        l2: float,
        user_id: str | None,
    ) -> RetrievalResult:
        if strategy == RetrievalStrategy.EXPANDED:
            expanded_k = min(k + 3, k * 2)
            return self._retriever.retrieve(
                query, k=expanded_k, lambda_1=l1, lambda_2=l2, user_id=user_id,
            )
        else:  # DENSE and HYBRID both use hybrid retrieval for consistent ranking
            return self._retriever.retrieve(
                query, k=k, lambda_1=l1, lambda_2=l2, user_id=user_id,
            )


# ---------------------------------------------------------------------------
# Reasoner Agent
# ---------------------------------------------------------------------------

class ReasonerAgent(BaseAgent):
    """Generates context-aware responses: R_t = G_phi(Q_t, K_t, M_t)."""

    name = "reasoner"

    def __init__(self, memory: SessionMemory, engine: ReasonerEngine) -> None:
        super().__init__(memory)
        self._engine = engine

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        is_recalibration: bool = False,
    ) -> ReasonerResult:
        """Generate a response using retrieved context and session memory.

        The Reasoner conditions on M_t by injecting session context into the
        system prompt — the LLM sees past query patterns and confidence trends.
        """
        # Build memory-augmented system prompt
        system_prompt = self._build_memory_aware_prompt(is_recalibration)

        result = self._engine.reason(query, chunks, system_prompt=system_prompt)

        # Post to shared memory
        self._post("evaluator", MessageType.GENERATION_RESULT, {
            "model_used": result.model_used,
            "fallback_level": result.fallback_level,
            "source_count": result.source_count,
            "has_error": result.error is not None,
        })

        return result

    def generate_without_retrieval(self, query: str) -> ReasonerResult:
        return self._engine.reason_without_retrieval(query)

    def _build_memory_aware_prompt(self, is_recalibration: bool) -> str:
        """Inject session memory context into the system prompt.

        Only injects session context during recalibration cycles to ensure
        deterministic initial responses for the same query.
        """
        base = RECALIBRATION_PROMPT if is_recalibration else SYSTEM_PROMPT

        if not is_recalibration:
            return base

        # Add session context only during recalibration
        session_ctx = self._memory.context_summary()
        if session_ctx:
            return f"{base}\n\nSession context:\n{session_ctx}"
        return base


# ---------------------------------------------------------------------------
# Evaluator Agent
# ---------------------------------------------------------------------------

class EvaluatorAgent(BaseAgent):
    """Evaluates responses and sends structured calibration feedback.

    Computes S_c (semantic consistency) and F_c (factual correctness) using
    models independent of the generator, then issues directives to the
    Retriever Agent via the shared memory bus.
    """

    name = "evaluator"

    def __init__(self, memory: SessionMemory, engine: EvaluatorEngine) -> None:
        super().__init__(memory)
        self._engine = engine

    def evaluate(
        self,
        response: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        """Run evaluation and post feedback for other agents."""
        result = self._engine.evaluate(response, chunks)

        # Determine specific feedback for Retriever Agent
        needs_calibration = result.confidence < calibration.confidence_threshold
        weak_component = self._identify_weak_component(result)

        # Post structured feedback to shared memory (negotiation protocol)
        self._post("retriever", MessageType.EVALUATION_FEEDBACK, {
            "s_c": result.semantic_consistency,
            "f_c": result.factual_correctness,
            "c_f": result.confidence,
            "threshold": calibration.confidence_threshold,
            "passed": not needs_calibration,
            "weak_component": weak_component,
            "needs_expansion": needs_calibration and result.confidence < 0.5,
            "directive": self._generate_directive(result, weak_component),
        })

        return result

    def should_calibrate(self, eval_result: EvaluationResult) -> bool:
        return eval_result.confidence < calibration.confidence_threshold

    def _identify_weak_component(self, result: EvaluationResult) -> str:
        if result.semantic_consistency < result.factual_correctness:
            return "semantic_consistency"
        elif result.factual_correctness < result.semantic_consistency:
            return "factual_correctness"
        return "balanced"

    def _generate_directive(self, result: EvaluationResult, weak: str) -> str:
        """Generate a human-readable calibration directive."""
        if result.confidence >= calibration.confidence_threshold:
            return "ACCEPT: confidence meets threshold"

        if weak == "semantic_consistency":
            return (
                f"CALIBRATE: S_c={result.semantic_consistency:.3f} is weak. "
                "Increase cosine weight (lambda_1) to improve retrieval relevance."
            )
        elif weak == "factual_correctness":
            return (
                f"CALIBRATE: F_c={result.factual_correctness:.3f} is weak. "
                "Increase density weight (lambda_2) and fetch new sources for better grounding."
            )
        return (
            f"CALIBRATE: Both components low (S_c={result.semantic_consistency:.3f}, "
            f"F_c={result.factual_correctness:.3f}). Expand retrieval and fetch new sources."
        )


# ---------------------------------------------------------------------------
# Calibration logic (used by the Orchestrator, informed by Evaluator feedback)
# ---------------------------------------------------------------------------

def compute_calibration_step(
    eval_result: EvaluationResult,
    feedback_msg: AgentMessage | None,
    current_k: int,
    current_l1: float,
    current_l2: float,
    cycle: int,
) -> CalibrationStep:
    """Compute parameter adjustments using evaluator feedback from memory."""
    c_f = eval_result.confidence
    s_c = eval_result.semantic_consistency
    f_c = eval_result.factual_correctness

    # k adjustment
    delta_k = math.ceil((1.0 - c_f) * calibration.k_increment_multiplier)
    new_k = current_k + delta_k

    # Lambda adjustment — informed by evaluator's weak-component analysis
    step = calibration.lambda_adjust_step
    weak = "balanced"
    if feedback_msg:
        weak = feedback_msg.payload.get("weak_component", "balanced")

    if weak == "semantic_consistency" or (weak == "balanced" and s_c <= f_c):
        new_l1 = min(0.9, current_l1 + step)
        new_l2 = max(0.1, current_l2 - step)
    else:
        new_l1 = max(0.1, current_l1 - step)
        new_l2 = min(0.9, current_l2 + step)

    total = new_l1 + new_l2
    if total > 0:
        new_l1 /= total
        new_l2 /= total

    logger.info(
        "Calibration cycle %d: C_f=%.3f, weak=%s, k: %d→%d, λ₁: %.2f→%.2f",
        cycle, c_f, weak, current_k, new_k, current_l1, new_l1,
    )

    return CalibrationStep(
        cycle=cycle,
        confidence_before=c_f,
        s_c=s_c,
        f_c=f_c,
        k_before=current_k,
        k_after=new_k,
        lambda_1_before=current_l1,
        lambda_1_after=round(new_l1, 2),
        lambda_2_before=current_l2,
        lambda_2_after=round(new_l2, 2),
    )
