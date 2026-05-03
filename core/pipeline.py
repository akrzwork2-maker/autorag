from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from core.config import calibration, retrieval
from core.embedder import Embedder
from core.vectorstore import VectorStore
from core.retriever import Retriever, RetrievalResult
from core.reasoner import Reasoner, ReasonerResult
from core.evaluator import Evaluator, EvaluationResult
from core.calibrator import Calibrator, CalibrationStep
from core.memory import (
    AgentMessage,
    SessionMemory,
    MessageType,
    QueryRecord,
    RetrievalStrategy,
)
from core.agents import (
    RetrieverAgent,
    ReasonerAgent,
    EvaluatorAgent,
    compute_calibration_step,
)

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    ENCODE = "encode"
    RETRIEVE = "retrieve"
    REASON = "reason"
    EVALUATE = "evaluate"
    CALIBRATE = "calibrate"
    FETCH_NEW = "fetch_new"
    DONE = "done"


@dataclass
class PipelineStep:
    step_type: StepType
    cycle: int
    detail: str
    duration_ms: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    answer: str
    model_used: str
    fallback_level: int

    semantic_consistency: float
    factual_correctness: float
    confidence: float

    sources: list[dict] = field(default_factory=list)
    total_sources_used: int = 0

    calibration_cycles: int = 0
    calibration_steps: list[CalibrationStep] = field(default_factory=list)

    trace: list[PipelineStep] = field(default_factory=list)

    total_time_ms: float = 0.0
    error: str | None = None


class Pipeline:
    """AutoRAG++ Orchestrator — coordinates three autonomous agents.

    The Retriever, Reasoner, and Evaluator agents share session-level
    memory (M_t) and negotiate through an asynchronous message protocol.
    The orchestrator drives the closed-loop: Retrieve → Reason → Evaluate
    → (Calibrate → Retrieve → Reason → Evaluate)* → Done.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reasoner: Reasoner | None = None,
        evaluator: Evaluator | None = None,
        calibrator: Calibrator | None = None,
        wiki_fetcher=None,
    ):
        self._embedder = embedder or Embedder()
        self._vectorstore = vectorstore or VectorStore(self._embedder)
        self._retriever = retriever or Retriever(self._vectorstore, self._embedder)
        self._reasoner = reasoner or Reasoner()
        self._evaluator = evaluator or Evaluator(self._embedder)
        self._calibrator = calibrator or Calibrator()
        self._wiki_fetcher = wiki_fetcher

        # Session memory persists across queries within the same Pipeline instance
        self._memory = SessionMemory()

        # Agents — each receives a reference to the shared memory
        self._retriever_agent = RetrieverAgent(
            self._memory, self._retriever, self._embedder, self._vectorstore,
        )
        self._reasoner_agent = ReasonerAgent(self._memory, self._reasoner)
        self._evaluator_agent = EvaluatorAgent(self._memory, self._evaluator)

    @property
    def session_memory(self) -> SessionMemory:
        return self._memory

    def run(self, query: str, on_step=None, user_id: str | None = None) -> PipelineResult:
        """Run the full AutoRAG++ agentic pipeline.

        Algorithm 1 from the paper: agents coordinate through M_t.
        """
        start = time.time()
        trace: list[PipelineStep] = []
        cal_steps: list[CalibrationStep] = []

        # Begin new query — clears per-query messages, keeps history
        self._memory.begin_query()

        if self._vectorstore.count(user_id) == 0:
            return PipelineResult(
                query=query, answer="", model_used="none", fallback_level=-1,
                semantic_consistency=0, factual_correctness=0, confidence=0,
                total_time_ms=round((time.time() - start) * 1000, 1),
                error="Knowledge Base is empty. Please seed or upload documents first.",
            )

        current_k = retrieval.initial_k
        current_l1 = retrieval.lambda_1
        current_l2 = retrieval.lambda_2

        def _emit(step: PipelineStep):
            trace.append(step)
            if on_step:
                on_step(step)

        # ── Step 1: Encode ──
        t0 = time.time()
        query_emb = self._embedder.encode(query)
        _emit(PipelineStep(
            step_type=StepType.ENCODE,
            cycle=0,
            detail=f"Encoded query → {len(query_emb)}-dim embedding",
            duration_ms=round((time.time() - t0) * 1000, 1),
            data={"embedding_dim": len(query_emb)},
        ))

        # ── Step 2: Retriever Agent — autonomous strategy selection + retrieval ──
        t0 = time.time()
        strategy, ret_result = self._retriever_agent.analyze_and_decide(
            query, k=current_k, lambda_1=current_l1, lambda_2=current_l2, user_id=user_id,
        )
        _emit(PipelineStep(
            step_type=StepType.RETRIEVE,
            cycle=0,
            detail=f"[{strategy.value}] Retrieved {len(ret_result.chunks)} chunks (k={current_k}, λ₁={current_l1})",
            duration_ms=round((time.time() - t0) * 1000, 1),
            data={
                "strategy": strategy.value,
                "k": current_k,
                "lambda_1": current_l1,
                "candidates": ret_result.candidates_fetched,
                "returned": len(ret_result.chunks),
                "top_score": ret_result.chunks[0].hybrid_score if ret_result.chunks else 0,
            },
        ))

        # ── Step 3: Reasoner Agent — memory-aware generation ──
        t0 = time.time()
        reason_result = self._reasoner_agent.generate(query, ret_result.chunks)
        _emit(PipelineStep(
            step_type=StepType.REASON,
            cycle=0,
            detail="Generated response via language engine",
            duration_ms=round((time.time() - t0) * 1000, 1),
            data={"fallback": reason_result.fallback_level},
        ))

        if reason_result.error:
            return self._build_error_result(query, reason_result, trace, start)

        # ── Step 4: Evaluator Agent — evaluate + post feedback to memory ──
        t0 = time.time()
        eval_result = self._evaluator_agent.evaluate(reason_result.answer, ret_result.chunks)
        _emit(PipelineStep(
            step_type=StepType.EVALUATE,
            cycle=0,
            detail=f"S_c={eval_result.semantic_consistency:.3f}, F_c={eval_result.factual_correctness:.3f}, C_f={eval_result.confidence:.3f}",
            duration_ms=round((time.time() - t0) * 1000, 1),
            data={
                "s_c": eval_result.semantic_consistency,
                "f_c": eval_result.factual_correctness,
                "c_f": eval_result.confidence,
                "threshold": calibration.confidence_threshold,
                "passed": eval_result.confidence >= calibration.confidence_threshold,
            },
        ))

        # ── Step 5: Calibration loop — agents negotiate through M_t ──
        cycle = 0
        while self._evaluator_agent.should_calibrate(eval_result) and cycle < calibration.max_cycles:
            cycle += 1
            self._memory.current_cycle = cycle

            # Evaluator's feedback message is already in memory from evaluate()
            feedback = self._memory.last_message(MessageType.EVALUATION_FEEDBACK)

            # Calibrate using evaluator feedback
            t0 = time.time()
            cal_step = compute_calibration_step(
                eval_result, feedback, current_k, current_l1, current_l2, cycle,
            )
            current_k = cal_step.k_after
            current_l1 = cal_step.lambda_1_after
            current_l2 = cal_step.lambda_2_after

            # Post calibration action to memory
            self._memory.post(AgentMessage(
                from_agent="orchestrator",
                to_agent="all",
                msg_type=MessageType.CALIBRATION_ACTION,
                payload={
                    "cycle": cycle,
                    "k": current_k,
                    "lambda_1": current_l1,
                    "lambda_2": current_l2,
                    "directive": feedback.payload.get("directive", "") if feedback else "",
                },
            ))

            _emit(PipelineStep(
                step_type=StepType.CALIBRATE,
                cycle=cycle,
                detail=f"Cycle {cycle}: k→{current_k}, λ₁→{current_l1}",
                duration_ms=round((time.time() - t0) * 1000, 1),
                data={
                    "k_before": cal_step.k_before, "k_after": cal_step.k_after,
                    "l1_before": cal_step.lambda_1_before, "l1_after": cal_step.lambda_1_after,
                    "l2_before": cal_step.lambda_2_before, "l2_after": cal_step.lambda_2_after,
                },
            ))

            # Dynamic Knowledge Update — fetch new sources
            if self._wiki_fetcher:
                t0 = time.time()
                try:
                    fetch_result = self._wiki_fetcher.fetch_and_index(query, self._vectorstore, user_id=user_id)
                    new_count = fetch_result["total"]
                    cal_step.wiki_docs_fetched = new_count
                    parts = []
                    if fetch_result["wiki"]:
                        parts.append(f"{fetch_result['wiki']} Wikipedia")
                    if fetch_result["web"]:
                        parts.append(f"{fetch_result['web']} Web")
                    detail = f"Fetched {' + '.join(parts)} chunks" if parts else "No new chunks found"
                    _emit(PipelineStep(
                        step_type=StepType.FETCH_NEW,
                        cycle=cycle,
                        detail=detail,
                        duration_ms=round((time.time() - t0) * 1000, 1),
                        data={"new_chunks": new_count, "wiki": fetch_result["wiki"], "web": fetch_result["web"]},
                    ))
                except Exception as e:
                    logger.warning("Wiki fetch failed in cycle %d: %s", cycle, e)

            # Retriever Agent re-retrieves with calibrated parameters
            t0 = time.time()
            ret_result = self._retriever_agent.re_retrieve(
                query, current_k, current_l1, current_l2, user_id=user_id,
            )
            _emit(PipelineStep(
                step_type=StepType.RETRIEVE,
                cycle=cycle,
                detail=f"Re-retrieved {len(ret_result.chunks)} chunks (k={current_k})",
                duration_ms=round((time.time() - t0) * 1000, 1),
                data={"k": current_k, "returned": len(ret_result.chunks)},
            ))

            # Reasoner Agent re-generates (is_recalibration=True for adjusted prompt)
            t0 = time.time()
            reason_result = self._reasoner_agent.generate(
                query, ret_result.chunks, is_recalibration=True,
            )
            _emit(PipelineStep(
                step_type=StepType.REASON,
                cycle=cycle,
                detail="Re-generated via language engine",
                duration_ms=round((time.time() - t0) * 1000, 1),
                data={},
            ))

            if reason_result.error:
                return self._build_error_result(query, reason_result, trace, start)

            # Evaluator Agent re-evaluates and posts new feedback
            t0 = time.time()
            eval_result = self._evaluator_agent.evaluate(reason_result.answer, ret_result.chunks)
            cal_step.confidence_after = eval_result.confidence
            cal_steps.append(cal_step)
            _emit(PipelineStep(
                step_type=StepType.EVALUATE,
                cycle=cycle,
                detail=f"S_c={eval_result.semantic_consistency:.3f}, F_c={eval_result.factual_correctness:.3f}, C_f={eval_result.confidence:.3f}",
                duration_ms=round((time.time() - t0) * 1000, 1),
                data={
                    "s_c": eval_result.semantic_consistency,
                    "f_c": eval_result.factual_correctness,
                    "c_f": eval_result.confidence,
                    "passed": eval_result.confidence >= calibration.confidence_threshold,
                },
            ))

        # ── Done ──
        _emit(PipelineStep(
            step_type=StepType.DONE,
            cycle=cycle,
            detail=f"Final C_f={eval_result.confidence:.3f} after {cycle} calibration cycle(s)",
            duration_ms=0,
        ))

        sources = [
            {
                "rank": c.rank,
                "doc_name": c.doc_name,
                "text": c.text,
                "hybrid_score": c.hybrid_score,
                "source": c.source,
            }
            for c in ret_result.chunks
        ]

        # ── Update session memory: M_{t+1} = UpdateMemory(M_t, Q_t, R_t, C_f) ──
        self._memory.record_query(QueryRecord(
            query=query,
            confidence=eval_result.confidence,
            s_c=eval_result.semantic_consistency,
            f_c=eval_result.factual_correctness,
            strategy_used=strategy,
            calibration_cycles=cycle,
            model_used=reason_result.model_used,
        ))

        return PipelineResult(
            query=query,
            answer=reason_result.answer,
            model_used=reason_result.model_used,
            fallback_level=reason_result.fallback_level,
            semantic_consistency=eval_result.semantic_consistency,
            factual_correctness=eval_result.factual_correctness,
            confidence=eval_result.confidence,
            sources=sources,
            total_sources_used=len(sources),
            calibration_cycles=cycle,
            calibration_steps=cal_steps,
            trace=trace,
            total_time_ms=round((time.time() - start) * 1000, 1),
        )

    def run_basic_rag(self, query: str, user_id: str | None = None) -> PipelineResult:
        """Single-pass retrieval (cosine only), no calibration — for X-Ray comparison.

        Uses raw engines directly (no agents) to avoid polluting session memory
        and to provide a clean baseline comparison.
        """
        start = time.time()

        ret_result = self._retriever.retrieve_basic(query, user_id=user_id)
        reason_result = self._reasoner.reason(query, ret_result.chunks)

        if reason_result.error:
            return PipelineResult(
                query=query, answer="", model_used="none", fallback_level=-1,
                semantic_consistency=0, factual_correctness=0, confidence=0,
                error=reason_result.error,
                total_time_ms=round((time.time() - start) * 1000, 1),
            )

        eval_result = self._evaluator.evaluate(reason_result.answer, ret_result.chunks)

        sources = [
            {"rank": c.rank, "doc_name": c.doc_name, "text": c.text,
             "hybrid_score": c.hybrid_score, "source": c.source}
            for c in ret_result.chunks
        ]

        return PipelineResult(
            query=query,
            answer=reason_result.answer,
            model_used=reason_result.model_used,
            fallback_level=reason_result.fallback_level,
            semantic_consistency=eval_result.semantic_consistency,
            factual_correctness=eval_result.factual_correctness,
            confidence=eval_result.confidence,
            sources=sources,
            total_sources_used=len(sources),
            calibration_cycles=0,
            total_time_ms=round((time.time() - start) * 1000, 1),
        )

    def run_llm_only(self, query: str) -> PipelineResult:
        """LLM-only mode — no retrieval — for X-Ray comparison.

        Uses raw engine directly (no agents) for clean baseline.
        """
        start = time.time()

        reason_result = self._reasoner.reason_without_retrieval(query)

        return PipelineResult(
            query=query,
            answer=reason_result.answer if not reason_result.error else "",
            model_used=reason_result.model_used,
            fallback_level=reason_result.fallback_level,
            semantic_consistency=0.0,
            factual_correctness=0.0,
            confidence=0.0,
            total_sources_used=0,
            calibration_cycles=0,
            total_time_ms=round((time.time() - start) * 1000, 1),
            error=reason_result.error,
        )

    def _build_error_result(self, query, reason_result, trace, start):
        return PipelineResult(
            query=query,
            answer="",
            model_used=reason_result.model_used,
            fallback_level=reason_result.fallback_level,
            semantic_consistency=0, factual_correctness=0, confidence=0,
            trace=trace,
            total_time_ms=round((time.time() - start) * 1000, 1),
            error=reason_result.error,
        )
