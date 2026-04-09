"""Session-level shared memory for agent coordination (M_t in the paper)."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent message types for the asynchronous negotiation protocol
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    QUERY_ANALYSIS = "query_analysis"
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    RETRIEVAL_RESULT = "retrieval_result"
    GENERATION_RESULT = "generation_result"
    EVALUATION_FEEDBACK = "evaluation_feedback"
    CALIBRATION_ACTION = "calibration_action"
    MEMORY_UPDATE = "memory_update"


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    EXPANDED = "expanded"


@dataclass
class AgentMessage:
    """Message exchanged between agents through the session memory bus."""
    from_agent: str
    to_agent: str
    msg_type: MessageType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryRecord:
    """Record of a past query and its outcome, stored in session memory."""
    query: str
    confidence: float
    s_c: float
    f_c: float
    strategy_used: RetrievalStrategy
    calibration_cycles: int
    model_used: str
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    """Shared memory M_t that all agents read from and write to.

    Implements the session-level memory described in the paper:
    M_{t+1} = UpdateMemory(M_t, Q_t, R_t, C_f)

    Agents post messages to the bus and read historical records
    to make informed autonomous decisions.
    """

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []
        self._query_history: list[QueryRecord] = []
        self._current_cycle: int = 0

    # -- Message bus ----------------------------------------------------------

    def post(self, msg: AgentMessage) -> None:
        """Post a message onto the shared bus."""
        self._messages.append(msg)

    def get_messages(
        self,
        *,
        to_agent: str | None = None,
        msg_type: MessageType | None = None,
    ) -> list[AgentMessage]:
        """Read messages, optionally filtered by recipient or type."""
        out = self._messages
        if to_agent is not None:
            out = [m for m in out if m.to_agent in (to_agent, "all")]
        if msg_type is not None:
            out = [m for m in out if m.msg_type == msg_type]
        return out

    def last_message(self, msg_type: MessageType) -> AgentMessage | None:
        """Return the most recent message of a given type, or None."""
        for m in reversed(self._messages):
            if m.msg_type == msg_type:
                return m
        return None

    # -- Query history (persistent within session) ----------------------------

    def record_query(self, record: QueryRecord) -> None:
        self._query_history.append(record)

    @property
    def query_history(self) -> list[QueryRecord]:
        return list(self._query_history)

    @property
    def query_count(self) -> int:
        return len(self._query_history)

    def avg_confidence(self) -> float:
        if not self._query_history:
            return 0.0
        return sum(r.confidence for r in self._query_history) / len(self._query_history)

    def recent_strategies(self, n: int = 5) -> list[RetrievalStrategy]:
        return [r.strategy_used for r in self._query_history[-n:]]

    # -- Current-query scratch pad (cleared each run) -------------------------

    def begin_query(self) -> None:
        """Clear per-query messages for a new pipeline run."""
        self._messages.clear()
        self._current_cycle = 0

    @property
    def current_cycle(self) -> int:
        return self._current_cycle

    @current_cycle.setter
    def current_cycle(self, value: int) -> None:
        self._current_cycle = value

    # -- Context summary for Reasoner Agent -----------------------------------

    def context_summary(self) -> str:
        """Build a concise textual summary for the Reasoner to condition on."""
        if not self._query_history:
            return ""
        recent = self._query_history[-3:]
        lines = ["Previous queries in this session:"]
        for r in recent:
            lines.append(
                f"- Q: \"{r.query[:80]}\" → C_f={r.confidence:.2f} "
                f"(strategy={r.strategy_used.value}, cycles={r.calibration_cycles})"
            )
        avg = self.avg_confidence()
        lines.append(f"Session average confidence: {avg:.2f}")
        return "\n".join(lines)
