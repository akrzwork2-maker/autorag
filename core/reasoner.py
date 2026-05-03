from __future__ import annotations

from dataclasses import dataclass

from core.llm import LLMResponse, call_llm
from core.retriever import RetrievedChunk


SYSTEM_PROMPT = """You are VerifAI, an AI research assistant powered by AutoRAG++.
You answer questions using the provided source documents.
Rules:
1. Base your answer on the provided sources. Cite them using [1], [2], etc.
2. Synthesize information from ALL provided sources to give the most complete answer possible.
3. If the sources only partially cover the question, answer what you can and briefly note which specific aspects are not covered.
4. Provide detailed, comprehensive answers. Explain concepts thoroughly with relevant context from the sources.
5. Structure longer answers with clear paragraphs. Use key points where appropriate.
6. Never fabricate information not present in the sources."""

RECALIBRATION_PROMPT = """You are VerifAI, an AI research assistant powered by AutoRAG++.
The system has fetched additional sources to better answer the user's question.
Rules:
1. You MUST use the newly provided sources to give a comprehensive answer. Cite using [1], [2], etc.
2. Combine information from all sources — both original documents and newly fetched ones.
3. Answer as thoroughly as possible using everything available.
4. If some aspects remain uncovered, answer what you can and briefly note the gap.
5. Structure your answer clearly with paragraphs and key points.
6. Never fabricate information not present in the sources."""


@dataclass
class ReasonerResult:
    answer: str
    model_used: str
    fallback_level: int
    prompt_sent: str
    source_count: int
    error: str | None = None


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _build_context_block(chunks: list[RetrievedChunk], max_tokens: int = 3000) -> str:
    if not chunks:
        return "No sources available."

    lines = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}] (from: {chunk.doc_name}, score: {chunk.hybrid_score})"
        entry = f"{header}\n{chunk.text}\n"
        entry_tokens = _estimate_tokens(entry)
        if used + entry_tokens > max_tokens and lines:
            break
        lines.append(entry)
        used += entry_tokens
    return "\n".join(lines)


def _build_prompt(query: str, chunks: list[RetrievedChunk], max_context_tokens: int = 3000) -> str:
    context = _build_context_block(chunks, max_tokens=max_context_tokens)
    return f"""Answer the following question using the provided sources.

SOURCES:
{context}

QUESTION: {query}

ANSWER (cite sources using [1], [2], etc.):"""


class Reasoner:

    def reason(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        system_prompt: str | None = None,
    ) -> ReasonerResult:
        prompt = _build_prompt(query, chunks)
        sys_prompt = system_prompt or SYSTEM_PROMPT

        llm_response: LLMResponse = call_llm(prompt, sys_prompt)

        if llm_response.error:
            return ReasonerResult(
                answer="",
                model_used=llm_response.model_used,
                fallback_level=llm_response.fallback_level,
                prompt_sent=prompt,
                source_count=len(chunks),
                error=llm_response.error,
            )

        return ReasonerResult(
            answer=llm_response.text,
            model_used=llm_response.model_used,
            fallback_level=llm_response.fallback_level,
            prompt_sent=prompt,
            source_count=len(chunks),
        )

    def reason_without_retrieval(self, query: str) -> ReasonerResult:
        prompt = f"""Answer the following question based on your knowledge.

QUESTION: {query}

ANSWER:"""

        sys_prompt = "You are a helpful AI assistant. Answer accurately and concisely."
        llm_response: LLMResponse = call_llm(prompt, sys_prompt)

        if llm_response.error:
            return ReasonerResult(
                answer="",
                model_used=llm_response.model_used,
                fallback_level=llm_response.fallback_level,
                prompt_sent=prompt,
                source_count=0,
                error=llm_response.error,
            )

        return ReasonerResult(
            answer=llm_response.text,
            model_used=llm_response.model_used,
            fallback_level=llm_response.fallback_level,
            prompt_sent=prompt,
            source_count=0,
        )
