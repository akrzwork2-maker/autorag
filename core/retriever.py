from __future__ import annotations

from dataclasses import dataclass, field

from core.config import retrieval
from core.embedder import Embedder
from core.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_name: str
    source: str
    cosine_similarity: float
    contextual_density: float
    hybrid_score: float
    rank: int


@dataclass
class RetrievalResult:
    query: str
    k: int
    lambda_1: float
    lambda_2: float
    candidates_fetched: int
    chunks: list[RetrievedChunk] = field(default_factory=list)


class Retriever:

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self._embedder = embedder or Embedder()
        self._vectorstore = vectorstore or VectorStore(self._embedder)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        lambda_1: float | None = None,
        lambda_2: float | None = None,
        user_id: str | None = None,
    ) -> RetrievalResult:
        k = k or retrieval.initial_k
        l1 = lambda_1 if lambda_1 is not None else retrieval.lambda_1
        l2 = lambda_2 if lambda_2 is not None else retrieval.lambda_2

        fetch_n = min(k * 3, max(k + 10, 15))

        raw_results = self._vectorstore.query(query, k=fetch_n, user_id=user_id)

        if not raw_results:
            return RetrievalResult(
                query=query,
                k=k,
                lambda_1=l1,
                lambda_2=l2,
                candidates_fetched=0,
                chunks=[],
            )

        scored: list[RetrievedChunk] = []
        for r in raw_results:
            cosine_sim = r["similarity"]

            density = self._embedder.contextual_density(r["text"])

            hybrid = l1 * cosine_sim + l2 * density

            scored.append(RetrievedChunk(
                chunk_id=r["id"],
                text=r["text"],
                doc_name=r["metadata"].get("doc_name", "unknown"),
                source=r["metadata"].get("source", "unknown"),
                cosine_similarity=round(cosine_sim, 4),
                contextual_density=round(density, 4),
                hybrid_score=round(hybrid, 4),
                rank=0,  # will be set after sorting
            ))


        scored.sort(key=lambda c: c.hybrid_score, reverse=True)
        top_k = scored[:k]
        for i, chunk in enumerate(top_k):
            chunk.rank = i + 1

        return RetrievalResult(
            query=query,
            k=k,
            lambda_1=l1,
            lambda_2=l2,
            candidates_fetched=len(raw_results),
            chunks=top_k,
        )

    def retrieve_basic(self, query: str, k: int | None = None, user_id: str | None = None) -> RetrievalResult:
        k = k or retrieval.initial_k
        raw_results = self._vectorstore.query(query, k=k, user_id=user_id)

        chunks = []
        for i, r in enumerate(raw_results):
            chunks.append(RetrievedChunk(
                chunk_id=r["id"],
                text=r["text"],
                doc_name=r["metadata"].get("doc_name", "unknown"),
                source=r["metadata"].get("source", "unknown"),
                cosine_similarity=round(r["similarity"], 4),
                contextual_density=0.0,
                hybrid_score=round(r["similarity"], 4),
                rank=i + 1,
            ))

        return RetrievalResult(
            query=query,
            k=k,
            lambda_1=1.0,
            lambda_2=0.0,
            candidates_fetched=len(raw_results),
            chunks=chunks,
        )
