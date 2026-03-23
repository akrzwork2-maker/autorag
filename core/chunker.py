from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.config import retrieval


@dataclass
class Chunk:
    text: str
    doc_name: str = ""
    chunk_index: int = 0
    source: str = "uploaded"
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4


_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    sentences = _SENT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


class Chunker:

    def __init__(
        self,
        target_min: int | None = None,
        target_max: int | None = None,
        overlap_sentences: int | None = None,
    ):
        self.target_min = target_min or retrieval.chunk_size_min
        self.target_max = target_max or retrieval.chunk_size_max
        self.overlap_sentences = overlap_sentences if overlap_sentences is not None else retrieval.chunk_overlap_sentences

    def _token_est(self, text: str) -> int:
        return len(text) // 4

    def chunk_text(
        self,
        text: str,
        doc_name: str = "",
        source: str = "uploaded",
        metadata: dict | None = None,
    ) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for sent in sentences:
            sent_tokens = self._token_est(sent)

            if current_tokens + sent_tokens > self.target_max and current_tokens >= self.target_min:
                chunk_text = " ".join(current_sents)
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_name=doc_name,
                    chunk_index=chunk_idx,
                    source=source,
                    metadata=metadata or {},
                ))
                chunk_idx += 1

                overlap = current_sents[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current_sents = list(overlap)
                current_tokens = sum(self._token_est(s) for s in current_sents)

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            chunk_text = " ".join(current_sents)
            if chunks and self._token_est(chunk_text) < self.target_min // 2:
                chunks[-1] = Chunk(
                    text=chunks[-1].text + " " + chunk_text,
                    doc_name=doc_name,
                    chunk_index=chunks[-1].chunk_index,
                    source=source,
                    metadata=metadata or {},
                )
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_name=doc_name,
                    chunk_index=chunk_idx,
                    source=source,
                    metadata=metadata or {},
                ))

        return chunks

    def chunk_pages(
        self,
        pages: list[str],
        doc_name: str = "",
        source: str = "uploaded",
    ) -> list[Chunk]:
        """Chunk a multi-page document (e.g., from PDF). Pages are joined then chunked."""
        full_text = "\n\n".join(pages)
        return self.chunk_text(full_text, doc_name=doc_name, source=source)
