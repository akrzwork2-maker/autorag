from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from core.chunker import Chunker, Chunk

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> list[str]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
    return pages


def load_and_chunk_pdf(
    pdf_path: str | Path,
    doc_name: str | None = None,
    source: str = "uploaded",
    chunker: Chunker | None = None,
) -> list[Chunk]:
    pdf_path = Path(pdf_path)
    doc_name = doc_name or pdf_path.stem
    chunker = chunker or Chunker()

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        logger.warning("No text extracted from %s", pdf_path)
        return []

    chunks = chunker.chunk_pages(
        pages,
        doc_name=doc_name,
        source=source,
    )
    logger.info("PDF %s: %d pages → %d chunks", doc_name, len(pages), len(chunks))
    return chunks


def load_pdf_bytes(
    pdf_bytes: bytes,
    doc_name: str,
    source: str = "uploaded",
    chunker: Chunker | None = None,
) -> list[Chunk]:
    chunker = chunker or Chunker()

    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

    if not pages:
        return []

    return chunker.chunk_pages(pages, doc_name=doc_name, source=source)
