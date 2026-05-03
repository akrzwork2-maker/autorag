from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config import models, verifier, calibration
from core.embedder import Embedder
from core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization for NLI — fixes garbled web-scraped chunks
# ---------------------------------------------------------------------------

def _normalize_for_nli(text: str) -> str:
    """Fix concatenated words from web scraping before sending to NLI.

    Handles patterns like 'ENISAThreatLandscape' → 'ENISA Threat Landscape'
    and 'Whatarethetopcybersecurity' → keeps as-is (needs wordpiece, fallback
    handles this via embedding scoring).
    """
    # Insert space before uppercase letter preceded by lowercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space between letter and digit transitions
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # Insert space before opening paren/bracket if preceded by alnum
    text = re.sub(r'([a-zA-Z0-9])([(\[])', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    nli_score: float
    evidence: str
    evidence_source: str


@dataclass
class EvaluationResult:
    semantic_consistency: float
    factual_correctness: float
    confidence: float
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)



_nli_tokenizer = None
_nli_model = None


def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        logger.info("Loading NLI model: %s", models.NLI_MODEL)
        _nli_tokenizer = AutoTokenizer.from_pretrained(models.NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(models.NLI_MODEL)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def nli_entailment_score(premise: str, hypothesis: str) -> float:
    try:
        tokenizer, model = _load_nli()
        premise = _normalize_for_nli(premise)
        inputs = tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = model.config.id2label
        entailment_idx = next(
            (i for i, label in id2label.items() if "entail" in label.lower()), 2
        )
        return float(probs[entailment_idx])
    except Exception as e:
        logger.error("NLI scoring failed: %s", e)
        return 0.5


def nli_batch_scores(pairs: list[tuple[str, str]], max_length: int = 512) -> list[float]:
    if not pairs:
        return []
    try:
        tokenizer, model = _load_nli()
        premises = [_normalize_for_nli(p[0]) for p in pairs]
        hypotheses = [p[1] for p in pairs]
        inputs = tokenizer(
            premises, hypotheses,
            return_tensors="pt", truncation=True, max_length=max_length, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        id2label = model.config.id2label
        ent_idx = next((i for i, l in id2label.items() if "entail" in l.lower()), 2)
        return [float(probs[i][ent_idx]) for i in range(len(pairs))]
    except Exception as e:
        logger.error("Batch NLI failed: %s", e)
        return [0.5] * len(pairs)


def extract_claims(text: str) -> list[str]:
    # Strip markdown formatting: **bold**, *italic*, numbered lists, bullet points
    cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # **bold** / *italic*
    cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', cleaned, flags=re.MULTILINE)  # 1. or 1)
    cleaned = re.sub(r'^\s*[-•–]\s*', '', cleaned, flags=re.MULTILINE)  # bullet points
    cleaned = re.sub(r'\s*[—–]\s*', '. ', cleaned)  # em-dashes as sentence breaks
    cleaned = re.sub(r'\[\d+\]', '', cleaned)  # strip citation markers [1], [2]

    # Split on sentence endings AND newlines (LLM often uses newlines between claims)
    parts = re.split(r'(?<=[.!?])\s+|\n+', cleaned.strip())
    claims = []
    for s in parts:
        s = s.strip().rstrip(".")
        # Skip intro/header lines ending with colon
        if s.endswith(":"):
            continue
        # Require minimum 20 chars and at least 3 words to be a real claim
        if len(s) >= 20 and len(s.split()) >= 3:
            claims.append(s)
    return claims


def verify_claim(
    claim: str,
    evidence_chunks: list[RetrievedChunk] | list[dict],
) -> ClaimVerdict:
    best_score = 0.0
    best_evidence = ""
    best_source = ""

    for chunk in evidence_chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("doc_name", "unknown")
        else:
            text = chunk.text
            source = chunk.doc_name

        score = nli_entailment_score(text, claim)
        if score > best_score:
            best_score = score
            best_evidence = text[:300]
            best_source = source

    if best_score >= verifier.verified_threshold:
        verdict = "VERIFIED"
    elif best_score <= verifier.contradicted_threshold:
        verdict = "CONTRADICTED"
    else:
        verdict = "UNVERIFIABLE"

    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        nli_score=round(best_score, 4),
        evidence=best_evidence,
        evidence_source=best_source,
    )


class Evaluator:

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder()

    def compute_semantic_consistency(
        self, response: str, chunks: list[RetrievedChunk]
    ) -> float:
        if not chunks:
            return 0.0

        resp_emb = self._embedder.encode(response)
        source_texts = [c.text for c in chunks]
        source_embs = self._embedder.encode_batch(source_texts)

        avg_source_emb = np.mean(source_embs, axis=0)
        avg_source_emb = avg_source_emb / (np.linalg.norm(avg_source_emb) + 1e-10)

        s_c = float(np.dot(resp_emb, avg_source_emb))
        return max(0.0, min(1.0, s_c))  

    def compute_factual_correctness(
        self, response: str, chunks: list[RetrievedChunk],
        max_claims: int = 4, max_chunks: int = 4,
    ) -> tuple[float, list[ClaimVerdict]]:
        if not chunks:
            return 0.0, []

        claims = extract_claims(response)
        if not claims:
            return 0.5, []

        eval_claims = claims[:max_claims]
        eval_chunks = chunks[:max_chunks]

        # --- NLI scoring ---
        pairs = []
        pair_map = []
        for ci, claim in enumerate(eval_claims):
            for chi, chunk in enumerate(eval_chunks):
                text = chunk.text if hasattr(chunk, 'text') else chunk.get("text", "")
                pairs.append((text, claim))
                pair_map.append((ci, chi))

        nli_scores = nli_batch_scores(pairs)

        # --- Embedding-based claim-chunk similarity (fallback signal) ---
        claim_embs = self._embedder.encode_batch(eval_claims)
        chunk_texts = [
            c.text if hasattr(c, 'text') else c.get("text", "")
            for c in eval_chunks
        ]
        chunk_embs = self._embedder.encode_batch(chunk_texts)

        verdicts = []
        for ci, claim in enumerate(eval_claims):
            best_nli = 0.0
            best_evidence = ""
            best_source = ""
            for j, (c_idx, ch_idx) in enumerate(pair_map):
                if c_idx == ci and nli_scores[j] > best_nli:
                    best_nli = nli_scores[j]
                    chunk = eval_chunks[ch_idx]
                    best_evidence = (chunk.text if hasattr(chunk, 'text') else chunk.get("text", ""))[:300]
                    best_source = chunk.doc_name if hasattr(chunk, 'doc_name') else chunk.get("metadata", {}).get("doc_name", "unknown")

            # Embedding similarity: max cosine between this claim and all chunks
            claim_vec = claim_embs[ci]
            embed_sim = max(
                float(np.dot(claim_vec, chunk_embs[k]) / (np.linalg.norm(claim_vec) * np.linalg.norm(chunk_embs[k]) + 1e-10))
                for k in range(len(eval_chunks))
            )
            embed_sim = max(0.0, embed_sim)  # clamp negatives

            # Blend: use NLI when it gives a real signal, otherwise lean on embedding similarity.
            # If NLI < 0.05 (garbled text / no direct entailment), use embedding sim scaled
            # into [0, 0.85] range so it can't fake a perfect score.
            if best_nli < 0.05:
                score = max(best_nli, embed_sim * 0.85)
            else:
                score = best_nli

            if score >= verifier.verified_threshold:
                verdict = "VERIFIED"
            elif score <= verifier.contradicted_threshold:
                verdict = "CONTRADICTED"
            else:
                verdict = "UNVERIFIABLE"
            verdicts.append(ClaimVerdict(claim=claim, verdict=verdict, nli_score=round(score, 4), evidence=best_evidence, evidence_source=best_source))

        f_c = sum(v.nli_score for v in verdicts) / len(verdicts)
        return round(f_c, 4), verdicts

    def evaluate(
        self, response: str, chunks: list[RetrievedChunk]
    ) -> EvaluationResult:
        s_c = self.compute_semantic_consistency(response, chunks)
        f_c, verdicts = self.compute_factual_correctness(response, chunks)

        # Weighted confidence: C_f = α·S_c + β·F_c + γ·R_c
        # R_c approximated as average retrieval score of top chunks
        r_c = 0.0
        if chunks:
            r_c = sum(c.hybrid_score for c in chunks[:4]) / min(len(chunks), 4)
            r_c = max(0.0, min(1.0, r_c))

        c_f = calibration.alpha * s_c + calibration.beta * f_c + calibration.gamma * r_c
        c_f = max(0.0, min(1.0, c_f))

        return EvaluationResult(
            semantic_consistency=round(s_c, 4),
            factual_correctness=round(f_c, 4),
            confidence=round(c_f, 4),
            claim_verdicts=verdicts,
        )

    def verify_text(
        self, text: str, evidence_chunks: list[RetrievedChunk] | list[dict],
        max_claims: int = 6, max_chunks_per_claim: int = 3,
    ) -> list[ClaimVerdict]:
        claims = extract_claims(text)[:max_claims]
        top_chunks = evidence_chunks[:max_chunks_per_claim]
        return [verify_claim(claim, top_chunks) for claim in claims]
