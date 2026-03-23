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
        inputs = tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
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
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []
    for s in sentences:
        s = s.strip().rstrip(".")
        if len(s) > 15:
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
        max_claims: int = 4, max_chunks: int = 2,
    ) -> tuple[float, list[ClaimVerdict]]:
        if not chunks:
            return 0.0, []

        claims = extract_claims(response)
        if not claims:
            return 0.5, []

        eval_claims = claims[:max_claims]
        eval_chunks = chunks[:max_chunks]

        pairs = []
        pair_map = []
        for ci, claim in enumerate(eval_claims):
            for chi, chunk in enumerate(eval_chunks):
                text = chunk.text if hasattr(chunk, 'text') else chunk.get("text", "")
                pairs.append((text, claim))
                pair_map.append((ci, chi))

        scores = nli_batch_scores(pairs)

        verdicts = []
        for ci, claim in enumerate(eval_claims):
            best_score = 0.0
            best_evidence = ""
            best_source = ""
            for j, (c_idx, ch_idx) in enumerate(pair_map):
                if c_idx == ci and scores[j] > best_score:
                    best_score = scores[j]
                    chunk = eval_chunks[ch_idx]
                    best_evidence = (chunk.text if hasattr(chunk, 'text') else chunk.get("text", ""))[:300]
                    best_source = chunk.doc_name if hasattr(chunk, 'doc_name') else chunk.get("metadata", {}).get("doc_name", "unknown")

            if best_score >= verifier.verified_threshold:
                verdict = "VERIFIED"
            elif best_score <= verifier.contradicted_threshold:
                verdict = "CONTRADICTED"
            else:
                verdict = "UNVERIFIABLE"
            verdicts.append(ClaimVerdict(claim=claim, verdict=verdict, nli_score=round(best_score, 4), evidence=best_evidence, evidence_source=best_source))

        f_c = sum(v.nli_score for v in verdicts) / len(verdicts)
        return round(f_c, 4), verdicts

    def evaluate(
        self, response: str, chunks: list[RetrievedChunk]
    ) -> EvaluationResult:
        s_c = self.compute_semantic_consistency(response, chunks)
        f_c, verdicts = self.compute_factual_correctness(response, chunks)

        if s_c + f_c > 0:
            c_f = 2 * s_c * f_c / (s_c + f_c)
        else:
            c_f = 0.0

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
