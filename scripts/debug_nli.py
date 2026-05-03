"""Debug NLI scoring with actual chunks — full pipeline verification."""
import sys
sys.path.insert(0, ".")

from core.evaluator import nli_entailment_score, Evaluator, _normalize_for_nli, extract_claims
from core.embedder import Embedder
from core.vectorstore import VectorStore

emb = Embedder()
vs = VectorStore(emb)

# 1) Test claim extraction with markdown-heavy LLM output
print("=" * 60)
print("1. Claim extraction from markdown LLM output")
print("=" * 60)
markdown_response = """The ENISA Threat Landscape 2025 identifies the following top cyber threats:

1. **Ransomware** — Remains the most impactful threat with increasing sophistication [1]
2. **Malware** — Continues to evolve with new variants targeting critical infrastructure [2]
3. **Social Engineering** — Phishing attacks remain prevalent across the EU [1]

- Threats against data including breaches and leaks are significant.
- Denial of service attacks target availability of systems.

Supply chain attacks targeting third-party dependencies are a growing concern."""

claims = extract_claims(markdown_response)
print(f"Extracted {len(claims)} claims:")
for i, c in enumerate(claims):
    print(f"  {i}: {c[:90]}{'...' if len(c)>90 else ''}")

# 2) Full evaluation pipeline
print("\n" + "=" * 60)
print("2. Full evaluate() — weighted confidence formula")
print("=" * 60)

from core.retriever import Retriever, RetrievedChunk
retriever = Retriever(vs)

chunks_result = retriever.retrieve(
    "What are the top cyber threats identified in the ENISA Threat Landscape 2025?", k=5
)
chunks = chunks_result.chunks
print(f"\nRetrieved {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: hybrid={c.hybrid_score:.4f}, {len(c.text)} chars, from {c.doc_name}")

evaluator = Evaluator()
result = evaluator.evaluate(markdown_response, chunks)
print(f"\nSemantic (S_c):   {result.semantic_consistency:.4f}")
print(f"Factual (F_c):    {result.factual_correctness:.4f}")
print(f"Confidence (C_f): {result.confidence:.4f}")
print(f"\nClaim verdicts:")
for v in result.claim_verdicts:
    print(f"  [{v.verdict:14s}] score={v.nli_score:.4f} | {v.claim[:70]}")

# 3) Edge cases
print("\n" + "=" * 60)
print("3. Edge cases")
print("=" * 60)

# Empty response
try:
    r = evaluator.evaluate("", chunks)
    print(f"Empty response:    S_c={r.semantic_consistency:.4f}, F_c={r.factual_correctness:.4f}, C_f={r.confidence:.4f}")
except Exception as e:
    print(f"Empty response: EXCEPTION: {e}")

# Empty chunks
try:
    r = evaluator.evaluate(markdown_response, [])
    print(f"Empty chunks:      S_c={r.semantic_consistency:.4f}, F_c={r.factual_correctness:.4f}, C_f={r.confidence:.4f}")
except Exception as e:
    print(f"Empty chunks: EXCEPTION: {e}")

# Very short response
try:
    r = evaluator.evaluate("Ransomware is a threat.", chunks)
    print(f"Short response:    S_c={r.semantic_consistency:.4f}, F_c={r.factual_correctness:.4f}, C_f={r.confidence:.4f}")
except Exception as e:
    print(f"Short response: EXCEPTION: {e}")

# Response with no claims (< 15 chars per sentence)
try:
    r = evaluator.evaluate("Yes. No. Maybe.", chunks)
    print(f"No claims:         S_c={r.semantic_consistency:.4f}, F_c={r.factual_correctness:.4f}, C_f={r.confidence:.4f}")
except Exception as e:
    print(f"No claims: EXCEPTION: {e}")

print("\n[DONE] All checks passed.")

