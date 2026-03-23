import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    from core.config import (
        EXAMPLE_QUERIES, XRAY_QUERIES, FACT_CHECK_EXAMPLES, paths
    )
    from core.embedder import Embedder
    from core.vectorstore import VectorStore
    from core.retriever import Retriever
    from core.reasoner import Reasoner, _build_prompt, SYSTEM_PROMPT
    from core.llm import call_llm, save_to_demo_cache, _cache_key

    logger.info("=" * 60)
    logger.info("VerifAI Demo Pre-Cacher")
    logger.info("=" * 60)

    emb = Embedder()
    vs = VectorStore(embedder=emb)
    retriever = Retriever(vs, emb)
    reasoner = Reasoner()

    kb_count = vs.count()
    logger.info("KB has %d chunks", kb_count)
    if kb_count == 0:
        logger.error("KB is empty! Run seed_kb.py first.")
        return

    all_queries = []
    all_queries.extend(EXAMPLE_QUERIES)
    all_queries.extend(XRAY_QUERIES)

    seen = set()
    unique_queries = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    logger.info("Total unique queries to cache: %d", len(unique_queries))

    cache_file = paths.DEMO_CACHE / "responses.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Existing cache has %d entries", len(cache))
    else:
        cache = {}

    cached_count = 0
    failed_count = 0

    for i, query in enumerate(unique_queries, 1):
        logger.info("\n[%d/%d] Query: %s", i, len(unique_queries), query[:80])

        try:
            ret_result = retriever.retrieve(query, k=5)
            if ret_result.chunks:
                prompt = _build_prompt(query, ret_result.chunks)
                key = _cache_key(prompt)

                if key in cache:
                    logger.info("  [AutoRAG++] Already cached ✓")
                else:
                    # Call LLM to get response
                    llm_resp = call_llm(prompt, SYSTEM_PROMPT)
                    if llm_resp.text and not llm_resp.error:
                        cache[key] = llm_resp.text
                        cached_count += 1
                        logger.info("  [AutoRAG++] Cached via %s ✓", llm_resp.model_used)
                    else:
                        logger.warning("  [AutoRAG++] LLM failed: %s", llm_resp.error)
                        failed_count += 1
            else:
                logger.warning("  [AutoRAG++] No chunks retrieved")
        except Exception as e:
            logger.error("  [AutoRAG++] Error: %s", e)
            failed_count += 1

        try:
            llm_only_prompt = f"""Answer the following question based on your knowledge.

QUESTION: {query}

ANSWER:"""
            llm_only_sys = "You are a helpful AI assistant. Answer accurately and concisely."
            key_llm = _cache_key(llm_only_prompt)

            if key_llm in cache:
                logger.info("  [LLM-only] Already cached ✓")
            else:
                llm_resp = call_llm(llm_only_prompt, llm_only_sys)
                if llm_resp.text and not llm_resp.error:
                    cache[key_llm] = llm_resp.text
                    cached_count += 1
                    logger.info("  [LLM-only] Cached via %s ✓", llm_resp.model_used)
                else:
                    logger.warning("  [LLM-only] LLM failed: %s", llm_resp.error)
                    failed_count += 1
        except Exception as e:
            logger.error("  [LLM-only] Error: %s", e)
            failed_count += 1

        try:
            ret_basic = retriever.retrieve_basic(query, k=5)
            if ret_basic.chunks:
                prompt_basic = _build_prompt(query, ret_basic.chunks)
                key_basic = _cache_key(prompt_basic)

                if key_basic in cache:
                    logger.info("  [BasicRAG] Already cached ✓")
                else:
                    llm_resp = call_llm(prompt_basic, SYSTEM_PROMPT)
                    if llm_resp.text and not llm_resp.error:
                        cache[key_basic] = llm_resp.text
                        cached_count += 1
                        logger.info("  [BasicRAG] Cached via %s ✓", llm_resp.model_used)
                    else:
                        logger.warning("  [BasicRAG] LLM failed: %s", llm_resp.error)
                        failed_count += 1
        except Exception as e:
            logger.error("  [BasicRAG] Error: %s", e)
            failed_count += 1

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("DONE: %d new entries cached, %d failed", cached_count, failed_count)
    logger.info("Total cache size: %d entries", len(cache))
    logger.info("Cache file: %s", cache_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
