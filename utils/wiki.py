from __future__ import annotations

import logging

import numpy as np
import wikipediaapi
import requests
from ddgs import DDGS

from core.config import calibration, SEED_WIKIPEDIA_TOPICS
from core.chunker import Chunker

logger = logging.getLogger(__name__)

_wiki = wikipediaapi.Wikipedia(
    user_agent="VerifAI/1.0 (student project; contact@verifai.local)",
    language="en",
)
_HEADERS = {"User-Agent": "VerifAI/1.0 (student project; contact@verifai.local)"}


def fetch_article(title: str) -> str | None:
    page = _wiki.page(title)
    if not page.exists():
        logger.warning("Wikipedia page not found: %s", title)
        return None
    return page.text


def fetch_article_sections(title: str, max_chars: int = 15000) -> str | None:
    text = fetch_article(title)
    if text:
        return text[:max_chars]
    return None


class WikiFetcher:

    def __init__(self, chunker: Chunker | None = None):
        self._chunker = chunker or Chunker()
        self._ddgs = DDGS()

    def _web_search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            results = list(self._ddgs.text(query, max_results=max_results))
            return [{"title": r["title"], "body": r["body"], "href": r["href"]}
                    for r in results if r.get("body")]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return []

    def _fetch_web_chunks(self, query: str, vectorstore, embedder, user_id=None, max_results: int = 5) -> int:
        results = self._web_search(query, max_results=max_results)
        if not results:
            return 0

        query_emb = embedder.encode(query)
        total_added = 0

        for r in results:
            slug = r["title"][:60].replace(" ", "_").replace("/", "_")
            doc_name = f"web_{slug}"
            if vectorstore.has_document(doc_name, user_id=user_id):
                continue

            text = f"{r['title']}\n\n{r['body']}"
            chunks = self._chunker.chunk_text(
                text, doc_name=doc_name, source="web",
                metadata={"triggered_by": query, "url": r["href"], "web_title": r["title"]},
            )

            filtered = []
            for chunk in chunks:
                chunk_emb = embedder.encode(chunk.text)
                sim = float(np.dot(query_emb, chunk_emb))
                if sim >= calibration.redundancy_min_similarity:
                    filtered.append(chunk)

            if filtered:
                added = vectorstore.add_chunks(filtered, user_id=user_id)
                total_added += added
                logger.info("Indexed %d web chunks from: %s", added, r["title"])

        return total_added

    def fetch_and_index(self, query: str, vectorstore, embedder=None, user_id: str | None = None) -> dict:
        from core.embedder import Embedder
        emb = embedder or Embedder()

        search_terms = self._extract_search_terms(query)
        all_terms = list(dict.fromkeys(search_terms))

        wiki_added = 0
        query_emb = emb.encode(query)

        for term in all_terms:
            doc_name = f"wiki_{term.replace(' ', '_')}"
            if vectorstore.has_document(doc_name, user_id=user_id):
                continue

            text = fetch_article_sections(term)
            if not text:
                continue

            chunks = self._chunker.chunk_text(
                text, doc_name=doc_name, source="auto_fetched",
                metadata={"triggered_by": query, "wiki_title": term},
            )

            filtered = []
            for chunk in chunks:
                chunk_emb = emb.encode(chunk.text)
                sim = float(np.dot(query_emb, chunk_emb))
                if sim >= calibration.redundancy_min_similarity:
                    filtered.append(chunk)

            if filtered:
                added = vectorstore.add_chunks(filtered, user_id=user_id)
                wiki_added += added
                logger.info("Indexed %d/%d chunks from wiki:%s", added, len(chunks), term)

        web_added = self._fetch_web_chunks(query, vectorstore, emb, user_id=user_id)

        return {"wiki": wiki_added, "web": web_added, "total": wiki_added + web_added}

    def _wiki_search(self, text: str, limit: int = 4) -> list[str]:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": text,
                    "srlimit": limit,
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [r["title"] for r in data.get("query", {}).get("search", [])]
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)
        return []

    def _extract_search_terms(self, query: str) -> list[str]:
        stop = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'was',
                'were', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 'or', 'can', 'does',
                'do', 'from', 'with', 'between', 'about', 'its', 'it', 'this', 'that', 'be',
                'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
                'not', 'but', 'if', 'then', 'than', 'so', 'very', 'just', 'also', 'into',
                'each', 'most', 'some', 'any', 'all', 'many', 'much', 'own', 'such', 'only'}
        words = [w.strip('?.,!"\'-()[]') for w in query.lower().split()]
        keywords = [w for w in words if w and len(w) > 2 and w not in stop]

        results = self._wiki_search(query, limit=4)

        if len(results) < 2 and keywords:
            combo = ' '.join(keywords[:4])
            results.extend(self._wiki_search(combo, limit=3))

        if len(results) < 2 and len(keywords) >= 2:
            for i in range(0, min(len(keywords) - 1, 3)):
                pair = ' '.join(keywords[i:i + 2])
                results.extend(self._wiki_search(pair, limit=2))

        seen = set()
        unique = []
        for t in results:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:6]

    def smart_seed(self, vectorstore, topic: str | None = None, user_id: str | None = None) -> dict:
        from core.vectorstore import VectorStore
        from core.embedder import Embedder
        uid = user_id or VectorStore.SYSTEM_USER

        if topic:
            search_topics = self._extract_search_terms(topic)
        else:
            docs = vectorstore.list_documents(uid)
            doc_names = [d["doc_name"] for d in docs if d["source"] == "uploaded"]
            if not doc_names:
                return {"wiki": 0, "web": 0, "total": 0}
            search_topics = []
            for name in doc_names[:3]:
                clean = name.replace("_", " ").replace("-", " ")
                search_topics.extend(self._extract_search_terms(clean))
            seen = set()
            search_topics = [t for t in search_topics if not (t in seen or seen.add(t))]

        wiki_added = 0
        for topic_name in search_topics[:6]:
            doc_name = f"wiki_{topic_name.replace(' ', '_').replace('(', '').replace(')', '')}"

            if vectorstore.has_document(doc_name, user_id=uid):
                continue

            text = fetch_article_sections(topic_name)
            if not text:
                continue

            chunks = self._chunker.chunk_text(
                text, doc_name=doc_name, source="auto_fetched",
                metadata={"wiki_title": topic_name},
            )
            if chunks:
                added = vectorstore.add_chunks(chunks, user_id=uid)
                wiki_added += added
                logger.info("Smart-seeded %d chunks from: %s", added, topic_name)

        web_query = topic or (search_topics[0] if search_topics else None)
        web_added = 0
        if web_query:
            emb = Embedder()
            web_added = self._fetch_web_chunks(web_query, vectorstore, emb, user_id=uid)

        return {"wiki": wiki_added, "web": web_added, "total": wiki_added + web_added}

    def seed_knowledge_base(self, vectorstore, topics: list[str] | None = None, user_id: str | None = None) -> int:
        from core.vectorstore import VectorStore
        system_uid = user_id or VectorStore.SYSTEM_USER
        topics = topics or SEED_WIKIPEDIA_TOPICS
        total = 0

        for topic in topics:
            doc_name = f"wiki_{topic.replace(' ', '_').replace('(', '').replace(')', '')}"

            if vectorstore.has_document(doc_name, user_id=system_uid):
                logger.info("Skipping %s (already indexed)", doc_name)
                continue

            text = fetch_article_sections(topic)
            if not text:
                continue

            chunks = self._chunker.chunk_text(
                text,
                doc_name=doc_name,
                source="seed",
                metadata={"wiki_title": topic},
            )

            if chunks:
                added = vectorstore.add_chunks(chunks, user_id=system_uid)
                total += added
                logger.info("Seeded %d chunks from: %s", added, topic)

        return total
