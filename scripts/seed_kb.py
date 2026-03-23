import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from core.embedder import Embedder
from core.vectorstore import VectorStore
from utils.wiki import WikiFetcher


def main():
    print("=== VerifAI Knowledge Base Seeder ===\n")

    emb = Embedder()
    vs = VectorStore(embedder=emb)
    fetcher = WikiFetcher()

    before = vs.count()
    print(f"KB before: {before} chunks")

    total = fetcher.seed_knowledge_base(vs)

    after = vs.count()
    print(f"\nSeeded {total} new chunks")
    print(f"KB after: {after} chunks")

    docs = vs.list_documents()
    print(f"\nDocuments in KB ({len(docs)}):")
    for d in docs:
        print(f"  {d['source']:15s} | {d['chunk_count']:3d} chunks | {d['doc_name']}")

    print("\n=== Seeding complete ===")


if __name__ == "__main__":
    main()
