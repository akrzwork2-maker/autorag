from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import paths, vectorstore_config
from core.embedder import Embedder
from core.chunker import Chunk


class VectorStore:

    SYSTEM_USER = "__system__"

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(
            path=str(paths.CHROMA_DB),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=vectorstore_config.collection_name,
            metadata={"hnsw:space": vectorstore_config.distance_metric},
        )

    def _user_filter(self, user_id: str | None) -> dict | None:
        if not user_id:
            return None
        return {"user_id": user_id}

    def _owner_filter(self, user_id: str | None) -> dict | None:
        if not user_id:
            return None
        return {"user_id": user_id}

    def add_chunks(self, chunks: list[Chunk], user_id: str | None = None) -> int:
        if not chunks:
            return 0

        uid = user_id or self.SYSTEM_USER
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode_batch(texts).tolist()

        ids = []
        metadatas = []
        for c in chunks:
            chunk_id = f"{uid}__{c.doc_name}__chunk_{c.chunk_index}"
            ids.append(chunk_id)
            metadatas.append({
                "doc_name": c.doc_name,
                "chunk_index": c.chunk_index,
                "source": c.source,
                "token_estimate": c.token_estimate,
                "user_id": uid,
                **(c.metadata or {}),
            })

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        k: int = 5,
        user_id: str | None = None,
    ) -> list[dict]:
        query_emb = self._embedder.encode(query_text).tolist()
        where = self._user_filter(user_id)

        total = self._collection.count()
        if total == 0:
            return []

        query_kwargs: dict = {
            "query_embeddings": [query_emb],
            "n_results": min(k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            retrieved.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": round(similarity, 4),
            })

        return retrieved

    def query_by_embedding(
        self,
        embedding: list[float],
        k: int = 5,
        user_id: str | None = None,
    ) -> list[dict]:
        if self._collection.count() == 0:
            return []

        where = self._user_filter(user_id)
        query_kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": min(k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            retrieved.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": round(similarity, 4),
            })

        return retrieved

    def count(self, user_id: str | None = None) -> int:
        where = self._user_filter(user_id)
        if not where:
            return self._collection.count()
        if self._collection.count() == 0:
            return 0
        results = self._collection.get(where=where, include=[])
        return len(results["ids"])

    def list_documents(self, user_id: str | None = None) -> list[dict]:
        if self._collection.count() == 0:
            return []

        where = self._user_filter(user_id)
        get_kwargs: dict = {"include": ["metadatas"]}
        if where:
            get_kwargs["where"] = where
        all_meta = self._collection.get(**get_kwargs)
        doc_map: dict[str, dict] = {}

        for meta in all_meta["metadatas"]:
            name = meta.get("doc_name", "unknown")
            if name not in doc_map:
                doc_map[name] = {
                    "doc_name": name,
                    "source": meta.get("source", "unknown"),
                    "chunk_count": 0,
                }
            doc_map[name]["chunk_count"] += 1

        return sorted(doc_map.values(), key=lambda d: d["doc_name"])

    def stats(self, user_id: str | None = None) -> dict:
        docs = self.list_documents(user_id)
        return {
            "total_chunks": self.count(user_id),
            "total_documents": len(docs),
            "embedding_dim": self._embedder.dim,
            "uploaded_docs": sum(1 for d in docs if d["source"] == "uploaded"),
            "auto_fetched_docs": sum(1 for d in docs if d["source"] == "auto_fetched"),
            "seed_docs": sum(1 for d in docs if d["source"] == "seed"),
        }

    def has_document(self, doc_name: str, user_id: str | None = None) -> bool:
        if self._collection.count() == 0:
            return False
        where = self._user_filter(user_id)
        if where:
            combined = {"$and": [{"doc_name": doc_name}, where]}
        else:
            combined = {"doc_name": doc_name}
        results = self._collection.get(
            where=combined,
            limit=1,
            include=[],
        )
        return len(results["ids"]) > 0

    def delete_document(self, doc_name: str, user_id: str | None = None) -> int:
        if self._collection.count() == 0:
            return 0
        where = self._owner_filter(user_id)
        if where:
            combined = {"$and": [{"doc_name": doc_name}, where]}
        else:
            combined = {"doc_name": doc_name}
        results = self._collection.get(
            where=combined,
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def reset(self):
        self._client.delete_collection(vectorstore_config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=vectorstore_config.collection_name,
            metadata={"hnsw:space": vectorstore_config.distance_metric},
        )

    def migrate_untagged_chunks(self, default_user_id: str | None = None):
        uid = default_user_id or self.SYSTEM_USER
        if self._collection.count() == 0:
            return 0
        all_data = self._collection.get(include=["metadatas"])
        ids_to_update = []
        metas_to_update = []
        for i, meta in enumerate(all_data["metadatas"]):
            if "user_id" not in meta:
                meta["user_id"] = uid
                ids_to_update.append(all_data["ids"][i])
                metas_to_update.append(meta)
        if ids_to_update:
            self._collection.update(ids=ids_to_update, metadatas=metas_to_update)
        return len(ids_to_update)
