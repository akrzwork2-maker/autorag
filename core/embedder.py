import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import models


class Embedder:

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or models.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return models.EMBEDDING_DIM

    def encode(self, text: str) -> np.ndarray:
        model = self._load()
        emb = model.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        model = self._load()
        embs = model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
        return np.asarray(embs, dtype=np.float32)

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        return float(np.dot(emb_a, emb_b))

    def similarity_scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        q_emb = self.encode(query)
        d_embs = self.encode_batch(documents)
        scores = d_embs @ q_emb
        return scores.tolist()

    def contextual_density(self, text: str) -> float:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) < 2:
            return 1.0
        embs = self.encode_batch(sentences)
        sim_matrix = embs @ embs.T
        n = len(sentences)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i, j]
                count += 1
        return float(total / count) if count > 0 else 1.0

    def is_loaded(self) -> bool:
        return self._model is not None
