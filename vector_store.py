from __future__ import annotations
from typing import List, Dict, Any

import numpy as np


class VectorStore:
    MODEL_NAME = "all-MiniLM-L6-v2" 

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._index = None  
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import streamlit as st
                self._model = st.session_state.get("_cached_model") or __import__(
                    "sentence_transformers", fromlist=["SentenceTransformer"]
                ).SentenceTransformer(self.MODEL_NAME)
            except Exception:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.MODEL_NAME)

    def _embed(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine via inner-product
        )
        return embeddings.astype(np.float32)

    def build(self, chunks: List[str], metadata: List[dict]) -> None:
        import faiss

        self._chunks = chunks
        self._metadata = metadata

        embeddings = self._embed(chunks)
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if self._index is None or len(self._chunks) == 0:
            return []

        q_emb = self._embed([query])
        actual_k = min(k, len(self._chunks))
        scores, indices = self._index.search(q_emb, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "text": self._chunks[idx],
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "score": float(score),
                }
            )

        return results