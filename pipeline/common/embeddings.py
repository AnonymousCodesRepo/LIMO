"""Thin client for the all-mpnet-base-v2 embedding server.

API: POST /embed {"texts": [...]} -> {"embeddings": [[...], ...], "dim": 768}
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import requests


DEFAULT_URL = os.environ.get(
    "PIPELINE_EMBED_URL", "http://localhost:8200/embed"
)


class EmbeddingClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 60.0,
                 batch_size: int = 64, max_retries: int = 3,
                 serialize_requests: bool = True):
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        # The embedding server uses a single sentence-transformer instance and
        # occasionally 500s under concurrent load. Serializing requests at the
        # client is cheap and avoids those 500s when the runner uses multiple
        # workers.
        self._lock = threading.Lock() if serialize_requests else None

    def _post(self, texts: list[str]) -> np.ndarray:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.url, json={"texts": texts}, timeout=self.timeout
                )
                r.raise_for_status()
                return np.asarray(r.json()["embeddings"], dtype=np.float32)
            except Exception as e:
                last_err = e
                # Transient 500 / connection resets: back off briefly and retry.
                time.sleep(0.25 * (2 ** attempt))
        assert last_err is not None
        raise last_err

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if self._lock is not None:
                with self._lock:
                    arr = self._post(batch)
            else:
                arr = self._post(batch)
            chunks.append(arr)
        mat = np.concatenate(chunks, axis=0)
        # Already L2-normalized by the server; no normalization needed.
        return mat

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
