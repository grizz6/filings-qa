"""Re-order retrieved chunks with a cross-encoder.

Vector search compares a question and a chunk that were embedded separately. A cross-encoder
reads the question and the chunk together, which ranks better but is too slow to run over
every chunk, so it only re-scores the candidates vector search returns.

Enabled by `retrieval.rerank_model` in config.yaml (off when empty). `score` keeps the vector
similarity, which the app shows and the low-confidence signal uses; the cross-encoder's score
is added as `rerank_score`.
"""

from __future__ import annotations

import threading
from functools import lru_cache

_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _build_reranker(model_name: str):
    from sentence_transformers import CrossEncoder  # heavy import, only when needed

    return CrossEncoder(model_name)


def load_reranker(model_name: str):
    """Load the model once per process."""
    with _LOCK:
        return _build_reranker(model_name)


def rerank(question: str, chunks: list[dict], k: int, score_fn) -> list[dict]:
    """The k chunks with the highest score_fn(question, texts), best first."""
    if not chunks:
        return []
    scores = score_fn(question, [c["text"] for c in chunks])
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -float(pair[1]))
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:k]]


def cross_encoder_scores(model_name: str):
    """score_fn backed by a cross-encoder model."""

    def score(question: str, texts: list[str]):
        return load_reranker(model_name).predict([(question, t) for t in texts])

    return score
