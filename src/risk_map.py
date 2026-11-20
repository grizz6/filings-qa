"""Risk map: how strongly each company's Risk Factors section speaks to each risk topic.

A topic is a short description ("cybersecurity breaches and data security incidents"). It is
embedded like a question, and each company's best-matching passage gives that company's
score for the topic: the cosine similarity, 0 to 1. Higher means the filing has a passage
closely about the topic. No language model is called, so the map costs nothing to build.

The default topics are `risk_topics` in config.yaml; visitors can add their own.
"""

from __future__ import annotations

import time

SNIPPET_CHARS = 400
CACHE_SECONDS = 3600  # the index changes only when a company files a new 10-K
MAX_TOPICS = 15
MAX_TOPIC_CHARS = 200

_cache: dict[tuple, tuple[float, dict]] = {}


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


def build(topics: dict[str, str], cfg: dict, best_fn) -> dict:
    """{topics, companies, cells}: one cell per topic and company with its best passage.

    best_fn(texts) returns, for each text, rows of (id, ticker, text, source_url, score).
    """
    names = list(topics)
    results = best_fn([topics[n] for n in names])
    companies = cfg["companies"]
    cells = []
    for name, rows in zip(names, results, strict=True):
        for chunk_id, ticker, text, source_url, score in rows:
            if ticker not in companies:
                continue
            cells.append(
                {
                    "topic": name,
                    "ticker": ticker,
                    "score": round(float(score), 4),
                    "chunk_id": chunk_id,
                    "snippet": _snippet(text),
                    "source_url": source_url,
                }
            )
    return {"topics": names, "companies": companies, "cells": cells}


def cached_build(topics: dict[str, str], cfg: dict, best_fn, now=time.time) -> dict:
    """build() with results kept for CACHE_SECONDS per set of topics."""
    key = tuple(sorted(topics.items()))
    hit = _cache.get(key)
    if hit and now() - hit[0] < CACHE_SECONDS:
        return hit[1]
    result = build(topics, cfg, best_fn)
    _cache[key] = (now(), result)
    return result
