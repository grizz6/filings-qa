"""Per-question search settings a visitor can choose (the Ask page sidebar).

- companies: search only these filings, instead of the companies named in the question;
- passages: how many passages the answer may use (top_k);
- rerank: re-order the passages with the cross-encoder (only if config.yaml names one);
- detail: "short" (2 to 5 sentences) or "detailed" (5 to 10 sentences).

Anything left out falls back to config.yaml.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_PASSAGES = 12
Detail = Literal["short", "detailed"]


class SearchSettings(BaseModel):
    companies: list[str] | None = Field(default=None, max_length=10)
    passages: int | None = Field(default=None, ge=1, le=MAX_PASSAGES)
    rerank: bool | None = None
    detail: Detail = "short"


def unknown_companies(settings: SearchSettings, cfg: dict) -> list[str]:
    return [t for t in settings.companies or [] if t not in cfg["companies"]]


def apply(settings: SearchSettings, cfg: dict) -> dict:
    """cfg with the settings' retrieval choices applied (the shared cfg is not changed)."""
    retrieval = dict(cfg["retrieval"])
    if settings.passages:
        retrieval["top_k"] = settings.passages
    if settings.rerank is False:
        retrieval["rerank_model"] = None
    return {**cfg, "retrieval": retrieval}
