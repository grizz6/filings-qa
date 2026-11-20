"""Shared helpers for the Streamlit pages: talk to the API, read the config.

Two ways to reach the API (src/api.py), with the same validation, limits and logging:
- API_URL set (Docker image): over HTTP to the API process next to the UI.
- API_URL not set (Streamlit Community Cloud, one process): the same FastAPI app is
  called in this process. Secrets come from Streamlit's secrets store.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

API_URL = os.environ.get("API_URL", "").rstrip("/")
TIMEOUT_SECONDS = 180  # an answer can wait out a Gemini rate limit
SECRETS = ("GEMINI_API_KEY", "SUPABASE_DB_URL")


class ApiError(RuntimeError):
    pass


def secrets_to_env() -> None:
    """Copy Streamlit secrets into the environment, where src/ reads them (env wins)."""
    try:
        import streamlit as st

        for name in SECRETS:
            if name in st.secrets and not os.environ.get(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:  # no secrets file (Docker, tests): the environment is the source
        pass


def _local_api():
    """The in-process API, rebuilt when the secrets it was built with change.

    On Streamlit Cloud the app can start before its secrets are saved; caching on which
    secrets are present keeps it from staying stuck without the database.
    """
    secrets_to_env()
    return _build_local_api(tuple(bool(os.environ.get(name, "").strip()) for name in SECRETS))


@lru_cache(maxsize=4)
def _build_local_api(secrets_present: tuple[bool, ...]):
    from fastapi.testclient import TestClient

    from src.api import create_app

    # raise_server_exceptions=False: an unexpected error becomes a 500, as over HTTP.
    return TestClient(create_app(), raise_server_exceptions=False)


def _load_models_quietly(retrieval: dict) -> None:
    from src import index, rerank

    try:
        index.load_embedder(retrieval["embedding_model"])
        if retrieval.get("rerank_model"):
            rerank.load_reranker(retrieval["rerank_model"])
    except Exception as exc:  # the first question will load them (and report errors) instead
        print(f"WARNING: model warm-up failed: {exc}", file=sys.stderr)


@lru_cache(maxsize=1)
def _warmup_thread() -> threading.Thread:
    retrieval = load_config()["retrieval"]
    thread = threading.Thread(target=_load_models_quietly, args=(retrieval,), daemon=True)
    thread.start()
    return thread


def start_warmup() -> threading.Thread | None:
    """Start loading the embedding model in the background as soon as the app starts.

    On Streamlit Cloud the app wakes on the first visit; without this, the first question
    also waited for the model to load. Only in-process mode needs it (no API_URL).
    """
    return None if API_URL else _warmup_thread()


def call(method: str, path: str, **kwargs) -> dict:
    if not API_URL:
        resp = _local_api().request(method, path, **kwargs)  # in-process: no network timeout
    else:
        try:
            resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_SECONDS, **kwargs)
        except requests.RequestException:
            raise ApiError(
                "The API is not reachable. It may still be starting; retry in a minute."
            ) from None
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, list):  # validation errors from FastAPI
            detail = "; ".join(d.get("msg", "") for d in detail)
        raise ApiError(detail or f"The API returned {resp.status_code}.")
    return resp.json()


def setup_problems() -> list[str]:
    """Names of the secrets the API reports as missing (empty when the app is ready)."""
    health = call("GET", "/health")
    flags = {"GEMINI_API_KEY": "llm_key_configured", "SUPABASE_DB_URL": "db_configured"}
    return [name for name, flag in flags.items() if not health.get(flag)]


def setting_defaults(cfg: dict) -> dict:
    return {
        "companies": [],
        "passages": cfg["retrieval"]["top_k"],
        "rerank": bool(cfg["retrieval"].get("rerank_model")),
        "detail": "short",
    }


def search_settings_sidebar(cfg: dict, choose_companies: bool = True) -> dict:
    """Sidebar controls for the search settings; returns them for POST /ask.

    choose_companies=False leaves out the company picker (pages that pick companies
    themselves); the returned companies are then None.

    Streamlit drops a widget's value when a page without that widget runs, so each value is
    also kept under its own key and restored, and the choices follow the visitor across pages.
    """
    import streamlit as st

    from src.settings import MAX_PASSAGES

    defaults = setting_defaults(cfg)
    for name, value in defaults.items():
        st.session_state.setdefault(f"keep_{name}", value)
        st.session_state[f"setting_{name}"] = st.session_state[f"keep_{name}"]

    def keep(name):
        st.session_state[f"keep_{name}"] = st.session_state[f"setting_{name}"]

    def reset():
        for name, value in defaults.items():
            st.session_state[f"keep_{name}"] = st.session_state[f"setting_{name}"] = value

    names = cfg["companies"]
    can_rerank = defaults["rerank"]
    with st.sidebar:
        st.header("Search settings")
        if choose_companies:
            st.multiselect(
                "Companies to search",
                list(names),
                format_func=lambda t: f"{names[t]} ({t})",
                key="setting_companies",
                on_change=keep,
                args=("companies",),
                placeholder="The ones named in the question",
                help="Leave empty to search the companies your question names (or all of them).",
            )
        st.slider(
            "Passages to use",
            1,
            MAX_PASSAGES,
            key="setting_passages",
            on_change=keep,
            args=("passages",),
            help="More passages can find more, but give the model more text to read.",
        )
        st.toggle(
            "Re-rank passages",
            key="setting_rerank",
            on_change=keep,
            args=("rerank",),
            disabled=not can_rerank,
            help="A cross-encoder re-orders the search results so the most relevant come first.",
        )
        st.radio(
            "Answer length",
            ["short", "detailed"],
            format_func=str.capitalize,
            key="setting_detail",
            on_change=keep,
            args=("detail",),
            horizontal=True,
        )
        st.button("Reset to defaults", on_click=reset)

    chosen = {name: st.session_state[f"keep_{name}"] for name in defaults}
    return {**chosen, "companies": (choose_companies and chosen["companies"]) or None}


def safe_markdown(text: str) -> str:
    """Model output as plain text with [n] citations: no images, outside links or HTML."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    return re.sub(r"https?://\S+", "", text).strip()


def series_color() -> str:
    """Categorical slot 1 (blue), stepped for the active light or dark theme."""
    try:
        import streamlit as st

        if st.context.theme.type == "dark":
            return "#3987e5"
    except Exception:
        pass
    return "#2a78d6"


# Sequential blue ramp (light -> dark) for magnitudes such as the risk map's scores; the
# low end recedes toward the page. .streamlit/config.toml sets a custom theme, which Streamlit
# draws light whatever the visitor's system setting, so the ramp is fixed for light.
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


__all__ = [
    "SEQUENTIAL",
    "ApiError",
    "call",
    "load_config",
    "safe_markdown",
    "search_settings_sidebar",
    "secrets_to_env",
    "series_color",
    "setup_problems",
]
