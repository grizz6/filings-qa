"""UI tests: run the Streamlit pages headless against fake API responses."""

import sys

import pytest
import requests

from src.config import ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))  # the pages import `client` from their own folder
import client  # noqa: E402

ANSWER = {
    "question": "What does Tesla say about supply chain risk?",
    "answer": "Component shortages could delay production [1].",
    "refused": False,
    "citations": [{"n": 1, "id": "TSLA_1A_0002", "ticker": "TSLA", "source_url": "https://sec/t"}],
    "sources": [{"id": "TSLA_1A_0002", "ticker": "TSLA", "score": 0.61,
                 "source_url": "https://sec/t", "snippet": "Tesla depends on suppliers…",
                 "text": "Tesla depends on suppliers. Component shortages could delay "
                         "production of our vehicles. Costs may rise.",
                 "rerank_score": 4.2},
                {"id": "TSLA_1A_0007", "ticker": "TSLA", "score": 0.44,
                 "source_url": "https://sec/t", "snippet": "Other…", "text": "Other risks."}],
    "latency_ms": 2300,
}  # fmt: skip
STATS = {
    "days": 14, "total": 3, "answered": 2, "refused": 1, "errors": 0,
    "refusal_rate": 1 / 3, "low_confidence_rate": 0.25,
    "latency_p50_ms": 2100, "latency_p95_ms": 4800,
    "daily": [{"date": f"2026-10-{d:02d}", "questions": d % 3, "refused": 0, "errors": 0,
               "latency_p50_ms": 2000 if d % 3 else None} for d in range(1, 15)],
    "recent": [{"ts": "2026-10-06T12:00:00+00:00", "status": "answered",
                "top_score": 0.6, "latency_ms": 2100}],
}  # fmt: skip


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture
def api(monkeypatch):
    """Route the pages' API calls to canned responses: api[path] = (status, body)."""
    routes, seen = {}, []

    def fake_request(method, url, timeout, **kwargs):
        path = url.removeprefix(client.API_URL)
        seen.append((method, path, kwargs))
        result = routes[path]
        if isinstance(result, Exception):
            raise result
        return Resp(*result)

    monkeypatch.setattr(client, "API_URL", "http://api.test")  # HTTP mode, as in Docker
    routes["/health"] = (200, {"status": "ok", "model": "m", "top_k": 8,
                               "llm_key_configured": True, "db_configured": True})  # fmt: skip
    monkeypatch.setattr(requests, "request", fake_request)
    routes["_seen"] = seen
    return routes


def page(name):
    at = AppTest.from_file(str(ROOT / "app" / name), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def ask(at, question):
    at.text_input[0].input(question)
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_client_turns_api_errors_into_readable_messages(api):
    api["/a"] = (503, {"detail": "The free Gemini quota is used up for today."})
    api["/b"] = (422, {"detail": [{"msg": "String too short"}, {"msg": "bad"}]})
    api["/c"] = (500, ValueError("not json"))
    api["/d"] = requests.ConnectionError("refused")
    for path, message in [("/a", "quota is used up"), ("/b", "String too short; bad"),
                          ("/c", "returned 500"), ("/d", "not reachable")]:  # fmt: skip
        with pytest.raises(client.ApiError, match=message):
            client.call("GET", path)
    api["/ok"] = (200, {"fine": True})
    assert client.call("GET", "/ok") == {"fine": True}


def test_ask_page_renders_examples_and_companies(api):
    at = page("Ask.py")
    assert at.title[0].value == "Filings Q&A"
    assert sum("?" in b.label for b in at.button) == 4  # example questions
    assert "Tesla (TSLA)" in at.caption[0].value


def test_example_button_fills_the_question(api):
    at = page("Ask.py")
    next(b for b in at.button if "Tesla" in b.label).click()
    at.run()
    assert at.text_input[0].value == "What does Tesla say about supply chain risk?"


def test_ask_shows_cited_answer_and_sources(api):
    api["/ask"] = (200, ANSWER)
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    text = " ".join(m.value for m in at.markdown)
    assert "Component shortages could delay production [1]." in text
    assert "[open the 10-K](https://sec/t)" in text
    assert any(e.label == "Retrieved excerpts (2)" for e in at.expander)
    assert any("2.3 s" in c.value for c in at.caption)
    method, path, kwargs = api["_seen"][-1]
    assert (method, path) == ("POST", "/ask")
    assert kwargs["json"] == {
        "question": "What does Tesla say about supply chain risk?",
        "settings": {"companies": None, "passages": client.load_config()["retrieval"]["top_k"],
                     "rerank": True, "detail": "short"},
    }  # fmt: skip


def test_ask_shows_refusal_as_info(api):
    api["/ask"] = (200, {**ANSWER, "answer": "Not found in the filings.", "refused": True,
                         "citations": []})  # fmt: skip
    at = ask(page("Ask.py"), "Who is the CEO of Microsoft?")
    assert "Not found in the filings" in at.info[0].value


def test_ask_shows_api_errors_as_warning(api):
    api["/ask"] = (429, {"detail": "The demo's daily limit of questions is reached."})
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    assert "daily limit" in at.warning[0].value


def test_monitoring_page_shows_kpis_charts_and_recent_questions(api):
    api["/stats"] = (200, STATS)
    at = page("pages/4_Monitoring.py")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "Questions": "3",
        "Refusal rate": "33%",
        "Latency p50 / p95": "2.1 s / 4.8 s",
        "Low-confidence retrievals": "25%",
    }
    assert len(at.get("vega_lite_chart")) == 2
    assert len(at.dataframe) >= 1
    assert api["_seen"][-1][2]["params"] == {"days": 14}


def test_monitoring_page_with_no_questions_yet(api):
    empty = {"total": 0, "answered": 0, "refused": 0, "refusal_rate": None,
             "low_confidence_rate": None, "latency_p50_ms": None, "latency_p95_ms": None,
             "recent": []}  # fmt: skip
    api["/stats"] = (200, {**STATS, **empty})
    at = page("pages/4_Monitoring.py")
    assert "No questions in this period yet" in at.info[0].value
    assert {m.label: m.value for m in at.metric}["Refusal rate"] == "–"


def test_monitoring_page_when_api_is_down(api):
    api["/stats"] = requests.ConnectionError("refused")
    at = page("pages/4_Monitoring.py")
    assert "not reachable" in at.warning[0].value


def test_without_api_url_the_api_runs_in_this_process(monkeypatch):
    # Streamlit Community Cloud runs one process: the UI calls the same FastAPI app in-process.
    from fastapi.testclient import TestClient

    from src import api, monitor

    def fake_answer(q, settings=None):
        return {"question": q, "answer": "Risk [1].", "refused": False,
                "citations": [{"n": 1, "id": "TSLA_1A_0002", "source_url": "https://sec/t"}],
                "chunks": [{"id": "TSLA_1A_0002", "ticker": "TSLA", "text": "t",
                            "source_url": "https://sec/t", "score": 0.6}]}  # fmt: skip

    local = TestClient(api.create_app(answer_fn=fake_answer, log=monitor.MemoryLog()))
    monkeypatch.setattr(client, "API_URL", "")
    monkeypatch.setattr(client, "_local_api", lambda: local)
    monkeypatch.setattr(requests, "request", lambda *a, **k: pytest.fail("no HTTP expected"))
    body = client.call("POST", "/ask", json={"question": "Tesla supply risk?"})
    assert body["answer"] == "Risk [1]."
    with pytest.raises(client.ApiError, match="String should have at least 3 characters"):
        client.call("POST", "/ask", json={"question": "hi"})  # same validation as over HTTP
    assert client.call("GET", "/stats", params={"days": 7})["total"] == 1


def test_streamlit_secrets_become_environment_variables(monkeypatch):
    import streamlit

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "already-set")
    secrets = {"GEMINI_API_KEY": "from-secrets", "SUPABASE_DB_URL": "from-secrets"}
    monkeypatch.setattr(streamlit, "secrets", secrets)
    client.secrets_to_env()
    assert client.os.environ["GEMINI_API_KEY"] == "from-secrets"
    assert client.os.environ["SUPABASE_DB_URL"] == "already-set"  # env wins, e.g. Docker


def test_missing_streamlit_secrets_are_fine(monkeypatch):
    import streamlit

    class NoSecrets:
        def __contains__(self, key):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(streamlit, "secrets", NoSecrets())
    client.secrets_to_env()  # does not raise


def test_streamlit_cloud_requirements_cover_the_app():
    # Streamlit Community Cloud installs app/requirements.txt (next to the entrypoint).
    import re

    lines = (ROOT / "app" / "requirements.txt").read_text().splitlines()
    names = {re.split(r"[<>=\[ ]", ln)[0].lower() for ln in lines if ln and ln[0].isalpha()}
    needed = {"requests", "pyyaml", "numpy", "sentence-transformers", "psycopg", "pgvector",
              "fastapi", "httpx", "streamlit"}  # fmt: skip
    assert needed <= names, needed - names
    assert any("download.pytorch.org/whl/cpu" in ln for ln in lines)  # small CPU-only PyTorch


def test_setup_problems_name_missing_secrets(api):
    health = {"status": "ok", "model": "m", "top_k": 8}
    api["/health"] = (200, {**health, "llm_key_configured": False, "db_configured": True})
    assert client.setup_problems() == ["GEMINI_API_KEY"]
    api["/health"] = (200, {**health, "llm_key_configured": True, "db_configured": True})
    assert client.setup_problems() == []


def test_ask_page_explains_missing_secrets(api):
    api["/health"] = (200, {"status": "ok", "model": "m", "top_k": 8,
                            "llm_key_configured": False, "db_configured": False})  # fmt: skip
    at = page("Ask.py")
    assert "GEMINI_API_KEY" in at.error[0].value and "SUPABASE_DB_URL" in at.error[0].value
    assert "Secrets" in at.error[0].value


def test_ask_page_has_no_setup_error_when_configured(api):
    at = page("Ask.py")  # the api fixture's default /health reports both secrets set
    assert len(at.error) == 0


def test_in_process_api_is_rebuilt_when_secrets_arrive(monkeypatch):
    # Streamlit Cloud: the app can start before secrets are saved; it must not stay stuck
    # with an in-memory log once SUPABASE_DB_URL appears.
    monkeypatch.setattr(client, "secrets_to_env", lambda: None)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    client._build_local_api.cache_clear()
    first = client._local_api()
    assert client._local_api() is first  # cached while nothing changes
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host:5432/db")
    assert client._local_api() is not first
    client._build_local_api.cache_clear()


def test_answer_text_cannot_render_images_or_links():
    # The answer comes from a model that also read the user's question: keep it plain text.
    text = "Risk [1]. ![x](http://evil.test/p.png) See [here](http://evil.test) and <b>hi</b>"
    safe = client.safe_markdown(text)
    assert "http://evil.test" not in safe and "![" not in safe
    assert "Risk [1]." in safe and "here" in safe
    assert "<b>" not in safe


def test_warmup_loads_the_model_once_in_the_background(monkeypatch):
    # Streamlit Cloud wakes the app on the first visit; loading the model then made the
    # first question slow. The warm-up starts loading as soon as the app starts.
    from src import index, rerank

    loaded = []
    monkeypatch.setattr(client, "API_URL", "")
    monkeypatch.setattr(index, "load_embedder", lambda name: loaded.append(name))
    monkeypatch.setattr(rerank, "load_reranker", lambda name: loaded.append(name))
    client._warmup_thread.cache_clear()
    client.start_warmup().join(timeout=5)
    client.start_warmup().join(timeout=5)
    retrieval = client.load_config()["retrieval"]
    assert loaded == [retrieval["embedding_model"], retrieval["rerank_model"]]
    client._warmup_thread.cache_clear()


def test_no_warmup_when_a_separate_api_serves_requests(monkeypatch):
    monkeypatch.setattr(client, "API_URL", "http://api.test")
    assert client.start_warmup() is None


def test_sidebar_settings_are_sent_with_the_question(api):
    api["/ask"] = (200, ANSWER)
    at = page("Ask.py")
    at.sidebar.multiselect[0].select("NKE")
    at.sidebar.slider[0].set_value(4)
    at.sidebar.toggle[0].set_value(False)
    at.sidebar.radio[0].set_value("detailed")
    at.run()
    ask(at, "How are tariffs described?")
    sent = api["_seen"][-1][2]["json"]["settings"]
    assert sent == {"companies": ["NKE"], "passages": 4, "rerank": False, "detail": "detailed"}


def test_sidebar_reset_restores_the_defaults(api):
    at = page("Ask.py")
    at.sidebar.slider[0].set_value(3)
    at.run()
    next(b for b in at.sidebar.button if b.label == "Reset to defaults").click()
    at.run()
    assert at.sidebar.slider[0].value == client.load_config()["retrieval"]["top_k"]


RISK_MAP = {
    "topics": ["Cybersecurity", "Climate"],
    "descriptions": {"Cybersecurity": "cyber attacks", "Climate": "climate change"},
    "companies": {"TSLA": "Tesla", "NKE": "Nike"},
    "cells": [
        {"topic": t, "ticker": k, "score": s, "chunk_id": f"{k}_1A_0001",
         "snippet": f"{k} passage on {t}", "source_url": f"https://sec/{k}"}
        for t, k, s in [("Cybersecurity", "TSLA", 0.62), ("Cybersecurity", "NKE", 0.48),
                        ("Climate", "TSLA", 0.55), ("Climate", "NKE", 0.41)]
    ],
}  # fmt: skip


@pytest.fixture
def fresh_cache():
    import streamlit as st

    st.cache_data.clear()  # the page caches the map; tests must not see each other's
    yield
    st.cache_data.clear()


def test_risk_explorer_draws_the_map_and_drills_into_a_cell(api, fresh_cache):
    api["/risk-map"] = (200, RISK_MAP)
    at = page("pages/2_Risk_Explorer.py")
    assert at.title[0].value == "Risk Explorer"
    assert at.multiselect[0].value == ["Cybersecurity", "Climate"]
    assert at.selectbox[1].value == "Tesla (TSLA)"  # strongest company for the topic first
    assert any("TSLA passage on Cybersecurity" in c.value for c in at.caption)
    assert any("[Open the 10-K](https://sec/TSLA)" in m.value for m in at.markdown)


def test_risk_explorer_adds_a_custom_topic(api, fresh_cache):
    api["/risk-map"] = (200, RISK_MAP)
    at = page("pages/2_Risk_Explorer.py")
    at.text_input[0].input("Water")
    at.text_input[1].input("water scarcity and drought")
    next(b for b in at.button if b.label == "Add topic").click()
    at.run()
    assert not at.exception, at.exception
    method, path, kwargs = api["_seen"][-1]
    assert (method, path) == ("POST", "/risk-map")
    assert kwargs["json"] == {"topics": {"Water": "water scarcity and drought"}}


def test_risk_explorer_reports_an_unavailable_map(api, fresh_cache):
    api["/risk-map"] = (503, {"detail": "The risk map is unavailable right now. Try again."})
    at = page("pages/2_Risk_Explorer.py")
    assert "unavailable" in at.warning[0].value


def compare_page():
    at = page("pages/3_Compare.py")
    assert at.title[0].value == "Compare companies"
    return at


def press(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_compare_asks_each_company_its_own_filing(api):
    api["/ask"] = (200, ANSWER)
    at = compare_page()
    assert not at.sidebar.multiselect  # this page picks companies itself
    at.text_input[0].input("How do they describe supply chain risk?")
    press(at, "Compare")
    asked = [kw["json"] for m, p, kw in api["_seen"] if p == "/ask"]
    assert [a["settings"]["companies"] for a in asked] == [["TSLA"], ["NKE"]]
    assert {a["question"] for a in asked} == {"How do they describe supply chain risk?"}
    text = " ".join(m.value for m in at.markdown)
    assert "#### Tesla (TSLA)" in text and "#### Nike (NKE)" in text
    assert text.count("Component shortages could delay production [1].") == 2


def test_compare_example_fills_question_and_companies(api):
    at = compare_page()
    press(at, "How could cybersecurity incidents affect them?")
    assert at.text_input[0].value == "How could cybersecurity incidents affect them?"
    assert at.multiselect[0].value == ["JPM", "MSFT", "NFLX"]


def test_compare_shows_refusals_and_errors_per_company(api):
    api["/ask"] = (200, {**ANSWER, "refused": True, "answer": "Not found in the filings."})
    at = compare_page()
    at.text_input[0].input("What is their revenue?")
    press(at, "Compare")
    assert sum("Not found in this filing" in i.value for i in at.info) == 2
    api["/ask"] = (429, {"detail": "The demo's daily limit of questions is reached."})
    press(at, "Compare")
    assert sum("daily limit" in w.value for w in at.warning) == 2


def test_compare_needs_a_question_and_two_companies(api):
    at = compare_page()
    press(at, "Compare")
    assert "Write a question first." in at.warning[0].value
    at.text_input[0].input("Supply risk?")
    at.multiselect[0].set_value(["TSLA"])
    press(at, "Compare")
    assert "at least 2 companies" in at.warning[0].value


def test_answer_view_highlights_support_charts_passages_and_offers_a_download(api):
    api["/ask"] = (200, ANSWER)
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    cited = next(e for e in at.expander if e.label == "[1] TSLA · TSLA_1A_0002")
    text = " ".join(m.value for m in cited.markdown)
    assert ":blue-background[Component shortages could delay production of our vehicles.]" in text
    assert len(at.get("vega_lite_chart")) == 1  # passage relevance chart
    assert at.get("download_button")[0].proto.label == "Download answer (Markdown)"


def test_answer_stays_on_screen_after_other_interactions(api):
    api["/ask"] = (200, ANSWER)
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    at.sidebar.radio[0].set_value("detailed")
    at.run()
    assert any("Component shortages" in m.value for m in at.markdown)


def test_answer_markdown_records_answer_sources_and_settings():
    md = client.answer_markdown(
        "Q?", ANSWER, {"companies": ["TSLA"], "passages": 8, "rerank": True,
                       "detail": "short"}, "m"
    )  # fmt: skip
    assert md.startswith("# Q?\n")
    assert "Component shortages could delay production [1]." in md
    assert "[1] TSLA 10-K, Item 1A, passage TSLA_1A_0002: https://sec/t" in md
    assert "companies: TSLA · passages: 8 · re-ranked: yes" in md


def test_filing_text_cannot_turn_into_markdown_or_latex():
    out = client.highlighted_passage("Costs of $5 *million* [see] # note", "Costs of $5")
    assert out == ":blue-background[Costs of \\$5] \\*million\\* \\[see\\] \\# note"
