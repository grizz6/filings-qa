import pytest

from src import retrieve
from src.config import load_config

CFG = load_config()
VECTOR_ONLY = {**CFG, "retrieval": {**CFG["retrieval"], "rerank_model": None}}


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What does Tesla say about supply chain risk?", ["TSLA"]),
        ("What are TSLA's main risks?", ["TSLA"]),
        ("How does JP Morgan describe interest rate risk?", ["JPM"]),
        ("What does JPMorgan Chase say about credit risk?", ["JPM"]),
        ("Does ExxonMobil mention climate regulation?", ["XOM"]),
        ("What does Exxon Mobil say about oil prices?", ["XOM"]),
        ("What fuel risks does Delta Air Lines face?", ["DAL"]),
        ("Compare Nike and Walmart on tariffs", ["NKE", "WMT"]),
        ("Which companies mention cybersecurity?", []),
    ],
)
def test_detect_companies(question, expected):
    assert retrieve.detect_companies(question, CFG) == expected


def test_aliases_are_whole_words():
    # "Apple" inside "pineapple" or "Delta" inside "deltas" must not filter.
    assert retrieve.detect_companies("Pineapple deltas and nikes", CFG) == []


def test_retrieve_passes_company_filter_and_top_k():
    seen = {}

    def fake_search(question, index_name, tickers, k):
        seen.update(question=question, index_name=index_name, tickers=tickers, k=k)
        return [("TSLA_1A_0003", "TSLA", "Supply text", "https://sec/x", 0.71)]

    chunks = retrieve.retrieve("Tesla supply chain?", cfg=VECTOR_ONLY, search_fn=fake_search)
    assert seen == {
        "question": "Tesla supply chain?",
        "index_name": "main",
        "tickers": ["TSLA"],
        "k": CFG["retrieval"]["top_k"],
    }
    assert chunks == [
        {
            "id": "TSLA_1A_0003",
            "ticker": "TSLA",
            "text": "Supply text",
            "source_url": "https://sec/x",
            "score": 0.71,
        }
    ]


def test_retrieve_without_company_searches_everything():
    seen = {}

    def fake_search(question, index_name, tickers, k):
        seen["tickers"] = tickers
        return []

    retrieve.retrieve("cybersecurity breaches", cfg=CFG, search_fn=fake_search)
    assert seen["tickers"] is None


def test_reranking_widens_the_search_and_keeps_the_best_k():
    cfg = {**CFG, "retrieval": {**CFG["retrieval"], "top_k": 2, "rerank_model": "ce",
                                "rerank_candidates": 4}}  # fmt: skip
    seen = {}

    def fake_search(question, index_name, tickers, k):
        seen["k"] = k
        return [(f"TSLA_1A_000{i}", "TSLA", f"text {i}", "https://sec/x", 0.9 - i / 10)
                for i in range(k)]  # fmt: skip

    def fake_scores(question, texts):  # the cross-encoder prefers the last candidates
        return [float(t.split()[-1]) for t in texts]

    chunks = retrieve.retrieve("Tesla?", cfg=cfg, search_fn=fake_search, rerank_fn=fake_scores)
    assert seen["k"] == 4
    assert [c["id"] for c in chunks] == ["TSLA_1A_0003", "TSLA_1A_0002"]
    assert chunks[0]["score"] == pytest.approx(0.6)  # vector similarity is kept
    assert chunks[0]["rerank_score"] == 3.0


def test_no_reranking_when_no_model_is_configured():
    cfg = {**CFG, "retrieval": {**CFG["retrieval"], "rerank_model": None}}

    def fake_rerank(question, texts):
        raise AssertionError("must not be called")

    def fake_search(question, index_name, tickers, k):
        return [("TSLA_1A_0001", "TSLA", "t", "u", 0.5)] * k

    chunks = retrieve.retrieve("Tesla?", cfg=cfg, search_fn=fake_search, rerank_fn=fake_rerank)
    assert len(chunks) == cfg["retrieval"]["top_k"]
