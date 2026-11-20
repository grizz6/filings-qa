from src import risk_map

CFG = {"companies": {"TSLA": "Tesla", "NKE": "Nike"}}


def fake_best(texts):
    rows = []
    for i, _ in enumerate(texts):
        rows.append([
            ("NKE_1A_0001", "NKE", "Nike " + "word " * 200, "https://sec/n", 0.40 + i / 10),
            ("TSLA_1A_0002", "TSLA", "Tesla text", "https://sec/t", 0.55),
            ("OLD_1A_0001", "OLD", "a company no longer configured", "https://sec/o", 0.9),
        ])  # fmt: skip
    return rows


def test_build_gives_one_cell_per_topic_and_configured_company():
    out = risk_map.build({"Cyber": "cyber attacks", "Climate": "climate"}, CFG, fake_best)
    assert out["topics"] == ["Cyber", "Climate"]
    assert out["companies"] == CFG["companies"]
    assert out["descriptions"] == {"Cyber": "cyber attacks", "Climate": "climate"}
    assert len(out["cells"]) == 4  # the unconfigured company is dropped
    cell = out["cells"][0]
    assert cell == {**cell, "topic": "Cyber", "ticker": "NKE", "score": 0.4,
                    "chunk_id": "NKE_1A_0001", "source_url": "https://sec/n"}  # fmt: skip
    assert len(cell["snippet"]) <= risk_map.SNIPPET_CHARS + 1


def test_cached_build_reuses_results_until_they_expire():
    calls = []

    def counting_best(texts):
        calls.append(texts)
        return fake_best(texts)

    risk_map._cache.clear()
    topics = {"Cyber": "cyber attacks"}
    clock = [1000.0]
    risk_map.cached_build(topics, CFG, counting_best, now=lambda: clock[0])
    risk_map.cached_build(topics, CFG, counting_best, now=lambda: clock[0])
    assert len(calls) == 1
    clock[0] += risk_map.CACHE_SECONDS
    risk_map.cached_build(topics, CFG, counting_best, now=lambda: clock[0])
    assert len(calls) == 2
    risk_map._cache.clear()
