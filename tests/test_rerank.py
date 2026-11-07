from src import rerank


def test_rerank_orders_by_score_and_keeps_k():
    chunks = [{"id": "a", "text": "x"}, {"id": "b", "text": "yy"}, {"id": "c", "text": "zzz"}]
    out = rerank.rerank("q", chunks, 2, lambda q, texts: [len(t) for t in texts])
    assert [c["id"] for c in out] == ["c", "b"]
    assert [c["rerank_score"] for c in out] == [3.0, 2.0]


def test_rerank_of_nothing_is_nothing():
    assert rerank.rerank("q", [], 5, lambda q, texts: []) == []


def test_cross_encoder_scores_pairs_the_question_with_each_text(monkeypatch):
    seen = []

    class FakeModel:
        def predict(self, pairs):
            seen.extend(pairs)
            return [0.1] * len(pairs)

    monkeypatch.setattr(rerank, "load_reranker", lambda name: FakeModel())
    assert list(rerank.cross_encoder_scores("ce")("q", ["a", "b"])) == [0.1, 0.1]
    assert seen == [("q", "a"), ("q", "b")]
