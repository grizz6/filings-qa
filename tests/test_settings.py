import pytest
from pydantic import ValidationError

from src import settings
from src.config import load_config

CFG = load_config()


def test_apply_overrides_only_what_was_chosen():
    out = settings.apply(settings.SearchSettings(passages=4, rerank=False), CFG)
    assert out["retrieval"]["top_k"] == 4 and out["retrieval"]["rerank_model"] is None
    assert out["retrieval"]["embedding_model"] == CFG["retrieval"]["embedding_model"]
    assert CFG["retrieval"]["rerank_model"]  # the shared config is not changed


def test_defaults_keep_config():
    assert settings.apply(settings.SearchSettings(), CFG) == CFG


def test_rerank_true_does_not_invent_a_model():
    cfg = {**CFG, "retrieval": {**CFG["retrieval"], "rerank_model": None}}
    assert (
        settings.apply(settings.SearchSettings(rerank=True), cfg)["retrieval"]["rerank_model"]
        is None
    )


def test_unknown_companies():
    chosen = settings.SearchSettings(companies=["TSLA", "ZZZ"])
    assert settings.unknown_companies(chosen, CFG) == ["ZZZ"]


@pytest.mark.parametrize("bad", [{"passages": 0}, {"passages": 13}, {"detail": "essay"}])
def test_out_of_range_settings_are_rejected(bad):
    with pytest.raises(ValidationError):
        settings.SearchSettings(**bad)
