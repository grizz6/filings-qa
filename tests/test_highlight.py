from src import highlight

PASSAGE = (
    "The Company depends on a limited number of suppliers. Shortages of battery cells could "
    "delay vehicle production and increase costs. Interest rates may also rise."
)
ANSWER = (
    "Tesla warns that battery cell shortages could delay production [1]. "
    "It also mentions rates [2]."
)


def test_supporting_sentence_finds_the_passage_sentence_behind_the_claim():
    assert highlight.supporting_sentence(ANSWER, 1, PASSAGE) == (
        "Shortages of battery cells could delay vehicle production and increase costs."
    )


def test_only_sentences_citing_that_number_count():
    assert highlight.citing_text(ANSWER, 2).strip() == "It also mentions rates ."
    assert "battery" not in highlight.citing_text(ANSWER, 2)


def test_no_support_when_too_little_overlaps():
    assert highlight.supporting_sentence("Nothing related here [1].", 1, PASSAGE) is None
    assert highlight.supporting_sentence(ANSWER, 3, PASSAGE) is None  # [3] is never cited


def test_multi_citations_like_1_3_are_split():
    answer = "Shortages could delay production [1][3]."
    assert highlight.citing_text(answer, 3) == highlight.citing_text(answer, 1)
