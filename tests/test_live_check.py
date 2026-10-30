import pytest

pytest.importorskip("playwright")

from checks import live_check  # noqa: E402

ANSWER = (
    "While Nike uses hedging activities to reduce volatility, these measures also limit the "
    "potential positive impact of a weaker U.S. Dollar [2]. Quota rules may apply [1]."
)


def test_answer_text_does_not_end_the_wait_early():
    # An answer that is still rendering (no footer yet) must not count as finished, even
    # when it mentions words like "limit" or "quota".
    assert not live_check.DONE.search(ANSWER)
    assert live_check.DONE.search(ANSWER + "\nAnswered in 16.6 s · model m")


@pytest.mark.parametrize(
    "message",
    [
        "Not found in the filings.",
        "The demo's daily limit of questions is reached. Try again tomorrow (UTC).",
        "The free Gemini quota is used up for today. Try again tomorrow.",
        "The language model is unavailable right now. Try again in a minute.",
        "Something went wrong while answering. Please try again.",
        "Questions are limited to 500 characters.",
        "The API is not reachable. It may still be starting; retry in a minute.",
        "Setup incomplete: missing `GEMINI_API_KEY`.",
    ],
)
def test_every_final_app_message_ends_the_wait(message):
    assert live_check.DONE.search(message)
