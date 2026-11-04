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


class FakeLocator:
    def __init__(self, n=0, text=""):
        self.n, self.text = n, text
        self.first = self

    def count(self):
        return self.n

    def click(self):
        pass

    def inner_text(self, timeout=None):
        return self.text


class FakeFrame:
    def __init__(self, url, text="", has_question_box=False):
        self.url, self.text, self.has_question_box = url, text, has_question_box

    def get_by_role(self, role, name=None):
        return FakeLocator(0)

    def get_by_label(self, label):
        return FakeLocator(int(self.has_question_box))

    def locator(self, selector):
        return FakeLocator(text=self.text)


class FakePage:
    """A page that stays blank until it has been reloaded `loads_until_ready` times."""

    def __init__(self, clock, loads_until_ready):
        self.clock, self.loads_until_ready = clock, loads_until_ready
        self.reloads, self.screenshots = 0, []

    @property
    def frames(self):
        ready = self.reloads >= self.loads_until_ready
        return [FakeFrame("https://filings.streamlit.app/", ""),
                FakeFrame("https://filings.streamlit.app/~/+/", "app", ready)]  # fmt: skip

    def wait_for_timeout(self, ms):
        self.clock[0] += ms / 1000

    def reload(self, **kwargs):
        self.reloads += 1

    def screenshot(self, path, full_page=False):
        self.screenshots.append(path)


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(live_check.time, "time", lambda: now[0])
    return now


def test_a_stalled_page_is_reloaded_until_the_app_appears(clock):
    page = FakePage(clock, loads_until_ready=2)
    frame = app_frame_or_exit(page, clock)
    assert frame.has_question_box
    assert page.reloads == 2  # once per RELOAD_SECONDS of blank page
    assert page.screenshots == []


def test_a_page_that_never_loads_fails_with_every_frame_and_a_screenshot(clock):
    page = FakePage(clock, loads_until_ready=10_000)
    with pytest.raises(SystemExit) as failure:
        app_frame_or_exit(page, clock)
    message = str(failure.value)
    assert "did not load within" in message
    assert "https://filings.streamlit.app/: (blank)" in message
    assert "https://filings.streamlit.app/~/+/: app" in message
    assert page.screenshots == [live_check.FAILURE_SCREENSHOT]


def app_frame_or_exit(page, clock):
    return live_check.app_frame(page, clock[0] + live_check.WAKE_SECONDS)
