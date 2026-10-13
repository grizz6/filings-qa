import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parents[1] / "checks"))

import scheduler  # noqa: E402

NY = ZoneInfo("America/New_York")


def fake_api(active=False):
    calls = []

    def api(path, method="GET", body=None):
        calls.append((method, path, body))
        if path.endswith("status=in_progress"):
            return {"total_count": int(active)}
        return {"total_count": 0}

    return api, calls


def test_starts_daily_traffic_when_questions_are_due(monkeypatch):
    api, calls = fake_api()
    monkeypatch.setattr(scheduler, "_api", api)
    monkeypatch.setattr(scheduler.daily_traffic, "previous_run_start", lambda now: None)
    due = scheduler.start_due_questions(datetime(2026, 10, 13, 13, 30, tzinfo=NY))
    assert due > 0
    assert calls[-1] == ("POST", "actions/workflows/daily-traffic.yml/dispatches", {"ref": "main"})


def test_does_not_start_a_second_run_while_one_is_active(monkeypatch):
    api, calls = fake_api(active=True)
    monkeypatch.setattr(scheduler, "_api", api)
    monkeypatch.setattr(scheduler.daily_traffic, "previous_run_start", lambda now: None)
    assert scheduler.start_due_questions(datetime(2026, 10, 13, 13, 30, tzinfo=NY)) > 0
    assert all(method == "GET" for method, _, _ in calls)


def test_nothing_happens_outside_the_window(monkeypatch):
    api, calls = fake_api()
    monkeypatch.setattr(scheduler, "_api", api)
    monkeypatch.setattr(scheduler.daily_traffic, "previous_run_start", lambda now: None)
    assert scheduler.start_due_questions(datetime(2026, 10, 13, 7, 0, tzinfo=NY)) == 0
    assert calls == []


def test_timing_keeps_the_app_awake_and_fits_in_one_job():
    assert scheduler.VISIT_EVERY <= 60 * 60  # far inside Streamlit's 12-hour sleep limit
    assert scheduler.RUN_FOR + 20 * 60 < 355 * 60  # the workflow timeout (GitHub max: 6 h)
