"""Run the live app's recurring jobs on time.

GitHub starts scheduled workflows late or drops them (often only a few runs a day), so the
app's recurring jobs are not left to GitHub's timer. Instead one long-running job:

- opens the app every VISIT_EVERY seconds so it never goes to sleep;
- every CHECK_EVERY seconds starts the Daily traffic workflow when questions are due;
- stops after RUN_FOR seconds, and the workflow then starts its own next run, so the cycle
  continues around the clock.

Run by .github/workflows/scheduler.yml.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime

import daily_traffic

CHECK_EVERY = 10 * 60
VISIT_EVERY = 30 * 60
RUN_FOR = int(5.5 * 3600)  # GitHub stops a job after 6 hours
MAX_FAILED_VISITS = 3  # in a row; then the run fails so GitHub emails the owner


def _api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def questions_run_active() -> bool:
    """True while a Daily traffic run is queued or running (it will ask what is due)."""
    for status in ("queued", "in_progress"):
        runs = _api(f"actions/workflows/{daily_traffic.WORKFLOW}/runs?status={status}")
        if runs.get("total_count"):
            return True
    return False


def start_due_questions(now: datetime) -> int:
    """Start the Daily traffic workflow if questions are due; return how many are due."""
    _, _, due = daily_traffic.plan(now, daily_traffic.load_pool(),
                                   daily_traffic.previous_run_start(now))  # fmt: skip
    if due and not questions_run_active():
        _api(f"actions/workflows/{daily_traffic.WORKFLOW}/dispatches", "POST", {"ref": "main"})
        print(f"{now:%H:%M} UTC: {len(due)} questions due; started Daily traffic", flush=True)
    return len(due)


def main() -> None:
    import keep_awake  # needs Playwright, which only this job installs

    started, last_visit, failed_visits = time.time(), 0.0, 0
    while time.time() - started < RUN_FOR:
        if time.time() - last_visit >= VISIT_EVERY:
            try:
                keep_awake.main()
                last_visit, failed_visits = time.time(), 0
            except (Exception, SystemExit) as e:  # retried at the next check
                failed_visits += 1
                print(f"Visit failed ({failed_visits} in a row): {e}", file=sys.stderr, flush=True)
                if failed_visits >= MAX_FAILED_VISITS:
                    sys.exit("The app did not load on several visits in a row")
        try:
            start_due_questions(datetime.now(UTC))
        except Exception as e:  # the next check retries; visits must go on
            print(f"Could not check for due questions: {e}", file=sys.stderr, flush=True)
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
