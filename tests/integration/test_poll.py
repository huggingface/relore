"""Section 5.2: the poll loop, and the three rules that keep it from losing threads.

The failure class under test is not a crash -- it is a pass that reports success while a
thread is never seen again.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine, func, select

from relore.duration import parse_interval
from relore.ingest import poll as poll_mod
from relore.ingest.poll import PASS, poll_once
from relore.store import repository as repo_layer
from relore.store import schema as s

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _poll(engine: Engine, fake: FakeGitHub):
    with fake.client() as client:
        return poll_once(engine, client, REPO)


def _mark(engine: Engine) -> dt.datetime | None:
    with engine.connect() as conn:
        return repo_layer.high_water(conn, REPO, PASS)


def _document_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(s.documents)).scalar_one())


@pytest.mark.parametrize(
    ("text", "seconds"), [("30s", 30), ("5m", 300), ("2h", 7200), ("1d", 86400), ("90", 90)]
)
def test_interval_parsing(text: str, seconds: float) -> None:
    assert parse_interval(text) == seconds


def test_bad_interval_says_what_it_wanted() -> None:
    with pytest.raises(ValueError, match="5m"):
        parse_interval("soon")


def test_a_first_pass_indexes_everything_and_sets_the_mark(
    engine: Engine, fake: FakeGitHub
) -> None:
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    fake.add_issue(2, updated_at="2026-02-01T00:00:00Z")

    result = _poll(engine, fake)

    assert (result.moved, result.indexed) == (2, 2)
    assert result.clean
    assert _mark(engine) == dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)


def test_a_second_pass_with_no_changes_writes_nothing(engine: Engine, fake: FakeGitHub) -> None:
    """The steady state. Section 5.1 makes the re-read free; this proves it stays free."""
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)
    before = _document_count(engine)

    second = _poll(engine, fake)

    assert second.documents_written == 0
    assert _document_count(engine) == before


def test_the_mark_is_a_high_water_not_a_wall_clock_window(engine: Engine, fake: FakeGitHub) -> None:
    """Miss a pass and a wall-clock window loses the gap permanently and invisibly.

    Here the poller is "down" while a thread moves; the persisted mark simply makes the
    next pass bigger, and nothing is lost.
    """
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)

    fake.add_issue(2, updated_at="2026-06-01T00:00:00Z")  # happened while we were away
    result = _poll(engine, fake)

    assert 2 in {row for row in _indexed_numbers(engine)}
    assert result.indexed >= 1
    assert _mark(engine) == dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def test_review_activity_is_found_by_the_second_list_query(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The reason there are two list queries at all.

    A review comment does not reliably bump its thread's ``updated_at``, so a poll driven
    only by ``/issues?since=`` would never re-read this PR. The PR here is older than the
    high-water mark, so the issue walk cannot surface it.
    """
    fake.add_issue(1, updated_at="2026-03-01T00:00:00Z")
    pr = fake.add_pr(2, updated_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)
    assert _mark(engine) == dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)

    fake.add_review_comment(
        pr, 300, "this belongs in the parent class", created_at="2026-04-01T00:00:00Z"
    )
    result = _poll(engine, fake)

    assert result.documents_written == 1
    with engine.connect() as conn:
        found = conn.execute(
            select(s.documents.c.body_text).where(s.documents.c.source_type == "review_comment")
        ).scalar_one()
    assert found == "this belongs in the parent class"


def test_the_checkpoint_stops_at_the_first_failure(
    engine: Engine, fake: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this shape exists to prevent.

    Threads arrive ascending. If the mark advanced past a failed one -- because a *later*
    thread committed fine -- that thread would never be seen again, and the pass would
    report a single failure and otherwise look healthy.
    """
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    fake.add_issue(2, updated_at="2026-02-01T00:00:00Z")
    fake.add_issue(3, updated_at="2026-03-01T00:00:00Z")

    real = poll_mod.fetch_thread

    def flaky(client, repo, number):
        if number == 2:
            raise RuntimeError("boom")
        return real(client, repo, number)

    monkeypatch.setattr(poll_mod, "fetch_thread", flaky)
    result = _poll(engine, fake)

    assert result.failed == [2]
    assert not result.clean
    assert result.indexed == 2, "later threads are still indexed; only the mark waits"
    assert _mark(engine) == dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    # And the next pass re-covers the remainder rather than skipping it.
    monkeypatch.setattr(poll_mod, "fetch_thread", real)
    again = _poll(engine, fake)
    assert 2 in set(_indexed_numbers(engine))
    assert again.clean
    assert _mark(engine) == dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)


def test_a_pull_request_gone_from_pulls_does_not_hold_the_mark(
    engine: Engine, fake: FakeGitHub
) -> None:
    """huggingface/trl#7121: listed as a PR by ``/issues``, 404 under ``/pulls``.

    The 404 is permanent. Failing the thread on it held the mark there, and every 5-minute
    pass re-read the 184 threads behind it, about 900 requests a pass, until the token's
    hourly budget ran out. The thread is indexed from ``/issues`` instead.
    """
    gone = fake.add_pr(1, updated_at="2026-01-01T00:00:00Z")
    gone.pulls_gone = {"", "reviews"}
    fake.add_review_comment(gone, 300, "still served", created_at="2026-01-01T00:00:00Z")
    fake.add_issue(2, updated_at="2026-02-01T00:00:00Z")

    result = _poll(engine, fake)

    assert result.clean
    assert set(_indexed_numbers(engine)) == {1, 2}
    assert _mark(engine) == dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    with engine.connect() as conn:
        found = conn.execute(
            select(s.documents.c.body_text).where(s.documents.c.source_type == "review_comment")
        ).scalar_one()
    assert found == "still served"


def test_reviews_github_stops_serving_are_kept_not_pruned(engine: Engine, fake: FakeGitHub) -> None:
    """A 404 says the endpoint is gone, not that the reviews were deleted."""
    pr = fake.add_pr(1, updated_at="2026-01-01T00:00:00Z")
    fake.add_review(pr, 500, "please split this", submitted_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)

    pr.pulls_gone = {"", "reviews"}
    pr.payload["updated_at"] = "2026-02-01T00:00:00Z"
    result = _poll(engine, fake)

    assert result.clean
    with engine.connect() as conn:
        kept = conn.execute(
            select(s.documents.c.body_text).where(s.documents.c.source_type == "review")
        ).scalar_one()
    assert kept == "please split this"


def _always_failing(monkeypatch: pytest.MonkeyPatch, bad: int) -> list[int]:
    """Make thread ``bad`` fail every fetch; return the numbers every fetch was asked for."""
    real = poll_mod.fetch_thread
    asked: list[int] = []

    def fetch(client, repo, number):
        asked.append(number)
        if number == bad:
            raise RuntimeError("boom, every time")
        return real(client, repo, number)

    monkeypatch.setattr(poll_mod, "fetch_thread", fetch)
    return asked


def test_a_thread_that_always_fails_holds_the_mark_but_costs_only_itself(
    engine: Engine, fake: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2 holds the mark for good; the window behind it is re-listed, not re-fetched."""
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    fake.add_issue(2, updated_at="2026-02-01T00:00:00Z")
    fake.add_pr(3, updated_at="2026-03-01T00:00:00Z")
    fake.add_issue(4, updated_at="2026-04-01T00:00:00Z")
    asked = _always_failing(monkeypatch, 2)
    _poll(engine, fake)

    asked.clear()
    second = _poll(engine, fake)

    assert asked == [2], "only the failing thread is fetched again"
    assert second.failed == [2] and second.skipped == 3
    assert _mark(engine) == dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    fake.threads[4].payload["updated_at"] = "2026-05-01T00:00:00Z"
    asked.clear()
    _poll(engine, fake)
    assert asked == [2, 4], "a thread that moved behind the held mark is still re-read"


def test_a_review_comment_behind_a_held_mark_is_still_fetched(
    engine: Engine, fake: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review comment does not bump ``updated_at``, so "current" is not decided by it."""
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    pr = fake.add_pr(2, updated_at="2026-02-01T00:00:00Z")
    _always_failing(monkeypatch, 1)
    _poll(engine, fake)

    fake.add_review_comment(pr, 300, "rename this", created_at="2026-03-01T00:00:00Z")
    result = _poll(engine, fake)

    assert result.documents_written == 1
    with engine.connect() as conn:
        found = conn.execute(
            select(s.documents.c.body_text).where(s.documents.c.source_type == "review_comment")
        ).scalar_one()
    assert found == "rename this"


def test_a_thread_sharing_the_high_water_second_is_not_dropped(
    engine: Engine, fake: FakeGitHub
) -> None:
    """``updated_at`` has one-second granularity, so the window overlaps by 60s."""
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)

    # Same second as the mark: a strictly-after resume would miss it entirely.
    fake.add_issue(2, updated_at="2026-01-01T00:00:00Z")
    result = _poll(engine, fake)

    assert 2 in set(_indexed_numbers(engine))
    assert (result.indexed, result.skipped) == (1, 1), "the overlap re-lists #1; it is current"


def test_a_capped_walk_still_makes_progress(engine: Engine, fake: FakeGitHub) -> None:
    """GitHub refuses page 3; the pass indexes the prefix, bounds the mark, and the next
    pass resumes from there. Three passes, no thread lost, nothing silent."""
    fake.page_size = 1
    fake.page_cap = 2
    for n in range(1, 5):
        fake.add_issue(n, updated_at=f"2026-0{n}-01T00:00:00Z")

    # Each pass covers two pages, and the 60s overlap re-reads the boundary thread for
    # free -- so four threads at two per pass takes three passes, not two.
    first = _poll(engine, fake)
    assert first.capped and not first.clean
    assert _mark(engine) == dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)

    assert _poll(engine, fake).capped
    assert _mark(engine) == dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)

    third = _poll(engine, fake)
    assert third.capped is False, "the tail fits in one page, so the cap is not reached"
    assert third.clean
    assert _mark(engine) == dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
    assert set(_indexed_numbers(engine)) == {1, 2, 3, 4}


def test_a_capped_walk_does_not_let_a_newer_thread_drag_the_mark(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The two list queries are ordered differently.

    The uncapped review-comment walk can surface a thread far newer than anything the
    capped issue walk reached. Indexing it is fine; letting it move the checkpoint over
    the threads we never saw is how they vanish.
    """
    fake.page_size = 1
    fake.page_cap = 1
    for n in range(1, 4):
        fake.add_issue(n, updated_at=f"2026-0{n}-01T00:00:00Z")
    newer = fake.add_pr(9, updated_at="2026-12-01T00:00:00Z")
    fake.add_review_comment(newer, 300, "inline", created_at="2026-12-01T00:00:00Z")

    result = _poll(engine, fake)

    assert result.capped
    assert 9 in set(_indexed_numbers(engine)), "it is still indexed"
    assert _mark(engine) == dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), (
        "but the mark stays behind the threads the capped walk never reached"
    )


def test_a_clean_pass_records_last_ok(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_issue(1, updated_at="2026-01-01T00:00:00Z")
    _poll(engine, fake)
    with engine.connect() as conn:
        row = conn.execute(
            select(s.sync_state.c.last_run_at, s.sync_state.c.last_ok_at).where(
                s.sync_state.c.repo == REPO
            )
        ).one()
    assert row.last_run_at is not None
    assert row.last_ok_at is not None


def _indexed_numbers(engine: Engine) -> list[int]:
    with engine.connect() as conn:
        return [
            r.github_number
            for r in conn.execute(select(s.threads.c.github_number).order_by(s.threads.c.id))
        ]
