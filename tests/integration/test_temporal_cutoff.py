"""The cutoff: what the index held at a past instant, on both dialects (section 12).

An evaluation that asks a retrieval system a question about a bug, and can retrieve the
comment written *after* that bug was fixed, measures nothing. Every test here names a way
the answer could reach the caller anyway.

The failure class is one-directional and worth stating: a cutoff that lets one future
document through is broken, and a cutoff that drops a legitimate pre-cutoff document only
costs recall. So every ambiguity here resolves towards exclusion -- see the NULL date.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine, update

from relore.ingest.index_thread import index_thread
from relore.search import SearchQuery, open_backend, search_expanded
from relore.search.queries import QueryError
from relore.store import schema as s

REPO = "owner/name"

#: The task cutoff every test writes against: documents dated 2026-01 are history, 2026-06
#: is the future the agent must not see.
CUTOFF = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)

BEFORE = "2026-01-15T00:00:00Z"
AFTER = "2026-06-01T00:00:00Z"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _index(engine: Engine, fake: FakeGitHub, *numbers: int) -> None:
    with fake.client() as client:
        for number in numbers:
            index_thread(engine, client, fake.repo, number)


def _query(**kwargs) -> SearchQuery:
    kwargs.setdefault("repos", (REPO,))
    return SearchQuery(**kwargs)


def _search(engine: Engine, **kwargs) -> list:
    return open_backend(engine).search(_query(**kwargs))


def _expanded(engine: Engine, **kwargs) -> list:
    return search_expanded(open_backend(engine), _query(**kwargs))


# -- the cutoff itself -----------------------------------------------------


def test_a_document_written_after_the_cutoff_is_unreachable(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="the mask marker as first reported")
    fake.add_comment(pr, 100, "the mask marker, explained after the fact", created_at=AFTER)
    _index(engine, fake, 1)

    assert {h.source_type for h in _search(engine, text="mask marker")} == {
        "body",
        "issue_comment",
    }
    assert {h.source_type for h in _search(engine, text="mask marker", before=CUTOFF)} == {"body"}


def test_the_cutoff_is_exclusive_at_its_own_instant(engine: Engine, fake: FakeGitHub) -> None:
    """``created_at < T``, not ``<=``. The brief's rule is written as a strict inequality
    and a benchmark whose cutoff is the moment the task became actionable must not return
    what was said in that same instant."""
    pr = fake.add_pr(1, body="unrelated opening")
    fake.add_comment(pr, 100, "the boundary marker", created_at="2026-03-01T00:00:00Z")
    _index(engine, fake, 1)

    assert _search(engine, text="boundary marker") != []
    assert _search(engine, text="boundary marker", before=CUTOFF) == []


def test_a_document_with_no_date_is_excluded_rather_than_admitted(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Fail closed. A document whose creation date this index does not know cannot be
    *shown* to predate the cutoff, and a leakage rule that admits what it cannot verify is
    not a rule. This is deliberately the opposite of ``_decay``, where the same NULL means
    "missing evidence, do not cost it rank"."""
    pr = fake.add_pr(1, body="unrelated opening")
    fake.add_comment(pr, 100, "the undated marker", created_at=BEFORE)
    _index(engine, fake, 1)
    with engine.begin() as conn:
        conn.execute(
            update(s.documents)
            .where(s.documents.c.source_type == "issue_comment")
            .values(github_created_at=None)
        )

    assert _search(engine, text="undated marker") != []
    assert _search(engine, text="undated marker", before=CUTOFF) == []


def test_the_cutoff_holds_through_every_expansion_leg(engine: Engine, fake: FakeGitHub) -> None:
    """Section 6's expansion fans one call into up to six queries. The cutoff has to ride
    on all of them, and it does because a leg is a ``replace()`` of the caller's query --
    this is the test that keeps that true if a leg is ever built some other way.

    What is asserted is that the future *document* is gone from every leg. The thread
    itself is still reachable through its pre-cutoff body, which is the separate leak
    :func:`test_a_thread_is_still_selected_by_evidence_that_did_not_exist_yet` covers.
    """
    pr = fake.add_pr(1, body="a report with nothing quotable in it")
    fake.add_comment(
        pr,
        100,
        "ValueError: expected 768 features, got 1024\n"
        "`compute_default_rope_parameters` in `src/mod.py`\n"
        "tests/test_mask.py::test_shapes fails",
        created_at=AFTER,
    )
    _index(engine, fake, 1)
    question = (
        "ValueError: expected 768 features, got 1024 compute_default_rope_parameters "
        "src/mod.py tests/test_mask.py::test_shapes"
    )

    assert any(h.source_type == "issue_comment" for h in _expanded(engine, text=question))
    hits = _expanded(engine, text=question, before=CUTOFF)
    assert hits != []  # the legs still run; it is the future document they must not return
    assert all(h.created_at < CUTOFF for h in hits)


def test_a_filter_cannot_return_a_future_document(engine: Engine, fake: FakeGitHub) -> None:
    """A filter match does not re-admit a row the cutoff dropped: the predicate that
    excludes the document is on ``documents`` and the filter is an ``EXISTS`` beside it."""
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    assert any(h.source_type == "review_comment" for h in _search(engine, files=("src/mod.py",)))
    assert "review_comment" not in {
        h.source_type for h in _search(engine, files=("src/mod.py",), before=CUTOFF)
    }


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the five thread_* signal tables carry no timestamp, so a path or symbol first "
        "named after the cutoff is attributed to the whole thread and selects that "
        "thread's pre-cutoff documents. No future *text* reaches the caller -- the leak "
        "is in the selection, not the payload -- but a real as-of-T deployment could not "
        "have made this match, so it inflates any condition that filters structurally. "
        "Closed by a `first_seen_at` column on those tables (temporal-cutoff-map.md 8); "
        "delete this marker when it lands."
    ),
)
def test_a_thread_is_still_selected_by_evidence_that_did_not_exist_yet(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    assert _search(engine, files=("src/mod.py",), before=CUTOFF) == []


def test_history_before_the_cutoff_still_comes_back(engine: Engine, fake: FakeGitHub) -> None:
    """The other half of the contract. A cutoff that empties the page is trivially
    leak-free and useless; the experiment needs what *was* said to remain reachable."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 100, "the rationale marker, settled early", created_at=BEFORE)
    fake.add_comment(pr, 101, "the rationale marker, revisited later", created_at=AFTER)
    _index(engine, fake, 1)

    hits = _search(engine, text="rationale marker", before=CUTOFF)

    assert [h.snippet for h in hits] == ["the rationale marker, settled early"]


def test_since_and_before_compose_into_a_window(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, body="the window marker, from the opening")
    fake.add_comment(pr, 100, "the window marker, mid-window", created_at="2026-02-01T00:00:00Z")
    fake.add_comment(pr, 101, "the window marker, after", created_at=AFTER)
    _index(engine, fake, 1)

    hits = _search(
        engine,
        text="window marker",
        since=dt.datetime(2026, 1, 20, tzinfo=dt.timezone.utc),
        before=CUTOFF,
    )

    assert [h.snippet for h in hits] == ["the window marker, mid-window"]


# -- refused at the boundary -----------------------------------------------


def test_a_naive_before_is_refused_at_the_boundary() -> None:
    with pytest.raises(QueryError):
        _query(before=dt.datetime(2026, 3, 1))


def test_an_empty_window_is_refused_rather_than_answered_with_silence() -> None:
    """A benchmark whose cutoff is misconfigured must not read as a corpus with nothing to
    say -- that is a baseline scoring zero for a reason nobody would look for."""
    with pytest.raises(QueryError):
        _query(
            since=CUTOFF,
            before=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        )


def test_a_window_of_one_instant_is_refused() -> None:
    with pytest.raises(QueryError):
        _query(since=CUTOFF, before=CUTOFF)


# -- the verbs that answer by number ---------------------------------------
#
# `search` can be made leak-free and the corpus still hand the answer over to anyone who
# asks for it by number. These three are that hole.


def _opened_at(thread, when: str):
    """The fixture pins every thread's creation to 2025; these tests need it moved."""
    thread.payload["created_at"] = when
    return thread


def test_a_thread_serves_only_what_it_had_said_by_the_cutoff(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="the opening, as reported")
    fake.add_comment(pr, 100, "an early observation", created_at=BEFORE)
    fake.add_comment(pr, 101, "the fix, explained afterwards", created_at=AFTER)
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1, before=CUTOFF)

    assert [c.snippet for c in view.comments] == ["an early observation"]
    assert view.total_documents == 1


def test_a_thread_opened_after_the_cutoff_is_not_found_rather_than_empty(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The two are different answers. An empty thread says "this exists and said nothing",
    which is a fact about a thread that did not exist."""
    _opened_at(fake.add_pr(1, body="the fix nobody had proposed yet"), AFTER)
    _index(engine, fake, 1)

    assert open_backend(engine).thread(REPO, 1) is not None
    assert open_backend(engine).thread(REPO, 1, before=CUTOFF) is None


def test_an_excluded_thread_is_not_found_by_number(engine: Engine, fake: FakeGitHub) -> None:
    """The cutoff cannot express this one: a benchmark's own task issue predates its
    cutoff and is still the answer."""
    fake.add_pr(1, body="the task's own thread")
    _index(engine, fake, 1)

    assert open_backend(engine).thread(REPO, 1, exclude=(1,)) is None


def test_an_excluded_thread_is_absent_from_search(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_pr(1, body="the excluded marker")
    fake.add_pr(2, body="the excluded marker, elsewhere")
    _index(engine, fake, 1, 2)

    assert {h.number for h in _search(engine, text="excluded marker")} == {1, 2}
    assert {h.number for h in _search(engine, text="excluded marker", exclude=(1,))} == {2}


def test_inflight_does_not_name_a_pull_request_that_did_not_exist_yet(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The sharpest leak in the codebase, and the reason this step exists.

    ``inflight`` walks the ``closes`` edge backwards with no trust floor and no filter --
    correctly, in production, because the deployment's own bot having an open fix is
    exactly the duplicate an agent must not create. Asked for a benchmark task's issue
    number it answers with the pull request that fixed it, which is the gold patch by
    number.
    """
    fake.add_issue(1, title="crashes for any rotary_pct != 1.0")
    _opened_at(fake.add_pr(2, title="fix: respect partial_rotary_factor", body="Fixes #1"), AFTER)
    _index(engine, fake, 1, 2)

    assert [c.number for c in open_backend(engine).inflight(REPO, 1).claims] == [2]
    assert open_backend(engine).inflight(REPO, 1, before=CUTOFF).claims == ()


def test_inflight_does_not_name_an_excluded_pull_request(engine: Engine, fake: FakeGitHub) -> None:
    """The gold pull request is usually *older* than the cutoff -- it is the thread the
    task was drawn from. Only the exclusion list reaches it."""
    fake.add_issue(1, title="crashes for any rotary_pct != 1.0")
    fake.add_pr(2, title="fix: respect partial_rotary_factor", body="Fixes #1")
    _index(engine, fake, 1, 2)

    assert [c.number for c in open_backend(engine).inflight(REPO, 1).claims] == [2]
    assert open_backend(engine).inflight(REPO, 1, exclude=(2,)).claims == ()


def test_a_claim_that_predates_the_cutoff_still_answers(engine: Engine, fake: FakeGitHub) -> None:
    """The other half: work genuinely in flight at `T` is exactly what the verb is for."""
    fake.add_issue(1, title="crashes for any rotary_pct != 1.0")
    fake.add_pr(2, title="an attempt already open at the cutoff", body="Fixes #1")
    _index(engine, fake, 1, 2)

    assert [c.number for c in open_backend(engine).inflight(REPO, 1, before=CUTOFF).claims] == [2]
