"""The pre-cutoff corpus export, and the post-filter for the leak `_filters` cannot close.

Both halves ask the index one question -- what did this thread have by ``T`` -- so the
tests that matter most are the ones where the two answers have to agree with each other and
with the SQL the search layer runs.

The failure class is one-directional, as in ``test_temporal_cutoff``: exporting or keeping
one post-cutoff document is broken, dropping a legitimate pre-cutoff one only costs recall.
Where the post-filter is deliberately stricter than the index, the test says so.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from fake_github import FakeGitHub, FakeGraphQL
from sqlalchemy import Engine, select, update

from relore.bench.corpus import EXPORT_VERSION, SignalPostFilter, export, signals_at
from relore.github.graphql import fetch_pr_details
from relore.ingest.index_thread import derive_thread, index_thread
from relore.search import SearchQuery, open_backend
from relore.store import repository as repo_layer
from relore.store import schema as s
from relore.store.dialect import utcnow

REPO = "owner/name"

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


def _with_details(engine: Engine, fake: FakeGitHub, number: int) -> None:
    """Run the per-PR pass, which is the only source of a changed-file list (section 3)."""
    graphql = FakeGraphQL(fake)
    with graphql.client() as client:
        details = fetch_pr_details(client, REPO, [number])
    with engine.begin() as conn:
        repo_layer.stage_raw(
            conn,
            REPO,
            [
                {
                    "repo": REPO,
                    "object_type": "pr_details",
                    "object_id": str(number),
                    "thread_number": number,
                    "payload": details[number],
                    "fetched_at": utcnow(),
                    "github_updated_at": None,
                }
            ],
        )
        derive_thread(conn, REPO, number)


def _query(**kwargs) -> SearchQuery:
    kwargs.setdefault("repos", (REPO,))
    return SearchQuery(**kwargs)


def _search(engine: Engine, **kwargs) -> list:
    return open_backend(engine).search(_query(**kwargs))


def _exported(engine: Engine, tmp_path: Path, **kwargs):
    """``(header, rows)`` of one export, read back off disk: the file is the deliverable."""
    kwargs.setdefault("repos", (REPO,))
    kwargs.setdefault("before", CUTOFF)
    out = tmp_path / "corpus.jsonl"
    export(engine, out, **kwargs)
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return lines[0]["corpus"], lines[1:]


def _identity(row: dict) -> tuple:
    return (
        row["repo"],
        row["number"],
        row["source_type"],
        row["source_id"],
        row["chunk_index"],
    )


# -- the export ------------------------------------------------------------


def test_a_document_written_after_the_cutoff_is_not_exported(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 100, "settled early", created_at=BEFORE)
    fake.add_comment(pr, 101, "revisited after the task", created_at=AFTER)
    _index(engine, fake, 1)

    _header, rows = _exported(engine, tmp_path)

    texts = [row["text"] for row in rows]
    assert "settled early" in texts
    assert "revisited after the task" not in texts


def test_an_undated_document_is_excluded_rather_than_exported(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """Fail-closed, as the query layer is: an unknown date cannot be shown to predate the
    cutoff."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 100, "a comment with no date the index knows", created_at=BEFORE)
    _index(engine, fake, 1)
    with engine.begin() as conn:
        conn.execute(
            update(s.documents)
            .where(s.documents.c.source_id == "100")
            .values(github_created_at=None)
        )

    _header, rows = _exported(engine, tmp_path)

    assert "100" not in {row["source_id"] for row in rows}


def test_an_excluded_thread_contributes_nothing(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """A thread that predates the cutoff and is still the answer has to be withholdable by
    number, or the baselines are handed it too."""
    fake.add_pr(1, body="the ordinary history")
    fake.add_pr(2, body="the pull request that fixed the task")
    _index(engine, fake, 1, 2)

    _header, rows = _exported(engine, tmp_path, exclude=(2,))

    assert {row["number"] for row in rows} == {1}


def test_the_header_carries_the_cutoff_and_a_digest_of_the_body(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """A corpus file that does not say what cutoff produced it gets quoted anyway."""
    import hashlib

    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 100, "settled early", created_at=BEFORE)
    _index(engine, fake, 1)

    out = tmp_path / "corpus.jsonl"
    stats = export(engine, out, repos=(REPO,), before=CUTOFF, exclude=(7,))
    lines = out.read_text().splitlines()
    header = json.loads(lines[0])["corpus"]

    assert header["version"] == EXPORT_VERSION
    assert header["before"] == CUTOFF.isoformat()
    assert header["exclude"] == [7]
    assert header["documents"] == stats.documents == len(lines) - 1
    assert header["known_limits"]
    body = "".join(line + "\n" for line in lines[1:]).encode()
    assert header["digest"] == stats.digest == hashlib.sha256(body).hexdigest()


def test_two_exports_of_one_corpus_have_the_same_digest(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """The digest is over the documents, not the header: the header carries a timestamp,
    and a corpus that hashes differently on every write compares to nothing."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 100, "settled early", created_at=BEFORE)
    _index(engine, fake, 1)

    first = export(engine, tmp_path / "a.jsonl", repos=(REPO,), before=CUTOFF)
    second = export(engine, tmp_path / "b.jsonl", repos=(REPO,), before=CUTOFF)

    assert first.digest == second.digest
    bodies = [(tmp_path / name).read_text().split("\n", 1)[1] for name in ("a.jsonl", "b.jsonl")]
    assert bodies[0] == bodies[1]


def test_the_export_holds_every_document_the_index_can_still_return(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """Why the predicate is shared rather than copied: a baseline scored over a corpus the
    index could answer from and the file could not is measuring the export."""
    pr = fake.add_pr(1, body="the marker in the opening")
    fake.add_comment(pr, 100, "the marker, settled early", created_at=BEFORE)
    fake.add_comment(pr, 101, "the marker, revisited later", created_at=AFTER)
    _index(engine, fake, 1)

    _header, rows = _exported(engine, tmp_path)
    exported = {_identity(row) for row in rows}
    hits = _search(engine, text="marker", before=CUTOFF)

    assert hits
    for hit in hits:
        assert (REPO, hit.number, hit.source_type, hit.source_id, hit.chunk_index) in exported


def test_the_export_is_ordered_by_identity_and_not_by_insertion(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """``documents.id`` is reassigned by a rebuild, so the file is ordered by the identity
    a human can name instead."""
    fake.add_pr(2, body="the later number, indexed first")
    fake.add_pr(1, body="the earlier number, indexed second")
    _index(engine, fake, 2, 1)

    _header, rows = _exported(engine, tmp_path)

    assert [row["number"] for row in rows] == sorted(row["number"] for row in rows)


def test_an_export_with_no_cutoff_is_written_and_says_so(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    """The control is a corpus too: same verb, and not mistakable for a pre-cutoff one."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_comment(pr, 101, "written after any plausible cutoff", created_at=AFTER)
    _index(engine, fake, 1)

    header, rows = _exported(engine, tmp_path, before=None)

    assert header["before"] is None
    assert "written after any plausible cutoff" in {row["text"] for row in rows}


def test_a_naive_cutoff_is_refused_at_the_boundary(engine: Engine, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export(engine, tmp_path / "c.jsonl", repos=(REPO,), before=dt.datetime(2026, 3, 1))


# -- the post-filter -------------------------------------------------------


def test_the_post_filter_drops_a_thread_selected_by_evidence_that_did_not_exist_yet(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The measured leak, and the reason this module exists.

    ``thread_files`` carries ``src/mod.py`` with no date, so the ``EXISTS`` matches and the
    thread's pre-cutoff opening comes back: every document on the page is genuinely
    pre-cutoff, and the leak is the *selection*. The strict xfail in
    ``test_temporal_cutoff`` asserts the same scenario against the query layer, where it is
    still open.
    """
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    query = _query(files=("src/mod.py",), before=CUTOFF)
    hits = open_backend(engine).search(query)
    assert hits, "the leak itself: the index still selects this thread"

    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == []


def test_a_path_the_thread_already_named_survives_the_post_filter(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The other half of the contract: a post-filter that empties every page measures
    nothing."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_review_comment(pr, 300, "the mask is wrong here", path="src/mod.py", created_at=BEFORE)
    _index(engine, fake, 1)

    query = _query(files=("src/mod.py",), before=CUTOFF)
    hits = open_backend(engine).search(query)

    assert hits
    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == hits


def test_a_symbol_first_named_after_the_cutoff_drops_the_thread(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="an opening with no symbol in it")
    fake.add_comment(pr, 100, "the cause is `rotate_half()`", created_at=AFTER)
    _index(engine, fake, 1)

    query = _query(symbols=("rotate_half",), before=CUTOFF)
    hits = open_backend(engine).search(query)
    assert hits

    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == []


def test_an_error_first_pasted_after_the_cutoff_drops_the_thread(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="an opening with no traceback in it")
    fake.add_comment(
        pr,
        100,
        "```\nTraceback (most recent call last):\nValueError: shapes 3 and 4 not aligned\n```",
        created_at=AFTER,
    )
    _index(engine, fake, 1)

    query = _query(errors=("ValueError: shapes 3 and 4 not aligned",), before=CUTOFF)
    hits = open_backend(engine).search(query)
    assert hits

    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == []


def test_a_query_with_no_signal_filter_is_untouched(engine: Engine, fake: FakeGitHub) -> None:
    """A selection leak has nothing to have selected: a text query is admitted by the
    documents themselves, which the cutoff already bounded."""
    pr = fake.add_pr(1, body="the marker in the opening")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    query = _query(text="marker", before=CUTOFF)
    hits = open_backend(engine).search(query)

    assert hits
    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == hits


def test_an_unpinned_post_filter_is_the_identity(engine: Engine, fake: FakeGitHub) -> None:
    """A control and a treatment come off the same code path, or the difference between
    them includes the code path. Same argument as ``AsOf()``."""
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    query = _query(files=("src/mod.py",))
    hits = open_backend(engine).search(query)

    assert hits
    assert SignalPostFilter(engine, None).keep(hits, query) == hits


def test_the_post_filter_reports_what_it_dropped(engine: Engine, fake: FakeGitHub) -> None:
    """The leak has to be sized, not assumed: a recall that moved does not say how far,
    and on a run where nothing drops it says nothing."""
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "this belongs upstream", path="src/mod.py", created_at=AFTER)
    _index(engine, fake, 1)

    query = _query(files=("src/mod.py",), before=CUTOFF)
    hits = open_backend(engine).search(query)

    assert SignalPostFilter(engine, CUTOFF).dropped(hits, query) == hits


# -- the diff, which no document carries -----------------------------------


def test_a_pull_request_merged_before_the_cutoff_keeps_its_changed_files(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The changed-file list comes from the per-PR node, not a document, so it has no date
    of its own. A pull request merged before ``T`` had *this* list at ``T``."""
    pr = fake.add_pr(1, body="an opening with no path in it", merged_at=BEFORE)
    pr.files = ["src/mod.py"]
    _index(engine, fake, 1)
    _with_details(engine, fake, 1)

    query = _query(files=("src/mod.py",), before=CUTOFF)
    hits = open_backend(engine).search(query)

    assert hits
    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == hits


def test_a_pull_request_merged_after_the_cutoff_contributes_none_of_its_diff(
    engine: Engine, fake: FakeGitHub
) -> None:
    """It had some prefix of this list and nothing here knows which, so it contributes
    none of it -- costing recall rather than admitting a diff assembled after the fact."""
    pr = fake.add_pr(1, body="an opening with no path in it", merged_at=AFTER)
    pr.files = ["src/mod.py"]
    _index(engine, fake, 1)
    _with_details(engine, fake, 1)

    query = _query(files=("src/mod.py",), before=CUTOFF)
    hits = open_backend(engine).search(query)
    assert hits

    assert SignalPostFilter(engine, CUTOFF).keep(hits, query) == []


# -- the two implementations of one predicate ------------------------------
#
# `satisfies` is a second implementation of filters whose first implementation is SQL --
# section 13.2 #3's hazard. These are the cases that would tell the two apart.


@pytest.mark.parametrize(
    "stored, asked, matches",
    [
        ("src/transformers/modeling_x.py", "src/transformers/modeling_x.py", True),
        ("src/transformers/modeling_x.py", "modeling_x.py", True),
        ("src/transformers/modeling_x.py", "transformers/modeling_x.py", True),
        # Anchored at `/`: why the SQL pattern is `%/needle`, not `%needle`.
        ("src/transformers/not_modeling_x.py", "modeling_x.py", False),
        # `_` is a LIKE wildcard the SQL side escapes; Python compares literally. This is
        # where the two could silently disagree.
        ("src/transformers/modeling_x.py", "modeling/x.py", False),
        ("src/mod.py", "./src/mod.py", True),
    ],
)
def test_the_python_path_match_agrees_with_the_sql_one(
    engine: Engine, fake: FakeGitHub, stored: str, asked: str, matches: bool
) -> None:
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_review_comment(pr, 300, "a remark", path=stored, created_at=BEFORE)
    _index(engine, fake, 1)

    query = _query(files=(asked,), before=CUTOFF)
    sql_matched = bool(open_backend(engine).search(query))
    with engine.connect() as conn:
        thread_id = conn.execute(select(s.threads.c.id)).scalar_one()
        python_matched = signals_at(conn, thread_id, CUTOFF).satisfies(query)

    assert sql_matched is matches
    assert python_matched is matches


def test_signals_at_reads_only_what_the_thread_had_said(engine: Engine, fake: FakeGitHub) -> None:
    """The stored signal rows are the timeless set this exists to correct, so it must not
    consult them."""
    pr = fake.add_pr(1, body="an opening")
    fake.add_review_comment(pr, 300, "early", path="src/early.py", created_at=BEFORE)
    fake.add_review_comment(pr, 301, "late", path="src/late.py", created_at=AFTER)
    _index(engine, fake, 1)

    with engine.connect() as conn:
        thread_id = conn.execute(select(s.threads.c.id)).scalar_one()
        stored = {row.path for row in conn.execute(select(s.thread_files.c.path))}
        at_cutoff = signals_at(conn, thread_id, CUTOFF)

    assert stored == {"src/early.py", "src/late.py"}
    assert at_cutoff.files == frozenset({"src/early.py"})
