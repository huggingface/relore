"""Section 5.1: ``index_thread`` is the only mutation primitive, and it is idempotent.

Every test runs on both dialects (tests/conftest.py). The load-bearing one is
:func:`test_reindexing_an_unchanged_thread_writes_zero_documents` -- that is milestone 0's
done-when, and if it ever goes red the whole incremental story is gone.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine, func, select

from relore.ingest.chunk import MAX_CHUNK_CHARS
from relore.ingest.index_thread import derive_thread, index_thread
from relore.store import schema as s

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _index(engine: Engine, fake: FakeGitHub, number: int):
    with fake.client() as client:
        return index_thread(engine, client, REPO, number)


def _documents(engine: Engine) -> dict[tuple[str, str], str]:
    with engine.connect() as conn:
        return {
            (row.source_type, row.source_id): row.body_text
            for row in conn.execute(
                select(s.documents.c.source_type, s.documents.c.source_id, s.documents.c.body_text)
            )
        }


def test_indexing_creates_one_document_per_github_unit(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, title="Fix the mask", body="the body")
    fake.add_comment(pr, 100, "a conversation comment")
    fake.add_review(pr, 200, "please move this")
    fake.add_review_comment(pr, 300, "this belongs in the parent class")

    result = _index(engine, fake, 1)

    assert result.stats.inserted == 5  # title, body, comment, review, review comment
    assert set(_documents(engine)) == {
        ("title", "1"),
        ("body", "1"),
        ("issue_comment", "100"),
        ("review", "200"),
        ("review_comment", "300"),
    }


def test_reindexing_an_unchanged_thread_writes_zero_documents(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "hello")
    fake.add_review_comment(pr, 300, "inline")

    first = _index(engine, fake, 1)
    second = _index(engine, fake, 1)

    assert first.stats.inserted == 4
    assert second.stats.wrote == 0, "a no-change poll must not write a single document"
    assert second.stats.unchanged == 4


def test_an_edited_comment_updates_exactly_one_document(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1)
    comment = fake.add_comment(pr, 100, "original text")
    fake.add_comment(pr, 101, "untouched")
    _index(engine, fake, 1)

    comment["body"] = "edited text"
    result = _index(engine, fake, 1)

    assert (result.stats.inserted, result.stats.updated, result.stats.deleted) == (0, 1, 0)
    assert _documents(engine)[("issue_comment", "100")] == "edited text"


def test_a_deleted_comment_is_removed_from_the_index(engine: Engine, fake: FakeGitHub) -> None:
    """The absent/present row of section 5.1's table -- how a removed comment leaves."""
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "will be deleted")
    fake.add_comment(pr, 101, "stays")
    _index(engine, fake, 1)

    pr.issue_comments = [c for c in pr.issue_comments if c["id"] != 100]
    result = _index(engine, fake, 1)

    assert result.stats.deleted == 1
    assert ("issue_comment", "100") not in _documents(engine)
    assert ("issue_comment", "101") in _documents(engine)


# -- events (issue #22) ----------------------------------------------------


def _thread_metadata(engine: Engine) -> dict:
    with engine.connect() as conn:
        return conn.execute(select(s.threads.c.metadata)).scalar_one()


def test_a_closure_is_indexed_as_an_event(engine: Engine, fake: FakeGitHub) -> None:
    """`state: closed` is the same string for withdrawn-by-author and closed-as-duplicate,
    and a closure has no prose, so `--focus` can never recover the difference."""
    pr = fake.add_pr(1, state="closed", author="truongsontung")
    pr.payload["state_reason"] = "duplicate"
    pr.payload["closed_by"] = {"login": "ydshieh", "id": 9, "type": "User", "site_admin": False}
    fake.add_review(pr, 200, "", state="APPROVED", author="vasqu")

    _index(engine, fake, 1)

    meta = _thread_metadata(engine)
    assert meta["state_reason"] == "duplicate"
    assert meta["closed_by"] == "ydshieh"
    assert meta["review_decision"] == "approved"
    assert meta["review_decision_by"] == ["vasqu"]


def test_the_latest_review_per_reviewer_is_the_one_that_counts(
    engine: Engine, fake: FakeGitHub
) -> None:
    """One `changes_requested` outranks any number of approvals, and a reviewer who only
    commented has neither approved nor blocked."""
    pr = fake.add_pr(1)
    fake.add_review(
        pr, 200, "", state="APPROVED", author="vasqu", submitted_at="2026-01-02T00:00:00Z"
    )
    fake.add_review(pr, 201, "", state="APPROVED", author="ydshieh")
    fake.add_review(
        pr,
        202,
        "one more thing",
        state="CHANGES_REQUESTED",
        author="vasqu",
        submitted_at="2026-01-03T00:00:00Z",
    )
    fake.add_review(pr, 203, "a thought", state="COMMENTED", author="stevhliu")

    _index(engine, fake, 1)

    meta = _thread_metadata(engine)
    assert meta["review_decision"] == "changes_requested"
    assert meta["review_decision_by"] == ["vasqu"]


def test_a_thread_nobody_reviewed_carries_no_decision(engine: Engine, fake: FakeGitHub) -> None:
    """Absent, not null: a key nothing wrote says the pass never saw one."""
    fake.add_pr(1)
    _index(engine, fake, 1)
    assert "review_decision" not in _thread_metadata(engine)


# -- signals (section 5.3) -------------------------------------------------


def _signal_rows(engine: Engine, table, *, dates: bool = False) -> list[tuple]:
    """One thread's signal rows as tuples. ``first_seen_at`` is left out unless asked for:
    every row has one, and a test about *which rows were extracted* reads better without a
    timestamp repeated down the column. :func:`_signal_dates` is where it is the subject."""
    skip = {"thread_id"} if dates else {"thread_id", "first_seen_at"}
    with engine.connect() as conn:
        columns = [c for c in table.columns if c.name not in skip]
        return sorted(tuple(row) for row in conn.execute(select(*columns)))


def test_extraction_fills_the_signal_tables_from_the_thread_it_derived(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body='  File "src/mod.py", line 9, in forward\nValueError: bad 3 shapes')
    fake.add_comment(pr, 100, "also tests/test_mod.py::test_shapes fails")

    _index(engine, fake, 1)

    # Both paths came out of prose -- a traceback frame and a node id -- so both are
    # `mentioned`, which is what keeps them out of "what this pull request changed"
    # (huggingface/relore#17).
    assert _signal_rows(engine, s.thread_files) == [
        ("src/mod.py", None, "mentioned"),
        ("tests/test_mod.py", None, "mentioned"),
    ]
    assert ("forward", "src/mod.py", "frame") in _signal_rows(engine, s.thread_symbols)
    assert _signal_rows(engine, s.thread_errors) == [("ValueError", "bad <n> shapes")]
    assert ("tests/test_mod.py::test_shapes", "test_shapes") in _signal_rows(engine, s.thread_tests)


def test_a_signal_cut_by_a_chunk_boundary_is_still_extracted(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A pasted traceback is one paragraph with no blank line in it, so once it passes
    ``MAX_CHUNK_CHARS`` the chunker has no natural boundary and cuts on length -- through
    the middle of a line. Extracting from the chunks then sees ``...ValueE`` and
    ``rror: ...`` and finds neither, while the person pasting that traceback into
    ``--error`` has all of it. So extraction reads the unchunked text.

    Found on `huggingface/transformers` #45237, whose long traceback is chunked and whose
    stored errors did not include the one its own paste resolves to.
    """
    frame = '  File "src/mod.py", line 41, in forward\n'
    error = "ValueError: the shapes do not match at all"
    # Land the error line across the hard cut: the frames before it fill the first chunk
    # to within a few characters of the limit.
    filler = frame * ((MAX_CHUNK_CHARS - 20) // len(frame))
    body = filler + " " * (MAX_CHUNK_CHARS - len(filler) - 12) + error
    assert len(body) > MAX_CHUNK_CHARS
    fake.add_pr(1, body=body)

    _index(engine, fake, 1)

    with engine.connect() as conn:
        rows = conn.execute(
            select(s.documents.c.chunk_index, s.documents.c.body_text)
            .where(s.documents.c.source_type == "body")
            .order_by(s.documents.c.chunk_index)
        ).all()
    assert len(rows) > 1, "the fixture has to be chunked for this to test anything"
    assert not any(error in row.body_text for row in rows), "and cut mid-error-line"
    assert _signal_rows(engine, s.thread_errors) == [
        ("ValueError", "the shapes do not match at all")
    ]


def test_rederiving_an_unchanged_thread_writes_no_signal_rows_either(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The section 5.1 idempotency claim, extended to section 5.3's rows. Extraction is
    not exempt just because its rows are cheap: a poll that rewrote every signal table on
    every pass would make a quiet index a write-heavy one, and the assertion untestable."""
    pr = fake.add_pr(1, body="see src/mod.py -- ValueError: bad shapes")
    fake.add_comment(pr, 100, "and tests/test_mod.py::test_shapes")

    first = _index(engine, fake, 1)
    second = _index(engine, fake, 1)

    assert first.signals.wrote > 0
    assert second.signals.wrote == 0, "a no-change poll must not rewrite a signal table"
    assert second.stats.wrote == 0


def test_a_signal_carries_the_date_of_the_document_it_was_read_out_of(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 13's cutoff reads this column, so what fills it is not an implementation
    detail of extraction -- it is the evidence the bound rests on."""
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_comment(pr, 100, "it is src/mod.py", created_at="2026-01-15T00:00:00Z")
    fake.add_comment(pr, 101, "and utils/helper.py", created_at="2026-06-01T00:00:00Z")

    _index(engine, fake, 1)

    assert _signal_rows(engine, s.thread_files, dates=True) == [
        ("src/mod.py", None, "mentioned", _at("2026-01-15")),
        ("utils/helper.py", None, "mentioned", _at("2026-06-01")),
    ]


def test_an_edited_comment_dates_its_signals_from_the_edit(
    engine: Engine, fake: FakeGitHub
) -> None:
    """``documents.body_markdown`` holds the *current* text, so a path added by an edit
    dates from the edit and not from the writing -- the same rule ``written_before``
    applies to the document itself, applied to the row that outlives it."""
    pr = fake.add_pr(1, body="an opening with no path in it")
    fake.add_comment(
        pr,
        100,
        "it is src/mod.py after all",
        created_at="2026-01-15T00:00:00Z",
        updated_at="2026-06-01T00:00:00Z",
    )

    _index(engine, fake, 1)

    assert _signal_rows(engine, s.thread_files, dates=True) == [
        ("src/mod.py", None, "mentioned", _at("2026-06-01"))
    ]


def _at(when: str):
    return dt.datetime.fromisoformat(when).replace(tzinfo=dt.timezone.utc)


def test_editing_a_comment_moves_only_the_signals_it_changed(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="see src/mod.py")
    comment = fake.add_comment(pr, 100, "and utils/helper.py")
    _index(engine, fake, 1)

    comment["body"] = "and utils/other.py"
    result = _index(engine, fake, 1)

    assert [row[0] for row in _signal_rows(engine, s.thread_files)] == [
        "src/mod.py",
        "utils/other.py",
    ]
    assert set(result.signals.rewritten) == {"files"}, "only the table that moved"


def test_a_deleted_comment_takes_its_signals_with_it(engine: Engine, fake: FakeGitHub) -> None:
    """The signals are derived from the documents in the same transaction, so a thread
    can never keep a signal for a body it no longer has -- the ``--file`` filter would
    otherwise answer from it."""
    pr = fake.add_pr(1, body="the body")
    fake.add_comment(pr, 100, "KeyError: missing mask")
    _index(engine, fake, 1)
    assert _signal_rows(engine, s.thread_errors) == [("KeyError", "missing mask")]

    pr.issue_comments = []
    _index(engine, fake, 1)

    assert _signal_rows(engine, s.thread_errors) == []


def test_the_per_pr_pass_supplies_the_changed_files_and_the_commits(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 3: there is no repository-wide reviews endpoint and no cheap file list, so
    the GraphQL node is the only source of a change type or a commit."""
    from fake_github import FakeGraphQL

    from relore.github.graphql import fetch_pr_details
    from relore.store import repository as repo_layer
    from relore.store.dialect import utcnow

    pr = fake.add_pr(1, body="no paths in this body")
    pr.files = ["src/mod.py"]
    pr.commits = [("abc1234", "fix the mask")]
    _index(engine, fake, 1)

    graphql = FakeGraphQL(fake)
    with graphql.client() as client:
        details = fetch_pr_details(client, REPO, [1])
    with engine.begin() as conn:
        repo_layer.stage_raw(
            conn,
            REPO,
            [
                {
                    "repo": REPO,
                    "object_type": "pr_details",
                    "object_id": "1",
                    "thread_number": 1,
                    "payload": details[1],
                    "fetched_at": utcnow(),
                    "github_updated_at": None,
                }
            ],
        )
        derive_thread(conn, REPO, 1)

    # `changed`: the per-PR pass's diff, which is the only source that answers "did
    # this thread touch it?" (huggingface/relore#17).
    assert _signal_rows(engine, s.thread_files) == [("src/mod.py", "MODIFIED", "changed")]
    assert _signal_rows(engine, s.thread_commits) == [("abc1234", "fix the mask")]


def test_a_label_change_touches_no_document(engine: Engine, fake: FakeGitHub) -> None:
    """Idempotent by construction: one ``threads`` row rewritten, zero documents."""
    issue = fake.add_issue(1, labels=("bug",))
    _index(engine, fake, 1)

    issue.payload["labels"] = [{"name": "bug"}, {"name": "good first issue"}]
    result = _index(engine, fake, 1)

    assert result.stats.wrote == 0
    with engine.connect() as conn:
        labels = conn.execute(select(s.thread_labels.c.label).order_by(s.thread_labels.c.label))
        assert [r.label for r in labels] == ["bug", "good first issue"]


def test_an_inline_review_comment_keeps_its_file_line_and_commit(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The file and line are what the code lens later turns into an enclosing symbol."""
    pr = fake.add_pr(1)
    fake.add_review_comment(pr, 300, "why is this here", path="src/models/foo.py", line=412)
    _index(engine, fake, 1)

    with engine.connect() as conn:
        row = conn.execute(
            select(s.documents.c.metadata, s.documents.c.commit_sha).where(
                s.documents.c.source_type == "review_comment"
            )
        ).one()
    assert row.metadata["path"] == "src/models/foo.py"
    assert row.metadata["line"] == 412
    assert row.metadata["diff_hunk"]
    assert row.commit_sha == "deadbeef"


def test_an_inline_review_comment_is_never_split(engine: Engine, fake: FakeGitHub) -> None:
    """One complete thought tied to one file and line; splitting destroys it."""
    pr = fake.add_pr(1)
    fake.add_review_comment(pr, 300, "x " * MAX_CHUNK_CHARS)
    _index(engine, fake, 1)

    with engine.connect() as conn:
        count = conn.execute(
            select(func.count())
            .select_from(s.documents)
            .where(s.documents.c.source_type == "review_comment")
        ).scalar_one()
    assert count == 1


def test_a_long_body_is_chunked_but_the_title_is_not(engine: Engine, fake: FakeGitHub) -> None:
    filler = "word " * 500
    body = "\n\n".join(f"## Part {i}\n\n{filler}" for i in range(6))
    fake.add_issue(1, title="t " * 5000, body=body)
    _index(engine, fake, 1)

    with engine.connect() as conn:
        by_type = dict(
            conn.execute(
                select(s.documents.c.source_type, func.count()).group_by(s.documents.c.source_type)
            ).all()
        )
    assert by_type["title"] == 1
    assert by_type["body"] > 1


@pytest.mark.parametrize(
    ("assoc", "bot", "expected"),
    [
        ("COLLABORATOR", False, "authoritative"),
        ("OWNER", False, "authoritative"),
        ("CONTRIBUTOR", False, "reported"),
        # Section 6.2's last resort: an unresolved MEMBER is a claim, not authority.
        # A maintainer demoted costs one lost result; a stranger promoted costs a patch.
        ("MEMBER", False, "reported"),
        ("CONTRIBUTOR", True, "machine"),
    ],
)
def test_trust_is_derived_from_the_payload(
    engine: Engine, fake: FakeGitHub, assoc: str, bot: bool, expected: str
) -> None:
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "a judgement", assoc=assoc, bot=bot)
    _index(engine, fake, 1)

    with engine.connect() as conn:
        row = conn.execute(
            select(s.documents.c.trust, s.documents.c.author_is_bot).where(
                s.documents.c.source_type == "issue_comment"
            )
        ).one()
    assert row.trust == expected
    assert row.author_is_bot is bot


def test_a_secret_is_redacted_in_both_stored_copies(engine: Engine, fake: FakeGitHub) -> None:
    """The markdown copy is what gets shown, so redaction cannot be text-only."""
    secret = "ghp_" + "z" * 36
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, f"here is my config\ntoken = {secret}\n")
    result = _index(engine, fake, 1)

    with engine.connect() as conn:
        row = conn.execute(
            select(s.documents.c.body_markdown, s.documents.c.body_text).where(
                s.documents.c.source_type == "issue_comment"
            )
        ).one()
    assert secret not in row.body_markdown
    assert secret not in row.body_text
    assert "[REDACTED:github-token]" in row.body_markdown
    assert result.redactions == {"github-token": 1}


@pytest.mark.parametrize("glue", ["\u200b", "\u202e", "\x00", "\u2028"])
def test_a_secret_split_by_an_invisible_byte_reaches_no_stored_copy(
    engine: Engine, fake: FakeGitHub, glue: str
) -> None:
    """``normalize`` drops zero-width characters on the way to ``body_text``, and serve
    drops the rest: a token ``redact`` missed would be reassembled out of both copies.

    U+2028 is the case the unit test above explains: no layer here treats it as invisible,
    and `normalize` removes it anyway, so it splits a secret past the scan and puts it back
    together in the column search reads."""
    secret = "ghp_" + "z" * 36
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, f"token = {secret[:4]}{glue}{secret[4:]}")
    result = _index(engine, fake, 1)

    with engine.connect() as conn:
        row = conn.execute(
            select(s.documents.c.body_markdown, s.documents.c.body_text).where(
                s.documents.c.source_type == "issue_comment"
            )
        ).one()
    assert secret not in row.body_markdown
    assert secret not in row.body_text
    assert "[REDACTED:github-token]" in row.body_markdown
    assert result.redactions == {"github-token": 1}


def test_derive_reruns_from_raw_objects_with_no_network(engine: Engine, fake: FakeGitHub) -> None:
    """Section 5.0's promise: re-deriving costs local CPU, never API budget.

    The client is closed before ``derive_thread`` runs, so a hidden fetch would raise
    rather than quietly succeed.
    """
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "a comment")
    fake.add_review_comment(pr, 300, "inline")
    _index(engine, fake, 1)
    requests_before = len(fake.requests)

    with engine.begin() as conn:
        again = derive_thread(conn, REPO, 1)

    assert again.stats.wrote == 0
    assert again.stats.unchanged == 4
    assert len(fake.requests) == requests_before, "derive must not touch the network"


def test_a_deleted_comment_is_pruned_from_raw_staging_too(engine: Engine, fake: FakeGitHub) -> None:
    """Staging is an upsert, so deletion has to be explicit.

    Without the prune the raw object survives and every later derive rebuilds its
    document -- section 5.1's delete row would never fire and nothing would look wrong.
    """
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "will be deleted")
    _index(engine, fake, 1)

    pr.issue_comments = []
    pr.payload["comments"] = 0
    result = _index(engine, fake, 1)

    assert result.raw_pruned == 1
    with engine.connect() as conn:
        staged = conn.execute(
            select(s.raw_objects.c.object_type, s.raw_objects.c.object_id).order_by(
                s.raw_objects.c.object_type
            )
        ).all()
    assert [(r.object_type, r.object_id) for r in staged] == [("pr", "1")]


def test_pruning_leaves_other_passes_staging_alone(engine: Engine, fake: FakeGitHub) -> None:
    """A poll knows nothing about ``pr_details``; the backfill's per-PR GraphQL pass stages
    it against the same thread number, and a poll must not delete it. The authoritative set
    is an allowlist for exactly this reason: a type a thread fetch does not name can never
    be pruned by one. Staged through the real fetch, so this is the production object and
    not a stand-in -- adding ``pr_details`` to a pruning path has to fail here."""
    from fake_github import FakeGraphQL

    from relore.github.graphql import fetch_pr_details
    from relore.store.dialect import utcnow
    from relore.store.repository import stage_raw

    pr = fake.add_pr(1)
    pr.files = ["src/mod.py"]
    _index(engine, fake, 1)

    graphql = FakeGraphQL(fake)
    with graphql.client() as client:
        details = fetch_pr_details(client, REPO, [1])
    with engine.begin() as conn:
        stage_raw(
            conn,
            REPO,
            [
                {
                    "repo": REPO,
                    "object_type": "pr_details",
                    "object_id": "1",
                    "thread_number": 1,
                    "payload": details[1],
                    "fetched_at": utcnow(),
                    "github_updated_at": None,
                }
            ],
        )

    _index(engine, fake, 1)

    with engine.connect() as conn:
        kinds = conn.execute(
            select(s.raw_objects.c.object_type).where(s.raw_objects.c.object_type == "pr_details")
        ).all()
    assert len(kinds) == 1


def test_a_configured_bot_account_is_the_machine_tier(
    engine: Engine, fake: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 6.2's bot-account list, which is deployment config because only the operator
    knows which of their ``User`` accounts are machines.

    ``HuggingFaceDocBuilderDev`` is the measured case: it presents as ``User`` and
    ``MEMBER``, so every other rule in section 6.2 makes its docs-build boilerplate
    admissible evidence. A deployment that indexes its own agent's comments and leaves this
    empty has reintroduced what section 11 refused.
    """
    monkeypatch.setenv("RELORE_BOT_ACCOUNTS", "HuggingFaceDocBuilderDev, someotherbot")
    pr = fake.add_pr(1)
    fake.add_comment(
        pr,
        100,
        "The docs for this PR live here.",
        author="HuggingFaceDocBuilderDev",
        assoc="MEMBER",
    )
    fake.add_comment(pr, 101, "a real maintainer comment", author="carol", assoc="COLLABORATOR")
    _index(engine, fake, 1)

    with engine.connect() as conn:
        tiers = {
            row.author: (row.trust, row.author_is_bot)
            for row in conn.execute(
                select(
                    s.documents.c.author, s.documents.c.trust, s.documents.c.author_is_bot
                ).where(s.documents.c.source_type == "issue_comment")
            )
        }

    assert tiers["HuggingFaceDocBuilderDev"] == ("machine", True)
    assert tiers["carol"] == ("authoritative", False)


def test_the_bot_list_is_case_insensitive_and_empty_by_default(
    engine: Engine, fake: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub logins are case-insensitive, and an operator will not match our casing.

    Empty is legal: it is the default, and it is why section 15.2 lists configuring this as
    a deployment step rather than a nicety.
    """
    monkeypatch.setenv("RELORE_BOT_ACCOUNTS", "huggingfacedocbuilderdev")
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "boilerplate", author="HuggingFaceDocBuilderDev", assoc="MEMBER")
    _index(engine, fake, 1)

    with engine.connect() as conn:
        trust = conn.execute(
            select(s.documents.c.trust).where(s.documents.c.source_type == "issue_comment")
        ).scalar_one()
    assert trust == "machine"

    monkeypatch.delenv("RELORE_BOT_ACCOUNTS")
    _index(engine, fake, 1)  # re-derive with no list configured
    with engine.connect() as conn:
        trust = conn.execute(
            select(s.documents.c.trust).where(s.documents.c.source_type == "issue_comment")
        ).scalar_one()
    assert trust == "reported"


# -- a full re-derive ------------------------------------------------------


def test_a_number_with_comments_but_no_thread_does_not_stop_a_full_rederive(
    engine: Engine, fake: FakeGitHub, monkeypatch, capsys
) -> None:
    """A comment walk stages a comment under its thread number, and the thread can be
    deleted or transferred upstream before the thread walk sees it — so a number can hold
    comments and no issue. Measured in production: 20 of the 47,928 numbers staged for
    `huggingface/transformers`, one of them 40 threads into the list.

    `derive` keeps no cursor, so treating every staged number as derivable made the Job
    die at the same number on every restart: a pass that restarts for ever and never
    progresses, which is exactly what section 5.2 says to assert against.
    """
    from relore import daemon
    from relore.ingest.timestamps import parse_timestamp
    from relore.store import repository as repo_layer
    from relore.store.dialect import utcnow

    fake.add_issue(1, body="a real thread")
    _index(engine, fake, 1)
    with engine.begin() as conn:
        repo_layer.stage_raw(
            conn,
            REPO,
            [
                {
                    "repo": REPO,
                    "object_type": "issue_comment",
                    "object_id": "9001",
                    "thread_number": 1998,  # nothing staged the issue itself
                    "payload": {"id": 9001, "body": "orphaned by a transfer upstream"},
                    "fetched_at": utcnow(),
                    "github_updated_at": parse_timestamp("2026-01-01T00:00:00Z"),
                }
            ],
        )

    monkeypatch.setattr(daemon, "_engine", lambda: engine)
    assert daemon.main(["derive", "--repo", REPO]) == 0

    out = capsys.readouterr().out
    assert "re-derived 1 threads" in out
    # Reported, not silently dropped (build plan section 13.3 #10).
    assert "skipped 1 numbers" in out and "relored sweep" in out
