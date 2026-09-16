"""Retrieval on a small hand-built corpus, on both dialects (section 12).

Section 4.1 is emphatic that the search layer does **not** port: ``ts_rank_cd`` is not
``bm25``, and a score from one says nothing about the other. So nothing here asserts a
score or an order that depends on one -- what is asserted on both is the *contract*: what
comes back, what never comes back, and how much of it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_github import FakeGitHub, FakeGraphQL
from sqlalchemy import Engine

from relore.ingest.backfill import backfill
from relore.ingest.index_thread import index_thread
from relore.search import SearchQuery, open_backend
from relore.search.queries import (
    BODY_SLACK_CHARS,
    MAX_BODY_CHARS,
    MAX_HITS,
    MAX_HITS_PER_THREAD,
    MAX_SNIPPET_CHARS,
    MAX_THREAD_COMMENTS,
    QueryError,
)

REPO = "owner/name"
OTHER = "owner/other"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _index(engine: Engine, fake: FakeGitHub, *numbers: int) -> None:
    with fake.client() as client:
        for number in numbers:
            index_thread(engine, client, fake.repo, number)


def _backfill(engine: Engine, fake: FakeGitHub) -> None:
    """The path that stages a merged PR's changed-file list: it comes from section 3's
    per-PR GraphQL pass and from nowhere else."""
    gql = FakeGraphQL(fake).client()
    with fake.client() as client:
        try:
            backfill(engine, client, fake.repo, gql=gql)
        finally:
            gql.close()


def _search(engine: Engine, **kwargs) -> list:
    kwargs.setdefault("repos", (REPO,))
    return open_backend(engine).search(SearchQuery(**kwargs))


# -- what comes back -------------------------------------------------------


def test_a_term_in_a_comment_is_findable(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, title="Fix the mask", body="nothing here")
    fake.add_review_comment(pr, 300, "this belongs in the parent class")
    _index(engine, fake, 1)

    hits = _search(engine, text="parent class")

    assert [(h.number, h.source_type) for h in hits] == [(1, "review_comment")]
    assert "parent class" in hits[0].snippet


def test_stemming_finds_an_inflected_form(engine: Engine, fake: FakeGitHub) -> None:
    """Both backends stem -- ``to_tsvector('english', ...)`` and FTS5's ``porter`` -- so a
    laptop is not surprising for a reason nobody would think to check (migration 3)."""
    fake.add_pr(1, body="the loader was loading twice")
    _index(engine, fake, 1)

    assert [h.number for h in _search(engine, text="load")] == [1]


def test_no_results_is_a_successful_empty_result(engine: Engine, fake: FakeGitHub) -> None:
    """Empty is not an error, always (AGENTS.md)."""
    fake.add_pr(1, body="something")
    _index(engine, fake, 1)

    assert _search(engine, text="nothing whatsoever matches this") == []


def test_a_pasted_traceback_searches_instead_of_erroring(engine: Engine, fake: FakeGitHub) -> None:
    """FTS5's ``MATCH`` takes a query language in which most of a traceback's punctuation
    means something, and an unbalanced quote is a hard error rather than an empty result.
    Postgres has the same trap in the raw ``to_tsquery``."""
    fake.add_pr(1, body='ValueError: expected "mask" got None (see mod.py:42)')
    _index(engine, fake, 1)

    hits = _search(engine, text='ValueError: expected "mask" got None (mod.py:42)')

    assert [h.number for h in hits] == [1]


def test_punctuation_only_query_returns_nothing_rather_than_everything(
    engine: Engine, fake: FakeGitHub
) -> None:
    fake.add_pr(1, body="a body")
    _index(engine, fake, 1)

    assert _search(engine, text="?!?  ...") == []


def test_a_filter_only_query_needs_no_text(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_pr(1, labels=("bug",))
    fake.add_pr(2, labels=("docs",))
    _index(engine, fake, 1, 2)

    assert {h.number for h in _search(engine, labels=("bug",))} == {1}


def test_the_file_filter_matches_a_path_named_in_prose(engine: Engine, fake: FakeGitHub) -> None:
    """Section 5.3's extraction is what makes the signal filters answer at all -- they
    were wired against empty tables at milestone 2, so this is the first test in which a
    filter both matches and excludes."""
    fake.add_pr(1, body="touches src/mod.py")
    fake.add_pr(2, body="touches something else entirely")
    _index(engine, fake, 1, 2)

    assert {h.number for h in _search(engine, files=("src/mod.py",))} == {1}
    assert [h.number for h in _search(engine, text="touches", files=("src/mod.py",))] == [1]
    assert _search(engine, files=("src/absent.py",)) == []


def test_the_file_filter_matches_a_basename_and_a_partial_path(
    engine: Engine, fake: FakeGitHub
) -> None:
    """One flag had three semantics: the full path hit the diff, a bare basename hit only
    prose mentions, and a partial path -- the natural thing to paste from a traceback --
    matched nothing and returned a clean empty page (huggingface/relore#48)."""
    pr = fake.add_pr(1, body="a fix", merged_at="2026-02-02T00:00:00Z")
    pr.files = ["src/transformers/models/gpt/modeling_gpt.py"]
    fake.add_pr(2, body="unrelated", merged_at="2026-02-02T00:00:00Z")
    _backfill(engine, fake)

    for form in (
        "src/transformers/models/gpt/modeling_gpt.py",
        "models/gpt/modeling_gpt.py",
        "modeling_gpt.py",
    ):
        assert {h.number for h in _search(engine, files=(form,))} == {1}, form


def test_the_file_filter_anchors_a_suffix_at_a_path_separator(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Suffix matching must not become substring matching: `modeling_gpt.py` is a
    different file from `not_modeling_gpt.py`, and `_` is a LIKE wildcard."""
    pr = fake.add_pr(1, body="a fix", merged_at="2026-02-02T00:00:00Z")
    pr.files = ["src/not_modeling_gpt.py"]
    _backfill(engine, fake)

    assert _search(engine, files=("modeling_gpt.py",)) == []
    # `_` escaped, so it cannot stand in for the `X`.
    assert _search(engine, files=("notXmodeling_gpt.py",)) == []


def test_the_error_filter_matches_a_pasted_traceback(engine: Engine, fake: FakeGitHub) -> None:
    """The two sides have to normalize identically (section 5.3). The caller pastes what
    their terminal printed -- addresses, counts and all -- and the indexed form has none
    of that in it, so a query that skipped ``error_query_form`` would match nothing and
    look like an index with nothing to say."""
    fake.add_pr(1, body="crashes with\n\nValueError: expected 768 features, got 1024")
    fake.add_pr(2, body="unrelated discussion")
    _index(engine, fake, 1, 2)

    pasted = 'Traceback (most recent call last):\n  File "m.py", line 9, in f\n'
    pasted += "ValueError: expected 512 features, got 4096"

    assert {h.number for h in _search(engine, errors=(pasted,))} == {1}
    assert _search(engine, errors=("KeyError: no such key",)) == []


def test_the_test_filter_takes_a_node_id_with_or_without_its_parameters(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 5.3 asks for both readings: the exact id that was written, and the
    parametrization-stripped form under which ``test_x[a]`` and ``test_x[b]`` unify."""
    fake.add_pr(1, body="tests/test_mod.py::TestMask::test_shapes[cuda] fails")
    _index(engine, fake, 1)

    exact = "tests/test_mod.py::TestMask::test_shapes[cuda]"
    assert {h.number for h in _search(engine, tests=(exact,))} == {1}
    assert {h.number for h in _search(engine, tests=(exact.removesuffix("[cuda]"),))} == {1}
    assert _search(engine, tests=("tests/test_mod.py::TestMask::test_other",)) == []


def test_a_symbol_filter_matches_a_stack_frame(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_pr(1, body='  File "src/mod.py", line 412, in forward\n    raise')
    _index(engine, fake, 1)

    assert {h.number for h in _search(engine, symbols=("forward",))} == {1}
    assert {h.number for h in _search(engine, files=("src/mod.py",))} == {1}


def test_a_thread_view_lists_the_files_extraction_found(engine: Engine, fake: FakeGitHub) -> None:
    """``ThreadView`` has read ``thread_files`` since milestone 2 and had nothing to read.
    Asserted here because it is the one place the signals are *rendered*.

    Both paths came out of the body, so both are `mentioned` and neither is a changed
    file: a pull request with no per-PR pass has no diff to report, and presenting one is
    what let a filename somebody typed answer a membership question
    (huggingface/relore#17).
    """
    fake.add_pr(1, body="see src/mod.py and docs/README.md")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.files_mentioned == ("docs/README.md", "src/mod.py")
    assert view.files_changed == ()


def test_since_excludes_older_documents(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, body="the masking marker, written long ago")
    fake.add_comment(pr, 100, "the masking marker again", created_at="2026-06-01T00:00:00Z")
    _index(engine, fake, 1)
    cutoff = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)

    assert {h.source_type for h in _search(engine, text="masking marker")} == {
        "body",
        "issue_comment",
    }
    assert {h.source_type for h in _search(engine, text="masking marker", since=cutoff)} == {
        "issue_comment"
    }


def test_a_naive_since_is_refused_at_the_boundary(engine: Engine) -> None:
    with pytest.raises(QueryError):
        SearchQuery(repos=(REPO,), since=dt.datetime(2026, 3, 1))


# -- what never comes back -------------------------------------------------


def test_machine_documents_are_excluded_from_a_default_result_set(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 6.2 and section 11: the corpus contains our own agent's comments, and
    returning one as prior discussion makes the agent's unreviewed output its own
    evidence. Excluded, not down-weighted -- at any score."""
    pr = fake.add_pr(1, body="human text about masking")
    fake.add_comment(pr, 100, "masking masking masking", author="sergereview[bot]", bot=True)
    _index(engine, fake, 1)

    assert {h.source_type for h in _search(engine, text="masking")} == {"body"}


def test_an_explicit_machine_query_is_the_only_way_to_see_them(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1, body="human text")
    fake.add_comment(pr, 100, "bot finding", author="sergereview[bot]", bot=True)
    _index(engine, fake, 1)

    hits = _search(engine, text="bot finding", trust="machine")

    assert [h.trust for h in hits] == ["machine"]


def test_raising_the_floor_drops_the_lower_tier(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, body="a drive-by claim about placement", assoc="CONTRIBUTOR")
    fake.add_review_comment(pr, 300, "placement belongs here", assoc="COLLABORATOR")
    _index(engine, fake, 1)

    assert {h.trust for h in _search(engine, text="placement")} == {"reported", "authoritative"}
    assert {h.trust for h in _search(engine, text="placement", trust="authoritative")} == {
        "authoritative"
    }


def test_a_repository_outside_the_token_scope_is_invisible(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 11 applies scoping in the query layer, not in a handler."""
    fake.add_pr(1, body="scoped text")
    _index(engine, fake, 1)

    assert [h.number for h in _search(engine, text="scoped", repos=(REPO,))] == [1]
    assert _search(engine, text="scoped", repos=(OTHER,)) == []


def test_an_empty_scope_returns_nothing_never_everything(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_pr(1, body="scoped text")
    _index(engine, fake, 1)

    assert _search(engine, text="scoped", repos=()) == []


def test_a_deleted_comment_leaves_the_full_text_index(engine: Engine, fake: FakeGitHub) -> None:
    """The SQLite half of this is migration 3's delete trigger and the Postgres half is a
    generated column, which is exactly why it is asserted on both: a stale FTS row would
    make a deleted comment findable forever, and nothing else would look wrong."""
    pr = fake.add_pr(1, body="unrelated")
    fake.add_comment(pr, 100, "a distinctive incantation")
    _index(engine, fake, 1)
    assert _search(engine, text="incantation") != []

    pr.issue_comments.clear()
    _index(engine, fake, 1)

    assert _search(engine, text="incantation") == []


def test_an_edited_comment_is_not_findable_by_its_old_text(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1)
    comment = fake.add_comment(pr, 100, "the original incantation")
    _index(engine, fake, 1)

    comment["body"] = "the replacement wording"
    _index(engine, fake, 1)

    assert _search(engine, text="incantation") == []
    assert [h.number for h in _search(engine, text="replacement")] == [1]


# -- how much comes back ---------------------------------------------------


def test_the_limit_is_clamped_not_refused(engine: Engine) -> None:
    """A client may ask for less. Never more -- and asking for more is not an error,
    because failing the call helps nobody."""
    assert SearchQuery(repos=(REPO,), limit=500).limit == MAX_HITS
    assert SearchQuery(repos=(REPO,), limit=0).limit == 1
    assert SearchQuery(repos=(REPO,), limit=3).limit == 3


def test_snippets_are_capped(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_pr(1, body="mask " + "filler " * 2000)
    _index(engine, fake, 1)

    hits = _search(engine, text="mask")

    assert hits and all(len(h.snippet) <= MAX_SNIPPET_CHARS + 2 for h in hits)


def test_one_chatty_thread_cannot_take_the_whole_page(engine: Engine, fake: FakeGitHub) -> None:
    chatty = fake.add_pr(1, title="masking", body="masking")
    for i in range(20):
        fake.add_comment(chatty, 100 + i, f"masking again {i}")
    quiet = fake.add_pr(2, title="masking elsewhere", body="masking")
    fake.add_comment(quiet, 500, "masking here too")
    _index(engine, fake, 1, 2)

    hits = _search(engine, text="masking")

    assert sum(1 for h in hits if h.number == 1) <= MAX_HITS_PER_THREAD
    assert 2 in {h.number for h in hits}


def test_hits_are_deduplicated_per_thread_and_source_type(engine: Engine, fake: FakeGitHub) -> None:
    """Section 6 keeps the best-scoring chunk per ``(thread, source_type)``, so a long
    body split into many chunks contributes one hit, not one per chunk."""
    fake.add_pr(1, body="\n\n".join(["masking paragraph"] * 400))
    _index(engine, fake, 1)

    hits = _search(engine, text="masking")

    assert [(h.number, h.source_type) for h in hits] == [(1, "body")]


# -- every hit carries its age and its tier --------------------------------


def test_every_hit_is_rendered_with_an_age_and_a_trust_tier(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Rendered, not implied (section 6, section 6.2): age changes what a model
    concludes, authority changes it more."""
    fake.add_pr(1, body="masking", assoc="OWNER")
    _index(engine, fake, 1)

    hit = _search(engine, text="masking")[0]

    assert hit.trust == "authoritative"
    assert hit.age and hit.age[-1] in "hdoy"  # h / d / mo / y
    assert hit.breakdown  # the human tuning the weights has nothing else (section 8)


# -- one thread ------------------------------------------------------------


def test_a_long_thread_is_never_returnable_in_full(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1, title="a long argument", body="the opening")
    for i in range(200):
        fake.add_comment(pr, 100 + i, f"comment number {i}")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert len(view.comments) == MAX_THREAD_COMMENTS
    assert view.total_documents == 200


def test_without_a_focus_a_thread_returns_its_opening_and_its_closing(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A long thread's resolution is at the end; truncating to the first N returns the
    part that was wrong and drops the part that settled it."""
    pr = fake.add_pr(1)
    for i in range(40):
        fake.add_comment(pr, 100 + i, f"comment {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    bodies = " ".join(c.snippet for c in view.comments)
    assert "comment 0" in bodies and "comment 39" in bodies


def test_a_focus_ranks_the_answering_comment_first(engine: Engine, fake: FakeGitHub) -> None:
    pr = fake.add_pr(1)
    for i in range(40):
        fake.add_comment(pr, 100 + i, f"filler {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    fake.add_comment(pr, 999, "the rotary embedding is the culprit")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1, focus="rotary embedding")

    assert view is not None
    assert view.comments[0].snippet == "the rotary embedding is the culprit"
    assert view.focus_matched == 1


def test_a_focus_orders_a_thread_and_never_empties_it(engine: Engine, fake: FakeGitHub) -> None:
    """The failure this exists for: every term is in the thread, no one comment carries
    them all, and the conjunction returned ``0 of 30`` on a thread with 30 comments --
    which reads as "nothing relevant here" and is not what it meant."""
    pr = fake.add_pr(1)
    fake.add_comment(pr, 100, "use_cache is the flag", created_at="2026-01-01T00:00:00Z")
    fake.add_comment(pr, 101, "gradient checkpointing is on", created_at="2026-01-01T00:01:00Z")
    fake.add_comment(pr, 102, "the warning is harmless", created_at="2026-01-01T00:02:00Z")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1, focus="use_cache gradient checkpointing warning")

    assert view is not None
    # No comment carries all four terms, so the honest denominator is zero -- and the
    # thread still comes back, best first.
    assert view.focus_matched == 0
    assert len(view.comments) == 3


def test_a_focus_no_comment_can_match_still_returns_the_thread(
    engine: Engine, fake: FakeGitHub
) -> None:
    pr = fake.add_pr(1)
    for i in range(4):
        fake.add_comment(pr, 100 + i, f"filler {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1, focus="nothing here says any of this")

    assert view is not None
    assert view.focus_matched == 0
    assert len(view.comments) == 4


def test_a_long_body_is_truncated_with_its_own_length_and_served_whole_on_request(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The gap that sent a diagnosis session back to `git clone`: an issue template spends
    its opening on environment boilerplate, so the reproduction began at the cut."""
    fake.add_issue(1, body="System Info " * 80 + "Reproduction: pass rotary_pct=0.25")
    _index(engine, fake, 1)

    capped = open_backend(engine).thread(REPO, 1)
    whole = open_backend(engine).thread(REPO, 1, full=True)

    assert capped is not None and whole is not None
    assert capped.body_truncated is True
    assert capped.body_chars == len(whole.body)
    assert "rotary_pct" not in capped.body  # the part that mattered, cut
    assert "rotary_pct" in whole.body
    assert whole.body_truncated is False


def test_normalizing_whitespace_is_not_truncation(engine: Engine, fake: FakeGitHub) -> None:
    """`snippet` collapses newlines before it cuts anything, so a body well under the cap
    came back two characters shorter than it was stored and the caller -- comparing the
    two lengths -- announced, in 40 characters, that it was withholding two
    (huggingface/relore#31)."""
    fake.add_issue(1, body="### System Info\n\nlinux\n\n### Reproduction\n\nrun it")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.body_truncated is False
    assert "Reproduction" in view.body


def test_a_body_a_sentence_over_the_cap_is_served_rather_than_announced(
    engine: Engine, fake: FakeGitHub
) -> None:
    """An advisory costs more than the tail it withholds below a sentence's worth -- and
    one that fires when nothing meaningful was withheld is one a reader learns to skip,
    and then misses the one that matters (huggingface/relore#31)."""
    body = "x" * (MAX_BODY_CHARS + BODY_SLACK_CHARS - 1)
    fake.add_issue(1, body=body)
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.body_truncated is False
    assert view.body == body, "the remainder is served, not announced"


def test_a_body_past_the_slack_is_still_cut_and_still_says_so(
    engine: Engine, fake: FakeGitHub
) -> None:
    fake.add_issue(1, body="y" * (MAX_BODY_CHARS + BODY_SLACK_CHARS + 200))
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.body_truncated is True
    assert view.body_chars == MAX_BODY_CHARS + BODY_SLACK_CHARS + 200


# -- what the trust floor took out (huggingface/relore#28) -----------------


def test_a_thread_counts_the_machine_comments_it_will_not_show(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Excluding them is right -- our own bot's output is not evidence (section 6.2) --
    but a thread whose only comment is one of them reported `0 of 0`, which is the
    sentence for a thread nobody has touched."""
    issue = fake.add_issue(1, body="the report")
    fake.add_comment(
        issue, 10, "Suggested jobs to run: run-slow gpt_neox", author="github-actions", bot=True
    )
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.comments == ()
    assert view.total_documents == 0, "the floor's denominator is what the cap composes with"
    assert view.machine_suppressed == 1


def test_a_thread_with_no_bots_suppresses_nothing(engine: Engine, fake: FakeGitHub) -> None:
    issue = fake.add_issue(1, body="the report")
    fake.add_comment(issue, 10, "I see it too")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.machine_suppressed == 0


# -- freshness, per document (huggingface/relore#32) -----------------------


def test_a_thread_carries_when_it_was_last_rebuilt(engine: Engine, fake: FakeGitHub) -> None:
    """`status`'s per-source high-water rows cannot be composed into an answer about one
    thread, and two field runs composed them wrongly in opposite directions."""
    before = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=5)
    fake.add_issue(1, body="the report")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.indexed_at is not None
    assert view.indexed_at >= before

    from relore.api.schemas import thread_json

    payload = thread_json(view)
    assert payload["indexed_at"] is not None
    assert payload["indexed_at_note"] == (
        "body and comments last indexed; later GitHub changes unknown"
    )


def test_a_truncated_changed_file_list_carries_its_denominator(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A short list of hits is a weak positive; a *missing entry* is read as a negative
    fact. `huggingface/transformers#39847` changed 323 files, the per-PR pass stages the
    first 100, and the dropped tail was the entry someone was checking."""
    pr = fake.add_pr(
        1, title="refactor", updated_at="2026-01-01T00:00:00Z", merged_at="2026-01-02T00:00:00Z"
    )
    pr.files = [f"src/m{i}/modeling_m{i}.py" for i in range(105)]
    _backfill(engine, fake)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.files_total == 105
    assert view.files_collected == 100
    # The numerator and the denominator are now the same quantity: `len(files_changed)`
    # *is* `files_collected`, where the old single array mixed in prose-derived paths and
    # disagreed with its own count by five (huggingface/relore#17).
    assert len(view.files_changed) == view.files_collected
    assert "src/m104/modeling_m104.py" not in view.files_changed  # and the count says so


def test_a_thread_with_no_changed_file_list_reports_none_rather_than_zero(
    engine: Engine, fake: FakeGitHub
) -> None:
    """An issue has no such list at all, which is not the same fact as "touched nothing"."""
    fake.add_issue(1, body="see src/mod.py")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.files_total is None
    assert view.files_collected == 0


def test_a_path_is_filed_under_where_it_came_from(engine: Engine, fake: FakeGitHub) -> None:
    """The three sources, told apart (huggingface/relore#17). The diff is what the pull
    request changed; an inline comment's anchor is also the diff and also not part of the
    collected page; a path in prose is somebody typing, and `config.json` here is the
    shape that answered "did this pull request touch it?" with a wrong yes."""
    pr = fake.add_pr(
        1,
        body="this breaks config.json handling",
        updated_at="2026-01-01T00:00:00Z",
        merged_at="2026-01-02T00:00:00Z",
    )
    pr.files = ["src/real/changed.py"]
    fake.add_review_comment(pr, 300, "move this", path="src/real/anchored.py")
    _backfill(engine, fake)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.files_changed == ("src/real/changed.py",)
    assert view.files_anchored == ("src/real/anchored.py",)
    assert "config.json" in view.files_mentioned
    # The one property the merged array could not hold: a prose filename is never
    # presented as something the pull request changed.
    assert "config.json" not in view.files_changed + view.files_anchored


def test_a_missing_thread_is_none_not_an_error(engine: Engine) -> None:
    assert open_backend(engine).thread(REPO, 4242) is None


# -- which ten comments, and how many comments there are -------------------


def test_the_middle_of_a_long_thread_is_reachable_without_a_focus(
    engine: Engine, fake: FakeGitHub
) -> None:
    """First-five-and-last-five is a sample that structurally excludes the middle, and a
    long thread is long because it was contested -- so the comment that settled it is in
    the middle by construction. On `huggingface/transformers#39847` the two that answered
    the question were at 51 and 79 of 97 and neither was reachable (huggingface/relore#16).
    """
    pr = fake.add_pr(1)
    for i in range(40):
        fake.add_comment(pr, 100 + i, f"comment {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert view.selection == "ends+middle"
    assert len(view.comments) == 10
    positions = [int(hit.snippet.split()[-1]) for hit in view.comments]
    assert positions[:2] == [0, 1], "the opening states the problem"
    assert positions[-2:] == [38, 39], "and the end usually resolves it"
    # The load-bearing assertion, and the one the old first-five-and-last-five fails:
    # its ten were 0-4 and 35-39, so nothing between 10 and 30 could ever appear.
    assert [p for p in positions if 10 <= p <= 30], "the middle is reachable"
    assert len(set(positions)) == 10, "and no slot is spent twice"


def test_a_review_beats_chatter_for_a_middle_slot(engine: Engine, fake: FakeGitHub) -> None:
    """A review with a state is an act; `🤗` is not. Nine of the ten default comments on
    #39847 were `[authoritative]`, which means the tier carried no signal and the slots
    went to emoji from the same people who wrote the verdict."""
    pr = fake.add_pr(1)
    for i in range(30):
        fake.add_comment(pr, 100 + i, f"chatter {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    fake.add_review(
        pr, 900, "revert this for models that do not need it", submitted_at="2026-01-01T00:15:00Z"
    )
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert any("revert this" in hit.snippet for hit in view.comments)


def test_a_short_thread_is_returned_whole(engine: Engine, fake: FakeGitHub) -> None:
    """The sample only applies where there is something to sample."""
    pr = fake.add_pr(1)
    for i in range(4):
        fake.add_comment(pr, 100 + i, f"comment {i}", created_at=f"2026-01-01T00:{i:02d}:00Z")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1)

    assert view is not None
    assert len(view.comments) == 4
    assert view.total_documents == 4


def test_one_comment_is_one_hit_however_many_chunks_it_became(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A 9,827-character comment is several documents, and two of them came back as two
    `[authoritative]` hits with one URL -- which reads as two people agreeing, and cost
    two of ten slots (huggingface/relore#18). One slot, and it says it is a passage.
    """
    pr = fake.add_pr(1)
    long_comment = "\n\n".join(f"paragraph {i} about rotary embeddings" for i in range(400))
    fake.add_comment(pr, 100, long_comment)
    fake.add_comment(pr, 101, "a short one about rotary embeddings")
    _index(engine, fake, 1)

    view = open_backend(engine).thread(REPO, 1, focus="rotary embeddings")

    assert view is not None
    ids = [hit.source_id for hit in view.comments]
    assert len(ids) == len(set(ids)), "no comment appears twice"
    urls = [hit.url for hit in view.comments]
    assert len(urls) == len(set(urls)), "and no URL appears twice"
    chunked = next(hit for hit in view.comments if hit.source_id == "100")
    assert chunked.passages > 1, "the hit knows it is one passage of a longer comment"
    # And the thread's own count is comments, not documents: the long one is one comment.
    assert view.total_documents == 2


# -- which engine answered -------------------------------------------------


def test_the_backend_names_itself_and_its_ranking_function(engine: Engine) -> None:
    """Section 4.1: ``status`` prints this so a surprising result set is diagnosable, and
    section 10's benchmark records it so a recall number measured on ``bm25`` is never
    compared with one measured on ``ts_rank_cd``."""
    info = open_backend(engine).info()

    assert info.name == engine.dialect.name
    assert info.ranking in ("ts_rank_cd", "bm25")
    assert "fulltext" in info.capabilities
    assert "vector" not in info.capabilities  # provisioned, unwritten (section 6.1)


# -- section 6.2's per-query-kind trust floors ------------------------------
#
# The floors were parked until `repo_authority` existed (section 13.1 #4): until an org
# repository's maintainers stop reading as unresolved MEMBER, a rationale floor filters out
# the very comments it exists to find. These assert the behaviours, not the tiers -- the
# tier derivation itself is tests/integration/test_authority.py.


def test_a_rationale_query_returns_no_reported_document(engine: Engine, fake: FakeGitHub) -> None:
    """The whole point of the floor: an unauthorised answer to "is this intentional" is a
    guess the agent will act on, and unlike a low-ranked hit nobody checks it."""
    pr = fake.add_pr(1, title="Fix the mask", body="nothing here")
    fake.add_comment(pr, 100, "the mask is intentional here", author="stranger", assoc="NONE")
    fake.add_review_comment(
        pr, 300, "the mask is intentional, by design", author="boss", assoc="OWNER"
    )
    _index(engine, fake, 1)

    plain = _search(engine, text="mask intentional")
    rationale = _search(engine, text="mask intentional", kind="rationale")

    assert {h.author for h in plain} == {"stranger", "boss"}
    assert {h.author for h in rationale} == {"boss"}
    assert all(h.trust == "authoritative" for h in rationale)


def test_a_failure_query_keeps_a_stranger_s_traceback(engine: Engine, fake: FakeGitHub) -> None:
    """A stranger reporting an exception is evidence. They are reporting, not adjudicating."""
    issue = fake.add_issue(1, title="crash", body="nothing")
    fake.add_comment(
        issue, 100, "AttributeError on the dtype attribute", author="stranger", assoc="NONE"
    )
    _index(engine, fake, 1)

    hits = _search(engine, text="AttributeError dtype", kind="failure")

    assert [h.author for h in hits] == ["stranger"]


def test_a_merged_pr_is_precedent_whoever_wrote_it(engine: Engine, fake: FakeGitHub) -> None:
    """Section 6.2: authority attaches to the merge, not to the author. A first-time
    contributor's merged change is precedent."""
    merged = fake.add_pr(
        1,
        title="add the widget",
        body="widget support, as discussed",
        author="newcomer",
        assoc="CONTRIBUTOR",
        merged_at="2026-02-02T00:00:00Z",
    )
    merged.merged_by = "boss"
    _index(engine, fake, 1)

    hits = _search(engine, text="widget", kind="precedent")

    assert hits, "a merged PR is precedent regardless of who authored it"
    assert {(h.author, h.trust) for h in hits} == {("newcomer", "reported")}


def test_an_unmerged_proposal_is_not_precedent(engine: Engine, fake: FakeGitHub) -> None:
    """The other half of the same sentence, and the reason the exemption is keyed on the
    merge rather than on the thread being a PR at all."""
    fake.add_pr(
        1,
        title="add the widget",
        body="widget support, as discussed",
        author="newcomer",
        assoc="CONTRIBUTOR",
    )
    _index(engine, fake, 1)

    assert _search(engine, text="widget") != []
    assert _search(engine, text="widget", kind="precedent") == []


def test_a_request_cannot_lower_a_kind_s_floor(engine: Engine, fake: FakeGitHub) -> None:
    """`trust` may only raise. Asking for `reported` on a rationale query is not a way
    around the floor -- the server decides the default, not the client (section 7)."""
    issue = fake.add_issue(1, title="why", body="nothing")
    fake.add_comment(issue, 100, "the widget is intentional", author="stranger", assoc="NONE")
    _index(engine, fake, 1)

    assert _search(engine, text="widget intentional") != []
    assert _search(engine, text="widget intentional", kind="rationale", trust="reported") == []


def test_raising_the_floor_drops_the_precedent_exemption(engine: Engine, fake: FakeGitHub) -> None:
    """Asking for `authoritative` is asking for authoritative documents, not for an
    exemption from your own request."""
    merged = fake.add_pr(
        1,
        title="add the widget",
        body="widget support",
        author="newcomer",
        assoc="CONTRIBUTOR",
        merged_at="2026-02-02T00:00:00Z",
    )
    merged.merged_by = "boss"
    _index(engine, fake, 1)

    assert _search(engine, text="widget", kind="precedent") != []
    assert _search(engine, text="widget", kind="precedent", trust="authoritative") == []


def test_a_kind_never_admits_a_machine_document(engine: Engine, fake: FakeGitHub) -> None:
    """The machine exclusion is a security property (section 11) and no query kind is a
    way through it -- including the one whose floor is lowest."""
    issue = fake.add_issue(1, title="crash", body="nothing")
    fake.add_comment(
        issue, 100, "AttributeError on dtype", author="ci", assoc="CONTRIBUTOR", bot=True
    )
    _index(engine, fake, 1)

    for kind in (None, "failure", "precedent", "rationale"):
        assert _search(engine, text="AttributeError dtype", kind=kind) == []
    assert [h.trust for h in _search(engine, text="AttributeError dtype", trust="machine")] == [
        "machine"
    ]
