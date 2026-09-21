"""Section 6's query expansion, on both dialects (section 12).

The fan-out is asserted as *data* -- which legs one call becomes -- because that half is
dialect-free and is the part a reimplementation gets wrong. The merged search is asserted
as a contract, never as an order that depends on ``ts_rank_cd`` or ``bm25`` (section 4.1):
what comes back, what never comes back, and how much of it.

The case that matters most is the one section 10.3 measured: a query whose terms live only
in a document the trust floor excludes. The plain search returns nothing for it. That is
not a ranking failure and no weight set can fix it, so it is the test that says whether
expansion did its job.
"""

from __future__ import annotations

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine

from relore.ingest.index_thread import index_thread
from relore.search import SearchQuery, expand, open_backend, search_expanded
from relore.search.expansion import MAX_LEGS, MAX_PER_KIND, RRF_K, rank_spec
from relore.search.queries import MAX_HITS_PER_THREAD

REPO = "owner/name"

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "src/transformers/models/foo/modeling_foo.py", line 412, in forward\n'
    "    hidden = self.proj(hidden)\n"
    "ValueError: expected 3 dimensions but got 4\n"
)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _index(engine: Engine, fake: FakeGitHub, *numbers: int) -> None:
    with fake.client() as client:
        for number in numbers:
            index_thread(engine, client, fake.repo, number)


def _legs(query: SearchQuery) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for leg in expand(query):
        out.setdefault(leg.name, []).append(leg.term)
    return out


# -- the fan-out ------------------------------------------------------------


def test_a_traceback_becomes_an_error_a_file_and_a_symbol_leg() -> None:
    """Section 6: "a traceback fans out into error, test-id, symbol, file and free-text"."""
    legs = _legs(SearchQuery(repos=(REPO,), text=TRACEBACK))

    assert legs["error"] == ["expected <n> dimensions but got <n>"]
    assert legs["file"] == ["src/transformers/models/foo/modeling_foo.py"]
    assert legs["symbol"] == ["forward"]
    assert legs["text"] == [TRACEBACK]


def test_the_error_leg_carries_section_5_3s_normal_form() -> None:
    """The one term whose two sides must agree. The index holds the normalized message, so
    a leg that filtered on the pasted line verbatim would match nothing -- indistinguishable
    from a corpus with nothing to say."""
    (term,) = _legs(SearchQuery(repos=(REPO,), text=TRACEBACK))["error"]

    assert "<n>" in term
    assert "expected 3" not in term


def test_the_exception_type_is_not_also_asked_as_a_symbol() -> None:
    """``ValueError`` passes every shape test and matches half a corpus on its own; the
    error leg already holds it in a more precise form."""
    legs = _legs(SearchQuery(repos=(REPO,), text=TRACEBACK))

    assert "ValueError" not in legs.get("symbol", [])


def test_a_paths_own_components_are_not_asked_as_symbols() -> None:
    """``tokenize`` shreds a path into words and ``modeling_foo`` then looks exactly like
    an identifier -- but it is a file, it already has a file leg, and asking it as a symbol
    spends a leg fusing the same evidence with itself."""
    legs = _legs(SearchQuery(repos=(REPO,), text=TRACEBACK))

    assert "modeling_foo" not in legs.get("symbol", [])


def test_a_bare_identifier_expands_without_backticks() -> None:
    """Indexing reads symbols out of code spans only, which is right for prose and wrong
    for a query: nobody types four words to emphasise one. Requiring backticks is exactly
    why section 10.3's queries expanded to nothing."""
    legs = _legs(SearchQuery(repos=(REPO,), text="_maybe_import_sdnq __new__"))

    assert legs["symbol"] == ["_maybe_import_sdnq", "__new__"]


def test_prose_expands_to_nothing_and_that_is_not_a_failure() -> None:
    """No identifier, no path, no traceback: there is no second question in the call, so
    there is nothing to fan out and no round trip to pay for."""
    legs = expand(SearchQuery(repos=(REPO,), text="why is this here"))

    assert [leg.name for leg in legs] == ["text"]


def test_a_signal_leg_drops_the_free_text_and_keeps_the_callers_filters() -> None:
    """The whole fix. The caller's terms are what ANDed to nothing; their ``--file`` is
    scope they chose, and dropping it would answer a wider question than was asked."""
    query = SearchQuery(repos=(REPO,), text="build_mask helper", files=("src/mod.py",))

    (leg,) = [leg for leg in expand(query) if leg.name == "symbol"]

    assert leg.query.text == ""
    assert leg.query.symbols == ("build_mask",)
    assert leg.query.files == ("src/mod.py",)


def test_the_callers_own_filters_become_a_textless_leg() -> None:
    """Section 10.3's control: on the leak-free rationale slice this leg alone scores
    0.857 and 1.000 with nothing empty, because the answer was reachable all along and the
    text was what excluded it."""
    query = SearchQuery(repos=(REPO,), text="some words", files=("src/mod.py",))

    (leg,) = [leg for leg in expand(query) if leg.name == "filters"]

    assert leg.query.text == ""
    assert leg.query.files == ("src/mod.py",)


def test_no_filters_means_no_filters_leg() -> None:
    """A textless leg with nothing to filter on is the whole repository, and its top ten
    would score as if the index had answered."""
    legs = _legs(SearchQuery(repos=(REPO,), text="build_mask"))

    assert "filters" not in legs


def test_the_fan_out_is_bounded() -> None:
    """A traceback naming twenty frames is one question, not twenty."""
    frames = "\n".join(f'  File "src/pkg/mod{i}.py", line {i}, in helper_{i}' for i in range(12))
    text = f"Traceback (most recent call last):\n{frames}\nValueError: boom happened here\n"

    legs = expand(SearchQuery(repos=(REPO,), text=text, files=("src/pkg/mod0.py",)))

    assert len(legs) <= MAX_LEGS
    assert len(_legs(SearchQuery(repos=(REPO,), text=text))["file"]) <= MAX_PER_KIND


def test_two_legs_asking_the_same_question_are_one_leg() -> None:
    """They would fuse as two votes, letting a duplicate outrank a hit that genuinely came
    back from two different questions."""
    query = SearchQuery(repos=(REPO,), text="build_mask", symbols=("build_mask",))

    legs = expand(query)

    assert len({(leg.query.text, leg.query.symbols) for leg in legs}) == len(legs)


SENTINEL_SYMBOL = "ReloreScopeSentinelNoSuchSymbol"
SENTINEL_FILE = "ReloreScopeSentinelNoSuchFile.py"
SENTINEL_ERROR = "ReloreScopeSentinelNoSuchError"
SENTINEL_TEST = "tests/sentinel.py::test_no_such_node"
DERIVED_SYMBOL_TEXT = "Qwen3_5MoeExperts CheckpointError"
NODE_ID_TEXT = "FAILED tests/models/test_foo.py::test_generate - boom"


def _assert_legs_keep_explicit_filters(query: SearchQuery) -> None:
    """Every expansion leg must keep the caller's *explicit* signal filters unchanged.

    Derived evidence may still fill a field the caller left empty -- that is a new
    question, not a dropped constraint. Same-kind values are ORed, so appending to a
    field the caller set would widen it; replacing it is how #75 leaked hits the page
    still claimed were filtered out.
    """
    for leg in expand(query):
        if query.files:
            assert leg.query.files == query.files
        if query.symbols:
            assert leg.query.symbols == query.symbols
        if query.errors:
            assert leg.query.errors == query.errors
        if query.tests:
            assert leg.query.tests == query.tests


def test_a_derived_symbol_does_not_replace_an_explicit_symbol_filter() -> None:
    """The field reproduction in huggingface/relore#75: expansion asked for the identifier
    in the prose and dropped ``--symbol`` on that leg, while the response still listed it."""
    query = SearchQuery(repos=(REPO,), text=DERIVED_SYMBOL_TEXT, symbols=(SENTINEL_SYMBOL,))

    assert any(
        leg.name == "symbol" for leg in expand(SearchQuery(repos=(REPO,), text=DERIVED_SYMBOL_TEXT))
    )
    _assert_legs_keep_explicit_filters(query)
    for leg in expand(query):
        assert SENTINEL_SYMBOL in leg.query.symbols
        assert "Qwen3_5MoeExperts" not in leg.query.symbols


def test_a_derived_file_does_not_replace_an_explicit_file_filter() -> None:
    query = SearchQuery(repos=(REPO,), text=TRACEBACK, files=(SENTINEL_FILE,))

    assert any(leg.name == "file" for leg in expand(SearchQuery(repos=(REPO,), text=TRACEBACK)))
    _assert_legs_keep_explicit_filters(query)


def test_a_derived_error_does_not_replace_an_explicit_error_filter() -> None:
    query = SearchQuery(repos=(REPO,), text=TRACEBACK, errors=(SENTINEL_ERROR,))

    assert any(leg.name == "error" for leg in expand(SearchQuery(repos=(REPO,), text=TRACEBACK)))
    _assert_legs_keep_explicit_filters(query)


def test_a_derived_test_id_does_not_replace_an_explicit_test_filter() -> None:
    query = SearchQuery(repos=(REPO,), text=NODE_ID_TEXT, tests=(SENTINEL_TEST,))

    assert any(leg.name == "test" for leg in expand(SearchQuery(repos=(REPO,), text=NODE_ID_TEXT)))
    _assert_legs_keep_explicit_filters(query)


def test_a_skipped_derived_leg_still_counts_toward_ranking() -> None:
    """The caller's ``--symbol`` suppresses the derived symbol *leg*, not the evidence: the
    identifier is still in the text, so it must still weigh in the score."""
    query = SearchQuery(repos=(REPO,), text=DERIVED_SYMBOL_TEXT, symbols=(SENTINEL_SYMBOL,))

    assert not any(
        leg.name == "symbol" and leg.term == "Qwen3_5MoeExperts" for leg in expand(query)
    )
    spec = rank_spec(query, expand(query))

    assert "Qwen3_5MoeExperts" in spec.symbols
    assert SENTINEL_SYMBOL in spec.symbols


def test_ranking_does_not_add_signals_for_a_kind_the_caller_left_open() -> None:
    """Only a *skipped* leg is folded in separately; an unscoped kind still arrives through
    its own leg, so nothing is counted twice."""
    query = SearchQuery(repos=(REPO,), text=DERIVED_SYMBOL_TEXT)
    spec = rank_spec(query, expand(query))

    assert spec.symbols.count("Qwen3_5MoeExperts") == 1


# -- the merged search, against a real index --------------------------------


def test_expansion_answers_what_the_trust_floor_made_unanswerable(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Section 10.3, reproduced small.

    The query's terms come from a *bot's* review finding, and section 6.2 excludes machine
    documents from reads -- so the only document holding both terms is one the index
    refuses to return, and the plain AND comes back empty. The maintainer's reply, which
    is the answer, shares no term with the query at all.
    """
    pr = fake.add_pr(1, merged_at="2026-02-01T00:00:00Z", title="quantization loading")
    finding = fake.add_review_comment(
        pr,
        10,
        "Gating this on `_maybe_import_sdnq` leaves `__new__` unable to raise.",
        path="src/pipeline_loading_utils.py",
        author="reviewbot",
        assoc="NONE",
        bot=True,
    )
    fake.add_review_comment(
        pr,
        11,
        "this was added to be consistent with modular diffusers per the suggestion",
        path="src/pipeline_loading_utils.py",
        author="carol",
        assoc="COLLABORATOR",
        in_reply_to=finding["id"],
    )
    _index(engine, fake, 1)
    backend = open_backend(engine)
    query = SearchQuery(
        repos=(REPO,),
        text="_maybe_import_sdnq __new__",
        kind="rationale",
        files=("src/pipeline_loading_utils.py",),
    )

    assert backend.search(query) == []
    assert [(hit.repo, hit.number) for hit in search_expanded(backend, query)] == [(REPO, 1)]


def test_expansion_never_loses_what_the_plain_search_found(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The ``text`` leg is the caller's query exactly as sent. Everything else is additive,
    so an expanded search is a superset by construction -- otherwise turning it on would be
    a regression nobody could see."""
    fake.add_issue(1, body="the build_mask helper is wrong")
    fake.add_issue(2, body="unrelated entirely")
    _index(engine, fake, 1, 2)
    backend = open_backend(engine)
    query = SearchQuery(repos=(REPO,), text="build_mask helper")

    plain = {(hit.repo, hit.number) for hit in backend.search(query)}
    expanded = {(hit.repo, hit.number) for hit in search_expanded(backend, query)}

    assert plain
    assert plain <= expanded


def test_the_per_thread_cap_survives_the_merge(engine: Engine, fake: FakeGitHub) -> None:
    """Each leg's SQL already caps per thread, but two legs can each return the same thread
    under a different source type -- so without a second pass one chatty thread reclaims
    the page the window functions were written to protect."""
    issue = fake.add_issue(1, body="build_mask lives in src/mod.py and is wrong")
    for comment_id in range(20, 28):
        fake.add_comment(issue, comment_id, "build_mask again, see src/mod.py")
    _index(engine, fake, 1)
    backend = open_backend(engine)

    hits = search_expanded(backend, SearchQuery(repos=(REPO,), text="build_mask src/mod.py"))

    assert len(hits) <= MAX_HITS_PER_THREAD


def test_a_hit_says_how_many_legs_agreed(engine: Engine, fake: FakeGitHub) -> None:
    """Section 6 puts every term of the score in ``breakdown`` for the person tuning it.
    A merged result whose provenance is invisible is one nobody can argue with."""
    fake.add_issue(1, body="build_mask lives in src/mod.py")
    _index(engine, fake, 1)
    backend = open_backend(engine)

    (hit, *_) = search_expanded(backend, SearchQuery(repos=(REPO,), text="build_mask src/mod.py"))

    assert hit.breakdown["legs"] >= 2
    assert hit.breakdown["fused"] == pytest.approx(hit.score, abs=1e-6)
    assert hit.score <= len(expand(SearchQuery(repos=(REPO,), text="build_mask src/mod.py"))) / (
        RRF_K + 1
    )


def test_an_empty_repo_scope_stays_empty_through_expansion(engine: Engine) -> None:
    """Section 11 fails closed: no repository in scope means nothing, never everything.
    Expansion runs several queries, so it is several chances to lose that."""
    assert search_expanded(open_backend(engine), SearchQuery(repos=(), text=TRACEBACK)) == []


def test_expansion_does_not_admit_rows_outside_an_explicit_symbol_filter(
    engine: Engine, fake: FakeGitHub
) -> None:
    """huggingface/relore#75: the JSON still reported ``--symbol``, but a derived symbol
    leg had already replaced it, so threads that never mentioned the sentinel came back.

    Backticks put ``Qwen3_5MoeExperts`` into ``thread_symbols`` the way a real review
    comment would, so the unconstrained expansion can find issue 47467. The constrained
    call must not.
    """
    fake.add_issue(47467, body="`Qwen3_5MoeExperts` CheckpointError while loading experts")
    fake.add_issue(2, body=f"`{SENTINEL_SYMBOL}` is the only thread this filter should keep")
    _index(engine, fake, 47467, 2)
    backend = open_backend(engine)
    unconstrained = SearchQuery(repos=(REPO,), text=DERIVED_SYMBOL_TEXT)
    constrained = SearchQuery(
        repos=(REPO,),
        text=DERIVED_SYMBOL_TEXT,
        symbols=(SENTINEL_SYMBOL,),
    )

    unconstrained_numbers = {hit.number for hit in search_expanded(backend, unconstrained)}
    constrained_numbers = {hit.number for hit in search_expanded(backend, constrained)}

    assert 47467 in unconstrained_numbers
    assert constrained_numbers == {2}
