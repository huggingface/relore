"""The coverage leg (huggingface/relore#79): threads ranked by the IDF of the rarest query
terms they hold, fused into every expanded search at the thread level.

Asserted on both dialects, because what is asserted is a property of the design rather
than of an engine: which threads are admitted, which terms count, how ties fuse and how
the page is rebuilt. No score is compared across backends (section 4.1).
"""

from __future__ import annotations

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine

from relore.ingest.index_thread import index_thread
from relore.search import SearchQuery, open_backend, search_best
from relore.search.expansion import COVERAGE_LEG, _shared_ranks
from relore.search.queries import Hit, coverage_words, is_plain_word

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _index(engine: Engine, fake: FakeGitHub, *numbers: int) -> None:
    with fake.client() as client:
        for number in numbers:
            index_thread(engine, client, fake.repo, number)


def _query(**kwargs: object) -> SearchQuery:
    return SearchQuery(repos=(REPO,), **kwargs)  # type: ignore[arg-type]


def _threads(hits: list[Hit]) -> list[int]:
    return list(dict.fromkeys(hit.number for hit in hits))


# -- the vocabulary ---------------------------------------------------------


def test_only_a_plain_word_is_prefix_matched() -> None:
    assert is_plain_word("qwen")
    assert is_plain_word("qwen3")
    assert not is_plain_word("create_masks_for_generate")
    assert not is_plain_word("41251")
    assert not is_plain_word("fa")


def test_words_are_lowercased_and_deduplicated_in_order() -> None:
    assert coverage_words("Qwen prefill QWEN regression") == ("qwen", "prefill", "regression")


def test_equal_scores_share_a_rank() -> None:
    """A tie inside the leg must be broken by the other legs, never by iteration order."""

    def hit(number: int) -> Hit:
        return Hit(
            repo=REPO,
            number=number,
            thread_type="issue",
            title="t",
            source_type="body",
            url="",
            author="a",
            trust="reported",
            age="1d",
            snippet="",
            score=0.0,
        )

    ranks = _shared_ranks([(hit(1), 9.0), (hit(2), 9.0), (hit(3), 4.0)])
    assert {key[1]: rank for key, (rank, _hit) in ranks.items()} == {1: 1, 2: 1, 3: 3}


# -- the leg ----------------------------------------------------------------


def test_rare_terms_held_across_a_thread_outrank_common_ones(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The measured case: the title carries one rare term and a comment the next. A thread
    repeating the common word scores below it however often it repeats it."""
    answer = fake.add_issue(1, title="[Qwen3] mask return type changed", body="see the trace")
    fake.add_comment(answer, 11, "prefill latency regressed by thirty percent")
    fake.add_issue(2, title="transformers is slow", body="transformers transformers transformers")
    for number in range(3, 8):
        fake.add_issue(number, title=f"thread {number}", body="a transformers question")
    _index(engine, fake, *range(1, 8))

    found = open_backend(engine).coverage(_query(text="qwen prefill transformers"))

    assert found.threads[0][0].number == 1
    assert "qwen" in found.terms


def test_a_plain_word_reaches_its_versioned_form(engine: Engine, fake: FakeGitHub) -> None:
    """``qwen`` matches nothing in the strict search, which stems but never prefixes, and a
    single term is never widened -- so before this leg the page was empty."""
    fake.add_issue(1, title="Qwen3 prefill latency", body="numbers attached")
    fake.add_issue(2, title="tokenizer question", body="nothing related")
    _index(engine, fake, 1, 2)
    backend = open_backend(engine)

    assert backend.search(_query(text="qwen")) == []
    hits, widened = search_best(backend, _query(text="qwen"))

    assert _threads(hits) == [1]
    assert widened == ""


def test_the_trust_floor_holds(engine: Engine, fake: FakeGitHub) -> None:
    """The leg reads through the same filters as ``search``: a term only a bot wrote is
    not evidence, so it neither counts nor represents the thread."""
    thread = fake.add_issue(1, title="a title", body="a body")
    fake.add_comment(thread, 11, "zebrafish zebrafish", author="ci-bot", bot=True)
    _index(engine, fake, 1)
    backend = open_backend(engine)

    assert backend.coverage(_query(text="zebrafish")).threads == ()
    assert search_best(backend, _query(text="zebrafish")) == ([], "")


def test_a_filter_excludes_what_the_leg_would_add(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_issue(1, title="Qwen3 prefill latency", body="numbers attached")
    _index(engine, fake, 1)

    hits, _ = search_best(open_backend(engine), _query(text="qwen", files=("src/absent.py",)))

    assert hits == []


# -- fusion and the page ----------------------------------------------------


def test_the_page_is_filled_breadth_first(engine: Engine, fake: FakeGitHub) -> None:
    """A thread the strict search answered three times must not spend the page before a
    thread only the leg found gets one slot."""
    strict = fake.add_issue(1, title="alpha beta", body="alpha beta again")
    fake.add_comment(strict, 11, "alpha beta once more")
    fake.add_issue(2, title="only beta here", body="nothing else")
    _index(engine, fake, 1, 2)

    hits, _ = search_best(open_backend(engine), _query(text="alpha beta"))

    assert [hit.number for hit in hits][:2] == [1, 2]
    assert [hit.number for hit in hits].count(1) == 3


def test_the_leg_is_disclosed(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_issue(1, title="Qwen3 prefill latency", body="numbers attached")
    _index(engine, fake, 1)
    trace: list = []

    search_best(open_backend(engine), _query(text="qwen latency"), trace=trace)

    assert [(leg.name, set(leg.term.split())) for leg in trace] == [
        (COVERAGE_LEG, {"qwen", "latency"})
    ]


def test_an_unexpanded_search_has_no_leg(engine: Engine, fake: FakeGitHub) -> None:
    """``expand=False`` is the plain search section 10 scores as ``index``; it stays that."""
    fake.add_issue(1, title="Qwen3 prefill latency", body="numbers attached")
    _index(engine, fake, 1)
    trace: list = []

    hits, _ = search_best(open_backend(engine), _query(text="qwen"), expand=False, trace=trace)

    assert hits == []
    assert trace == []


def test_newest_orders_the_fused_page(engine: Engine, fake: FakeGitHub) -> None:
    """``sort`` presents a page and never selects one (section 10.7) -- so it applies after
    fusion, or the fused order would silently replace the one the caller asked for."""
    old = fake.add_issue(1, title="beta", body="nothing")
    fake.add_comment(old, 11, "alpha beta", created_at="2026-01-01T00:00:00Z")
    new = fake.add_issue(2, title="gamma", body="nothing")
    fake.add_comment(new, 21, "beta", created_at="2026-03-01T00:00:00Z")
    _index(engine, fake, 1, 2)

    hits, _ = search_best(open_backend(engine), _query(text="alpha beta", sort="newest"))
    stamps = [hit.created_at for hit in hits]

    assert stamps == sorted(stamps, reverse=True)


def test_newest_orders_an_expanded_page_the_leg_left_alone(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Expansion fuses its own legs by score; with no coverage contribution to fuse, the
    caller's ``newest`` must still hold."""
    old = fake.add_issue(1, title="t", body="nothing")
    fake.add_comment(old, 11, "see `build_mask` here", created_at="2026-01-01T00:00:00Z")
    new = fake.add_issue(2, title="t", body="nothing")
    fake.add_comment(new, 21, "and `build_mask` there", created_at="2026-03-01T00:00:00Z")
    _index(engine, fake, 1, 2)

    hits, _ = search_best(open_backend(engine), _query(text="build_mask", sort="newest"))
    stamps = [hit.created_at for hit in hits]

    assert len(stamps) >= 2
    assert stamps == sorted(stamps, reverse=True)
