"""Section 6.1's semantic leg: off unless configured, prose only, never stale, never a veto
on the widening, and absent rather than broken where there is no vector tier.

The embedder is a fake with a tiny concept table, so a paraphrase is something a test can
construct: ``slower`` and ``latency`` land on the same axis with no shared term.
"""

from __future__ import annotations

import math

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine, func, select

from relore.embedding import DIMENSIONS
from relore.ingest.embed import embed_pending, pending
from relore.ingest.index_thread import index_thread
from relore.render import semantic_line
from relore.search import SearchQuery, open_backend, search_best
from relore.search.expansion import SEMANTIC_LEG, WIDENED_ANY_TERM
from relore.search.queries import is_prose
from relore.store.dialect import has_vector_tier, write_embeddings
from relore.store.repository import index_summary
from relore.store.schema import documents

REPO = "owner/name"
MODEL = "fake/concepts-v1"

#: Words that mean the same thing to the fake model, and nothing else does.
CONCEPTS = {
    "slower": "speed",
    "slow": "speed",
    "latency": "speed",
    "regressed": "speed",
    "tokenizer": "text",
    "vocabulary": "text",
}


class FakeEmbedder:
    model = MODEL

    def __init__(self) -> None:
        self.queries: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            concept = CONCEPTS.get(word.strip(".,"))
            if concept:
                vector[hash(concept) % DIMENSIONS] += 1.0
        vector[DIMENSIONS - 1] += 0.01  # never the zero vector
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


@pytest.fixture
def vectors(engine: Engine) -> Engine:
    if not has_vector_tier(engine):
        pytest.skip("no vector tier on this dialect (section 4.1)")
    return engine


def _index(engine: Engine, fake: FakeGitHub, *numbers: int) -> None:
    with fake.client() as client:
        for number in numbers:
            index_thread(engine, client, fake.repo, number)


def _query(**kwargs: object) -> SearchQuery:
    return SearchQuery(repos=(REPO,), **kwargs)  # type: ignore[arg-type]


PARAPHRASE = "generation got slower after the upgrade"


def _paraphrase_corpus(fake: FakeGitHub) -> None:
    fake.add_issue(1, title="prefill numbers", body="end to end prefill latency regressed")
    fake.add_issue(2, title="vocabulary question", body="the tokenizer splits oddly")


# -- the gate ---------------------------------------------------------------


def test_only_prose_is_embedded() -> None:
    assert is_prose(PARAPHRASE)
    assert not is_prose("model_inputs device")
    assert not is_prose("layer_types reference_overrides")
    assert not is_prose("`build_mask` `forward()` Qwen3")


def test_a_bag_of_identifiers_never_reaches_the_model(vectors: Engine, fake: FakeGitHub) -> None:
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    embedder = FakeEmbedder()
    embed_pending(vectors, REPO, embedder)

    search_best(open_backend(vectors), _query(text="prefill_latency"), embedder=embedder)

    assert embedder.queries == []


# -- the leg ----------------------------------------------------------------


def test_a_paraphrase_reaches_its_thread(vectors: Engine, fake: FakeGitHub) -> None:
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    embedder = FakeEmbedder()
    embed_pending(vectors, REPO, embedder)
    backend = open_backend(vectors)
    trace: list = []

    without, _ = search_best(backend, _query(text=PARAPHRASE))
    hits, _ = search_best(backend, _query(text=PARAPHRASE), embedder=embedder, trace=trace)

    assert 1 not in {hit.number for hit in without}
    assert [hit.number for hit in hits][:1] == [1]
    assert (SEMANTIC_LEG, MODEL) in [(leg.name, leg.term) for leg in trace]


def test_the_leg_never_vetoes_the_widening(vectors: Engine, fake: FakeGitHub) -> None:
    """A nearest-neighbour leg is never empty. Deciding the widening on the fused page let
    it suppress the fallback, and the page became the vectors' alone (relore#79)."""
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    embedder = FakeEmbedder()
    embed_pending(vectors, REPO, embedder)

    _hits, widened = search_best(open_backend(vectors), _query(text=PARAPHRASE), embedder=embedder)

    assert widened == WIDENED_ANY_TERM


def test_the_trust_floor_holds(vectors: Engine, fake: FakeGitHub) -> None:
    thread = fake.add_issue(1, title="a title", body="nothing to see")
    fake.add_comment(thread, 11, "latency regressed badly", author="ci-bot", bot=True)
    _index(vectors, fake, 1)
    embedder = FakeEmbedder()
    embed_pending(vectors, REPO, embedder)

    near = open_backend(vectors).semantic(
        _query(text=PARAPHRASE), embedder.query(PARAPHRASE), MODEL
    )

    assert all(hit.trust != "machine" for hit, _score in near)


def test_an_edited_document_is_not_served_until_embedded_again(
    vectors: Engine, fake: FakeGitHub
) -> None:
    """The reconcile resets ``embedding_model`` on a rewrite, and the leg reads only the
    configured model's rows -- so the old text's vector is unreachable, not merely stale."""
    thread = fake.add_issue(1, title="prefill numbers", body="end to end prefill latency regressed")
    _index(vectors, fake, 1)
    embedder = FakeEmbedder()
    embed_pending(vectors, REPO, embedder)
    backend = open_backend(vectors)
    assert backend.semantic(_query(text=PARAPHRASE), embedder.query(PARAPHRASE), MODEL)

    thread.payload["body"] = "this is about the tokenizer vocabulary now"
    thread.payload["title"] = "vocabulary question"
    _index(vectors, fake, 1)

    assert backend.semantic(_query(text=PARAPHRASE), embedder.query(PARAPHRASE), MODEL) == []
    assert pending(vectors, REPO, MODEL) > 0


# -- the pass ---------------------------------------------------------------


def test_the_pass_is_idempotent(vectors: Engine, fake: FakeGitHub) -> None:
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    embedder = FakeEmbedder()

    first = embed_pending(vectors, REPO, embedder)
    second = embed_pending(vectors, REPO, embedder)

    assert first.embedded > 0
    assert second.embedded == 0
    assert pending(vectors, REPO, MODEL) == 0


def test_a_vector_is_not_written_over_text_that_changed(vectors: Engine, fake: FakeGitHub) -> None:
    """The race: a poll rewrites the document between the read that fed the model and the
    write. The hash guard leaves it pending instead of stamping old text as current."""
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1)
    with vectors.begin() as conn:
        wrote = write_embeddings(conn, MODEL, [(1, "not-the-current-hash", [0.1] * DIMENSIONS)])

    assert wrote == 0


def test_status_reports_coverage_rather_than_on(vectors: Engine, fake: FakeGitHub) -> None:
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    embed_pending(vectors, REPO, FakeEmbedder(), limit=1)
    with vectors.connect() as conn:
        summary = index_summary(conn)

    line = semantic_line({**summary, "semantic_model": MODEL})

    assert summary["embedded"] == {MODEL: 1}
    assert f"1 of {summary['documents']} documents embedded" in line


# -- where there is no tier -------------------------------------------------


def test_without_a_vector_tier_the_leg_is_absent_not_broken(
    engine: Engine, fake: FakeGitHub
) -> None:
    if has_vector_tier(engine):
        pytest.skip("this dialect has a vector tier")
    _paraphrase_corpus(fake)
    _index(engine, fake, 1, 2)
    trace: list = []

    search_best(open_backend(engine), _query(text=PARAPHRASE), embedder=FakeEmbedder(), trace=trace)

    assert SEMANTIC_LEG not in [leg.name for leg in trace]
    with engine.begin() as conn, pytest.raises(RuntimeError, match="no vector tier"):
        write_embeddings(conn, MODEL, [(1, "h", [0.0] * DIMENSIONS)])


def test_the_poll_hook_embeds_newest_first_and_bounded(vectors: Engine, fake: FakeGitHub) -> None:
    """A fresh index has its whole history pending; the poll's share is what it just wrote,
    capped, so the loop keeps polling while ``pass.sh embed`` takes the history."""
    _paraphrase_corpus(fake)
    _index(vectors, fake, 1, 2)
    total = pending(vectors, REPO, MODEL)

    result = embed_pending(vectors, REPO, FakeEmbedder(), limit=1, newest_first=True)

    assert result.embedded == 1
    assert pending(vectors, REPO, MODEL) == total - 1
    with vectors.connect() as conn:
        newest = conn.execute(select(func.max(documents.c.id))).scalar()
        embedded = conn.execute(
            select(documents.c.id).where(documents.c.embedding_model == MODEL)
        ).scalar()
    assert embedded == newest
