"""Section 6's query expansion: one call fans out into several, and the results merge.

The plain search ANDs every content term, which is what stops a pasted sentence matching
everything -- and what makes a pasted *traceback* match nothing. Expansion is the other
half of that contract: a caller sends one thing, the server asks the index the four or
five questions that thing actually contains, and merges the answers.

**Why this comes before the weights.** Section 10.3 measured the leak-free rationale slice
and `relore` lost to both baselines, but not by ranking badly: 6 of 9 queries returned
*nothing at all*. The query is drawn from a bot's review finding, and in 10 of the 14
examples carrying text terms the only document in the answer's thread holding all of them
is that bot finding -- which section 6.2 excludes from reads. So the AND asked for the one
document the index refuses to return. A weight set cannot reorder an empty result; running
the legs independently and merging is what reaches the answer at all.

**The legs**, in the order section 6 names them:

* ``text``   -- the caller's query exactly as sent, so expansion can never *lose* a result
  the plain search would have found. Everything else is additive.
* ``error``  -- an error line pulled out of the pasted text, in section 5.3's normal form.
* ``test``   -- a pytest node id.
* ``symbol`` -- an identifier.
* ``file``   -- a path.
* ``filters`` -- the caller's own signal filters with the free text dropped.

Every leg keeps the caller's explicit filters, because those are scope they chose; only the
free text moves. The signal legs carry no free text at all, which is the whole point: a
term set that ANDs to nothing still reaches its threads through an overlap filter.

**The floor under a textless leg is gone, and the weights are what removed it.** This
module used to note that with no text every score is 0, so a signal leg's tie broke
arbitrarily -- section 10.4 #1 measured exactly that as ``precedent`` recall rising a tenth
while its MRR stayed flat. Section 6's weighted score fixes it from the other side:
:func:`rank_spec` hands every leg the *whole* question to score against while leaving its
filters alone, so a row is ordered by how much of the call it carries even when the leg
that found it matched one path and no words. Decay by query kind arrived with the same
change and is applied by the backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from relore.ingest.extract import extract_signals, is_symbol
from relore.search.queries import (
    MATCH_ANY,
    MAX_HITS,
    MAX_HITS_PER_THREAD,
    Hit,
    SearchQuery,
    tokenize,
)
from relore.search.ranking import RankSpec

if TYPE_CHECKING:  # pragma: no cover - a type, not a dependency
    from relore.search.backends import SearchBackend

#: Section 6 names five legs; ``filters`` is the sixth and only appears when the caller
#: supplied one. The cap bounds how many round trips one call can cost -- a traceback
#: naming twenty frames is one question, not twenty.
MAX_LEGS = 6

#: How many signals of one kind become legs. A traceback's *last* frame is the one that
#: raised, and section 5.3 sorts signals rather than preserving frame order, so taking
#: more than a couple here buys distractors rather than recall.
MAX_PER_KIND = 2

#: Reciprocal-rank fusion's constant. **Deliberately the literature default and
#: deliberately untuned**: it was set from first principles before section 10.4's first
#: run and not touched afterwards, and it stayed untouched through the weights. The set is
#: frozen now, so moving it is finally *permitted* -- which is a reason to do it against a
#: measurement, not a reason to do it.
RRF_K = 60

#: The legs, in section 6's order. Order is the tie-break when two hits fuse equal, so it
#: is the one place a leg is preferred over another -- and the caller's own text wins.
LEG_NAMES = ("text", "error", "test", "symbol", "file", "filters")

#: What :func:`search_best` reports when it had to widen. One value today, and a name
#: rather than a boolean so a second rung can be added without changing the field.
WIDENED_ANY_TERM = "any-term"

#: The shortest query worth widening. One term already *is* its own disjunction, so
#: widening it can only return the same page under a different name.
MIN_WIDEN_TERMS = 2


@dataclass(frozen=True)
class Leg:
    """One question the expansion decided the call contained."""

    name: str
    query: SearchQuery
    #: What it was expanded from, for section 8's "why did this come back?" view.
    term: str = ""


def expand(query: SearchQuery) -> tuple[Leg, ...]:
    """Fan one request out into section 6's legs. Always at least the caller's own.

    Extraction runs through :func:`relore.ingest.extract.extract_signals` -- the *same*
    function that filled the tables these legs filter on. That pairing is not a
    convenience: an error term matches only in section 5.3's normal form, so a leg that
    normalized the pasted line differently would filter on a string the index does not
    hold, and match nothing in a way indistinguishable from a quiet corpus.
    """
    legs = [Leg(name="text", query=query, term=query.text)]
    text = query.text.strip()
    if not text:
        return tuple(legs)

    signals = extract_signals([{"body_markdown": text, "body_text": text, "metadata": {}}])
    derived: list[tuple[str, str, dict[str, tuple[str, ...]]]] = []
    for message in _take(row["message_norm"] for row in signals.errors):
        derived.append(("error", message, {"errors": (message,)}))
    for test_id in _take(row["test_id"] for row in signals.tests):
        derived.append(("test", test_id, {"tests": (test_id,)}))
    for symbol in _take(_symbols(text, signals)):
        derived.append(("symbol", symbol, {"symbols": (symbol,)}))
    for path in _take(row["path"] for row in signals.files):
        derived.append(("file", path, {"files": (path,)}))

    for name, term, override in derived:
        # No free text on a signal leg. That is the fix, not an omission: the caller's
        # terms are what ANDed to nothing, and the overlap filter is thread-level.
        field = next(iter(override))
        if getattr(query, field):
            # The caller already scoped this kind. ``replace(..., **override)`` would
            # drop that constraint (huggingface/relore#75); appending would OR-widen
            # it, because values inside one signal filter are disjoined. Skip the
            # derived leg -- the text and filters-only legs still carry the evidence.
            continue
        legs.append(Leg(name=name, query=replace(query, text="", **override), term=term))

    # The caller's own filters, asked without their text. Section 10.3's control: on the
    # leak-free rationale slice this alone scores 0.857 and 1.000 with nothing empty,
    # because the answer thread is reachable and it was the text that excluded it.
    if query.has_signal_filter:
        legs.append(Leg(name="filters", query=replace(query, text=""), term="(filters only)"))

    return tuple(_dedupe_legs(legs))[:MAX_LEGS]


def search_expanded(backend: SearchBackend, query: SearchQuery) -> list[Hit]:
    """Run every leg, fuse the rankings, and re-apply section 6's caps across the union.

    The per-thread cap has to be re-applied here even though each leg's SQL already
    enforced it (``backends/base._select``): two legs can each return the same thread
    under a different ``source_type``, and three legs agreeing is exactly the case the
    fusion is meant to reward -- so without a second pass one thread reclaims the page
    that the window functions were written to protect.
    """
    legs = expand(query)
    if len(legs) == 1:
        # Nothing was expanded, so there is nothing to fuse and no reason to pay for a
        # second pass over the same rows.
        return backend.search(query)

    spec = rank_spec(query, legs)
    fused: dict[tuple[str, int, str], _Fused] = {}
    for order, leg in enumerate(legs):
        # Each leg capped, then merged (section 6). A leg is asked for the full page
        # rather than the caller's limit: a hit at rank 9 of two legs outranks a hit at
        # rank 1 of one, and clipping the legs first would hide it.
        for rank, hit in enumerate(
            backend.search(replace(leg.query, limit=MAX_HITS), rank_as=spec), start=1
        ):
            key = (hit.repo, hit.number, hit.source_type)
            entry = fused.get(key)
            if entry is None:
                entry = fused[key] = _Fused(hit=hit, leg_order=order)
            entry.score += 1.0 / (RRF_K + rank)
            entry.legs.append(leg.name)
            # Keep the chunk the strongest leg surfaced, which is section 6's "best
            # chunk per (thread, source_type)" applied across the union rather than
            # inside one query.
            if order < entry.leg_order:
                entry.hit, entry.leg_order = hit, order

    ranked = sorted(fused.values(), key=lambda e: (-e.score, e.leg_order, e.hit.repo, e.hit.number))
    out: list[Hit] = []
    per_thread: dict[tuple[str, int], int] = {}
    for entry in ranked:
        thread = (entry.hit.repo, entry.hit.number)
        if per_thread.get(thread, 0) >= MAX_HITS_PER_THREAD:
            continue
        per_thread[thread] = per_thread.get(thread, 0) + 1
        out.append(_explained(entry))
        if len(out) == query.limit:
            break
    return out


def rank_spec(query: SearchQuery, legs: Sequence[Leg]) -> RankSpec:
    """The one question every leg is *scored* against, however it was filtered.

    This is the half of section 6's weights that expansion needs and could not have
    without them (see ``ranking.py``). Each leg filters on a single derived term and
    carries no free text, so scored against itself every row in it overlaps exactly one
    value and the order is a tie -- which is precisely what section 10.4 #1 measured when
    ``precedent`` recall rose a tenth and its MRR did not move.

    So the spec unions what the caller filtered on with what expansion pulled out of their
    text, and keeps the caller's own words for the lexical term. A thread carrying three of
    the derived symbols then outranks one carrying a single one, inside every leg, with no
    text having matched anywhere.

    The terms are evidence, not scope: widening the *scoring* question cannot admit a row
    the leg's own filters excluded, because the filters are unchanged.
    """
    spec = RankSpec.of(query)
    for leg in legs:
        spec = spec.with_signals(RankSpec.of(leg.query))
    return spec


@dataclass
class _Fused:
    hit: Hit
    leg_order: int
    score: float = 0.0
    legs: list[str] = field(default_factory=list)


def _explained(entry: _Fused) -> Hit:
    """The fused rank, and how many legs agreed, in the hit's own breakdown.

    Section 6 puts every term of the score in ``breakdown`` for the person tuning it. A
    merged result whose provenance is invisible is one nobody can argue with, and the leg
    count is the only part of this that is not the plain score.
    """
    breakdown = dict(entry.hit.breakdown)
    breakdown["fused"] = round(entry.score, 6)
    breakdown["legs"] = float(len(entry.legs))
    return replace(entry.hit, score=entry.score, breakdown=breakdown)


def _symbols(text: str, signals) -> list[str]:
    """Identifiers in the query, whether or not the caller wrote them in backticks.

    Indexing takes symbols out of code spans only, and that is right for prose: a bare
    lowercase word in a comment is usually emphasis, so ``is_symbol`` alone would admit
    half a sentence. **A query is not prose.** Nobody types four words to emphasise one,
    so a query token shaped like an identifier *is* the subject of the question -- and
    requiring backticks means ``relore search "_maybe_import_sdnq __new__"`` expands to
    nothing, which is section 10.3's failing case exactly.

    Two things are dropped rather than asked twice, and both are the argument
    ``is_symbol`` already makes for not counting a path as a symbol:

    * **the exception type.** ``ValueError`` passes every shape test and matches half a
      corpus on its own, and the error leg holds it in a more precise form.
    * **a path's own components.** ``tokenize`` shreds
      ``src/models/foo/modeling_foo.py`` into words, and ``modeling_foo`` then looks
      exactly like an identifier -- but it is a *file*, it already has a file leg, and
      asking it as a symbol spends a leg to fuse the same evidence with itself.
    """
    raised = {row["exception_type"] for row in signals.errors if row["exception_type"]}
    from_paths = {token for row in signals.files for token in tokenize(row["path"])}
    found = [row["symbol"] for row in signals.symbols if row["symbol"] not in raised]
    seen = set(found) | raised | from_paths
    for token in tokenize(text):
        if token not in seen and is_symbol(token):
            seen.add(token)
            found.append(token)
    return found


def _take(values) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
        if len(out) == MAX_PER_KIND:
            break
    return out


def _dedupe_legs(legs: Sequence[Leg]) -> list[Leg]:
    """Two legs asking the same question cost a round trip and skew the fusion.

    They fuse as two votes, which would let a duplicate outrank a hit that genuinely
    came back from two different questions.
    """
    out: list[Leg] = []
    seen: set[tuple] = set()
    for leg in legs:
        key = (
            leg.query.text,
            leg.query.files,
            leg.query.symbols,
            leg.query.errors,
            leg.query.tests,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(leg)
    return out


def search_best(
    backend: SearchBackend, query: SearchQuery, *, expand: bool = True
) -> tuple[list[Hit], str]:
    """Section 6's search, asked so that a query with terms in it never dead-ends.

    **The measured problem.** Expansion fixed the pasted *traceback* -- a signal leg
    reaches the thread the AND could not. It does nothing for pasted *prose*, because
    prose derives no signals: :func:`expand` returns the caller's own leg alone and the
    conjunction is the whole call. Across the three head-to-head runs in
    ``relore/docs/gh-vs-relore-2026-09-14.md`` every zero-result page had that shape --
    six to nine ordinary words, no error, no path, no identifier --

        search 'FA2 static cache fullgraph compile generate override'   -> 0 hits

    where dropping half the terms answers at rank 1. And a zero costs a *turn*: the agent
    reformulates, the whole conversation is re-sent, ~59k prompt tokens for a page that
    said nothing. That is the same cost the ``why`` dead end had (section 13.3) and the
    same shape :meth:`~relore.search.backends.base.SearchBackend._focused` fixed inside
    one thread -- here it is the last place the empty page survived.

    **So: ask again with the terms disjoined, and only then.** The strict conjunction runs
    first and unchanged, so no query that had an answer can be widened out of one; the
    widening is reached only from an empty page, where there is nothing to lose. The score
    keeps all-terms rows first (both backends), so what widening adds is a tail rather
    than a reshuffle.

    **It is disclosed, always.** The second return value is what the page says it did, and
    it is set whenever widening was *tried* -- including when it also came back empty,
    which is the more useful of the two answers: it tells the caller their words are not
    the problem and reformulating them again will not help.
    """

    def ask(asked: SearchQuery) -> list[Hit]:
        return search_expanded(backend, asked) if expand else backend.search(asked)

    hits = ask(query)
    if hits or len(tokenize(query.text)) < MIN_WIDEN_TERMS:
        return hits, ""
    return ask(replace(query, match=MATCH_ANY)), WIDENED_ANY_TERM
