"""The Postgres backend: ``tsvector`` + GIN, ranked by ``ts_rank_cd``.

The only supported deployment (section 4.1), and the engine section 6's weights are being
fitted to. ``documents.fts`` is a generated column created by migration 2 and deliberately
absent from ``store/schema.py``, so it is named here as a column expression rather than
read off the table.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import reduce
from typing import Any

from sqlalchemy import Select, case, func, literal, literal_column, select

from relore.search.backends.base import SearchBackend
from relore.search.queries import (
    MATCH_ALL,
    MATCH_ANY,
    SEMANTIC_DEPTH,
    SEMANTIC_POOL,
    BackendInfo,
    Hit,
    SearchQuery,
    is_plain_word,
    tokenize,
)
from relore.store import schema as s

_FTS = literal_column("documents.fts")

#: How many terms a *scoring* query may carry. A pasted traceback tokenizes into hundreds,
#: and a disjunction over all of them is both slow and undiscriminating -- every document
#: in the corpus matches `self` or `line`. The caller's first two dozen words are what the
#: question is about; section 6's signal legs carry the precise terms.
MAX_SCORE_TERMS = 24

#: ``ts_rank_cd``'s normalization flag 32: ``rank / (rank + 1)``, which maps cover density
#: into ``[0, 1)``. Only the widened score uses it, and only so that density can be made a
#: strict tie-break under coverage -- an unbounded rank cannot be.
RANK_NORM_SCALE = 32


class PostgresBackend(SearchBackend):
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="postgresql",
            ranking="ts_rank_cd",
            capabilities=frozenset({"fulltext", "weighted"}),
        )

    def _tsquery(self, text: str) -> Any:
        """Not ``websearch_to_tsquery``, whose quoted phrases, ``OR`` and ``-negation``
        have no SQLite counterpart. ``plainto_tsquery`` is a plain AND of the terms, which
        is what the FTS5 backend can also express -- so the two engines answer the same
        *question* and differ only in how they score it. Both take arbitrary user text
        without raising, which the raw ``to_tsquery`` does not.
        """
        return func.plainto_tsquery("english", text)

    def _tsquery_any(self, text: str) -> Any | None:
        """The same terms, disjoined -- or ``None`` when the text lexes to nothing.

        `tokenize` yields `[A-Za-z0-9_]+` only, so this cannot reach `to_tsquery` as
        syntax -- which is the reason the raw form is usable here at all (see
        `_tsquery`). An all-stopword query lexes to the empty tsquery and ranks 0, which
        is the right answer and not an error.
        """
        terms = tokenize(text)[:MAX_SCORE_TERMS]
        return func.to_tsquery("english", " | ".join(terms)) if terms else None

    def fts_score(self, text: str, *, match: str = MATCH_ALL) -> Any:
        """Cover density, except on a widened page, where **coverage decides and density
        only breaks ties**.

        ``ts_rank_cd`` measures how *densely* the matched lexemes sit, so a document
        saying ``cache`` nine times outranks one carrying six of the caller's seven
        distinct terms once each. Where the filter is the conjunction that is harmless --
        every surviving row carries all of them and only density is left to say -- and it
        is the ranking section 10.5 fitted, so it is untouched.

        Under :data:`~relore.search.queries.MATCH_ANY` it is wrong, and measured wrong. On
        the query that motivated the widening, against 41,857 documents of the
        ``transformers`` sample, 2 documents carry six of the seven terms, 14 carry five,
        36 carry four and 3,111 carry one. Ranked by density the page opened on a
        two-term document and the six-term one came *sixth* -- section 10.6's failure
        wearing a different hat: not wrong so much as not ranked, and invisible to
        Recall@k. So the widened score is section 6's own overlap shape, ``matched /
        asked``, over the caller's terms -- the same ``[0, 1]`` scale as every signal term
        -- with normalized density divided by ``len(terms) + 1`` underneath it. That
        divisor is what makes the order exactly lexicographic rather than approximately:
        the tie-break is strictly smaller than ``1 / len(terms)``, the narrowest gap
        coverage can produce, so no amount of density can buy a document past one that
        carries more of the question.
        """
        terms = tokenize(text)[:MAX_SCORE_TERMS] if match == MATCH_ANY else []
        if not terms:
            return func.ts_rank_cd(_FTS, self._tsquery(text))
        density = func.ts_rank_cd(
            _FTS, func.to_tsquery("english", " | ".join(terms)), RANK_NORM_SCALE
        )
        return self._coverage(terms) + density / literal(float(len(terms) + 1))

    def semantic(
        self,
        query: SearchQuery,
        vector: Sequence[float],
        model: str,
        *,
        depth: int = SEMANTIC_DEPTH,
    ) -> list[tuple[Hit, float]]:
        """The nearest documents under ``search``'s filters, one per thread, best first.

        Only rows embedded by ``model`` are read: the reconcile resets ``embedding_model``
        when a document is rewritten, so a vector of text that no longer exists is never
        served. The distance is exact -- no ANN index -- which is the right trade at the
        size this was measured on; an HNSW index is the change when it stops being.
        """
        if not query.repos:
            return []
        # Floats this method formats itself, so inlining them is not an injection path, and
        # it keeps the parameter out of a statement Postgres would otherwise re-plan.
        literal_vector = "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"
        distance = literal_column(f"(documents.embedding <=> '{literal_vector}'::vector)")
        stmt = (
            select(s.documents.c.id, s.documents.c.thread_id, distance.label("distance"))
            .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
            .where(*self._filters(query), s.documents.c.embedding_model == model)
            .order_by(distance)
            .limit(SEMANTIC_POOL)
        )
        nearest: dict[int, tuple[int, float]] = {}
        with self.engine.connect() as conn:
            for row in conn.execute(stmt):
                nearest.setdefault(int(row.thread_id), (int(row.id), float(row.distance)))
                if len(nearest) == depth:
                    break
        hits = self.hits_for_documents([doc for doc, _d in nearest.values()], query)
        return [(hits[doc], 1.0 - dist) for doc, dist in nearest.values() if doc in hits]

    def term_match(self, word: str) -> Any:
        """``to_tsquery`` with ``:*`` for a plain word, ``plainto_tsquery`` otherwise.

        The raw ``to_tsquery`` is safe here for the same reason as in :meth:`_tsquery_any`:
        a plain word is ``[a-z][a-z0-9]+``, so it cannot reach the parser as syntax. A
        stopword lexes to the empty query, matches nothing, and drops out of the leg.
        """
        if is_plain_word(word):
            return _FTS.op("@@")(func.to_tsquery("english", f"{word}:*"))
        return _FTS.op("@@")(func.plainto_tsquery("english", word))

    def _coverage(self, terms: list[str]) -> Any:
        """How many of ``terms`` this row carries at all, over how many were asked.

        Deliberately the same expression as
        :meth:`~relore.search.backends.base.SearchBackend._overlap`, which is what section
        6 scores a signal filter with: one ``case`` per value, summed, over the count. A
        term is counted once however often it occurs -- that is the whole difference from
        density, and the reason this exists.
        """
        matched = reduce(
            operator.add,
            (
                case((_FTS.op("@@")(func.to_tsquery("english", term)), 1.0), else_=0.0)
                for term in terms
            ),
        )
        return matched / literal(float(len(terms)))

    def fts_filter(self, stmt: Select, text: str, *, match: str = MATCH_ALL) -> Select:
        disjunction = self._tsquery_any(text) if match == MATCH_ANY else None
        if disjunction is not None:
            return stmt.where(_FTS.op("@@")(disjunction))
        return stmt.where(_FTS.op("@@")(self._tsquery(text)))

    def unfiltered_fts_score(self, text: str) -> Any | None:
        """Relevance to text this query is not filtering on -- and it **ORs** the terms.

        ``ts_rank_cd`` is an ordinary function of a stored ``tsvector``, so it needs no join
        and no ``@@``: a row matching nothing simply scores 0. That is what lets a signal
        leg of an expanded call prefer the rows the caller's words are about
        (``search/ranking.py``), and it is a capability the FTS5 backend does not have.

        **Filtering ANDs; scoring ORs, and conflating the two is a real bug that shipped.**
        :meth:`_tsquery` is a conjunction, which is right where it decides *admission* --
        section 10.4: it is what stops a pasted sentence matching everything. Used to
        *order* a set the text did not select, the same conjunction scores 0 for almost
        every row, so the whole page ties and falls through to the recency tie-break.

        Measured on the query that exposed it -- `--file src/transformers/masking_utils.py`,
        `kind=rationale`, 861 admissible documents on the matching threads: the conjunction
        ranks **1** of them above zero, the disjunction 126. The other 860 tied, and the
        recency fallback then reliably picked each thread's *last* document, which on a
        merged pull request is the approval. Nine of ten slots came back ``LGTM``, ``Thx``,
        ``Yep`` -- retrieval that is not wrong so much as not ranked, and invisible to
        Recall@k because the right thread was still on the page.

        A disjunction is the correct scoring question anyway: "how much of what they asked
        about does this document carry" is graded, and section 6 already spends the
        conjunction on the ``text`` leg where every surviving row satisfies it.
        """
        disjunction = self._tsquery_any(text)
        if disjunction is None:
            return None
        # Both, not one: a document carrying *every* term really is stronger evidence than
        # one carrying some, and the conjunction is what says so. The disjunction then
        # orders everything underneath it instead of leaving it tied at zero. Measured, the
        # sum and the disjunction alone are within a thousandth of each other on every
        # slice -- the conjunction is nonzero too rarely to reorder much -- so this is
        # chosen for using both signals rather than for the number (section 10.6).
        #
        # **Not** the widened score, though it answers a related question. This one was
        # fitted against the frozen set (section 10.5) and every signal leg of every
        # expanded call is ordered by it; `fts_score`'s `MATCH_ANY` branch is reachable
        # only from a page that came back empty, and changing one to match the other would
        # move numbers that are the record.
        return func.ts_rank_cd(_FTS, self._tsquery(text)) + func.ts_rank_cd(_FTS, disjunction)

    def age_days(self, column: Any) -> Any:
        return func.extract("epoch", func.now() - column) / literal(86400.0)
