"""The SQLite backend: FTS5, ranked by ``bm25``. A development affordance, not a
deployment (section 4.1).

Two things about it are load-bearing, and both are traps rather than preferences.

**The score is negated.** FTS5's ``bm25()`` returns a value where *more negative* is more
relevant. :meth:`SqliteBackend.fts_score` flips it so "higher is better" holds on both
backends and the caps in ``queries.py`` need no dialect knowledge.

**It cannot rank text it did not MATCH.** ``bm25()`` is an auxiliary function of the FTS5
index, so it is an error anywhere but a query that joined and matched it -- which means
:meth:`SqliteBackend.unfiltered_fts_score` has to return ``None`` and section 6's lexical
term is simply absent from a textless expansion leg here. The signal overlaps and the decay
still score, so those legs are ordered rather than tied; they are ordered by less. Section
4.1 already refuses to compare a number across backends, and this is one of the reasons.

**User text never reaches ``MATCH`` as written.** ``MATCH`` takes a query *language* --
``"``, ``*``, ``:``, ``^``, ``-``, ``(`` and the bare words ``AND``/``OR``/``NOT`` all
mean something in it, and an unbalanced quote is a hard error rather than an empty result.
A pasted traceback contains most of those characters. So the text is tokenized and
re-emitted as quoted terms, which is also what makes this an AND of the terms and
therefore the same question ``plainto_tsquery`` answers on the other side.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, literal, literal_column, select, table
from sqlalchemy import text as sa_text

from relore.search.backends.base import SearchBackend
from relore.search.queries import MATCH_ALL, MATCH_ANY, BackendInfo, is_plain_word, tokenize
from relore.store import schema as s

#: Created by migration 3, external-content over ``documents``, and deliberately not in
#: ``store/schema.py``: it is not part of the portable core, and ``metadata.create_all``
#: must never try to make it on Postgres.
documents_fts = table("documents_fts")

_MATCH = sa_text("documents_fts MATCH :fts_query")
_BM25 = literal_column("bm25(documents_fts)")


class SqliteBackend(SearchBackend):
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="sqlite", ranking="bm25", capabilities=frozenset({"fulltext", "weighted"})
        )

    def fts_score(self, text: str, *, match: str = MATCH_ALL) -> Any:
        """``bm25`` either way, and under :data:`MATCH_ANY` it is already the graded score.

        The Postgres side has to *add* a disjunction to keep a widened page ordered (see
        that backend); here the widened MATCH is itself the disjunction, and bm25 over it
        scores a row carrying four of the terms above one carrying two. The two engines
        order a widened page by different numbers, which section 4.1 already says of every
        number here.
        """
        if not _match_expression(text, match=match):
            return literal(0.0)
        return -_BM25

    def fts_filter(self, stmt: Select, text: str, *, match: str = MATCH_ALL) -> Select:
        expression = _match_expression(text, match=match)
        if not expression:
            # Every term was punctuation. An empty MATCH is a syntax error in FTS5, and
            # "no results" is the honest answer, so ask for a contradiction rather than
            # for everything.
            return stmt.where(literal(False))
        return stmt.join(
            documents_fts, literal_column("documents_fts.rowid") == s.documents.c.id
        ).where(_MATCH.bindparams(fts_query=expression))

    def unfiltered_fts_score(self, text: str) -> Any | None:
        """Nothing. See the module docstring: FTS5 refuses ``bm25()`` outside a MATCH."""
        return None

    def term_match(self, word: str) -> Any:
        """The word quoted, and starred when it is plain -- FTS5's prefix query.

        Quoted like every other term this module emits, so the only operator in the
        expression is the ``*`` added here. A subquery rather than a join, because
        :meth:`~relore.search.backends.base.SearchBackend.coverage` asks several of these
        in one statement and ``MATCH`` takes one expression per table reference.
        """
        pattern = f'"{word}"*' if is_plain_word(word) else f'"{word}"'
        return s.documents.c.id.in_(
            select(literal_column("rowid"))
            .select_from(documents_fts)
            .where(literal_column("documents_fts").op("MATCH")(literal(pattern)))
        )

    def age_days(self, column: Any) -> Any:
        """``julianday`` is in days already, and ``'now'`` is UTC -- which is what the
        column holds, since ``store/dialect.UTCDateTime`` normalizes on the way in."""
        return func.julianday(literal("now")) - func.julianday(column)


def _match_expression(query: str, *, match: str = MATCH_ALL) -> str:
    """User text as an FTS5 ``MATCH`` expression: every term quoted, implicitly ANDed.

    Quoting is what neutralizes the query language, so a pasted traceback searches for its
    own words instead of erroring on its own punctuation.

    Under :data:`~relore.search.queries.MATCH_ANY` the terms are joined by an explicit
    ``OR`` -- the one place this module *uses* the query language instead of neutralizing
    it. The terms stay quoted, so that connective is the only operator in the expression.
    """
    terms = [f'"{term}"' for term in tokenize(query)]
    return (" OR " if match == MATCH_ANY else " ").join(terms)
