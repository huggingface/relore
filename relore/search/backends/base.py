"""What both backends share, and the five things they do not.

Section 4.1: the portable core behaves identically on both dialects, but **the search
layer does not port** -- ``tsvector``+GIN is not FTS5, and ``ts_rank_cd`` is not ``bm25``.
So filters, scoping, caps, section 6's weighted score and the thread view live here, once,
and each subclass supplies only what its engine expresses differently:

* :meth:`SearchBackend.info` -- which engine, and what it can answer
* :meth:`SearchBackend.fts_score` -- the relevance expression
* :meth:`SearchBackend.fts_filter` -- the predicate, and any join it needs
* :meth:`SearchBackend.unfiltered_fts_score` -- the same relevance, for text this query is
  *not* filtering on. Postgres can; FTS5 cannot, and says so by returning ``None``.
* :meth:`SearchBackend.age_days` -- how old a row is, for section 6's decay

Everything else is ordinary SQL over the shared connection, which is why a filter bug
cannot be dialect-specific. Section 6's *weights* are shared for the same reason: the
overlap terms are counted with the same predicates the filters use, so a term can never
score what the filter would not have matched.
"""

from __future__ import annotations

import datetime as dt
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Any, NamedTuple

from sqlalchemy import (
    Engine,
    Select,
    and_,
    case,
    distinct,
    exists,
    func,
    literal,
    nulls_last,
    or_,
    select,
)

from relore.ingest.extract import CHANGED, FROM_COMMENT, MENTIONED
from relore.search.queries import (
    AFTER_CURSOR,
    BODY_SLACK_CHARS,
    COMMENT_VIEW,
    ENDS_AND_MIDDLE,
    HUMAN_TRUST,
    MACHINE_TRUST,
    MATCH_ALL,
    MAX_BODY_CHARS,
    MAX_CLAIMS,
    MAX_HITS_PER_THREAD,
    MAX_OUTLINE_COMMENTS,
    MAX_SNIPPET_CHARS,
    MAX_THREAD_COMMENTS,
    OUTLINE_CHARS,
    OUTLINE_VIEW,
    PASSAGE_OVERFETCH,
    THREAD_ENDS,
    BackendInfo,
    Claim,
    Hit,
    InflightView,
    SearchQuery,
    ThreadView,
    WhyView,
    admissible_trust,
    render_age,
    render_stamp,
    snippet,
    tokenize,
    trust_policy,
)
from relore.search.ranking import WEIGHTS, RankSpec, half_life
from relore.store import schema as s


class _Comments(NamedTuple):
    """What one thread's comment query produced: a page, or an outline standing in for one.

    A tuple of six positional values grew a seventh and an eighth with the outline's focus
    (huggingface/relore#71), and the caller unpacking them by position is the kind of line
    where a reordering is silent. Named, and private to this module.
    """

    hits: tuple[Hit, ...]
    total: int
    matched: int | None
    selection: str
    suppressed: int
    outline: tuple[Hit, ...] = ()
    outline_matched: int | None = None
    outline_widened: bool = False
    comment_status: str = ""


def _written_before(before: dt.datetime | None) -> tuple[Any, ...]:
    """Section 13's cutoff over ``documents``, as predicates to splat into a ``where``.

    Fails closed twice over. A NULL ``github_created_at`` compares NULL rather than TRUE,
    so an undated document is dropped; and an absent cutoff returns no predicate at all,
    so a caller that forgets one is not silently given a bound it did not ask for. Both
    directions matter: this is the filter that makes an evaluation honest, and the way it
    breaks is by quietly doing nothing.
    """
    return () if before is None else (s.documents.c.github_created_at < before,)


def _opened_before(before: dt.datetime | None) -> tuple[Any, ...]:
    """The same, over ``threads``. A thread opened at or after the cutoff did not exist."""
    return () if before is None else (s.threads.c.created_at < before,)


def _reachable(
    conn: Any,
    repo: str,
    number: int,
    before: dt.datetime | None,
    exclude: Sequence[int],
) -> bool:
    """Whether one thread may be named at all, by number, under this cutoff."""
    if number in tuple(exclude):
        return False
    if before is None:
        return True
    return (
        conn.execute(
            select(s.threads.c.id).where(
                s.threads.c.repo == repo,
                s.threads.c.github_number == number,
                *_opened_before(before),
            )
        ).one_or_none()
        is not None
    )


class SearchBackend(ABC):
    """Read-only retrieval over one engine.

    The engine handed in is expected to be the read-only one (section 11). Nothing here
    writes, and nothing here is allowed to: the API's whole connection is ``SELECT``-only
    on both dialects, so a write would fail at the driver rather than be caught by review.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # -- the dialect-specific three --------------------------------------

    @abstractmethod
    def info(self) -> BackendInfo: ...

    @abstractmethod
    def fts_score(self, text: str, *, match: str = MATCH_ALL) -> Any:
        """A column expression: higher is more relevant.

        ``match`` is the *admission* question the score is ordering the answers to, and
        under :data:`~relore.search.queries.MATCH_ANY` the two stop agreeing: a row that
        carries two of seven terms is on the page, and the conjunction scores it 0 along
        with every other partial row. So a widened score has to be graded, and both
        backends fold the disjunction in -- which is the same correction
        :meth:`unfiltered_fts_score` documents for a different reason.
        """

    @abstractmethod
    def fts_filter(self, stmt: Select, text: str, *, match: str = MATCH_ALL) -> Select:
        """Narrow ``stmt`` to rows matching ``text``, adding whatever join that needs.

        ``match`` chooses the connective: :data:`~relore.search.queries.MATCH_ALL` is the
        conjunction every corpus-wide search is asked with, and
        :data:`~relore.search.queries.MATCH_ANY` is the widening the fallback asks for
        after it came back empty.
        """

    @abstractmethod
    def unfiltered_fts_score(self, text: str) -> Any | None:
        """Relevance to ``text`` for a query that is **not** filtering on it, or ``None``.

        Section 6's expansion runs signal legs that carry no free text at all, and the
        caller's words are still evidence about which of those rows is the best one. On
        Postgres ``ts_rank_cd`` is an ordinary expression and can say so. On SQLite it
        cannot: ``bm25()`` is an auxiliary function of the FTS5 index and is an error
        anywhere but a query that MATCHed it, so that backend returns ``None`` and the
        lexical term is simply absent from those legs. Section 4.1 already says a number
        from one backend describes a different engine; this is one of the places it does.
        """

    @abstractmethod
    def age_days(self, column: Any) -> Any:
        """Age of ``column`` in days, as a float expression. Section 6's decay input."""

    # -- search ----------------------------------------------------------

    def search(self, query: SearchQuery, *, rank_as: RankSpec | None = None) -> list[Hit]:
        """Section 6: full text, filters, the trust floor, and the weighted score.

        ``rank_as`` is "filter by the leg, score by the whole question" (see
        ``ranking.py``). Expansion passes the caller's own terms plus everything it derived
        from them, so a leg that filters on one path still orders its rows by how much of
        the *call* each thread carries. Left ``None`` -- every caller but expansion -- a
        query is scored against itself, which is what it always meant.

        No results is a successful call with nothing in it. That includes the case where
        the token's scope is empty -- section 11 fails closed, so *no* repository in scope
        means nothing, never everything.
        """
        if not query.repos:
            return []
        spec = rank_as or RankSpec.of(query)
        with self.engine.connect() as conn:
            return [self._hit(row, query) for row in conn.execute(self._select(query, spec))]

    def files_blind_spot(self, query: SearchQuery) -> int:
        """How many threads ``--file`` could not *test*, rather than did not match.

        A thread with no changed-file row is indistinguishable, in the filter, from one
        that never touched the path: both are simply absent. ``thread`` discloses that per
        row; this is the same disclosure for the aggregate.

        **Pull requests only.** An issue has no diff and never will, so its absence from a
        ``--file`` page is the right answer rather than a gap -- and counting issues buries
        the real number in them: on ``transformers`` it is 19,460 issues against 4 open PRs
        actually missing a list.

        Every filter *except* ``--file`` is applied, the text match included, so the number
        means "pull requests this query otherwise matched, which have no collected
        changed-file list" rather than a corpus-wide total identical on every page.
        """
        if not query.files or not query.repos:
            return 0
        uncollected = ~exists().where(
            and_(
                s.thread_files.c.thread_id == s.documents.c.thread_id,
                s.thread_files.c.source == CHANGED,
            )
        )
        stmt = (
            select(func.count(distinct(s.documents.c.thread_id)))
            .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
            .where(
                *self._filters(query, include_files=False),
                s.threads.c.thread_type == "pr",
                uncollected,
            )
        )
        if query.text.strip():
            stmt = self.fts_filter(stmt, query.text)
        with self.engine.connect() as conn:
            return int(conn.execute(stmt).scalar() or 0)

    def _select(self, query: SearchQuery, spec: RankSpec | None = None) -> Select:
        """Section 6's dedup and per-thread cap, in SQL rather than after the fetch.

        This has to be the database's job, and the reason is a bug that looked like a
        passing test. Fetching N rows and capping them in Python caps the wrong set: a
        thread with forty matching comments fills the fetch before a second thread is ever
        read, so the cap has nothing left to choose and the page is one thread regardless.
        Two window functions -- ``row_number`` over ``(thread, source_type)`` and then over
        the thread -- mean the ``LIMIT`` applies to rows that already survived both, which
        is correct at any corpus size.

        The score is materialized in its own subquery first, and that layer is not
        cosmetic: FTS5 refuses its auxiliary functions anywhere but a direct query over the
        index ("unable to use function bm25 in the requested context"), so a window that
        orders by ``bm25()`` is an error rather than a slow plan. Ranking by an already
        computed column is portable and gives both dialects the same shape.
        """
        spec = spec or RankSpec.of(query)
        filtering_on_text = bool(query.text.strip())
        terms = self._score_terms(query, spec)
        score = self._weighted(terms, spec)
        scored = score is not None

        stmt = (
            select(
                s.documents.c.id,
                s.documents.c.thread_id,
                s.threads.c.repo,
                s.threads.c.github_number,
                s.threads.c.thread_type,
                s.threads.c.title,
                s.documents.c.source_type,
                # Which comment, and which piece of it: a hit is a *document*, and a long
                # comment is several. Two of them on one page share a URL and read as two
                # people agreeing unless the page can say otherwise (huggingface/relore#18).
                s.documents.c.source_id,
                s.documents.c.chunk_index,
                s.documents.c.url,
                s.documents.c.author,
                s.documents.c.trust,
                s.documents.c.body_text,
                s.documents.c.github_created_at,
                (score if scored else literal(0.0)).label("score"),
                # Section 8's per-hit score breakdown: every term as its own number, for
                # the person tuning the weights. They are already computed for the sum, so
                # carrying them out costs nothing and is the only way a ranking complaint
                # can be specific rather than "the order looks wrong".
                *(expression.label(f"term_{name}") for name, expression in terms.items()),
            )
            .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
            .where(*self._filters(query))
        )
        if filtering_on_text:
            stmt = self.fts_filter(stmt, query.text, match=query.match)
        rows = stmt.subquery("scored")

        chunks = select(
            rows,
            func.row_number()
            .over(
                partition_by=[rows.c.thread_id, rows.c.source_type],
                order_by=_order(rows, scored),
            )
            .label("chunk_rank"),
        ).subquery("chunks")

        best = (
            select(
                chunks,
                func.row_number()
                .over(partition_by=[chunks.c.thread_id], order_by=_order(chunks, scored))
                .label("thread_rank"),
            )
            .where(chunks.c.chunk_rank == 1)
            .subquery("best")
        )
        return (
            select(best)
            .where(best.c.thread_rank <= MAX_HITS_PER_THREAD)
            .order_by(*_present_order(best, scored, query.sort))
            .limit(query.limit)
        )

    # -- section 6's score -----------------------------------------------

    def _score_terms(self, query: SearchQuery, spec: RankSpec) -> dict[str, Any]:
        """Every term of section 6's sum that this call can express, unweighted.

        A term is *absent* rather than zero when the question does not contain it: a query
        with no error in it should not carry an ``error`` column of zeros through three
        subqueries, and section 8's breakdown reads better when it lists what was actually
        asked. A weight of 0 removes a term for the same reason -- which is how
        ``relationship`` stays out of the SQL until milestone 4 fills ``thread_links``.
        """
        candidates = {
            "error": (
                WEIGHTS.error,
                self._overlap(s.thread_errors, s.thread_errors.c.message_norm, spec.errors, True),
            ),
            "test": (
                WEIGHTS.test,
                self._overlap(s.thread_tests, s.thread_tests.c.test_id, spec.tests),
            ),
            "symbol": (
                WEIGHTS.symbol,
                self._overlap(s.thread_symbols, s.thread_symbols.c.symbol, spec.symbols),
            ),
            "file": (
                WEIGHTS.file,
                self._overlap(s.thread_files, s.thread_files.c.path, spec.files),
            ),
            "lexical": (WEIGHTS.lexical, self._lexical(query, spec)),
        }
        return {
            name: expression
            for name, (weight, expression) in candidates.items()
            if weight and expression is not None
        }

    def _weighted(self, terms: dict[str, Any], spec: RankSpec) -> Any | None:
        """Section 6's sum, decayed. ``None`` when there is nothing to score at all.

        Returning ``None`` rather than a bound zero is what keeps ``_order`` honest: a
        query with no terms is not a query whose every row ties, it is one that should come
        back newest-first and say so.
        """
        if not terms:
            return None
        total = reduce(
            operator.add,
            (literal(getattr(WEIGHTS, name)) * expression for name, expression in terms.items()),
        )
        decay = self._decay(spec.kind)
        return total if decay is None else total * decay

    def _overlap(
        self, table: Any, column: Any, values: tuple[str, ...], fuzzy: bool = False
    ) -> Any | None:
        """How much of what was asked this thread carries, in ``[0, 1]``.

        A *fraction* rather than a count, so the term is comparable across questions of
        different sizes: a thread matching both of two requested files should not outrank
        one matching all four of four. Each value is tested with the **same predicate the
        filter uses** (:meth:`_thread_has`), which is not tidiness -- section 13.2 #3
        records what happens when the two sides of a term disagree about normal form, and
        sharing the predicate makes disagreeing impossible.
        """
        if not values:
            return None
        matched = reduce(
            operator.add,
            (
                case((self._thread_has(table, column, (value,), fuzzy=fuzzy), 1.0), else_=0.0)
                for value in values
            ),
        )
        return matched / literal(float(len(values)))

    def _lexical(self, query: SearchQuery, spec: RankSpec) -> Any | None:
        """The full-text term, from whichever side of the join is available.

        When the query filters on its own text the FTS join exists and both engines can
        score it. When it does not -- every signal leg of an expanded call -- only a
        backend that can rank text it did not MATCH has anything to say, which is what
        :meth:`unfiltered_fts_score` answers.
        """
        if query.text.strip():
            return self.fts_score(query.text, match=query.match)
        if spec.text.strip():
            return self.unfiltered_fts_score(spec.text)
        return None

    def _decay(self, kind: str | None) -> Any | None:
        """Section 6's kind-aware decay as a column expression, or ``None`` for no decay.

        ``rationale`` has no half-life on purpose -- the oldest thread is often the answer
        -- and neither does an unspecified kind, because decay is the only factor that can
        demote a correct old result and the safe default is to leave the order alone.

        Two guards that are not decoration. A row with no timestamp decays by nothing
        rather than to nothing: an unknown date is missing evidence, not old evidence. And
        a *future* timestamp is clamped to age zero rather than allowed to push the
        denominator below 1 and score above the undecayed maximum -- GitHub's clock is not
        ours, and a skewed row must not be able to buy rank with it.
        """
        life = half_life(kind)
        if life is None:
            return None
        age = self.age_days(s.documents.c.github_created_at)
        fresh = case((age > literal(0.0), age), else_=literal(0.0))
        return func.coalesce(literal(1.0) / (literal(1.0) + fresh / literal(life)), literal(1.0))

    @staticmethod
    def _trust_filter(query: SearchQuery) -> Any:
        """Section 6.2's floor, including its one disjunction.

        A precedent query answers from the ``authoritative`` tier *or* from any human tier
        on a merged pull request, because merging is the maintainer's act and authority
        attaches to the merge rather than to the author. Expressed here rather than in
        ``queries.py`` so that module stays free of a database import.
        """
        policy = trust_policy(query.trust, query.kind)
        condition = s.documents.c.trust.in_(policy.tiers)
        if policy.merged_pr_is_precedent:
            condition = or_(
                condition,
                and_(
                    s.threads.c.merged_at.isnot(None),
                    s.documents.c.trust.in_(HUMAN_TRUST),
                ),
            )
        return condition

    def _filters(self, query: SearchQuery, *, include_files: bool = True) -> list[Any]:
        """Every predicate that is not full text.

        Repo scoping is first and unconditional: section 11 applies it here, in the query
        layer, rather than in a handler, so that adding an endpoint cannot drop it.
        """
        where: list[Any] = [
            s.threads.c.repo.in_(query.repos),
            self._trust_filter(query),
        ]
        if query.since is not None:
            where.append(s.documents.c.github_created_at >= query.since)
        # Section 13's temporal cutoff, and it fails closed on its own. A NULL
        # `github_created_at` compares NULL rather than TRUE, so a document whose date this
        # index does not know is *excluded* from an `as_of` query instead of admitted to
        # it. That is the right way round: an unknown date cannot be shown to predate the
        # cutoff, and a leakage rule that admits what it cannot verify is not a rule. It is
        # the opposite of `_decay`'s treatment of the same NULL -- there an unknown date is
        # missing evidence and must not cost rank; here it is missing evidence and must not
        # buy admission.
        if query.before is not None:
            where.append(s.documents.c.github_created_at < query.before)
        if query.exclude:
            where.append(s.threads.c.github_number.notin_(query.exclude))
        if query.labels:
            where.append(self._thread_has(s.thread_labels, s.thread_labels.c.label, query.labels))
        # The four signal filters below read tables that milestone 3's extraction pass
        # fills. They are wired now because they are versioned API surface (section 7) and
        # because a filter written later against a populated table is a filter nobody
        # tested empty -- but until that pass runs they correctly match nothing.
        if query.files and include_files:
            where.append(self._thread_has_path(query.files))
        if query.symbols:
            where.append(
                self._thread_has(s.thread_symbols, s.thread_symbols.c.symbol, query.symbols)
            )
        if query.errors:
            where.append(
                self._thread_has(
                    s.thread_errors, s.thread_errors.c.message_norm, query.errors, fuzzy=True
                )
            )
        if query.tests:
            where.append(self._thread_has(s.thread_tests, s.thread_tests.c.test_id, query.tests))
        return where

    @staticmethod
    def _thread_has_path(values: tuple[str, ...]) -> Any:
        """``--file``, matched exactly *or* on a path-segment boundary.

        ``thread_files`` holds diff entries under their repository-root path and prose
        mentions under whatever somebody typed, so exact matching made one flag mean three
        different things. Suffix matching collapses them: a bare basename and a partial
        path both find the changed-file rows the full path finds.

        Anchored at ``/`` so ``modeling_x.py`` cannot match ``not_modeling_x.py``, and
        ``ESCAPE``'d because a path may contain ``_``, which is a LIKE wildcard.
        """
        clauses: list[Any] = []
        for value in values:
            needle = value.strip().lstrip("./")
            if not needle:
                continue
            clauses.append(s.thread_files.c.path == needle)
            pattern = "%/" + needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append(s.thread_files.c.path.like(pattern, escape="\\"))
        if not clauses:
            return literal(False)
        return exists().where(
            and_(s.thread_files.c.thread_id == s.documents.c.thread_id, or_(*clauses))
        )

    @staticmethod
    def _thread_has(
        table: Any, column: Any, values: tuple[str, ...], *, fuzzy: bool = False
    ) -> Any:
        """``EXISTS`` against a signal table, as OR across the requested values.

        Section 4's signal tables are separate from ``documents`` so that an exact match
        can be weighted independently of prose; at this milestone that means filtered
        independently of it. ``fuzzy`` is a ``LIKE`` containment test, used for error
        strings where the caller pastes more than the indexed normalized form -- it is not
        the trigram tier, which is Postgres-only and lands with the ranking work.
        """
        if fuzzy:
            match = or_(*[column.like(f"%{value}%") for value in values])
        else:
            match = column.in_(values)
        return exists().where(and_(table.c.thread_id == s.documents.c.thread_id, match))

    def _hit(self, row: Any, query: SearchQuery) -> Hit:
        return Hit(
            repo=row.repo,
            number=int(row.github_number),
            thread_type=row.thread_type,
            title=row.title,
            source_type=row.source_type,
            url=row.url,
            author=row.author,
            trust=row.trust,
            age=render_age(row.github_created_at),
            snippet=snippet(row.body_text, query.terms, limit=query.snippet_chars),
            score=float(row.score or 0.0),
            created_at=row.github_created_at,
            source_id=str(row.source_id or ""),
            chunk_index=int(row.chunk_index or 0),
            breakdown=_breakdown(row),
        )

    # -- one thread ------------------------------------------------------

    def thread(
        self,
        repo: str,
        number: int,
        *,
        focus: str = "",
        full: bool = False,
        after: str = "",
        outline: bool = False,
        comment: str = "",
        before: dt.datetime | None = None,
        exclude: Sequence[int] = (),
    ) -> ThreadView | None:
        """One thread, capped (section 6).

        ``focus`` **orders** which comments come back and never selects them -- see
        :meth:`_focused`. Without it the order is the opening and the closing of the
        thread rather than the first N: on a long thread the resolution is at the end, and
        returning only the beginning reliably returns the part that was wrong. With
        ``outline`` it *narrows* instead, which is the one inversion of that rule and is
        argued where it happens (:meth:`_outline`).

        ``outline`` and ``after`` are the two ways out of the page cap, and they exist
        because of what the cap does to a caller who needs the *whole* thread. Measured on
        one agent run: ``transformers#37866`` has 70 comments and the page serves ten, so
        the agent came back **six times** with six different ``--focus`` strings, spent
        3,558 tokens, and saw overlapping samples -- the shape this project keeps finding,
        one level up from a file re-read at a different line range each time.

        Raising the cap is not the answer (section 6: the caps are the contract). So:
        ``outline`` serves one *line* per comment for the whole thread, which is the
        ``defs`` move applied to a discussion -- ask for the shape, then read the parts --
        and ``after`` turns the repeated sample into a sweep by starting the page after a
        comment the caller has already seen.

        ``outline`` short-circuits the page: nothing selects the ten rows a caller is not
        going to be shown (:meth:`_comments`).

        ``before`` and ``exclude`` are the cutoff, and this verb needs them more than
        ``search`` does: a benchmark can withhold a thread from every ranked page and still
        hand it over to anyone who asks for it by number. A thread opened at or after the
        cutoff, or named in ``exclude``, is **not found** rather than empty -- the two are
        different answers and only one of them is true.
        """
        if number in tuple(exclude):
            return None
        with self.engine.connect() as conn:
            row = conn.execute(
                select(s.threads).where(
                    s.threads.c.repo == repo, s.threads.c.github_number == number
                )
            ).one_or_none()
            if row is None:
                return None
            # `created_at` and not `updated_at`: a thread that existed before the cutoff is
            # legitimately readable, and its later comments are dropped document by
            # document below rather than by hiding the thread that carries them.
            if before is not None and (row.created_at is None or row.created_at >= before):
                return None
            labels = tuple(
                str(label)
                for (label,) in conn.execute(
                    select(s.thread_labels.c.label)
                    .where(s.thread_labels.c.thread_id == row.id)
                    .order_by(s.thread_labels.c.label)
                )
            )
            paths = _files_by_provenance(conn, row.id)
            # No join: the target *number* is the claim (section 13.3), and joining to
            # `threads` would drop exactly the edges worth reporting -- a pull request
            # closing an issue this index has never seen.
            links = tuple(
                {"relationship": rel, "target": int(target), "indexed": resolved is not None}
                for rel, target, resolved in conn.execute(
                    select(
                        s.thread_links.c.relationship,
                        s.thread_links.c.target_number,
                        s.thread_links.c.target_thread_id,
                    )
                    .where(s.thread_links.c.thread_id == row.id)
                    .order_by(s.thread_links.c.target_number)
                )
            )
            # The denominator's numerator: only the diff counts against `changed_files`,
            # so this and `files_changed` are the same quantity counted once.
            collected = len(paths[CHANGED])
            # Indexed, not `row.metadata`: `Row` shadows it.
            meta = row._mapping["metadata"] or {}
            changed = meta.get("changed_files")
            body, body_chars, body_truncated = self._body(conn, row.id, full=full, before=before)
            page = self._comments(
                conn, row, focus, after=after, outline=outline, comment=comment, before=before
            )

        return ThreadView(
            repo=row.repo,
            number=int(row.github_number),
            thread_type=row.thread_type,
            title=row.title,
            url=row.url,
            author=row.author,
            state=row.state,
            age=render_age(row.created_at),
            created_at=row.created_at,
            labels=labels,
            body=body,
            body_chars=body_chars,
            body_truncated=body_truncated,
            comments=page.hits,
            files_changed=paths[CHANGED],
            files_anchored=paths[FROM_COMMENT],
            files_mentioned=paths[MENTIONED],
            files_total=int(changed) if changed is not None else None,
            files_collected=collected,
            links=links,
            state_reason=meta.get("state_reason"),
            closed_by=meta.get("closed_by"),
            merged=row.merged_at is not None,
            review_decision=meta.get("review_decision"),
            review_decision_by=tuple(meta.get("review_decision_by") or ()),
            requested_reviewers=tuple(meta.get("requested_reviewers") or ()),
            total_documents=page.total,
            machine_suppressed=page.suppressed,
            indexed_at=row.indexed_at,
            selection=page.selection,
            focus=focus,
            focus_matched=page.matched,
            outline=page.outline,
            outline_total=len(page.outline),
            outline_matched=page.outline_matched,
            outline_widened=page.outline_widened,
            comment=comment,
            comment_status=page.comment_status,
            after=after,
        )

    # -- why is this line like this (issue #9) ----------------------------

    def why(
        self,
        repo: str,
        path: str,
        line: int,
        *,
        sha: str,
        summary: str = "",
        window: int = 25,
        history: Sequence[Any] = (),
        origin: Any = None,
        before: dt.datetime | None = None,
        exclude: Sequence[int] = (),
    ) -> WhyView:
        """What the index knows about the commit blame named, and about this line.

        ``git blame`` gives the commit; this gives the argument. The pull request is
        resolved from ``thread_commits`` first and from the squash-merge convention
        ``(#1234)`` in the summary second, because a commit staged by no per-PR pass is
        common and answering "no idea" while the number is sitting in the subject is not.

        Review comments are matched on the path and a window of lines around the anchor.
        The anchor is a *name* in GitHub's payload and a line number in ours, so an exact
        match would silently drop every comment written against a since-edited file --
        which is most of them.

        ``history`` is the line's revision chain, resolved to pull requests here because
        the caller has the clone and this has the index (relore#57). Blame names the last
        commit, which on a reformatted line is a cosmetic pull request standing in front of
        the one that argued the behaviour.

        ``origin`` is the *string* chain -- ``git log -S`` on a word from the line, chosen
        by :func:`relore.code.blame.origin_history`. It exists because the revision chain
        follows a range and therefore loses a block that was rewritten rather than moved,
        which is the common case and was the measured one: on ``generation/utils.py:2301``
        the whole eight-commit revision chain postdates #37866, the pull request that
        argued the behaviour. Resolved through the same query as the revisions, so a
        reader can tell the two lists apart by what they are, not by how they look.
        """
        shas = [str(commit.sha) for commit in history]
        origin_commits = tuple(getattr(origin, "commits", ()) or ())
        shas += [str(commit.sha) for commit in origin_commits]
        with self.engine.connect() as conn:
            row = conn.execute(
                select(s.threads.c.github_number, s.threads.c.title)
                .select_from(
                    s.thread_commits.join(s.threads, s.thread_commits.c.thread_id == s.threads.c.id)
                )
                .where(s.threads.c.repo == repo, s.thread_commits.c.sha.like(f"{sha[:12]}%"))
                .limit(1)
            ).one_or_none()
            resolved = _threads_for_shas(conn, repo, shas)
            chain = tuple(
                {
                    "sha": commit.sha,
                    "date": commit.date,
                    "summary": commit.summary,
                    **(
                        resolved.get(str(commit.sha))
                        or {"number": _number_from_summary(str(commit.summary))}
                    ),
                }
                for commit in history
            )
            introduced = tuple(
                {
                    "sha": commit.sha,
                    "date": commit.date,
                    "summary": commit.summary,
                    **(
                        resolved.get(str(commit.sha))
                        or {"number": _number_from_summary(str(commit.summary))}
                    ),
                }
                for commit in origin_commits
            )
            number = int(row.github_number) if row else _number_from_summary(summary)
            # A pull request the cutoff has not reached yet is *no answer*, not a hidden
            # one: the blamed line is in a tree the agent was given, but the argument that
            # produced it had not been made. Same shape as a commit that reached the tree
            # through no indexed pull request, which this already answers.
            if number is not None and not _reachable(conn, repo, number, before, exclude):
                number = None
            if number is None:
                return WhyView(
                    repo=repo,
                    path=path,
                    line=line,
                    sha=sha,
                    summary=summary,
                    history=chain,
                    origin=introduced,
                    origin_term=str(getattr(origin, "term", "") or ""),
                    origin_considered=tuple(getattr(origin, "considered", ()) or ()),
                    origin_declined=str(getattr(origin, "declined", "") or ""),
                )
            at_line, on_file, reviews, level = _argument(
                conn, number, path, line, window, before=before
            )
        return WhyView(
            repo=repo,
            path=path,
            line=line,
            sha=sha,
            summary=summary,
            number=number,
            resolved_by="commit" if row else "summary",
            thread=self.thread(repo, number, before=before, exclude=exclude),
            anchored=tuple(at_line),
            on_file=tuple(on_file),
            reviews=tuple(reviews),
            level=level,
            history=chain,
            origin=introduced,
            origin_term=str(getattr(origin, "term", "") or ""),
            origin_considered=tuple(getattr(origin, "considered", ()) or ()),
            origin_declined=str(getattr(origin, "declined", "") or ""),
        )

    # -- what is already being worked on ---------------------------------

    def inflight(
        self,
        repo: str,
        number: int,
        *,
        before: dt.datetime | None = None,
        exclude: Sequence[int] = (),
    ) -> InflightView:
        """Which threads claim to close ``number``.

        The most expensive mistake an agent makes is writing a patch for something already
        in review, and it is preventable in one hop: this walks section 13.3's edges
        backwards. Ordered open first and then by recency, because a stale draft and an
        approved pull request imply opposite next actions.

        **No trust floor, deliberately.** This reads `threads`, not `documents`, so
        section 6.2's machine exclusion does not apply -- and it should not: the
        deployment's own bot having an open fix pull request is precisely the duplicate an
        agent must not create. Excluding a document as evidence and hiding a pull request
        as *work in progress* are different questions.
        """
        newest = nulls_last(s.threads.c.updated_at.desc())
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    s.threads.c.repo,
                    s.threads.c.github_number,
                    s.threads.c.thread_type,
                    s.threads.c.title,
                    s.threads.c.url,
                    s.threads.c.author,
                    s.threads.c.state,
                    s.threads.c.merged_at,
                    s.threads.c.created_at,
                    s.threads.c.metadata,
                    s.thread_links.c.relationship,
                )
                .select_from(
                    s.thread_links.join(s.threads, s.thread_links.c.thread_id == s.threads.c.id)
                )
                .where(
                    s.threads.c.repo == repo,
                    s.thread_links.c.target_number == number,
                    *_opened_before(before),
                    *([s.threads.c.github_number.notin_(tuple(exclude))] if exclude else []),
                )
                .order_by(case((s.threads.c.state == "open", 0), else_=1), newest)
            ).all()
            indexed = int(
                conn.execute(
                    select(func.count())
                    .select_from(
                        s.thread_links.join(s.threads, s.thread_links.c.thread_id == s.threads.c.id)
                    )
                    .where(s.threads.c.repo == repo)
                ).scalar_one()
                or 0
            )

        claims = tuple(
            Claim(
                repo=row.repo,
                number=int(row.github_number),
                thread_type=row.thread_type,
                title=row.title,
                url=row.url,
                author=row.author,
                state=row.state,
                # Indexed, not `row.metadata`: `Row` shadows it.
                draft=bool((row._mapping["metadata"] or {}).get("draft")),
                merged=row.merged_at is not None,
                age=render_age(row.created_at),
                created_at=row.created_at,
                relationship=row.relationship,
                state_reason=(row._mapping["metadata"] or {}).get("state_reason"),
                closed_by=(row._mapping["metadata"] or {}).get("closed_by"),
                review_decision=(row._mapping["metadata"] or {}).get("review_decision"),
            )
            for row in rows[:MAX_CLAIMS]
        )
        return InflightView(
            repo=repo, number=number, claims=claims, total=len(rows), links_indexed=indexed
        )

    def _body(
        self, conn: Any, thread_id: int, *, full: bool, before: dt.datetime | None = None
    ) -> tuple[str, int, bool]:
        """The opening post, how long it really is, and whether any of it was withheld.

        The cap is a token budget, not the "never returnable in full" contract: that one
        is about the *comments*, which are unbounded -- a 200-comment argument is the case
        section 6 exists for -- whereas a body is one document whose length is whoever
        opened the thread. Cutting it silently is what sent a diagnosis session back to
        ``git clone``: on ``huggingface/transformers`` the issue template spends its first
        ~700 characters on ``### System Info`` and ``### Who can help?``, so
        ``### Reproduction`` -- the part an agent came for -- started at almost exactly the
        800th. So ``full`` serves all of it on request, and the truncated form now says
        how much it is holding back.

        **Truncation is reported, not inferred** (huggingface/relore#31). The caller used
        to compare the served length against ``len(whole)``, and :func:`snippet` collapses
        whitespace before it cuts anything -- so a 244-character body with blank lines in
        it came back as 242 and announced, in 40 characters, that it was withholding two
        that had never been withheld. Only this function knows the difference, so it says.

        The same number is why :data:`BODY_SLACK_CHARS` exists: a remainder shorter than
        the sentence announcing it is worth serving rather than announcing.
        """
        chunks = (
            conn.execute(
                select(s.documents.c.body_text)
                .where(
                    s.documents.c.thread_id == thread_id,
                    s.documents.c.source_type == "body",
                    *_written_before(before),
                )
                .order_by(s.documents.c.chunk_index.asc())
            )
            .scalars()
            .all()
        )
        # Every chunk, not chunk 0: past 6,000 characters the chunker splits a body on its
        # headings, and reading the first piece would cap `full` at a boundary the reporter
        # never wrote. The join is the boundary it split on.
        whole = "\n\n".join(text for text in chunks if text)
        if full:
            return whole, len(whole), False
        # `snippet` normalizes whitespace whether or not it cuts, so the comparison that
        # decides "was anything withheld?" has to be against the normalized form -- and a
        # body only a sentence over the cap is served whole instead.
        flat = " ".join(whole.split())
        if len(flat) <= MAX_BODY_CHARS + BODY_SLACK_CHARS:
            return flat, len(whole), False
        return snippet(whole, limit=MAX_BODY_CHARS), len(whole), True

    def _comments(
        self,
        conn: Any,
        thread: Any,
        focus: str,
        *,
        after: str = "",
        outline: bool = False,
        comment: str = "",
        before: dt.datetime | None = None,
    ) -> _Comments:
        def documents(tiers: tuple[str, ...]) -> Select:
            return select(
                s.documents.c.id,
                s.documents.c.source_type,
                s.documents.c.source_id,
                s.documents.c.chunk_index,
                s.documents.c.url,
                s.documents.c.author,
                s.documents.c.trust,
                s.documents.c.body_text,
                s.documents.c.github_created_at,
            ).where(
                s.documents.c.thread_id == thread.id,
                s.documents.c.source_type.notin_(("title", "body")),
                s.documents.c.trust.in_(tiers),
                # The cutoff rides inside `documents` rather than on `base`, so every
                # derived select -- the machine-tier count below included -- is bounded by
                # construction. A count that saw past the cutoff would announce comments
                # the page cannot serve.
                *_written_before(before),
            )

        base = documents(admissible_trust(None))
        # The sweep cursor, applied before anything counts or selects. `total` therefore
        # stays the thread's whole comment count -- the denominator a reader needs is "of
        # this thread", never "of what is left after where I resumed" -- while everything
        # that *chooses* rows sees only what comes after the cursor.
        remaining = _after(conn, base, after) if after.strip() else base
        # Comments, not rows: a 9,827-character comment is several documents and counting
        # them called one comment two (huggingface/relore#18). `passages` is the same
        # aggregate read the other way, and it is what lets a hit say it is a piece.
        passages = _passage_counts(conn, base)
        total = len(passages)
        # What the floor above took out. The exclusion is right -- our own bot's output is
        # not evidence (section 6.2) -- but unannounced it turns into `0 of 0 comments` on
        # a thread we hold one for, which is the sentence for a thread nobody has touched
        # (huggingface/relore#28). Counted here because only this query knows.
        suppressed = len(_passage_counts(conn, documents((MACHINE_TRUST,))))

        if comment.strip():
            # One addressed document, served whole, and it short-circuits everything the
            # page would have computed. Nothing here selects: the caller named the row.
            served, status = self._one_comment(conn, thread, comment.strip(), passages)
            return _Comments(
                hits=served,
                total=total,
                matched=None,
                selection=COMMENT_VIEW,
                suppressed=suppressed,
                comment_status=status,
            )

        if outline:
            # The outline *replaces* the page (``render_thread`` returns after it), so the
            # page's own selection is work nobody reads -- and on a focused outline it is
            # the expensive half, an FTS pass and an overfetch for ten rows that are
            # discarded. Measured on ``transformers#46419`` (644 comments) the whole call
            # is single-digit milliseconds either way; this is here because computing an
            # answer in order to throw it away is how the two selections drift apart.
            sketch, sketch_matched, widened = self._outline(conn, remaining, passages, focus)
            return _Comments(
                hits=(),
                total=total,
                matched=None,
                selection=OUTLINE_VIEW,
                suppressed=suppressed,
                outline=sketch,
                outline_matched=sketch_matched,
                outline_widened=widened,
            )

        matched: int | None = None
        if focus.strip():
            matched = len(
                {
                    (row.source_type, row.source_id)
                    for row in conn.execute(self.fts_filter(remaining, focus))
                }
            )
            rows, selection = self._focused(conn, remaining, focus), "focus"
        elif after.strip():
            # A cursor means a sweep, and a sweep is sequential. `ends+middle` is a
            # *sampling* strategy for "show me this thread", and composed with a cursor it
            # is a trap: the sample's last row is the thread's last comment, so the obvious
            # next call -- after the last id on the page -- returns nothing and reads as
            # "that was the end of it". Chronological from the cursor is what `--after`
            # means, and it is the only selection that can actually reach every comment.
            rows, selection = (
                _one_per_comment(_chronological(conn, remaining, None))[:MAX_THREAD_COMMENTS],
                AFTER_CURSOR,
            )
        else:
            rows, selection = (
                _ends_and_middle(conn, remaining, MAX_THREAD_COMMENTS),
                ENDS_AND_MIDDLE,
            )

        terms = tokenize(focus)
        hits = tuple(
            Hit(
                repo=thread.repo,
                number=int(thread.github_number),
                thread_type=thread.thread_type,
                title=thread.title,
                source_type=row.source_type,
                url=row.url,
                author=row.author,
                trust=row.trust,
                age=render_age(row.github_created_at),
                snippet=snippet(row.body_text, terms),
                score=float(row.score or 0.0),
                created_at=row.github_created_at,
                source_id=str(row.source_id or ""),
                chunk_index=int(row.chunk_index or 0),
                passages=passages.get((row.source_type, row.source_id), 1),
                # Measured against the raw text, not inferred from the ellipsis the window
                # ends with: the page uses this to decide whether it owes the caller the
                # address of the rest, and a fact about withholding is not something to
                # read back off a glyph.
                truncated=len(" ".join((row.body_text or "").split())) > MAX_SNIPPET_CHARS,
            )
            for row in rows
        )
        return _Comments(
            hits=hits, total=total, matched=matched, selection=selection, suppressed=suppressed
        )

    def _one_comment(
        self, conn: Any, thread: Any, comment: str, passages: dict[tuple[str, str], int]
    ) -> tuple[tuple[Hit, ...], str]:
        """One comment, whole, by the id the page prints beside it (huggingface/relore#71).

        **The id on a page was an ornament until this existed.** A comment longer than
        :data:`~relore.search.queries.MAX_SNIPPET_CHARS` comes back as a window with an
        ellipsis -- which says *that* it was cut and offers nothing that undoes it:
        ``--full`` is the opening post, ``--focus`` re-ranks and snippets again, ``--after``
        serves the *next* comments. So the page disclosed a gap it could not close, which
        is the shape this project keeps re-finding, and every comment carried a 76-character
        URL that was not an address either -- it opens a browser, and the caller reading the
        page is not a browser. One line per page naming this flag costs ~20 tokens where
        ten URLs cost ~250.

        Whole, with no cap of its own, on the same bargain ``--full`` already makes: the
        caps exist so a *sample* is not mistaken for a corpus, and a document asked for by
        its id is the opposite of a sample. Measured on the transformers corpus, a comment's
        median length is 120 characters and its 99th percentile 2,186 -- the cap would fire
        on a handful of threads and leave them with no way through, which is what it would
        have been added to prevent.

        Three outcomes, named rather than left to an empty answer: ``served``, ``absent``
        (this thread holds no such comment), and ``suppressed`` (it does, and it is our own
        bot's, so section 6.2 excluded it). A caller cannot tell the last two apart from
        outside, and they are opposite next actions -- stop looking, or ask
        ``search --trust machine``.
        """

        def whole(tiers: tuple[str, ...]) -> list[Any]:
            return list(
                conn.execute(
                    select(
                        s.documents.c.source_type,
                        s.documents.c.source_id,
                        s.documents.c.chunk_index,
                        s.documents.c.url,
                        s.documents.c.author,
                        s.documents.c.trust,
                        s.documents.c.body_text,
                        s.documents.c.github_created_at,
                    )
                    .where(
                        s.documents.c.thread_id == thread.id,
                        s.documents.c.source_type.notin_(("title", "body")),
                        s.documents.c.source_id == comment,
                        s.documents.c.trust.in_(tiers),
                    )
                    .order_by(s.documents.c.chunk_index.asc())
                )
            )

        rows = whole(admissible_trust(None))
        if not rows:
            return (), "suppressed" if whole((MACHINE_TRUST,)) else "absent"
        head = rows[0]
        # The chunker splits on boundaries and the pieces are re-joined in order, so what
        # comes back is the comment as it was written -- not a concatenation of windows.
        body = "\n".join(row.body_text or "" for row in rows).strip()
        return (
            Hit(
                repo=thread.repo,
                number=int(thread.github_number),
                thread_type=thread.thread_type,
                title=thread.title,
                source_type=head.source_type,
                url=head.url,
                author=head.author,
                trust=head.trust,
                age=render_age(head.github_created_at),
                snippet=body,
                score=0.0,
                created_at=head.github_created_at,
                source_id=str(head.source_id or ""),
                chunk_index=0,
                passages=passages.get((head.source_type, head.source_id), len(rows)),
            ),
        ), "served"

    def _outline(
        self,
        conn: Any,
        base: Select,
        passages: dict[tuple[str, str], int],
        focus: str = "",
    ) -> tuple[tuple[Hit, ...], int | None, bool]:
        """Every comment, one line's worth each, oldest first (relore#70).

        Honours ``--after``, so a thread too long to outline whole is swept rather than
        truncated. That is the opposite of the first design, and the measurement is why:
        a 644-comment thread cannot be served complete at any line length, so a view that
        ignored the cursor would put rows 101 onwards permanently out of reach -- which is
        the silent incompleteness this verb exists to end, reintroduced by the fix for it.

        Chronological, never ranked. A ranked outline would be a second answer to the
        question ``--focus`` already answers, and the reason to read a whole thread is
        usually that ranking it has not worked -- which is exactly the run this came from.
        :data:`OUTLINE_CHARS` is deliberately short of quotable: this says *which* comment
        to ask for, and a line long enough to answer with would make it another page.

        **``focus`` narrows it, and does not reorder it** (huggingface/relore#71). Both
        times the measured run reached for an outline it ran ``--outline --json`` through a
        client-side filter -- 411 and 236 tokens against ~2,593 for the rendered outline of
        the same thread. It wanted the ids carrying a word, not the index; so that is a
        flag rather than a pipeline. It is the one place in this codebase where a focus
        *selects*, and the reason the rule holds elsewhere is the reason it inverts here:
        an outline's rows are already complete, so a narrowed one withholds nothing a
        second call cannot have, while a narrowed *page* would be ten of seventy chosen by
        a conjunction nobody could see.

        Which is why the match is a **disjunction, in Python, over every chunk**. A
        conjunction is what empties a thread page as soon as a caller passes a sentence
        (:meth:`_focused`), and an outline that came back empty would read as "no comment
        in this thread mentions that" -- the one thing it did not mean. Over every chunk
        because the outline's row is a comment's *first* document and its terms may be in
        its fourth. And if nothing carries a term the view widens back to the whole
        outline and says so, rather than handing back a page whose emptiness is a property
        of the question (huggingface/relore#47).
        """
        rows = _chronological(conn, base, None)
        matched: int | None = None
        widened = False
        # Lowercased on both sides: `tokenize` does not case-fold (it feeds engines that
        # do), and a caller who types a word the way GitHub renders it -- `RoPE`, `Cache`
        # -- must not get an empty narrowing out of it.
        terms = tuple(term.lower() for term in tokenize(focus))
        if terms:
            carrying = {
                (row.source_type, row.source_id)
                for row in rows
                if any(term in (row.body_text or "").lower() for term in terms)
            }
            matched = len(carrying)
            if carrying:
                rows = [row for row in rows if (row.source_type, row.source_id) in carrying]
            else:
                widened = True
        sketch = _one_per_comment(rows)[:MAX_OUTLINE_COMMENTS]
        return (
            tuple(
                Hit(
                    repo="",
                    number=0,
                    thread_type="",
                    title="",
                    source_type=row.source_type,
                    url=row.url,
                    author=row.author,
                    trust=row.trust,
                    age=render_age(row.github_created_at),
                    snippet=snippet(row.body_text, limit=OUTLINE_CHARS),
                    score=0.0,
                    created_at=row.github_created_at,
                    source_id=str(row.source_id or ""),
                    chunk_index=int(row.chunk_index or 0),
                    passages=passages.get((row.source_type, row.source_id), 1),
                )
                for row in sketch
            ),
            matched,
            widened,
        )

    def _focused(self, conn: Any, base: Select, focus: str) -> list[Any]:
        """Order a thread's comments by a focus query. **Never filter on it.**

        The thread *is* the admission decision, and there is none left to make inside it.
        Both engines' text filter is a conjunction -- ``plainto_tsquery`` is a plain AND,
        FTS5 gets one quoted term per word -- which is right for corpus-wide ``search``,
        where it is what stops a pasted sentence matching everything. Applied inside one
        thread it empties the page as soon as the caller passes a sentence, and ``--help``
        invites exactly that ("select the comments that answer this").

        Measured on ``huggingface/transformers#28056``, 30 comments: ``cache`` returned 10,
        ``use_cache gradient`` 2, ``use_cache gradient checkpointing warning`` **0**. Every
        term is in the thread; no single comment carries all four. A monotonic decrease
        with term count ending in an empty page is the signature, and an empty page reads
        as "this thread has nothing relevant" -- the one thing it did not mean.

        So the focus scores, and the page is never empty on a thread that has comments.
        Postgres ranks the whole thread with the disjunction
        :meth:`unfiltered_fts_score` already builds, so an all-terms match still sorts
        first and a partial match follows it. FTS5 cannot score what it did not ``MATCH``
        (see the sqlite backend), so there the matched comments lead and the chronological
        remainder fills the rest of the page.

        Chunks collapse here too, which is why the fetch is :data:`PASSAGE_OVERFETCH`
        times the page: two pieces of one comment ranked first and second is two slots
        spent on one voice, and it reads as corroboration (huggingface/relore#18).
        """
        wanted = MAX_THREAD_COMMENTS * PASSAGE_OVERFETCH
        ranking = self.unfiltered_fts_score(focus)
        if ranking is not None:
            rows = list(
                conn.execute(
                    base.add_columns(ranking.label("score"))
                    .order_by(ranking.desc(), s.documents.c.github_created_at.asc().nullslast())
                    .limit(wanted)
                )
            )
            return _one_per_comment(rows)[:MAX_THREAD_COMMENTS]

        score = self.fts_score(focus)
        rows = _one_per_comment(
            list(
                conn.execute(
                    self.fts_filter(base.add_columns(score.label("score")), focus)
                    .order_by(score.desc())
                    .limit(wanted)
                )
            )
        )[:MAX_THREAD_COMMENTS]
        if len(rows) >= MAX_THREAD_COMMENTS:
            return rows
        chosen = {(row.source_type, row.source_id) for row in rows}
        remainder = [
            row
            for row in _one_per_comment(_chronological(conn, base, None))
            if (row.source_type, row.source_id) not in chosen
        ]
        return rows + _ends(remainder, MAX_THREAD_COMMENTS - len(rows))


_SQUASHED = re.compile(r"\(#(\d+)\)\s*$")


def _number_from_summary(summary: str) -> int | None:
    """``Fix the mask (#48672)`` -- the squash-merge convention, used as a fallback."""
    match = _SQUASHED.search(summary or "")
    return int(match.group(1)) if match else None


def _argument(
    conn: Any,
    number: int,
    path: str,
    line: int,
    window: int,
    *,
    before: dt.datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """What was said about this line, widening until something answers (relore#64).

    Three groups and the level that carried the answer, because ``0 comments on this line``
    is a fact about a *window* and reads as a fact about the line. Measured: the line this
    was written for has no anchored comment in `transformers`#43121 and its whole argument
    one level up, and the verb's full stop cost an agent eleven further calls.

    * **line** -- review comments within ``window`` lines of the anchor. What it always did.
    * **file** -- review comments on the same path, further away. Still about this code.
    * **review** -- the pull request's review *bodies* (relore#63). These carry no line
      anchor at all, so no widening of a line window could ever have reached them, and they
      are where an approval states its conditions: `transformers`#37866's approving review
      is the whole reason the line under it exists.

    Filtered in Python rather than in SQL: the anchor lives in ``documents.metadata``, and
    reaching into JSON is the one thing ``test_no_dialect_leak`` forbids the store from
    doing. A thread's review rows are tens of rows, so the cost is nothing.
    """
    rows = conn.execute(
        select(
            s.documents.c.author,
            s.documents.c.trust,
            s.documents.c.url,
            s.documents.c.body_text,
            s.documents.c.metadata,
            s.documents.c.github_created_at,
            s.documents.c.source_type,
        )
        .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
        .where(
            s.threads.c.github_number == number,
            s.documents.c.source_type.in_(("review_comment", "review")),
            *_written_before(before),
        )
    ).all()

    at_line: list[dict[str, Any]] = []
    on_file: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for row in rows:
        meta = row._mapping["metadata"] or {}
        entry = {
            "author": row.author,
            "trust": row.trust,
            "url": row.url,
            "line": None,
            "age": render_age(row.github_created_at),
            # The age is what the page prints; this is what a citation needs. `why` is the
            # verb most often asked for "the comment, with its author and its date", and
            # its own revision rows have carried a date since relore#57 -- so without this
            # the page was inconsistent with itself (huggingface/relore#66).
            "date": render_stamp(row.github_created_at),
            "text": row.body_text,
        }
        if row.source_type == "review":
            # A review body is the verdict, not a remark on a line: its state is what makes
            # "approved, on condition that…" different from a passing thought.
            entry["state"] = meta.get("state")
            reviews.append(entry)
            continue
        if meta.get("path") != path:
            continue
        anchor = meta.get("line") or meta.get("original_line")
        entry["line"] = anchor
        if anchor is not None and abs(int(anchor) - line) > window:
            on_file.append(entry)
        else:
            at_line.append(entry)

    level = "line" if at_line else "file" if on_file else "review" if reviews else "none"
    return at_line, on_file, reviews, level


def _threads_for_shas(conn: Any, repo: str, shas: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve each sha to the pull request that carried it, for the line's history.

    One query for the whole chain: a `git log -L` walk is a handful of commits and a
    round trip each would make the widening cost more than the dead end it replaces.
    """
    if not shas:
        return {}
    rows = conn.execute(
        select(
            s.thread_commits.c.sha,
            s.threads.c.github_number,
            s.threads.c.title,
            s.threads.c.url,
            s.threads.c.state,
        )
        .select_from(
            s.thread_commits.join(s.threads, s.thread_commits.c.thread_id == s.threads.c.id)
        )
        .where(
            s.threads.c.repo == repo,
            or_(*[s.thread_commits.c.sha.like(f"{sha[:12]}%") for sha in shas]),
        )
    ).all()
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        for sha in shas:
            if row.sha.startswith(sha[:12]):
                found[sha] = {
                    "number": int(row.github_number),
                    "title": row.title,
                    "url": row.url,
                    "state": row.state,
                }
    return found


def _files_by_provenance(conn: Any, thread_id: int) -> dict[str, tuple[str, ...]]:
    """One thread's paths, one list per source (huggingface/relore#17).

    Merged into a single array these answer a membership question affirmatively and
    wrongly: ``config.json`` is named in ``huggingface/transformers#39847``'s discussion
    and is not in its 323-file diff, and a bare ``modeling_rope_utils.py`` sat next to the
    real ``src/transformers/modeling_rope_utils.py`` with nothing to tell them apart.

    Three lists rather than two, because the middle one is neither. A path an inline
    review comment hangs on **is** in the diff -- GitHub will not anchor a comment
    anywhere else -- so it is evidence, unlike prose; but it is not part of the page the
    per-PR pass collected, so counting it against ``changed_files`` would make the
    numerator and the denominator two different quantities again. It is often a path past
    the 100-row cap, which makes it the one source that recovers what the cap dropped.

    ``source`` is null on rows written before migration 7, and the fallback is the old
    inference: right for the diff (only it has a ``change_type``) and unable to tell an
    anchor from prose, which is exactly the conflation that was there before and no worse.
    A ``relored derive`` fixes it for real.
    """
    out: dict[str, list[str]] = {CHANGED: [], FROM_COMMENT: [], MENTIONED: []}
    rows = conn.execute(
        select(s.thread_files.c.path, s.thread_files.c.change_type, s.thread_files.c.source)
        .where(s.thread_files.c.thread_id == thread_id)
        .distinct()
        .order_by(s.thread_files.c.path)
    )
    for path, change_type, source in rows:
        source = source or (CHANGED if change_type is not None else MENTIONED)
        out.get(source, out[MENTIONED]).append(str(path))
    return {source: tuple(dict.fromkeys(paths)) for source, paths in out.items()}


def _after(conn: Any, base: Select, after: str) -> Select:
    """The same query, starting strictly after one comment (relore#70).

    The cursor is a comment's ``source_id`` -- the id in its own GitHub URL, which is what
    a caller has in front of them after reading a page. An id this thread does not hold
    narrows nothing and is **not** an error: the alternative is a 4xx on a sweep that has
    simply reached a comment the trust floor excludes, and a caller cannot tell those
    apart from outside.

    Ordered on ``(github_created_at, source_type, source_id)`` and compared as a widened
    tuple rather than by timestamp alone. Two comments on one thread can share a second --
    a review and its first inline comment routinely do -- and a cursor that compared only
    the timestamp would skip whichever of them sorted second. Skipping one comment of a
    sweep is precisely the silent incompleteness this verb exists to stop, so the
    comparison is spelled out rather than approximated. Written as ``or_``/``and_`` and not
    as a row-value comparison, which SQLite only learned in 3.15.
    """
    cursor = conn.execute(
        base.with_only_columns(
            s.documents.c.github_created_at,
            s.documents.c.source_type,
            s.documents.c.source_id,
        )
        .where(s.documents.c.source_id == after)
        .limit(1)
    ).one_or_none()
    if cursor is None:
        return base
    when, kind, ident = cursor
    same = or_(
        s.documents.c.source_type > kind,
        and_(s.documents.c.source_type == kind, s.documents.c.source_id > ident),
    )
    if when is None:
        # An undated comment sorts last (`_chronological` is `nullslast`), so what follows
        # it is the rest of the undated ones. Explicit, because `col > NULL` is NULL and a
        # cursor that quietly matched nothing would end a sweep early and look finished.
        return base.where(and_(s.documents.c.github_created_at.is_(None), same))
    return base.where(
        or_(
            s.documents.c.github_created_at > when,
            and_(s.documents.c.github_created_at == when, same),
            # The undated tail, which sorts after every dated comment.
            s.documents.c.github_created_at.is_(None),
        )
    )


def _passage_counts(conn: Any, base: Select) -> dict[tuple[str, str], int]:
    """How many documents each comment was chunked into (huggingface/relore#18).

    One aggregate over the thread, which is what makes both numbers honest: the thread's
    comment count stops counting chunks, and a hit can say *"passage 2 of 7"* instead of
    arriving as a second comment with the same URL, author and tier as the first.
    """
    counted = base.with_only_columns(
        s.documents.c.source_type,
        s.documents.c.source_id,
        func.count().label("passages"),
    ).group_by(s.documents.c.source_type, s.documents.c.source_id)
    return {(row.source_type, row.source_id): int(row.passages) for row in conn.execute(counted)}


def _chronological(conn: Any, base: Select, limit: int | None) -> list[Any]:
    """A thread's comments oldest-first, scored zero: the order with nothing to rank."""
    rows = list(
        conn.execute(
            base.add_columns(literal(0.0).label("score")).order_by(
                s.documents.c.github_created_at.asc().nullslast()
            )
        )
    )
    return rows if limit is None else _ends(rows, limit)


def _ends_and_middle(conn: Any, base: Select, limit: int) -> list[Any]:
    """The unfocused page: both ends, and the middle actually represented.

    Taking the first five and the last five is defensible in the abstract -- an opening
    states the problem and an ending states the resolution -- and it fails on exactly the
    threads ``thread`` is for. A long thread is long *because* it was contested, and a
    contested thread resolves in the middle, before the CI churn that fills the tail. On
    ``huggingface/transformers#39847`` the ten it returned were four emoji, a ``cc @``, a
    ``run-slow:`` line and CI chatter, while the two comments that answered the question
    sat at positions 51 and 79 of 97 (huggingface/relore#16).

    So: :data:`THREAD_ENDS` at each end, and the remaining slots spread across everything
    between them -- **reviews first**, because a review with a state is an act rather than
    a remark, then evenly spaced so no region of the middle is structurally unreachable.
    Chunks of one comment collapse into one slot on the way (huggingface/relore#18): ten
    slots spent on five comments is a real cost at this cap.

    Deterministic, and it claims nothing it cannot support: the page says it is a sample,
    not a ranking (:data:`ENDS_AND_MIDDLE`), and ``--focus`` is the way to rank.
    """
    rows = _one_per_comment(_chronological(conn, base, None))
    if len(rows) <= limit:
        return rows
    ends = min(THREAD_ENDS, limit // 2)
    head, tail = rows[:ends], rows[len(rows) - ends :]
    middle = rows[ends : len(rows) - ends]
    return head + _spread(middle, limit - 2 * ends) + tail


def _one_per_comment(rows: list[Any]) -> list[Any]:
    """One row per comment, the first chunk standing for the rest, order preserved."""
    best: dict[tuple[str, str], Any] = {}
    for row in rows:
        best.setdefault((row.source_type, row.source_id), row)
    return list(best.values())


def _spread(rows: list[Any], slots: int) -> list[Any]:
    """``slots`` of ``rows``: every review first, then an even sample of the remainder.

    Even rather than "the longest" or "the most replied to": length is not merit and the
    reply count is not in the payload, whereas a gap in coverage is the actual defect --
    any region of the middle must be able to appear.
    """
    if slots <= 0 or not rows:
        return []
    if len(rows) <= slots:
        return rows
    chosen = {
        id(row) for row in rows if str(row.source_type).startswith("review")
    }  # an act, not a remark
    if len(chosen) > slots:
        reviews = [row for row in rows if id(row) in chosen]
        chosen = {id(row) for row in _even(reviews, slots)}
    rest = [row for row in rows if id(row) not in chosen]
    chosen |= {id(row) for row in _even(rest, slots - len(chosen))}
    return [row for row in rows if id(row) in chosen]


def _even(rows: list[Any], slots: int) -> list[Any]:
    """``slots`` rows spaced evenly across ``rows``, endpoints included."""
    if slots <= 0:
        return []
    if len(rows) <= slots:
        return rows
    step = (len(rows) - 1) / (slots - 1) if slots > 1 else 0
    return [rows[round(i * step)] for i in range(slots)]


def _breakdown(row: Any) -> dict[str, float]:
    """Section 8's per-hit score breakdown, read off the columns ``_select`` carried out.

    Only the terms this question actually contained appear, which is the point: a
    breakdown padded with zeros for everything the caller did not ask says less than one
    that lists what was weighed.
    """
    out = {"score": float(row.score or 0.0)}
    for key in row._mapping:
        if str(key).startswith("term_"):
            out[str(key)[len("term_") :]] = float(row._mapping[key] or 0.0)
    return out


def _order(source: Any, scored: bool) -> list[Any]:
    """Best first, newest first, and stable. **The selection order.**

    Score leads only when there *is* one: ordering by a bound constant is at best ignored
    and at worst a type error, and recency is the honest order for a query with nothing to
    rank. Since section 6's weighted score, "there is one" means the question carried a
    term of any kind -- not merely that it carried text, which is the change that gives a
    textless expansion leg an order at all (section 10.4 #1).

    The id tail makes the order total, so a page is reproducible -- which matters when the
    next thing built on top of it is a labelling UI (section 8).

    This one is used by the two ``row_number()`` windows that *choose* which document
    represents a chunk and a thread, and it stays on relevance whatever the caller asked
    to sort by -- see :func:`_present_order`.
    """
    lead = [source.c.score.desc()] if scored else []
    return [*lead, source.c.github_created_at.desc(), source.c.id.desc()]


def _present_order(source: Any, scored: bool, sort: str) -> list[Any]:
    """How the chosen hits are *presented*, which is not how they were chosen.

    ``newest`` must not reach the ``row_number()`` windows. Section 10.6 is the record of
    what happens when recency decides which document represents a thread: it picks the
    thread's *last* one, which on a merged pull request is the approving review, and a real
    query came back nine hits of ``LGTM``, ``Thx``, ``Nice!``, ``Yep``. Selecting on
    relevance and then ordering by date gives the best answer *from* each thread, most
    recent first -- which is what someone asking for a date sort wants, and not what
    sorting the whole candidate set by date would return.

    ``nulls_last`` because a document with no timestamp must not lead a date sort, and the
    two dialects disagree on where a NULL goes by default: Postgres puts it first on
    ``DESC``, SQLite last. As a tie-break that was invisible; as the leading key it decides
    the page.
    """
    if sort == "newest":
        return [nulls_last(source.c.github_created_at.desc()), source.c.id.desc()]
    return _order(source, scored)


def _ends(rows: list[Any], limit: int) -> list[Any]:
    """The opening and the closing of a thread, in chronological order.

    A long thread's resolution is at the end. Truncating to the first N returns the part
    that was wrong and drops the part that settled it.
    """
    if len(rows) <= limit:
        return rows
    head = limit // 2
    return rows[:head] + rows[len(rows) - (limit - head) :]
