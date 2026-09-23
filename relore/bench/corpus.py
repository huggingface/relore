"""The corpus as it stood at ``T``, and the signal-table leak the cutoff cannot reach.

Both halves ask one question -- which documents did this thread have by ``T`` -- so they
are one module.

**The export.** Section 10's baselines have to read the same documents the index read, so
the rows come out under :func:`relore.search.backends.base.corpus_scope`, the predicate
``_filters`` itself applies, rather than a second copy of it.

**The post-filter.** The five ``thread_*`` tables carry only ``thread_id`` and a value
(section 4): no timestamp. So a path first named after ``T`` is attributed to the whole
thread, and ``--file X --before T`` selects that thread's pre-cutoff documents on evidence
that did not exist yet. :class:`SignalPostFilter` re-derives each hit thread's signals from
the documents it had by ``T`` and drops what the index could not have selected.

It fixes filtering only. ``_overlap`` has already scored the same timeless rows, so
ordering stays inflated until those tables carry a ``first_seen_at``; any number from a
post-filtered run has to say so.

Two ceilings, both erring towards dropping a hit rather than admitting a future document:

* Extraction reads unchunked text (``ingest/extract``); this rejoins chunks, so a term cut
  by the hard split past ``MAX_CHUNK_CHARS`` is lost.
* A merged diff has no date of its own, so ``thread_files`` rows with source ``changed``
  are readmitted only when the thread merged before ``T``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Engine, and_, or_, select

from relore.ingest.extract import CHANGED, extract_signals
from relore.search.backends.base import corpus_scope, written_before
from relore.search.queries import Hit, SearchQuery
from relore.store import schema as s

#: Bumped when the shape of an exported line changes.
EXPORT_VERSION = 1

#: What this corpus is still wrong about, carried in the header so a reader need not come
#: here for it.
KNOWN_LIMITS = (
    "signal-table-ranking: `_overlap` scores thread_* rows that carry no timestamp, so a "
    "run's ordering is inflated until those tables carry a first_seen_at",
    "edited-bodies: documents.body_markdown holds the current text, so a body edited "
    "after the cutoff is exported as edited",
    "rendered-age: relore renders a hit's age from the wall clock, not from the cutoff",
)

#: The export's order and a row's identity. Not ``documents.id``, which a rebuild
#: reassigns: this is ``documents_identity_uq`` with the thread named the way a human does.
IDENTITY = ("repo", "number", "source_type", "source_id", "chunk_index")


# -- the export ------------------------------------------------------------


@dataclass(frozen=True)
class ExportStats:
    """What one export wrote."""

    path: Path
    documents: int
    threads: int
    repos: tuple[str, ...]
    before: dt.datetime | None
    exclude: tuple[int, ...]
    #: sha256 over the document lines only: the header carries a timestamp, so including
    #: it would make two exports of one corpus hash differently.
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "documents": self.documents,
            "threads": self.threads,
            "repos": list(self.repos),
            "before": self.before.isoformat() if self.before else None,
            "exclude": list(self.exclude),
            "digest": self.digest,
        }


def export(
    engine: Engine,
    out: Path,
    *,
    repos: Sequence[str],
    before: dt.datetime | None,
    exclude: Sequence[int] = (),
) -> ExportStats:
    """Write every in-scope document written before ``before`` to ``out``, as JSONL.

    Header first, as ``bench/dataset`` does: the window a number was produced under travels
    in the file. The body goes to a temporary file first so that header can carry a digest
    of what follows it.
    """
    if before is not None and before.tzinfo is None:
        raise ValueError("before must be timezone-aware")
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    documents = 0
    threads: set[tuple[str, int]] = set()

    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as body, engine.connect() as conn:
            for row in _rows(conn, repos=repos, before=before, exclude=exclude):
                line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                digest.update(line.encode("utf-8"))
                body.write(line)
                documents += 1
                threads.add((row["repo"], row["number"]))
        stats = ExportStats(
            path=out,
            documents=documents,
            threads=len(threads),
            repos=tuple(repos),
            before=before,
            exclude=tuple(exclude),
            digest=digest.hexdigest(),
        )
        header = {
            "corpus": {
                "version": EXPORT_VERSION,
                "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "identity": list(IDENTITY),
                "known_limits": list(KNOWN_LIMITS),
                **stats.as_dict(),
            }
        }
        del header["corpus"]["path"]  # where the file is is not a property of the corpus
        with open(out, "w", encoding="utf-8") as final:
            final.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
            with open(tmp, encoding="utf-8") as body_read:
                for line in body_read:
                    final.write(line)
    finally:
        tmp.unlink(missing_ok=True)
    return stats


def _rows(
    conn: Connection,
    *,
    repos: Sequence[str],
    before: dt.datetime | None,
    exclude: Sequence[int],
) -> Iterator[dict[str, Any]]:
    """One dict per exported document, in :data:`IDENTITY` order.

    ``body_text``, not ``body_markdown``: the normalized form is what the index scores, so
    it is what a baseline gets. ``trust`` rides along and is not applied -- the floor is a
    property of a query (section 6.2), not of a corpus.
    """
    stmt = (
        select(
            s.threads.c.repo,
            s.threads.c.github_number,
            s.threads.c.thread_type,
            s.threads.c.title,
            s.documents.c.source_type,
            s.documents.c.source_id,
            s.documents.c.chunk_index,
            s.documents.c.author,
            s.documents.c.trust,
            s.documents.c.url,
            s.documents.c.github_created_at,
            s.documents.c.body_text,
        )
        .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
        .where(*corpus_scope(repos, before, exclude))
        .order_by(
            s.threads.c.repo,
            s.threads.c.github_number,
            s.documents.c.source_type,
            s.documents.c.source_id,
            s.documents.c.chunk_index,
        )
    )
    for row in conn.execute(stmt):
        yield {
            "repo": row.repo,
            "number": int(row.github_number),
            "thread_type": row.thread_type,
            "title": row.title,
            "source_type": row.source_type,
            "source_id": str(row.source_id or ""),
            "chunk_index": int(row.chunk_index or 0),
            "author": row.author,
            "trust": row.trust,
            "url": row.url,
            "created_at": row.github_created_at.isoformat() if row.github_created_at else None,
            "text": row.body_text,
        }


# -- the post-filter -------------------------------------------------------


@dataclass(frozen=True)
class ThreadSignals:
    """What one thread carried by ``T``, in the four shapes a query can filter on.

    ``links`` and ``commits`` are absent on purpose: neither is reachable from
    :class:`~relore.search.queries.SearchQuery`, so neither can select a document.
    """

    files: frozenset[str]
    symbols: frozenset[str]
    #: A sequence, not a set: the error filter is a containment test, not equality.
    errors: tuple[str, ...]
    tests: frozenset[str]

    def satisfies(self, query: SearchQuery) -> bool:
        """Whether this thread could have been selected by ``query`` at the cutoff.

        Each group is a disjunction over its values and the groups conjoin -- what
        ``_filters`` does with four ``EXISTS`` clauses. This is a second implementation of
        predicates whose first is SQL (section 13.2 #3 is the record of what disagreeing
        costs), so ``tests/integration/test_corpus.py`` holds the two to one answer.
        """
        if query.files and not any(_has_path(self.files, value) for value in query.files):
            return False
        if query.symbols and not (self.symbols & set(query.symbols)):
            return False
        if query.errors and not any(
            any(value in stored for stored in self.errors) for value in query.errors
        ):
            return False
        # noqa on the ladder rather than an inlined negation: four filters are refused the
        # same way, and the shape is the argument that none of them is special.
        if query.tests and not (self.tests & set(query.tests)):  # noqa: SIM103
            return False
        return True


def _has_path(paths: Iterable[str], value: str) -> bool:
    """``--file``, matched the way :meth:`SearchBackend._thread_has_path` matches it:
    exact, or on a path-segment boundary. The SQL side escapes ``_`` and ``%`` for LIKE;
    here the comparison is literal already, which is the only difference between them."""
    needle = value.strip().lstrip("./")
    if not needle:
        return False
    return any(path == needle or path.endswith("/" + needle) for path in paths)


def signals_at(conn: Connection, thread_id: int, before: dt.datetime) -> ThreadSignals:
    """Re-derive one thread's signals from the documents it had by ``before``.

    The stored rows are the timeless set this exists to correct, so they are consulted for
    nothing a document can produce -- the merged diff excepted, which no document carries.
    """
    rows = conn.execute(
        select(
            s.documents.c.source_type,
            s.documents.c.source_id,
            s.documents.c.chunk_index,
            s.documents.c.body_markdown,
            s.documents.c.metadata,
        )
        .where(s.documents.c.thread_id == thread_id, *written_before(before))
        .order_by(
            s.documents.c.source_type,
            s.documents.c.source_id,
            s.documents.c.chunk_index,
        )
    ).all()

    sources: list[dict[str, Any]] = []
    for _key, group in groupby(rows, key=lambda r: (r.source_type, r.source_id)):
        pieces = list(group)
        sources.append(
            {
                # Blank line between chunks: `chunk` strips each piece, and every boundary
                # it chose but the hard cut was a heading or a paragraph break anyway.
                "body_markdown": "\n\n".join(p.body_markdown or "" for p in pieces),
                "metadata": pieces[0].metadata or {},
            }
        )

    signals = extract_signals(sources, detail=None)
    files = {row["path"] for row in signals.files}
    files |= _merged_diff_paths(conn, thread_id, before)
    return ThreadSignals(
        files=frozenset(files),
        symbols=frozenset(row["symbol"] for row in signals.symbols),
        errors=tuple(row["message_norm"] for row in signals.errors),
        tests=frozenset(row["test_id"] for row in signals.tests),
    )


def _merged_diff_paths(conn: Connection, thread_id: int, before: dt.datetime) -> set[str]:
    """The changed-file list, if it was final before the cutoff.

    A merged pull request's diff is what it is at the merge. One open at ``T`` had some
    prefix of this list that nothing here can reconstruct, so it contributes none of it.
    """
    merged_at = conn.execute(
        select(s.threads.c.merged_at).where(s.threads.c.id == thread_id)
    ).scalar()
    if merged_at is None or merged_at >= before:
        return set()
    return {
        row.path
        for row in conn.execute(
            select(s.thread_files.c.path).where(
                s.thread_files.c.thread_id == thread_id,
                s.thread_files.c.source == CHANGED,
            )
        )
    }


class SignalPostFilter:
    """Drop the hits a thread's *future* signals selected.

    Benchmark-only and outside the query layer: no production semantics change, no
    migration, ``O(hits)``. An unpinned filter (``before=None``) is the identity, matching
    ``AsOf()``, so a control and a treatment run come off the same code path.
    """

    def __init__(self, engine: Engine, before: dt.datetime | None) -> None:
        if before is not None and before.tzinfo is None:
            raise ValueError("before must be timezone-aware")
        self.engine = engine
        self.before = before

    def keep(self, hits: Sequence[Hit], query: SearchQuery) -> list[Hit]:
        """``hits``, minus those whose thread did not carry the filtered-on value at the
        cutoff. A query with no signal filter is untouched: a selection leak has nothing to
        have selected."""
        if self.before is None or not query.has_signal_filter or not hits:
            return list(hits)
        with self.engine.connect() as conn:
            ids = _thread_ids(conn, {(hit.repo, hit.number) for hit in hits})
            verdict: dict[tuple[str, int], bool] = {}
            for ref, thread_id in ids.items():
                verdict[ref] = signals_at(conn, thread_id, self.before).satisfies(query)
        # A hit whose thread is not in `ids` cannot be shown to have carried anything, and
        # this module excludes what it cannot verify.
        return [hit for hit in hits if verdict.get((hit.repo, hit.number), False)]

    def dropped(self, hits: Sequence[Hit], query: SearchQuery) -> list[Hit]:
        """The complement of :meth:`keep`: how far the leak reached on this run, reported
        rather than inferred from a recall that moved."""
        kept = {id(hit) for hit in self.keep(hits, query)}
        return [hit for hit in hits if id(hit) not in kept]


def _thread_ids(conn: Connection, refs: Iterable[tuple[str, int]]) -> Mapping[tuple[str, int], int]:
    """``(repo, number) -> threads.id`` for one page.

    An ``OR`` of pairs rather than a row-value ``IN``: the clause is tiny at one page, and
    row-value comparison is a thing the two dialects disagree about quietly.
    """
    refs = list(refs)
    if not refs:
        return {}
    clauses = [
        and_(s.threads.c.repo == repo, s.threads.c.github_number == number) for repo, number in refs
    ]
    rows = conn.execute(
        select(s.threads.c.repo, s.threads.c.github_number, s.threads.c.id).where(or_(*clauses))
    )
    return {(row.repo, int(row.github_number)): int(row.id) for row in rows}
