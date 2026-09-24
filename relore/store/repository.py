"""Data access. **No dialect branches live here** -- they belong in ``store/dialect.py``,
and ``tests/unit/test_no_dialect_leak.py`` asserts it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, delete, func, insert, select

from relore.store import schema as s
from relore.store.dialect import upsert, utcnow

DocumentKey = tuple[str, str, int]

#: Derived columns whose value comes from state *outside* this document's own payload,
#: and which therefore cannot ride on ``content_hash`` (section 5.1).
#:
#: ``trust`` is derived from the payload **plus** ``repo_authority`` (section 6.2), so
#: resolving an author's write access changes the tier without changing a byte of the
#: text. Comparing hashes alone would report "unchanged" and leave the stale tier in
#: place for ever -- and section 6.2 requires that a rebuild reproduce the tier, which is
#: only true if the reconcile can see it move. ``enclosing_symbol`` joins this tuple when
#: the code lens lands (milestone 4); it has the same shape of input.
DERIVED_COLUMNS: tuple[str, ...] = ("trust", "author_is_bot")


@dataclass(frozen=True)
class ReconcileStats:
    """What section 5.1's set diff actually did.

    ``unchanged`` is the overwhelming majority in steady state, and a poll of an
    untouched thread must report zeroes everywhere else -- that is the idempotency claim,
    and it is a test rather than an aspiration.
    """

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0

    @property
    def wrote(self) -> int:
        return self.inserted + self.updated + self.deleted


@dataclass(frozen=True)
class SignalStats:
    """What :func:`reconcile_signals` did. ``rewritten`` is empty in steady state."""

    rows: int = 0
    rewritten: dict[str, int] = field(default_factory=dict)

    @property
    def wrote(self) -> int:
        return sum(self.rewritten.values())


# -- raw staging -----------------------------------------------------------


def stage_raw(conn: Connection, repo: str, rows: list[dict[str, Any]]) -> None:
    rows = [{**row, "payload": _without_nul(row["payload"])} for row in rows]
    upsert(conn, s.raw_objects, rows, key=("repo", "object_type", "object_id"))


def _without_nul(payload: Any) -> Any:
    """Drop U+0000 on the way into ``raw_objects`` -- the one place staging is not verbatim.

    Postgres JSONB rejects it and SQLite does not, and the pass wedges at a thread it cannot
    index (section 5.2 rule 5) rather than skipping it. `scrub` removes it at serve time
    anyway, so no reader loses anything.
    """
    if isinstance(payload, str):
        return payload.replace("\x00", "") if "\x00" in payload else payload
    if isinstance(payload, dict):
        return {key: _without_nul(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_without_nul(value) for value in payload]
    return payload


def prune_raw(
    conn: Connection,
    repo: str,
    number: int,
    *,
    keep: set[tuple[str, str]],
    object_types: tuple[str, ...],
) -> int:
    """Drop staged objects of ``object_types`` for this thread that the fetch did not see.

    Staging is an upsert, so without this a deleted comment stays in ``raw_objects``
    forever and every subsequent derive faithfully re-emits its document -- the deletion
    row of section 5.1's table would never fire, and nothing would look wrong.

    Scoped to the object types the caller is *authoritative* for. A poll re-reads a
    thread's conversation and reviews, so it may prune those; it knows nothing about
    ``pr_details``, which another pass stages against the same thread number, so it must
    not touch them.
    """
    existing = {
        (row.object_type, row.object_id)
        for row in conn.execute(
            select(s.raw_objects.c.object_type, s.raw_objects.c.object_id).where(
                s.raw_objects.c.repo == repo,
                s.raw_objects.c.thread_number == number,
                s.raw_objects.c.object_type.in_(object_types),
            )
        )
    }
    stale = existing - keep
    for object_type, object_id in stale:
        conn.execute(
            delete(s.raw_objects).where(
                s.raw_objects.c.repo == repo,
                s.raw_objects.c.object_type == object_type,
                s.raw_objects.c.object_id == object_id,
            )
        )
    return len(stale)


def load_raw(conn: Connection, repo: str, number: int) -> dict[str, list[dict[str, Any]]]:
    """Every staged object for one thread, grouped by ``object_type``.

    Derive reads from here and never from a fetch result, so a field that was not staged
    cannot silently work at poll time and vanish on a rebuild.
    """
    stmt = select(
        s.raw_objects.c.object_type, s.raw_objects.c.object_id, s.raw_objects.c.payload
    ).where(s.raw_objects.c.repo == repo, s.raw_objects.c.thread_number == number)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for object_type, object_id, payload in conn.execute(stmt):
        grouped.setdefault(object_type, []).append({"_object_id": object_id, **payload})
    return grouped


def staged_thread_numbers(conn: Connection, repo: str, *, after: int | None = None) -> list[int]:
    stmt = select(s.raw_objects.c.thread_number.distinct()).where(
        s.raw_objects.c.repo == repo, s.raw_objects.c.thread_number.isnot(None)
    )
    if after is not None:
        stmt = stmt.where(s.raw_objects.c.thread_number > after)
    return sorted(int(n) for (n,) in conn.execute(stmt))


def merged_pr_numbers(conn: Connection, repo: str, *, after: int | None = None) -> list[int]:
    """Merged PRs, ascending. What counts as precedent (section 3): an unmerged PR's file
    list is not evidence of anything that shipped.

    Read off ``threads.merged_at`` rather than out of a raw payload: section 4.1 forbids
    querying *into* JSON, since a JSON path predicate is not portable.
    """
    stmt = select(s.threads.c.github_number).where(
        s.threads.c.repo == repo,
        s.threads.c.thread_type == "pr",
        s.threads.c.merged_at.isnot(None),
    )
    if after is not None:
        stmt = stmt.where(s.threads.c.github_number > after)
    return sorted(int(n) for (n,) in conn.execute(stmt))


def open_pr_numbers(conn: Connection, repo: str) -> list[int]:
    """Still-open PRs, ascending.

    Separate from :func:`merged_pr_numbers` because the two answer different questions and
    are staged under different rules. A merged PR's diff is final, so it is fetched once
    and skipped forever after. An open one is still moving, so it is re-fetched every run
    -- which is affordable precisely because "open" is a small, self-limiting set.

    Closed-unmerged PRs are in neither: they did not ship and are not in flight.
    """
    stmt = select(s.threads.c.github_number).where(
        s.threads.c.repo == repo,
        s.threads.c.thread_type == "pr",
        s.threads.c.state == "open",
        # Both conditions, not just the state: a merged PR is not in flight whatever its
        # state column says, and re-walking it every run would be the expensive half of
        # the corpus paying the open half's refresh cost.
        s.threads.c.merged_at.is_(None),
    )
    return sorted(int(n) for (n,) in conn.execute(stmt))


def staged_object_ids(conn: Connection, repo: str, object_type: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            select(s.raw_objects.c.object_id).where(
                s.raw_objects.c.repo == repo, s.raw_objects.c.object_type == object_type
            )
        )
    }


def threads_of_objects(
    conn: Connection, repo: str, object_type: str, object_ids: set[str]
) -> set[int]:
    if not object_ids:
        return set()
    return {
        int(row[0])
        for row in conn.execute(
            select(s.raw_objects.c.thread_number).where(
                s.raw_objects.c.repo == repo,
                s.raw_objects.c.object_type == object_type,
                s.raw_objects.c.object_id.in_(sorted(object_ids)),
                s.raw_objects.c.thread_number.isnot(None),
            )
        )
    }


def delete_raw(conn: Connection, repo: str, object_type: str, object_ids: set[str]) -> int:
    if not object_ids:
        return 0
    result = conn.execute(
        delete(s.raw_objects).where(
            s.raw_objects.c.repo == repo,
            s.raw_objects.c.object_type == object_type,
            s.raw_objects.c.object_id.in_(sorted(object_ids)),
        )
    )
    return int(result.rowcount or 0)


# -- threads ---------------------------------------------------------------


def upsert_thread(conn: Connection, row: dict[str, Any]) -> int:
    upsert(conn, s.threads, [row], key=("repo", "github_number"))
    thread_id = conn.execute(
        select(s.threads.c.id).where(
            s.threads.c.repo == row["repo"],
            s.threads.c.github_number == row["github_number"],
        )
    ).scalar_one()
    return int(thread_id)


def set_labels(conn: Connection, thread_id: int, labels: list[str]) -> None:
    conn.execute(delete(s.thread_labels).where(s.thread_labels.c.thread_id == thread_id))
    if labels:
        upsert(
            conn,
            s.thread_labels,
            [{"thread_id": thread_id, "label": label} for label in sorted(set(labels))],
            key=("thread_id", "label"),
        )


# -- documents -------------------------------------------------------------


def reconcile_documents(
    conn: Connection, thread_id: int, desired: list[dict[str, Any]]
) -> ReconcileStats:
    """Section 5.1's table, as one function.

    present/absent -> insert; hash differs -> update; hash equal -> **no-op**;
    absent/present -> delete, which is how a removed comment leaves the index.

    "Hash equal" means equal on ``content_hash`` *and* on :data:`DERIVED_COLUMNS`, whose
    inputs live outside the document's payload. The hash is still the identity of the
    document's content; it was never the identity of the row.
    """
    compare = ("content_hash", *DERIVED_COLUMNS)
    stored: dict[DocumentKey, tuple[Any, ...]] = {
        (row.source_type, row.source_id, row.chunk_index): tuple(
            getattr(row, column) for column in compare
        )
        for row in conn.execute(
            select(
                s.documents.c.source_type,
                s.documents.c.source_id,
                s.documents.c.chunk_index,
                *(s.documents.c[column] for column in compare),
            ).where(s.documents.c.thread_id == thread_id)
        )
    }
    wanted = {(row["source_type"], row["source_id"], row["chunk_index"]): row for row in desired}

    to_write: list[dict[str, Any]] = []
    superseded: list[tuple[DocumentKey, str]] = []
    inserted = updated = unchanged = 0
    for key, row in wanted.items():
        if key not in stored:
            inserted += 1
            to_write.append(row)
        elif stored[key] != tuple(row[column] for column in compare):
            updated += 1
            to_write.append(row)
            # Only a *text* change supersedes anything. A promotion between trust tiers
            # moves a DERIVED_COLUMN without changing a byte of the document (section 5.1),
            # and recording that as an edit would fill the history with rows in which
            # nothing the author wrote is different.
            if stored[key][0] != row["content_hash"]:
                superseded.append((key, "edited"))
        else:
            unchanged += 1

    stale = [key for key in stored if key not in wanted]
    superseded.extend((key, "deleted") for key in stale)

    # Strictly before the upsert and before the delete, inside the same transaction: this
    # is the only moment the previous text still exists. Running it afterwards reads back
    # the *replacement* and records that as the old version -- which is not a crash, it is
    # a history that quietly says every document always said what it says now.
    keep_versions(conn, thread_id, superseded)

    if to_write:
        upsert(
            conn,
            s.documents,
            [{"thread_id": thread_id, **row} for row in to_write],
            key=("thread_id", "source_type", "source_id", "chunk_index"),
        )

    for source_type, source_id, chunk_index in stale:
        conn.execute(
            delete(s.documents).where(
                s.documents.c.thread_id == thread_id,
                s.documents.c.source_type == source_type,
                s.documents.c.source_id == source_id,
                s.documents.c.chunk_index == chunk_index,
            )
        )

    return ReconcileStats(inserted, updated, unchanged, len(stale))


def keep_versions(
    conn: Connection, thread_id: int, superseded: Sequence[tuple[DocumentKey, str]]
) -> int:
    """Section 14.1: copy a document's current text into ``documents_history`` before it is
    replaced or removed. Append-only; returns how many rows were written.

    Called from inside :func:`reconcile_documents`' transaction on purpose. The history and
    the replacement have to commit together or not at all -- a history row for a write that
    rolled back is a version of a document that never existed, and the reverse loses the
    only copy.

    Three properties this has to keep, and each is a way it could quietly go wrong:

    * **It reads ``documents``, never ``raw_objects``.** Section 11.3 redacts on the way
      into ``documents``, so copying the stored row is what makes the history redacted too.
      Re-reading the staged payload here would un-redact every secret the ingest path had
      removed, in a table nobody looks at.
    * **It writes nothing when nothing changed.** Section 12's load-bearing assertion is
      that re-indexing an unchanged thread performs **zero** document writes; a history row
      is a document write. Only text changes and deletions reach here.
    * **It is not retrievable.** Nothing searches this table and no API returns it. Section
      6.2's tiers and section 11's envelope are about what is served, and a superseded
      claim served as current would be worse than not having kept it. ``relored status``
      counts the rows so the capture is verifiable; a read verb is a later decision.
    """
    if not superseded:
        return 0
    reasons = dict(superseded)
    rows = [
        {
            "thread_id": thread_id,
            "source_type": row.source_type,
            "source_id": row.source_id,
            "chunk_index": row.chunk_index,
            "body_markdown": row.body_markdown,
            "body_text": row.body_text,
            "content_hash": row.content_hash,
            "author": row.author,
            "github_updated_at": row.github_updated_at,
            "superseded_at": utcnow(),
            "reason": reasons[(row.source_type, row.source_id, row.chunk_index)],
        }
        for row in conn.execute(
            select(
                s.documents.c.source_type,
                s.documents.c.source_id,
                s.documents.c.chunk_index,
                s.documents.c.body_markdown,
                s.documents.c.body_text,
                s.documents.c.content_hash,
                s.documents.c.author,
                s.documents.c.github_updated_at,
            ).where(s.documents.c.thread_id == thread_id)
        )
        if (row.source_type, row.source_id, row.chunk_index) in reasons
    ]
    if rows:
        conn.execute(insert(s.documents_history), rows)
    return len(rows)


# -- signals (section 5.3) -------------------------------------------------

#: The signal tables and the columns that *are* the row. There is no surrogate key and no
#: unique constraint: a signal row is its own value, so the identity of a set of them is
#: the tuple, and :func:`reconcile_signals` compares sets rather than keys. A duplicate is
#: unrepresentable rather than rejected -- both sides of that comparison are sets.
#:
#: ``first_seen_at`` is on the five extracted tables and not on ``links``: section 6's
#: ``w_rel`` is 0, so no query filters or scores by an edge yet, and a column no predicate
#: reads is the mistake section 5.3 declines for keywords. It is part of the *row* like
#: ``trust`` and ``target_thread_id`` are -- a value that appears earlier than the stored
#: one is a different row and the reconcile has to notice, which it does because the
#: comparison below is over exactly these columns.
SIGNAL_TABLES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("files", s.thread_files, ("path", "change_type", "source", "first_seen_at")),
    ("symbols", s.thread_symbols, ("symbol", "path", "symbol_type", "first_seen_at")),
    ("errors", s.thread_errors, ("exception_type", "message_norm", "first_seen_at")),
    ("tests", s.thread_tests, ("test_id", "test_function", "first_seen_at")),
    ("commits", s.thread_commits, ("sha", "message", "first_seen_at")),
    # Section 13.3. `target_thread_id` is part of the row rather than an update to it, so
    # a target that becomes indexed later changes the row and the reconcile notices --
    # the same reason `trust` is in DERIVED_COLUMNS.
    ("links", s.thread_links, ("relationship", "target_number", "target_thread_id")),
)


def reconcile_signals(
    conn: Connection, thread_id: int, signals: Mapping[str, Sequence[Mapping[str, Any]]]
) -> SignalStats:
    """Replace one thread's signal rows, writing nothing when nothing moved.

    Per table: compare the stored set of rows against the extracted one and, only if they
    differ, delete this thread's rows and insert the new set. Whole-table-per-thread
    rather than row-by-row because these sets are small (tens of rows) and a row has no
    key to update *by* -- and because the property that has to hold is the one section
    5.1 states for documents: re-deriving an unchanged thread performs **zero** writes.
    Extraction is not free of that rule just because its rows are cheap; a poll that
    rewrites every signal table on every pass makes the idempotency assertion untestable
    and turns a quiet index into a write-heavy one.

    ``signals`` is a plain mapping keyed by the names in :data:`SIGNAL_TABLES` -- the
    store layer does not import the extractor, and does not need to.
    """
    changed: dict[str, int] = {}
    for name, table, columns in SIGNAL_TABLES:
        stored = {
            tuple(row)
            for row in conn.execute(
                select(*(table.c[column] for column in columns)).where(
                    table.c.thread_id == thread_id
                )
            )
        }
        wanted = {tuple(row[column] for column in columns) for row in signals.get(name, ())}
        if stored == wanted:
            continue
        conn.execute(delete(table).where(table.c.thread_id == thread_id))
        if wanted:
            conn.execute(
                insert(table),
                [
                    {"thread_id": thread_id, **dict(zip(columns, row, strict=True))}
                    for row in sorted(
                        wanted, key=lambda r: tuple("" if v is None else str(v) for v in r)
                    )
                ],
            )
        changed[name] = len(wanted)
    return SignalStats(
        rows=sum(len(signals.get(name, ())) for name, _t, _c in SIGNAL_TABLES), rewritten=changed
    )


# -- authority (section 6.2) -----------------------------------------------


def get_authority(conn: Connection, repo: str) -> dict[str, str]:
    """``login -> permission`` for every author already resolved on this repository.

    Read once per derive rather than per document: it is a few hundred rows on a
    repository with tens of thousands of threads, because only ``MEMBER`` authors are
    ever resolved.
    """
    return {
        row.login: row.permission
        for row in conn.execute(
            select(s.repo_authority.c.login, s.repo_authority.c.permission).where(
                s.repo_authority.c.repo == repo
            )
        )
        if row.permission is not None
    }


def authority_checked_at(conn: Connection, repo: str) -> dict[str, dt.datetime]:
    """When each login was last resolved, so a refresh can skip the fresh ones."""
    return {
        row.login: row.checked_at
        for row in conn.execute(
            select(s.repo_authority.c.login, s.repo_authority.c.checked_at).where(
                s.repo_authority.c.repo == repo
            )
        )
    }


def upsert_authority(conn: Connection, rows: list[dict[str, Any]]) -> None:
    if rows:
        upsert(conn, s.repo_authority, rows, key=("repo", "login"))


def member_authors(conn: Connection, repo: str) -> list[str]:
    """Distinct human authors presenting as ``MEMBER`` -- the only tier worth resolving.

    ``OWNER`` and ``COLLABORATOR`` are already unambiguous write access and ``CONTRIBUTOR``
    /``NONE`` are not organisation members at all, so neither costs a request (section
    6.2). This is what bounds the resolution pass to a few hundred calls rather than one
    per document.
    """
    return sorted(
        {
            row.author
            for row in conn.execute(
                select(s.documents.c.author.distinct())
                .select_from(s.documents.join(s.threads, s.documents.c.thread_id == s.threads.c.id))
                .where(
                    s.threads.c.repo == repo,
                    s.documents.c.author_assoc == "MEMBER",
                    s.documents.c.author_is_bot.is_(False),
                    s.documents.c.author.isnot(None),
                )
            )
            if row.author
        }
    )


def merged_by_logins(conn: Connection, repo: str) -> set[str]:
    """Everyone who has ever merged a pull request here.

    Section 6.2's fallback, for a token without push access on the repository: merging
    *is* the write act, so the answer is definitive where it is positive. Read out of
    ``threads.metadata`` in Python rather than with a JSON operator, because ``->>`` and
    ``json_extract`` are not the same function and a dialect branch does not belong in
    this module.
    """
    return {
        (row[0] or {}).get("merged_by")  # indexed, not `row.metadata`: `Row` shadows it
        for row in conn.execute(
            select(s.threads.c.metadata).where(
                s.threads.c.repo == repo, s.threads.c.merged_at.isnot(None)
            )
        )
    } - {None}


def thread_ids_by_number(conn: Connection, repo: str, numbers: Iterable[int]) -> dict[int, int]:
    """``github_number -> threads.id`` for the numbers that are indexed.

    A link's target is a number the source thread named, so resolving it to a row is a
    lookup that can legitimately come back empty: an open pull request may well close an
    issue nobody has indexed yet. The caller stores the claim either way (section 13.3).
    """
    wanted = list(dict.fromkeys(numbers))
    if not wanted:
        return {}
    return {
        int(number): int(thread_id)
        for number, thread_id in conn.execute(
            select(s.threads.c.github_number, s.threads.c.id).where(
                s.threads.c.repo == repo, s.threads.c.github_number.in_(wanted)
            )
        )
    }


def threads_authored_by(conn: Connection, repo: str, logins: set[str]) -> list[int]:
    """Thread numbers carrying a document by any of ``logins``.

    A resolution pass re-derives these and nothing else: a tier that did not move needs
    no write, and on a large repository most threads have no ``MEMBER`` author at all.
    """
    if not logins:
        return []
    return sorted(
        {
            row.github_number
            for row in conn.execute(
                select(s.threads.c.github_number.distinct())
                .select_from(s.threads.join(s.documents, s.documents.c.thread_id == s.threads.c.id))
                .where(s.threads.c.repo == repo, s.documents.c.author.in_(sorted(logins)))
            )
        }
    )


def thread_freshness(conn: Connection, repo: str, thread_type: str) -> dict[int, dt.datetime]:
    """``github_number -> updated_at`` for one kind of thread.

    Read in one query so a widening sample (section 5.5) can tell, from the listing
    payload it already has, which threads it is holding a current copy of -- and skip the
    two-to-five requests it would take to re-confirm that.
    """
    return {
        int(row.github_number): row.updated_at
        for row in conn.execute(
            select(s.threads.c.github_number, s.threads.c.updated_at).where(
                s.threads.c.repo == repo, s.threads.c.thread_type == thread_type
            )
        )
        if row.updated_at is not None
    }


# -- the sampled floor (section 10) ----------------------------------------


def record_sample(
    conn: Connection,
    repo: str,
    thread_type: str,
    *,
    indexed_from: dt.datetime,
    selector: str,
) -> None:
    """Declare that this corpus starts somewhere.

    Written before the first thread is fetched, so a run killed after one thread has
    still put a floor under what it built. Re-sampling the same type with an *earlier*
    floor lowers it; a later one does not raise it, because the rows already indexed are
    still there and the corpus really does start where it started.
    """
    current = conn.execute(
        select(s.repo_sample.c.indexed_from).where(
            s.repo_sample.c.repo == repo, s.repo_sample.c.thread_type == thread_type
        )
    ).scalar_one_or_none()
    upsert(
        conn,
        s.repo_sample,
        [
            {
                "repo": repo,
                "thread_type": thread_type,
                "indexed_from": min(indexed_from, current) if current else indexed_from,
                "selector": selector,
                "sampled_at": utcnow(),
            }
        ],
        key=("repo", "thread_type"),
    )


def samples(conn: Connection, repo: str | None = None) -> list[dict[str, Any]]:
    """Every declared floor, for ``status`` to print. Empty means no repository here was
    sampled -- which is the only thing that makes a recall number comparable with one from
    a full index (section 10)."""
    query = select(s.repo_sample).order_by(s.repo_sample.c.repo, s.repo_sample.c.thread_type)
    if repo is not None:
        query = query.where(s.repo_sample.c.repo == repo)
    return [
        {
            "repo": row.repo,
            "thread_type": row.thread_type,
            "indexed_from": row.indexed_from,
            "selector": row.selector,
            "sampled_at": row.sampled_at,
        }
        for row in conn.execute(query)
    ]


# -- sync state ------------------------------------------------------------


def high_water(conn: Connection, repo: str, pass_name: str) -> dt.datetime | None:
    return conn.execute(
        select(s.sync_state.c.high_water).where(
            s.sync_state.c.repo == repo, s.sync_state.c["pass"] == pass_name
        )
    ).scalar_one_or_none()


def advance_high_water(conn: Connection, repo: str, pass_name: str, moment: dt.datetime) -> None:
    """Move the checkpoint forward, never backward.

    Section 5.2 rule 2: the mark advances per *committed* thread, to that thread's
    ``updated_at``. Monotonicity is enforced here rather than trusted, because threads
    arrive sorted only as long as nothing retries out of order.
    """
    current = high_water(conn, repo, pass_name)
    if current is not None and moment <= current:
        touch_pass(conn, repo, pass_name)
        return
    upsert(
        conn,
        s.sync_state,
        [{"repo": repo, "pass": pass_name, "high_water": moment, "last_run_at": utcnow()}],
        key=("repo", "pass"),
        update=("high_water", "last_run_at"),
    )


def get_cursor(conn: Connection, repo: str, pass_name: str) -> str | None:
    return conn.execute(
        select(s.sync_state.c.cursor).where(
            s.sync_state.c.repo == repo, s.sync_state.c["pass"] == pass_name
        )
    ).scalar_one_or_none()


def set_cursor(conn: Connection, repo: str, pass_name: str, value: str | None) -> None:
    """Where an *interrupted* pass got to. Cleared when the pass completes.

    Distinct from ``high_water``, which is an incremental position and persists: a cursor
    that survived a completed pass would make the next run skip work it should redo, and
    "a re-run mutates zero documents" would pass for the wrong reason.
    """
    upsert(
        conn,
        s.sync_state,
        [{"repo": repo, "pass": pass_name, "cursor": value, "last_run_at": utcnow()}],
        key=("repo", "pass"),
        update=("cursor", "last_run_at"),
    )


def touch_pass(conn: Connection, repo: str, pass_name: str, *, ok: bool = False) -> None:
    row: dict[str, Any] = {"repo": repo, "pass": pass_name, "last_run_at": utcnow()}
    update = ["last_run_at"]
    if ok:
        row["last_ok_at"] = row["last_run_at"]
        update.append("last_ok_at")
    upsert(conn, s.sync_state, [row], key=("repo", "pass"), update=tuple(update))


# -- status ----------------------------------------------------------------


def index_summary(conn: Connection) -> dict[str, Any]:
    from relore.freshness import COLLECTOR_NOTE, pass_note

    def count(table: Any) -> int:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())

    passes = [
        {
            "repo": row.repo,
            "pass": getattr(row, "pass"),
            "note": pass_note(getattr(row, "pass")),
            "high_water": row.high_water.isoformat() if row.high_water else None,
            "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
            "last_ok_at": row.last_ok_at.isoformat() if row.last_ok_at else None,
            # Where a long pass has got to, and the reason a reader can tell a pass that
            # is *running* from one that stopped: `last_run_at` moves per committed
            # thread, `last_ok_at` only when the pass finishes, and the cursor is cleared
            # on completion. So run-after-ok means in flight, and a cursor says how far.
            "cursor": row.cursor,
        }
        for row in conn.execute(select(s.sync_state).order_by(s.sync_state.c.repo))
    ]
    return {
        "threads": count(s.threads),
        "documents": count(s.documents),
        "raw_objects": count(s.raw_objects),
        # Section 14.1. Nothing searches this table, so the count is the only way to see
        # that versions are being kept at all -- and "0 after a month of polling" is the
        # symptom of a capture that silently stopped working.
        "superseded_documents": count(s.documents_history),
        "passes": passes,
        "passes_note": COLLECTOR_NOTE,
        # Section 10: a recall number from a sampled corpus is only comparable against a
        # baseline restricted to the same window, so the window is part of the status.
        "samples": [
            {
                **row,
                "indexed_from": row["indexed_from"].isoformat(),
                "sampled_at": row["sampled_at"].isoformat(),
            }
            for row in samples(conn)
        ],
    }
