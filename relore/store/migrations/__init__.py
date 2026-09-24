"""Versioned, forward-only migrations.

Each step is recorded in ``schema_migrations``, so ``relored migrate`` is idempotent and
says what it actually did. Steps are dialect-aware: section 4.1's search layer is two
steps, each a no-op on the other dialect and recorded as applied either way so it is not
retried forever.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import Connection, Engine, func, insert, inspect, select, text

from relore.store.dialect import UTCDateTime, utcnow
from relore.store.schema import (
    documents_history,
    metadata,
    repo_sample,
    schema_migrations,
    thread_commits,
    thread_errors,
    thread_files,
    thread_links,
    thread_symbols,
    thread_tests,
)

log = logging.getLogger(__name__)


def _portable_core(conn: Connection) -> None:
    """Every table that must behave identically on both dialects."""
    metadata.create_all(bind=conn, checkfirst=True)


def _postgres_search_layer(conn: Connection) -> None:
    """tsvector, pg_trgm and pgvector -- the half of the schema that does not port.

    On SQLite this does nothing; :func:`_sqlite_search_layer` is its counterpart.
    """
    if conn.engine.dialect.name != "postgresql":
        log.info("skipping the Postgres search layer on %s", conn.engine.dialect.name)
        return

    # Required: milestone 3's ranking depends on both. Fail loudly if unavailable.
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    conn.execute(
        text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS fts tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', body_text)) STORED"
        )
    )
    conn.execute(text("CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING gin (fts)"))
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS thread_errors_trgm ON thread_errors "
            "USING gin (message_norm gin_trgm_ops)"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS thread_symbols_trgm ON thread_symbols "
            "USING gin (symbol gin_trgm_ops)"
        )
    )

    # Optional, and in its own savepoint: `CREATE EXTENSION vector` needs privileges the
    # application role often lacks, and a failed statement would otherwise abort the whole
    # migration transaction. Nothing writes the column in v1 (section 6.1), so an index
    # without a vector tier is fully functional -- warn and carry on rather than refusing
    # to migrate over a column no code reads yet.
    try:
        with conn.begin_nested():
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector(384)")
            )
    except Exception as exc:  # noqa: BLE001 -- the reason is reported, not swallowed
        log.warning(
            "pgvector not provisioned (%s); the embedding column is absent. "
            "v1 writes no vectors, so this is not fatal -- see build-plan section 6.1.",
            exc.__class__.__name__,
        )


#: The FTS5 index and the three triggers that maintain it. Written as one list so the
#: statements and the reason they exist stay next to each other.
_SQLITE_FTS = (
    # External content: the text lives in `documents` and FTS5 reads it back on demand,
    # which is what makes this an index rather than a second copy of the corpus.
    #
    # `porter unicode61` because the Postgres side is `to_tsvector('english', ...)`.
    # Neither the tokenizer nor the ranking function matches -- `bm25` is not
    # `ts_rank_cd`, and section 4.1 is emphatic that a number from one says nothing about
    # the other -- but stemming at least means "loading" finds "load" on both, so a
    # laptop is not surprising for a reason nobody would think to check.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        body_text,
        content='documents',
        content_rowid='id',
        tokenize='porter unicode61'
    )
    """,
    # Maintained by triggers rather than by the ingest path, because `repository.py` may
    # not know which dialect it is talking to (tests/unit/test_no_dialect_leak.py) and a
    # trigger keeps that true: the write path stays one portable upsert, and SQLite keeps
    # its own index in step underneath it. An upsert's DO UPDATE fires the update trigger,
    # so section 5.1's reconcile is covered in all four of its cases.
    """
    CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, body_text) VALUES (new.id, new.body_text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, body_text)
            VALUES ('delete', old.id, old.body_text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, body_text)
            VALUES ('delete', old.id, old.body_text);
        INSERT INTO documents_fts(rowid, body_text) VALUES (new.id, new.body_text);
    END
    """,
    # An index built under milestone 1 has documents but no FTS rows, and a laptop that
    # migrates and then cannot find anything looks like broken retrieval rather than a
    # missing backfill. `rebuild` is the external-content table's own idempotent
    # reconcile, and on an empty database it costs nothing.
    "INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')",
)


def _sqlite_search_layer(conn: Connection) -> None:
    """The SQLite half of section 4.1's search layer: FTS5, and no tier above it.

    No trigram and no vector index, now or later -- ``pg_trgm`` and ``pgvector`` have no
    SQLite equivalent worth the name, so the fuzzy symbol and error matching in section 6
    degrades to a ``LIKE`` scan here. That is a real capability gap, declared by the
    backend and printed by ``status`` rather than left for someone to infer from a thin
    result set.
    """
    if conn.engine.dialect.name != "sqlite":
        log.info("skipping the SQLite search layer on %s", conn.engine.dialect.name)
        return
    for statement in _SQLITE_FTS:
        conn.execute(text(statement))


def _sampled_floor(conn: Connection) -> None:
    """``repo_sample`` (section 10), for databases created before it existed.

    A table added to the portable core is created by step 1 on a fresh database and by
    nothing at all on an existing one -- ``create_all(checkfirst=True)`` only skips what
    is already there, and step 1 has long since been recorded as applied. So new portable
    tables need their own step. ``checkfirst`` keeps this a no-op on a fresh database,
    where step 1 has just created it.
    """
    repo_sample.create(bind=conn, checkfirst=True)


def _document_history(conn: Connection) -> None:
    """``documents_history`` (section 14.1, settled 2026-09-09: keep versions).

    Same shape as step 4 and for the same reason: a table added to the portable core needs
    its own step, because step 1 is already recorded as applied on every existing database.

    **It must land before the first production backfill**, which is the point of section
    15.1: from the moment a poll replaces a comment body, the previous text is gone, and no
    later migration can recover what GitHub no longer serves.
    """
    documents_history.create(bind=conn, checkfirst=True)


def _link_claims(conn: Connection) -> None:
    """``thread_links`` reshaped for section 13.3: a claim, plus its resolution.

    The old shape had a not-null foreign key on both ends, which is why nothing ever wrote
    it -- an edge was unstorable until its target thread was indexed, so the whole corpus
    pass had to exist before a single edge did. The new shape carries ``target_number``
    (known from the source thread alone) and a nullable ``target_thread_id``.

    **Recreated rather than altered**, because the table is empty by construction on every
    database that exists: nothing has ever populated it. SQLite cannot drop a NOT NULL
    without rewriting the table anyway. Rows here would mean a version of this code nobody
    has shipped, so the count is checked and a surprise refuses the migration rather than
    dropping somebody's data.
    """
    if conn.dialect.has_table(conn, "thread_links"):
        rows = conn.execute(select(func.count()).select_from(thread_links)).scalar_one()
        if rows:
            raise RuntimeError(
                f"thread_links holds {rows} rows; this step recreates the table and would "
                "drop them. Nothing in any shipped version writes it, so migrate by hand."
            )
        thread_links.drop(bind=conn)
    thread_links.create(bind=conn)


def _file_provenance(conn: Connection) -> None:
    """``thread_files.source`` (huggingface/relore#17).

    A column added to an existing table, so ``create_all`` will not do it -- the table has
    been there since step 1. Portable ``ADD COLUMN``, nullable, no backfill: the value is
    a **derived** one, so every existing row gets it from the next ``relored derive``, and
    until then a null reads as "this index predates the split" rather than as a claim
    about where the path came from. :meth:`SearchBackend.thread` says exactly that.
    """
    if "source" in {column["name"] for column in inspect(conn).get_columns("thread_files")}:
        return
    conn.execute(text("ALTER TABLE thread_files ADD COLUMN source TEXT"))


def _signal_first_seen(conn: Connection) -> None:
    """``first_seen_at`` on the five extracted signal tables (temporal-cutoff-map.md 8).

    Section 13's cutoff is enforced on ``documents`` and the five ``thread_*`` tables carry
    no date, so a path or a symbol first named *after* the cutoff was attributed to the
    whole thread and selected that thread's pre-cutoff documents. No future text reached
    the caller -- the leak was in the selection, not the payload -- but an as-of-``T``
    deployment could not have made the match, which inflates anything that filters or ranks
    structurally.

    Same shape as step 7 and for the same reason: columns on tables that have existed since
    step 1, so ``create_all(checkfirst=True)`` will not add them. Nullable, and **no
    backfill** -- the value is derived, so every row gets one from the next ``relored
    derive``, which reads ``raw_objects`` and needs no network. Until that runs, null reads
    as "this index predates the column", and the cutoff drops the row rather than admitting
    it: an unbounded query is unaffected, and a bounded one is conservative. That is the
    right direction for a migration whose whole purpose is to stop over-admitting.
    """
    column_type = UTCDateTime().compile(conn.dialect)
    for table in (thread_files, thread_symbols, thread_errors, thread_tests, thread_commits):
        if "first_seen_at" in {c["name"] for c in inspect(conn).get_columns(table.name)}:
            continue
        conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN first_seen_at {column_type}"))


MIGRATIONS: tuple[tuple[int, str, Callable[[Connection], None]], ...] = (
    (1, "portable_core", _portable_core),
    (2, "postgres_search_layer", _postgres_search_layer),
    (3, "sqlite_search_layer", _sqlite_search_layer),
    (4, "sampled_floor", _sampled_floor),
    (5, "document_history", _document_history),
    (6, "link_claims", _link_claims),
    (7, "file_provenance", _file_provenance),
    (8, "signal_first_seen", _signal_first_seen),
)


def applied_versions(engine: Engine) -> set[int]:
    schema_migrations.create(bind=engine, checkfirst=True)
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(select(schema_migrations.c.version))}


def migrate(engine: Engine) -> list[int]:
    """Apply every unapplied step. Returns the versions applied by *this* call."""
    already = applied_versions(engine)
    newly: list[int] = []
    for version, name, step in MIGRATIONS:
        if version in already:
            continue
        with engine.begin() as conn:
            log.info("applying migration %d (%s)", version, name)
            step(conn)
            conn.execute(insert(schema_migrations).values(version=version, applied_at=utcnow()))
        newly.append(version)
    return newly


def describe(engine: Engine) -> dict[str, Any]:
    already = applied_versions(engine)
    return {
        "dialect": engine.dialect.name,
        "applied": sorted(already),
        "pending": sorted(v for v, _n, _s in MIGRATIONS if v not in already),
    }
