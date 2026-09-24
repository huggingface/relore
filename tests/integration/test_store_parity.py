"""Dialect parity: the portable core must behave *identically* on both dialects.

Every test here runs twice -- once on in-memory SQLite, once on Postgres if
``RELORE_TEST_POSTGRES_URL`` is set. See tests/conftest.py, and section 4.1 for the four
traps these pin down.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine, delete, func, insert, select, text

from relore.store import repository as repo_layer
from relore.store import schema as s
from relore.store.dialect import UTC, upsert, utcnow
from relore.store.migrations import MIGRATIONS, describe, migrate


def _thread(engine: Engine, number: int = 1, **overrides: object) -> int:
    row = {
        "repo": "owner/name",
        "github_number": number,
        "thread_type": "issue",
        "title": f"thread {number}",
        "indexed_at": utcnow(),
        "metadata": {},
        **overrides,
    }
    with engine.begin() as conn:
        return repo_layer.upsert_thread(conn, row)


def test_migrate_is_idempotent(engine: Engine) -> None:
    assert migrate(engine) == [], "the fixture already migrated; a second call must be a no-op"
    state = describe(engine)
    assert state["pending"] == []
    # Every step, including the one that is a no-op on this dialect: section 4.1's search
    # layer is per-dialect, and a skipped step still has to be recorded or it is retried
    # for ever.
    assert state["applied"] == [version for version, _name, _step in MIGRATIONS]


def test_the_file_provenance_column_lands_on_a_database_that_predates_it(
    engine: Engine,
) -> None:
    """Step 7 is an ``ALTER TABLE`` rather than a ``create_all``: ``thread_files`` has
    existed since step 1, so a new column on it reaches an existing index only through its
    own step -- and has to be a no-op on a fresh one, where step 1 just created it with
    the column already there (huggingface/relore#17).
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE thread_files"))
        conn.execute(
            text("CREATE TABLE thread_files (thread_id BIGINT, path TEXT, change_type TEXT)")
        )

    _file_provenance_step()(engine)
    _file_provenance_step()(engine)  # idempotent: the column is checked for, not assumed

    with engine.connect() as conn:
        conn.execute(select(s.thread_files.c.source))


def test_the_first_seen_at_columns_land_on_a_database_that_predates_them(
    engine: Engine,
) -> None:
    """Step 8, same shape as step 7 and on five tables at once (temporal-cutoff-map.md 8).

    Deliberately no backfill: the value is derived, so the next ``relored derive`` fills it
    from ``raw_objects`` with no network. Until then it is null, and a null fails closed --
    an unbounded query is unaffected and a bounded one drops the row rather than admitting
    it, which is the safe direction for a migration whose purpose is to stop over-admitting.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE thread_symbols"))
        conn.execute(
            text(
                "CREATE TABLE thread_symbols (thread_id BIGINT, symbol TEXT, path TEXT, "
                "symbol_type TEXT)"
            )
        )

    _step(8)(engine)
    _step(8)(engine)  # idempotent: the column is checked for, not assumed

    with engine.connect() as conn:
        assert conn.execute(select(s.thread_symbols.c.first_seen_at)).all() == []
        # And a value written through it comes back aware, which is the whole reason the
        # DDL renders `UTCDateTime` per dialect rather than hardcoding one spelling.
        conn.execute(select(s.thread_files.c.first_seen_at))


def _file_provenance_step():
    return _step(7)


def _step(version: int):
    """One migration step, run outside :func:`migrate` because the fixture has already
    recorded every version as applied."""
    step = dict((v, function) for v, _name, function in MIGRATIONS)[version]

    def run(engine: Engine) -> None:
        with engine.begin() as conn:
            step(conn)

    return run


def test_surrogate_keys_autoincrement(engine: Engine) -> None:
    """SQLite auto-increments only a column declared exactly ``INTEGER PRIMARY KEY``.

    A ``BIGINT`` there is not a rowid alias and stays NULL, which is why every surrogate
    key goes through ``pk_type()``.
    """
    first = _thread(engine, 1)
    second = _thread(engine, 2)
    assert isinstance(first, int) and isinstance(second, int)
    assert second > first


def test_timestamps_come_back_tz_aware(engine: Engine) -> None:
    """Section 5.2 compares timestamps; a naive/aware mix there silently loses threads."""
    moment = dt.datetime(2026, 3, 1, 12, 30, tzinfo=UTC)
    _thread(engine, 7, updated_at=moment)
    with engine.connect() as conn:
        got = conn.execute(
            select(s.threads.c.updated_at).where(s.threads.c.github_number == 7)
        ).scalar_one()
    assert got.tzinfo is not None
    assert got == moment


def test_a_naive_datetime_is_refused_at_the_boundary(engine: Engine) -> None:
    """Better a loud failure than a value whose meaning depends on the writer's machine."""
    with pytest.raises(Exception, match="naive datetime"), engine.begin() as conn:
        conn.execute(
            insert(s.threads).values(
                repo="owner/name",
                github_number=99,
                thread_type="issue",
                title="x",
                indexed_at=dt.datetime(2026, 1, 1, 0, 0),  # noqa: DTZ001 -- the point
                metadata={},
            )
        )


def test_upsert_updates_rather_than_duplicating(engine: Engine) -> None:
    _thread(engine, 3, title="before")
    _thread(engine, 3, title="after")
    with engine.connect() as conn:
        rows = conn.execute(select(s.threads.c.title).where(s.threads.c.github_number == 3)).all()
    assert [r.title for r in rows] == ["after"]


def test_upsert_with_no_updatable_columns_does_not_raise(engine: Engine) -> None:
    """``thread_labels`` is all key and no payload, so the ON CONFLICT has nothing to set."""
    thread_id = _thread(engine, 4)
    for _ in range(2):
        with engine.begin() as conn:
            upsert(
                conn,
                s.thread_labels,
                [{"thread_id": thread_id, "label": "bug"}],
                key=("thread_id", "label"),
            )
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(s.thread_labels)).scalar_one() == 1


def test_deleting_a_thread_cascades_to_its_documents(engine: Engine) -> None:
    """SQLite ignores ON DELETE unless foreign keys are switched on per connection."""
    thread_id = _thread(engine, 5)
    with engine.begin() as conn:
        repo_layer.reconcile_documents(
            conn,
            thread_id,
            [
                {
                    "source_type": "body",
                    "source_id": "5",
                    "chunk_index": 0,
                    "body_markdown": "b",
                    "body_text": "b",
                    "content_hash": "h",
                    "chunking_version": 1,
                    "extractor_version": 1,
                    "metadata": {},
                }
            ],
        )
    with engine.begin() as conn:
        conn.execute(delete(s.threads).where(s.threads.c.id == thread_id))
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(s.documents)).scalar_one() == 0


def test_json_columns_round_trip(engine: Engine) -> None:
    payload = {"draft": False, "nested": {"a": [1, 2, 3]}, "text": "héllo"}
    _thread(engine, 6, metadata=payload)
    with engine.connect() as conn:
        got = conn.execute(
            select(s.threads.c.metadata).where(s.threads.c.github_number == 6)
        ).scalar_one()
    assert got == payload


def test_a_nul_byte_in_a_payload_is_stageable_on_both_dialects(engine: Engine) -> None:
    """SQLite stores U+0000, Postgres JSONB refuses it -- and the pass wedges at the thread
    it cannot index rather than skipping it."""
    with engine.begin() as conn:
        repo_layer.stage_raw(
            conn,
            "owner/name",
            [
                {
                    "repo": "owner/name",
                    "object_type": "issue_comment",
                    "object_id": "1",
                    "thread_number": 1,
                    "payload": {"body": "token = ghp_\x00abc", "nested": ["a\x00b"]},
                    "fetched_at": utcnow(),
                    "github_updated_at": None,
                }
            ],
        )
    with engine.connect() as conn:
        payload = conn.execute(select(s.raw_objects.c.payload)).scalar_one()
    assert payload == {"body": "token = ghp_abc", "nested": ["ab"]}


def test_the_high_water_mark_never_moves_backwards(engine: Engine) -> None:
    later = dt.datetime(2026, 5, 1, tzinfo=UTC)
    earlier = dt.datetime(2026, 1, 1, tzinfo=UTC)
    with engine.begin() as conn:
        repo_layer.advance_high_water(conn, "owner/name", "threads", later)
        repo_layer.advance_high_water(conn, "owner/name", "threads", earlier)
    with engine.connect() as conn:
        assert repo_layer.high_water(conn, "owner/name", "threads") == later
