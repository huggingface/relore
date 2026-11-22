"""The only module allowed to know whether it is talking to Postgres or SQLite.

``store/repository.py`` containing an ``if dialect ==`` is a bug, and
``tests/unit/test_no_dialect_leak.py`` says so. See the build plan section 4.1 for
why the portability is deliberately partial and where it stops.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Engine, Table, TypeDecorator, text
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.pool import StaticPool

UTC = dt.timezone.utc


def utcnow() -> dt.datetime:
    """The only clock this package reads.

    Timestamp defaults are set in Python, never in the DDL: ``func.now()`` renders as
    ``CURRENT_TIMESTAMP``, which is naive -- and on SQLite a *string*. Section 5.2's delta
    detection compares timestamps, and a naive/aware mix there silently loses threads.
    """
    return dt.datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """A timestamp that is tz-aware UTC on both dialects, or raises.

    Postgres round-trips ``timestamptz`` natively. SQLite has no such type: it stores
    whatever it is handed and returns it naive. So normalize on the way in and re-attach
    UTC on the way out, and **refuse a naive value at the boundary** rather than storing
    something whose meaning depends on the writer's machine.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: dt.datetime | None, dialect: Any) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "naive datetime reached the database boundary; "
                "use relore.store.dialect.utcnow() or attach tzinfo explicitly"
            )
        return value.astimezone(UTC)

    def process_result_value(self, value: dt.datetime | None, dialect: Any) -> dt.datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def pk_type() -> Any:
    """A surrogate key that autoincrements on both dialects.

    SQLite auto-increments only a column declared *exactly* ``INTEGER PRIMARY KEY``; a
    ``BIGINT`` is not a rowid alias and stays NULL. On Postgres this is ``bigserial``.
    """
    return BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


def json_type() -> Any:
    """``jsonb`` on Postgres, TEXT-backed JSON on SQLite.

    Never query *into* a payload: ``derive`` reads whole objects (section 5.0), so the
    difference costs nothing as long as nobody adds a JSON path predicate.
    """
    return JSON().with_variant(postgresql.JSONB(), "postgresql")


def make_engine(url: str, *, read_only: bool = False) -> Engine:
    """Section 11's read-only invariant, on both dialects.

    Postgres enforces it with a ``SELECT``-only role, which is the deployment's job and
    not something a connection string can claim. SQLite has a real equivalent -- a
    ``mode=ro`` URI -- and the API uses it, because portability is not a licence to lose
    an invariant.
    """
    url = normalize_url(url)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        if _is_memory(url):
            # An in-memory SQLite database lives *inside* one connection: with the default
            # pool, the second connection is a different, empty database. Everything here
            # opens more than one, so share exactly one.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        elif read_only:
            path = url.split("///", 1)[1] if "///" in url else ""
            if path and not path.startswith("file:"):
                url = f"sqlite:///file:{path}?mode=ro&uri=true"
    engine = _sa_create_engine(url, future=True, **kwargs)
    if engine.dialect.name == "sqlite":
        # Without this SQLite does not enforce the ON DELETE CASCADEs the schema declares,
        # so a deleted thread would leave orphaned documents behind and the reconcile in
        # section 5.1 would silently under-count.
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def normalize_url(url: str) -> str:
    """Point a bare ``postgresql://`` URL at the driver we actually ship.

    SQLAlchemy resolves ``postgresql://`` to psycopg2, and this project depends on
    psycopg 3 -- so the URL in every doc and every deployment note would fail with
    ``ModuleNotFoundError: psycopg2`` for no reason a reader could guess. Rewriting it
    here is cheaper than making every caller remember ``postgresql+psycopg://``.
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):  # the libpq/Heroku spelling
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    return url


def _is_memory(url: str) -> bool:
    return ":memory:" in url or url.rstrip("/") == "sqlite:"


def is_deployment_grade(engine_or_conn: Any) -> bool:
    """Whether this database is a *deployment* rather than a development affordance.

    Postgres is the only supported one (section 4.1); SQLite is what lets the tests run
    in-memory and a laptop try ``relored poll`` with nothing provisioned. The question
    lives here because this is the module allowed to know which database it is talking to
    -- ``relored serve`` asks it before binding a port, and gets an answer rather than a
    dialect name to compare.
    """
    return _dialect_name(engine_or_conn) == "postgresql"


def dialect_name(engine_or_conn: Any) -> str:
    """Which database this is, as a plain string.

    Exists so ``search/`` can pick its backend without becoming a second dialect seam.
    Section 4.1 puts retrieval behind one implementation *per* dialect precisely because
    the two engines are different -- so the choice is a lookup in a registry, and the
    answer to "which one" comes from here, the module allowed to know.
    """
    return _dialect_name(engine_or_conn)


def _dialect_name(engine_or_conn: Any) -> str:
    dialect = getattr(engine_or_conn, "dialect", None)
    if dialect is None:  # a Connection wrapping an Engine
        dialect = engine_or_conn.engine.dialect
    return str(dialect.name)


def upsert(
    conn: Any,
    table: Table,
    rows: list[dict[str, Any]],
    *,
    key: tuple[str, ...],
    update: tuple[str, ...] | None = None,
) -> None:
    """Insert-or-update ``rows`` keyed on ``key``.

    The construct is dialect-specific -- ``postgresql.insert`` and ``sqlite.insert`` each
    carry their own ``on_conflict_do_update`` -- which is the whole reason this function
    exists instead of the callers reaching for one directly.
    """
    if not rows:
        return
    module = postgresql if _dialect_name(conn) == "postgresql" else sqlite
    stmt = module.insert(table)
    if update is None:
        update = tuple(c.name for c in table.columns if c.name not in key and not c.primary_key)
    if update:
        stmt = stmt.on_conflict_do_update(
            index_elements=list(key),
            set_={name: stmt.excluded[name] for name in update},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(key))
    conn.execute(stmt, rows)


def has_vector_tier(engine_or_conn: Any) -> bool:
    """Whether ``documents.embedding`` exists: Postgres with pgvector provisioned.

    Migration 2 adds the column only when ``CREATE EXTENSION vector`` succeeds, so the
    dialect alone does not answer it.
    """
    if _dialect_name(engine_or_conn) != "postgresql":
        return False
    conn = engine_or_conn
    if hasattr(engine_or_conn, "connect"):
        with engine_or_conn.connect() as conn:
            return _has_embedding_column(conn)
    return _has_embedding_column(conn)


def _has_embedding_column(conn: Any) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'documents' AND column_name = 'embedding'"
            )
        ).first()
    )


def write_embeddings(conn: Any, model: str, rows: list[tuple[int, str, list[float]]]) -> int:
    """Store ``(document id, content_hash, vector)`` rows, each only if its text is still
    the text that was embedded. Returns how many were written.

    The hash guard is the race: a poll can rewrite a document between the read that fed the
    model and this write. Without it the old text's vector is stamped with the current
    model and served as current; with it the row stays unembedded for the next pass.
    """
    if not rows:
        return 0
    if not has_vector_tier(conn):
        raise RuntimeError("no vector tier: Postgres with pgvector is required to embed")
    statement = text(
        "UPDATE documents SET embedding = CAST(:vector AS vector), embedding_model = :model "
        "WHERE id = :id AND content_hash = :hash"
    )
    result = conn.execute(
        statement,
        [
            {
                "vector": "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]",
                "model": model,
                "id": document_id,
                "hash": content_hash,
            }
            for document_id, content_hash, vector in rows
        ],
    )
    return max(int(result.rowcount or 0), 0)
