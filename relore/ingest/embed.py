"""``relored embed``: fill the vector tier for one repository (section 6.1).

A document needs a vector when its ``embedding_model`` is not the configured model: never
embedded, embedded by another model, or rewritten since (the reconcile resets the column on
every write). So the pass is idempotent -- a second run over an unchanged index embeds
nothing -- and resumable, since it walks by id and commits per batch.

``relored poll`` runs it after every pass when ``RELORE_EMBED_MODEL`` is set, which is what
keeps the tier current; this module is also the backfill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import Engine, func, or_, select

from relore.store import schema as s
from relore.store.dialect import write_embeddings

log = logging.getLogger(__name__)

#: What one poll pass may embed. A fresh index has its whole history pending, and an
#: unbounded hook would hold the poll loop for hours; the history is ``pass.sh embed``'s.
POLL_LIMIT = 1024

#: Documents per transaction. Large enough that the embedder's length sort has lengths to
#: group; the model itself is called in much smaller batches (``EMBED_BATCH``).
BATCH = 1024


class DocumentEmbedder(Protocol):
    model: str

    def documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbedResult:
    embedded: int
    #: Read, then rewritten by a poll before the write landed. Left for the next pass.
    changed: int


def _stale(repo: str, model: str):
    return (
        s.documents.c.thread_id.in_(select(s.threads.c.id).where(s.threads.c.repo == repo)),
        or_(s.documents.c.embedding_model.is_(None), s.documents.c.embedding_model != model),
    )


def pending(engine: Engine, repo: str, model: str) -> int:
    """How many of ``repo``'s documents lack a ``model`` vector."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(s.documents).where(*_stale(repo, model))
            ).scalar()
            or 0
        )


def embed_pending(
    engine: Engine,
    repo: str,
    embedder: DocumentEmbedder,
    *,
    batch: int = BATCH,
    limit: int | None = None,
    newest_first: bool = False,
) -> EmbedResult:
    """``newest_first`` is the poll hook's order: what the pass just wrote has the highest
    ids, and walking down while a backfill Job walks up keeps the two off the same rows."""
    embedded = changed = 0
    bound: int | None = None
    while limit is None or embedded + changed < limit:
        size = batch if limit is None else min(batch, limit - embedded - changed)
        column = s.documents.c.id
        position = (
            () if bound is None else ((column < bound) if newest_first else (column > bound),)
        )
        with engine.connect() as conn:
            rows = conn.execute(
                select(column, s.documents.c.content_hash, s.documents.c.body_text)
                .where(*_stale(repo, embedder.model), *position)
                .order_by(column.desc() if newest_first else column)
                .limit(size)
            ).all()
        if not rows:
            break
        vectors = embedder.documents([row.body_text or "" for row in rows])
        with engine.begin() as conn:
            wrote = write_embeddings(
                conn,
                embedder.model,
                [
                    (int(row.id), str(row.content_hash), vector)
                    for row, vector in zip(rows, vectors, strict=True)
                ],
            )
        embedded += wrote
        changed += len(rows) - wrote
        bound = int(rows[-1].id)
        log.info("%s: %d embedded (%s)", repo, embedded, embedder.model)
    return EmbedResult(embedded=embedded, changed=changed)
