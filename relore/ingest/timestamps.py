"""Reading and writing GitHub's timestamps, in one place.

Both directions matter and both have a trap. Parsing: ``fromisoformat`` does not accept a
trailing ``Z`` before Python 3.11 and this project's floor is 3.10, so the substitution is
required rather than defensive. Formatting: GitHub wants ``Z``, not ``+00:00``.

Everything downstream is aware UTC -- section 5.2 compares timestamps, and a naive/aware
mix there silently loses threads.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

UTC = dt.timezone.utc


def parse_timestamp(value: Any) -> dt.datetime | None:
    """Parse a GitHub timestamp as aware UTC. ``None`` and ``""`` pass through."""
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso_utc(moment: dt.datetime) -> str:
    """Format for a ``since=`` parameter, in the spelling GitHub returns."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Document types whose ``github_updated_at`` GitHub defines **per object**, so a later
#: value is evidence the text itself changed. A thread's ``title`` and ``body`` carry the
#: issue's own ``updated_at`` instead, which any activity bumps -- a new comment, a label, a
#: close. Measured on a `huggingface/transformers` sample: 99.6% of bodies and titles carry a
#: timestamp later than their creation, against 8.7% of issue comments. Applying the rule
#: below to those would drop about 5% of a corpus, including the opening statement of every
#: active thread, to catch almost nothing.
#:
#: Here rather than in the search backend because two callers need it and they are on
#: opposite sides of the index: :func:`relore.search.backends.base.written_before` applies
#: it to a stored document at query time, and :func:`text_existed_at` applies it to the same
#: document at extraction time so a signal read out of that text inherits the same bound.
EDITABLE_PER_OBJECT = ("issue_comment", "review_comment")


def text_existed_at(
    source_type: str,
    created_at: dt.datetime | None,
    updated_at: dt.datetime | None,
) -> dt.datetime | None:
    """The earliest instant at which this document's *current* text is known to have existed.

    The mirror of :func:`relore.search.backends.base.written_before`, and deliberately the
    same rule: a comment written in 2024 and edited in 2026 holds 2026 words, so anything
    read out of it dates from the edit and not from the writing. For the two editable types
    that is ``updated_at``; for a title or a body it is ``created_at``, because their
    ``updated_at`` is the thread's and says nothing about the text.

    ``None`` propagates rather than defaulting: an undated document says nothing about when
    a value first appeared, so :func:`first_seen_at` passes over it rather than guessing.
    """
    if source_type in EDITABLE_PER_OBJECT and updated_at is not None:
        # ``max`` rather than ``updated_at`` outright: GitHub's clock is not ours, and a row
        # whose edit predates its creation must not buy an earlier bound with the skew.
        return updated_at if created_at is None else max(created_at, updated_at)
    return created_at


def first_seen_at(moments: Iterable[dt.datetime | None]) -> dt.datetime | None:
    """The earliest of several :func:`text_existed_at` answers, ignoring the unknown ones.

    One row stands for every document that produced the value, so the question is what the
    row lets a cutoff *claim*: ``first_seen_at < T`` has to mean the thread certainly
    carried this value before ``T``. A dated contributor settles that on its own -- the
    thread demonstrably held the value at that instant -- and an undated one can only mean
    the true first appearance is *earlier*, never later. So skipping it cannot admit a
    value the thread did not have, which is the only direction that matters.

    Dropping the whole row instead would be the stricter reading and it is the wrong one.
    Measured on the 3,064-thread pilot index, every ``review`` document carries no
    ``github_created_at`` (GitHub dates a review with ``submitted_at``), and letting one
    poison the rows it shares with dated comments made 3.1% of ``thread_files`` unusable
    under any cutoff -- a recall cost with no soundness to show for it.

    ``None`` when *nothing* is dated, which is the case where nothing is known and the
    ``<`` comparison then drops the row. That is the same fail-closed direction section 13
    takes everywhere else, applied where it is actually load-bearing.
    """
    dated = [moment for moment in moments if moment is not None]
    return min(dated) if dated else None
