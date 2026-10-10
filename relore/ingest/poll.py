"""The poll loop -- section 5.2's delta detection, in two layers.

Layer A asks *which threads moved*; layer B re-reads each one and lets
:func:`~relore.ingest.index_thread.index_payload` do the set diff, because GitHub never
says *what* changed inside a thread.

Almost every line here is defence against one failure class: a pass that loses threads
permanently while looking successful. Three rules do the work, and each earns its
complexity.

1. **Ascending by timestamp, and the checkpoint advances per committed thread.** Not per
   pass, and never to ``now``. A process killed mid-pass resumes where it stopped.
2. **The checkpoint stops at the first failure.** Threads arrive ascending, so advancing
   past a failed one would skip it forever -- and the pass would still report success.
   Later threads are still indexed (that is idempotent and useful); only the mark waits.
   A thread that fails on every pass holds the mark for good, so the window behind it
   is re-listed each pass -- and only re-listed: a thread already held at the listing's
   ``updated_at`` is not re-fetched. Without that, one stuck trl thread re-read 184
   threads every 5 minutes, about 900 requests a pass, and spent the token's hourly
   budget. Threads the review-comment walk surfaces are always fetched, because a
   review comment does not reliably bump ``updated_at``.
3. **A capped walk bounds how far the mark may move.** The two list queries are ordered
   differently, so the uncapped one can surface a thread far newer than anything the
   capped one reached. Indexing it is fine; letting it drag the checkpoint over the
   threads we never saw is not.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine

from relore.github.client import GitHubClient, PaginationCapReached
from relore.github.fetch_thread import fetch_thread
from relore.ingest.index_thread import index_payload
from relore.ingest.timestamps import iso_utc, parse_timestamp
from relore.store import repository as repo_layer

log = logging.getLogger(__name__)

PASS = "threads"

# `updated_at` has one-second granularity and threads can share a timestamp, so resuming
# strictly after the high-water second can drop one that shared it. Re-processing is free
# -- section 5.1 makes it all no-ops -- so the overlap costs nothing and removes the whole
# class of boundary bug, including the inclusive-vs-exclusive question about `since`.
OVERLAP = dt.timedelta(seconds=60)

_PR_NUMBER = re.compile(r"/pulls?/(\d+)")


@dataclass
class Discovery:
    """What layer A found."""

    candidates: dict[int, str] = field(default_factory=dict)
    """Thread number -> the best timestamp we know for it, used only for ordering.

    For a thread from the issue walk this is its own ``updated_at``. For one surfaced only
    by a review comment it is that comment's ``created_at``, which is at or before the
    thread's real ``updated_at`` -- so it sorts conservatively early, never late.
    """

    commented: set[int] = field(default_factory=set)
    """Threads the review-comment walk surfaced: fetched even when ``updated_at`` stood still."""

    capped: bool = False
    """GitHub refused deeper pagination: ``candidates`` is a prefix of what moved."""

    safe_bound: str | None = None
    """The checkpoint may not advance beyond this. Set only when the walk was capped."""


@dataclass
class PollResult:
    repo: str
    since: dt.datetime | None
    moved: int = 0
    #: Moved threads already held at the listing's ``updated_at``, so not re-fetched.
    skipped: int = 0
    indexed: int = 0
    documents_written: int = 0
    #: Section 5.3's rows. Counted separately from documents, and reported here rather
    #: than only in a test, because the steady-state number is the one that shows a
    #: regression: a poll of an unchanged repository must write neither.
    signals_written: int = 0
    failed: list[int] = field(default_factory=list)
    capped: bool = False

    @property
    def clean(self) -> bool:
        return not self.failed and not self.capped


def discover_moved(client: GitHubClient, repo: str, since: dt.datetime | None) -> Discovery:
    """Layer A: the union of two list queries.

    The second query exists because review activity does not reliably bump a thread's
    ``updated_at`` the way a conversation comment does, and ``pulls/comments`` is uncapped
    and cheap (section 3) -- one extra request buys certainty instead of a guess about
    GitHub's internals.
    """
    found = Discovery()
    stamp = iso_utc(since) if since else None

    issue_params: dict[str, Any] = {"state": "all", "sort": "updated", "direction": "asc"}
    if stamp:
        issue_params["since"] = stamp
    walked: list[str] = []
    try:
        for issue in client.paginate(f"repos/{repo}/issues", issue_params):
            number = int(issue["number"])
            updated = str(issue.get("updated_at") or "")
            found.candidates[number] = updated
            walked.append(updated)
    except PaginationCapReached:
        # Not fatal, and not silent. Whatever we collected still gets indexed, and the
        # checkpoint advances through it -- so the next pass starts later and covers the
        # remainder. That is the resumability claim, exercised.
        found.capped = True
        found.safe_bound = max(walked) if walked else None
        log.warning(
            "%s: GitHub refused deeper pagination on the thread walk. Indexing the %d "
            "threads found so far and advancing the checkpoint no further than %s; the "
            "next pass resumes from there. For a first full index use `relored backfill`.",
            repo,
            len(found.candidates),
            found.safe_bound,
        )

    comment_params: dict[str, Any] = {"sort": "created", "direction": "asc"}
    if stamp:
        comment_params["since"] = stamp
    for comment in client.paginate(f"repos/{repo}/pulls/comments", comment_params):
        number = _pr_number(comment)
        if number is None:
            continue
        found.candidates.setdefault(number, str(comment.get("created_at") or ""))
        found.commented.add(number)

    return found


def poll_once(
    engine: Engine,
    client: GitHubClient,
    repo: str,
    *,
    since: dt.datetime | None = None,
) -> PollResult:
    with engine.connect() as conn:
        mark = since or repo_layer.high_water(conn, repo, PASS)
        current = {
            **repo_layer.thread_freshness(conn, repo, "issue"),
            **repo_layer.thread_freshness(conn, repo, "pr"),
        }
    window = mark - OVERLAP if mark else None
    if window is None:
        log.info("%s: no high-water mark yet, walking from the start", repo)

    found = discover_moved(client, repo, window)
    result = PollResult(repo=repo, since=window, moved=len(found.candidates), capped=found.capped)
    bound = parse_timestamp(found.safe_bound) if found.safe_bound else None
    blocked = False

    for number, key in sorted(found.candidates.items(), key=lambda kv: (kv[1], kv[0])):
        stamp = parse_timestamp(key)
        if number not in found.commented and stamp is not None and current.get(number) == stamp:
            # Rule 2's held mark, kept cheap: an unmoved `updated_at` means nothing to read.
            result.skipped += 1
            if may_advance(stamp, blocked=blocked, bound=bound):
                with engine.begin() as conn:
                    repo_layer.advance_high_water(conn, repo, PASS, stamp)
            continue
        try:
            # Network outside the transaction; the commit and the checkpoint inside it.
            payload = fetch_thread(client, repo, number)
        except Exception:
            log.exception("%s#%s: fetch failed", repo, number)
            result.failed.append(number)
            blocked = True
            continue
        try:
            with engine.begin() as conn:
                indexed = index_payload(conn, payload)
                if may_advance(indexed.updated_at, blocked=blocked, bound=bound):
                    assert indexed.updated_at is not None
                    repo_layer.advance_high_water(conn, repo, PASS, indexed.updated_at)
            result.indexed += 1
            result.documents_written += indexed.wrote
            result.signals_written += indexed.signals.wrote
            if indexed.redactions:
                # Names only. The matched value is never logged or used as a metric label.
                log.info("%s#%s: redacted %s", repo, number, dict(indexed.redactions))
        except Exception:
            log.exception("%s#%s: index failed", repo, number)
            result.failed.append(number)
            blocked = True

    with engine.begin() as conn:
        repo_layer.touch_pass(conn, repo, PASS, ok=result.clean)
    return result


def may_advance(
    updated_at: dt.datetime | None, *, blocked: bool, bound: dt.datetime | None
) -> bool:
    """Rules 2, 5 and 6 of this module's docstring, as one predicate.

    Public because ``ingest/sample.py`` walks a different list under a different pass
    name and must not reimplement them -- a checkpoint that advances past a failure is
    the silent-success class this whole module is defence against.
    """
    if updated_at is None or blocked:
        return False
    return bound is None or updated_at <= bound


def poll_forever(
    engine: Engine,
    client: GitHubClient,
    repo: str,
    interval_seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_passes: int | None = None,
) -> list[PollResult]:
    results: list[PollResult] = []
    while max_passes is None or len(results) < max_passes:
        started = time.monotonic()
        results.append(poll_once(engine, client, repo))
        result = results[-1]
        log.info(
            "%s: %d moved, %d current, %d indexed, %d document writes, %d signal writes%s",
            repo,
            result.moved,
            result.skipped,
            result.indexed,
            result.documents_written,
            result.signals_written,
            f", {len(result.failed)} failed" if result.failed else "",
        )
        if max_passes is not None and len(results) >= max_passes:
            break
        sleep(max(0.0, interval_seconds - (time.monotonic() - started)))
    return results


def _pr_number(comment: dict[str, Any]) -> int | None:
    for key in ("pull_request_url", "html_url"):
        match = _PR_NUMBER.search(str(comment.get(key) or ""))
        if match:
            return int(match.group(1))
    return None
