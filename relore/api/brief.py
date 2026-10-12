"""``GET /api/v1/brief/{n}`` -- what a person needs before reading a new issue or PR.

A maintainer on watch duty sees a thread arrive with none of its history: who opened it,
whether anybody with write access is already on it, what else claims the same fix, and
what this project said before about the files it touches. Every piece already has a verb;
this asks them all in one request and returns one page, so the answer costs one command
rather than four.

**Mechanical, never summarized** (section 1): every line is a row the index holds, and
each section says how many it left out. Composed here rather than in the client because
the client may not reach the backend (AGENTS.md invariant 1), and one request is one
charge against the rate limit rather than six.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from relore.api.schemas import claim_json, hit_json, thread_json
from relore.search import SearchQuery, search_best
from relore.search.backends.base import SearchBackend
from relore.search.queries import (
    MAX_BRIEF_PEOPLE,
    MAX_BRIEF_THREADS,
    MAX_HITS,
    Hit,
    render_age,
    render_stamp,
)

#: The changed files the history section asks about. Each one is an ``EXISTS`` in the
#: overlap score, and a 300-file refactor would make that the slowest query this API
#: runs; the page says how many of how many it used.
MAX_BRIEF_PATHS = 20

#: The ``closes #N`` targets whose other claimants are looked up, one ``inflight`` each.
MAX_BRIEF_TARGETS = 5

#: Paths that say least about what a change is *for*: almost every pull request touches
#: one, so overlap on them ranks noise. Asked about last, never dropped.
_GENERIC_PREFIXES = ("tests/", "docs/", "utils/", ".github/")

#: What the thread head keeps from ``thread_json``. A brief is not a thread page: the
#: body and the comments are one ``relore thread`` away.
_HEAD_KEYS = (
    "repo",
    "number",
    "type",
    "title",
    "url",
    "author",
    "state",
    "age",
    "date",
    "labels",
    "state_reason",
    "closed_by",
    "merged",
    "review_decision",
    "review_decision_by",
    "requested_reviewers",
    "files_total",
    "files_collected",
    "comments_total",
    "comments_machine_suppressed",
    "indexed_at",
)


def brief_json(
    backend: SearchBackend,
    repo: str,
    number: int,
    *,
    before: dt.datetime | None = None,
    exclude: Sequence[int] = (),
    compact: bool = False,
) -> dict[str, Any] | None:
    """The brief, or ``None`` if ``number`` is not indexed (or not reachable as of ``before``)."""
    view = backend.thread(repo, number, before=before, exclude=exclude)
    if view is None:
        return None
    whole = thread_json(view, compact=compact)
    head = {key: whole[key] for key in _HEAD_KEYS}
    withheld = (*exclude, number)

    people, machine = backend.participants(repo, number, before=before)
    record = backend.author_record(repo, number, before=before, exclude=exclude)

    targets = [int(link["target"]) for link in view.links if link.get("relationship") == "closes"]
    heads = backend.thread_heads(repo, targets[:MAX_BRIEF_TARGETS], before=before)
    closes = [
        {"number": target, "indexed": target in heads}
        | ({"thread": claim_json(heads[target])} if target in heads else {})
        for target in targets[:MAX_BRIEF_TARGETS]
    ]

    # Other work on the same problem. For an issue: what claims to close it. For a pull
    # request: what *else* claims to close each issue it closes -- itself withheld.
    asked = [number] if view.thread_type == "issue" else targets[:MAX_BRIEF_TARGETS]
    competing: dict[int, dict[str, Any]] = {}
    competing_total = links_indexed = 0
    for target in asked:
        inflight = backend.inflight(repo, target, before=before, exclude=withheld)
        competing_total += inflight.total
        links_indexed = inflight.links_indexed
        for claim in inflight.claims:
            competing.setdefault(claim.number, claim_json(claim))

    shown = set(withheld) | set(competing) | set(targets)
    file_history = _file_history(backend, view, repo, before, withheld, shown)
    shown |= {t["number"] for t in (file_history or {}).get("threads", [])}
    same_error = _same_error(backend, repo, number, before, withheld, shown)
    shown |= {t["number"] for t in (same_error or {}).get("threads", [])}
    similar = _similar(backend, view.title, repo, before, (*withheld, *shown))

    return {
        "thread": head,
        "opener": _opener_json(record),
        "people": {
            "listed": [
                {
                    "author": p.author,
                    "trust": p.trust,
                    "comments": p.comments,
                    "age": render_age(p.last_at),
                    "date": render_stamp(p.last_at),
                }
                for p in people[:MAX_BRIEF_PEOPLE]
            ],
            "returned": min(len(people), MAX_BRIEF_PEOPLE),
            "total": len(people),
            "authoritative": sum(1 for p in people if p.trust == "authoritative"),
            "machine_comments": machine,
        },
        "closes": closes,
        "closes_total": len(targets),
        "competing": {
            "claims": list(competing.values()),
            "returned": len(competing),
            "total": competing_total,
            "links_indexed": links_indexed,
        },
        "file_history": file_history,
        "same_error": same_error,
        "similar": similar,
    }


def _opener_json(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "author": record.author,
        "trust": record.trust,
        "issues": record.issues,
        "prs": record.prs,
        "merged": record.merged,
        "first_age": render_age(record.first_at) if record.first_at else None,
        "first_date": render_stamp(record.first_at),
    }


def _paths(view: Any) -> tuple[str, list[str]]:
    """Which paths this thread is *about*, and on what evidence.

    A pull request's diff where we have it -- collected files, then the paths inline
    review comments hang on, which recover some of what the 100-row page dropped. An issue
    has no diff, so the paths somebody named in it: an extraction, and labelled as one.
    """
    if view.thread_type == "pr" and (view.files_changed or view.files_anchored):
        paths = list(dict.fromkeys((*view.files_changed, *view.files_anchored)))
        basis = "changed"
    else:
        paths = list(view.files_mentioned)
        basis = "mentioned"
    paths.sort(key=lambda path: path.startswith(_GENERIC_PREFIXES))
    return basis, paths


def _file_history(
    backend: SearchBackend,
    view: Any,
    repo: str,
    before: dt.datetime | None,
    withheld: Sequence[int],
    shown: set[int],
) -> dict[str, Any] | None:
    basis, paths = _paths(view)
    if not paths:
        return None
    used = tuple(paths[:MAX_BRIEF_PATHS])
    query = SearchQuery(
        repos=(repo,),
        files=used,
        before=before,
        exclude=tuple(withheld),
        limit=MAX_HITS,
        # A brief quotes one passage per thread as a pointer, never as the answer.
        compact=True,
    )
    threads = _threads(backend, repo, backend.search(query), before, shown)
    return {
        "basis": basis,
        "paths_used": list(used),
        "paths_total": len(paths),
        # Pull requests with no collected file list could not be tested either way.
        "untested": backend.files_blind_spot(query),
        "threads": threads,
    }


def _same_error(
    backend: SearchBackend,
    repo: str,
    number: int,
    before: dt.datetime | None,
    withheld: Sequence[int],
    shown: set[int],
) -> dict[str, Any] | None:
    error = backend.thread_error(repo, number)
    if error is None:
        return None
    query = SearchQuery(
        repos=(repo,),
        errors=(error,),
        kind="failure",
        before=before,
        exclude=tuple(withheld),
        limit=MAX_HITS,
        # A brief quotes one passage per thread as a pointer, never as the answer.
        compact=True,
    )
    return {
        "error": error,
        "threads": _threads(backend, repo, backend.search(query), before, shown),
    }


def _similar(
    backend: SearchBackend,
    title: str,
    repo: str,
    before: dt.datetime | None,
    withheld: Sequence[int],
) -> dict[str, Any]:
    """Threads whose words match the title, with everything listed above withheld so the
    slots go to threads this page has not already named."""
    query = SearchQuery(
        repos=(repo,),
        text=title,
        before=before,
        exclude=tuple(dict.fromkeys(withheld)),
        limit=MAX_HITS,
        # A brief quotes one passage per thread as a pointer, never as the answer.
        compact=True,
    )
    hits, widened = search_best(backend, query, expand=True)
    return {
        "query": title,
        "widened": widened,
        "threads": _threads(backend, repo, hits, before, set()),
    }


def _threads(
    backend: SearchBackend,
    repo: str,
    hits: list[Hit],
    before: dt.datetime | None,
    shown: set[int],
) -> list[dict[str, Any]]:
    """One row per thread, in rank order: its head, and the best-ranked passage from it.

    A search page holds up to three passages per thread; a brief lists threads, so the
    first passage is the evidence and the rest are counted.
    """
    best: dict[int, Hit] = {}
    passages: dict[int, int] = {}
    for hit in hits:
        if hit.repo != repo or hit.number in shown:
            continue
        best.setdefault(hit.number, hit)
        passages[hit.number] = passages.get(hit.number, 0) + 1
    numbers = list(best)[:MAX_BRIEF_THREADS]
    heads = backend.thread_heads(repo, numbers, before=before)
    return [
        {
            **(claim_json(heads[n]) if n in heads else {"repo": repo, "number": n}),
            "evidence": hit_json(best[n], compact=True),
            "passages": passages[n],
        }
        for n in numbers
    ]
