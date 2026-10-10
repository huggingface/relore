"""Fetch one thread's canonical current state.

This is the network half of section 5.1. It returns raw payloads and derives nothing:
everything is staged in ``raw_objects`` first so that re-chunking and re-extracting later
cost local CPU instead of another day of API budget (section 5.0).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from relore.github.client import GitHubClient, GitHubError

# The self-link every payload carries. `html_url` is what a document cites, so it stays.
_DROP_KEYS = frozenset({"url", "_links", "repository", "performed_via_github_app"})
# Nested account objects are ~20 fields of which we read three.
_ACCOUNT_KEEP = frozenset({"login", "id", "type", "site_admin"})


def strip_mirror_fields(payload: Any) -> Any:
    """Drop the API's duplicated link fields before storing.

    Section 4: these are pure duplication and most of the payload size -- roughly 2-3 GB
    of JSONB for a large repository once they are gone, considerably more if they are not.
    """
    if isinstance(payload, list):
        return [strip_mirror_fields(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    if "login" in payload and "type" in payload:
        return {k: v for k, v in payload.items() if k in _ACCOUNT_KEEP}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _DROP_KEYS:
            continue
        if key.endswith("_url") and key != "html_url":
            continue
        out[key] = strip_mirror_fields(value)
    return out


# The object types a thread fetch is authoritative for. Notably *not* `pr_details`,
# which the backfill's per-PR pass stages against the same thread number. This is an
# allowlist for exactly that reason: a type nobody here names can never be pruned.
AUTHORITATIVE_TYPES = ("issue", "pr", "issue_comment", "review", "review_comment")


@dataclass
class ThreadPayload:
    """One thread as fetched: the raw objects that will be staged, and nothing derived."""

    repo: str
    number: int
    thread_type: str  # issue | pr
    thread: dict[str, Any]
    issue_comments: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    review_comments: list[dict[str, Any]] = field(default_factory=list)
    #: Object types GitHub would not serve for this thread. Their staged rows are kept:
    #: an empty list here means "not read", not "deleted".
    unread: set[str] = field(default_factory=set)

    def raw_objects(self) -> list[tuple[str, str, dict[str, Any]]]:
        """``(object_type, object_id, payload)`` for every object in this thread."""
        rows: list[tuple[str, str, dict[str, Any]]] = [
            (self.thread_type, str(self.number), self.thread)
        ]
        for kind, items in (
            ("issue_comment", self.issue_comments),
            ("review", self.reviews),
            ("review_comment", self.review_comments),
        ):
            rows.extend((kind, str(item["id"]), item) for item in items)
        return rows


def fetch_thread(client: GitHubClient, repo: str, number: int) -> ThreadPayload:
    """Read a thread and everything attached to it.

    Two to five requests. A PR is fetched twice on purpose: ``GET /issues/{n}`` is the
    endpoint that works for both kinds, but ``merged_at`` and ``merged_by`` exist only on
    ``GET /pulls/{n}`` -- and ``merged_by`` is what section 6.2's authority fallback needs,
    available nowhere cheaper.

    GitHub can list a pull request under ``/issues`` and answer 404 under ``/pulls``
    (huggingface/trl#7121: closed, unmerged, ``pulls/{n}`` and ``pulls/{n}/reviews`` gone).
    The 404 is permanent, so raising on it fails the thread on every pass, and since the
    poll's checkpoint stops at the first failure, every pass re-reads the whole window
    behind it. A ``/pulls`` 404 therefore keeps what ``/issues`` said, which already
    carries ``merged_at``, and marks the missing kinds :attr:`ThreadPayload.unread`.
    """
    issue = client.get_json(f"repos/{repo}/issues/{number}")
    is_pr = "pull_request" in issue
    thread = dict(issue)
    unread: set[str] = set()
    if is_pr:
        detail = _unless_gone(lambda: client.get_json(f"repos/{repo}/pulls/{number}"))
        if detail is not None:
            thread.update(detail)

    payload = ThreadPayload(
        repo=repo,
        number=number,
        thread_type="pr" if is_pr else "issue",
        thread=strip_mirror_fields(thread),
        unread=unread,
    )
    if issue.get("comments"):
        payload.issue_comments = [
            strip_mirror_fields(c)
            for c in client.paginate(f"repos/{repo}/issues/{number}/comments")
        ]
    if is_pr:
        for kind, attr, tail in (
            ("review", "reviews", "reviews"),
            ("review_comment", "review_comments", "comments"),
        ):
            items = _unless_gone(
                lambda tail=tail: list(client.paginate(f"repos/{repo}/pulls/{number}/{tail}"))
            )
            if items is None:
                unread.add(kind)
            else:
                setattr(payload, attr, [strip_mirror_fields(item) for item in items])
    return payload


def _unless_gone(read: Callable[[], Any]) -> Any:
    """``read()``, or None when GitHub answers 404. Every other error still raises."""
    try:
        return read()
    except GitHubError as exc:
        if exc.status == 404:
            return None
        raise
