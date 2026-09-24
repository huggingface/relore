"""A GitHub good enough to exercise the real client.

Served through ``httpx.MockTransport``, so ``GitHubClient``'s own pagination, Link-header
following and error handling run for real. Mocking the client instead would test nothing
about the two limits in section 3, which are the whole reason that code is shaped as it is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from relore.github.client import GitHubClient
from relore.github.graphql import GraphQLClient

BASE = "https://api.github.test"


def account(login: str, *, bot: bool = False) -> dict[str, Any]:
    return {"login": login, "id": abs(hash(login)) % 10**6, "type": "Bot" if bot else "User"}


#: ``files(first: N)`` / ``commits(first: N)`` in ``github/graphql.py``. Duplicated rather
#: than imported so the fixture describes GitHub, not our own constant.
_GRAPHQL_PAGE = 100


@dataclass
class FakeThread:
    number: int
    payload: dict[str, Any]
    issue_comments: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    review_comments: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    #: ``(sha, headline)`` per commit -- the per-PR pass is the only source of these, and
    #: section 5.3 extracts a commit only from here, never from a sha written in prose.
    commits: list[tuple[str, str]] = field(default_factory=list)
    # Who merged it -- section 6.2's authority fallback, and available nowhere but the
    # per-PR pass. Per thread rather than fixed, because "who merged this" is the whole
    # signal.
    merged_by: str = "maintainer"
    #: Issue numbers GitHub itself resolved from this PR's closing keywords (section 13.3).
    #: Served with their repository, because the real field can point at another one.
    closing_references: list[int] = field(default_factory=list)

    @property
    def is_pr(self) -> bool:
        return "pull_request" in self.payload


class FakeGitHub:
    def __init__(self, repo: str = "owner/name", *, page_size: int = 100) -> None:
        self.repo = repo
        self.page_size = page_size
        self.threads: dict[int, FakeThread] = {}
        self.requests: list[str] = []
        # Emulate section 3's HTTP 422: refuse to serve a page beyond this index.
        self.page_cap: int | None = None
        # Section 6.2's collaborator-permission endpoint: login -> permission. A login
        # that is absent answers 404 ("not a collaborator"), which is an answer.
        self.permissions: dict[str, str] = {}
        # The endpoint needs push access on the repository; a token without it gets 403
        # for every login, which is the case the merged_by fallback exists for.
        self.permission_endpoint_forbidden = False

    # -- building ----------------------------------------------------------

    def add_issue(
        self,
        number: int,
        *,
        title: str = "a title",
        body: str = "a body",
        updated_at: str = "2026-01-01T00:00:00Z",
        author: str = "alice",
        assoc: str = "CONTRIBUTOR",
        labels: tuple[str, ...] = (),
        state: str = "open",
        bot: bool = False,
    ) -> FakeThread:
        thread = FakeThread(
            number=number,
            payload={
                "number": number,
                "node_id": f"I_{number}",
                "title": title,
                "body": body,
                "state": state,
                "user": account(author, bot=bot),
                "author_association": assoc,
                "labels": [{"name": name} for name in labels],
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": updated_at,
                "closed_at": None,
                "html_url": f"https://github.test/{self.repo}/issues/{number}",
                "url": f"{BASE}/repos/{self.repo}/issues/{number}",
                "comments": 0,
            },
        )
        self.threads[number] = thread
        return thread

    def add_pr(self, number: int, *, merged_at: str | None = None, **kwargs: Any) -> FakeThread:
        thread = self.add_issue(number, **kwargs)
        # The bulk /issues walk nests merged_at here; GET /pulls/{n} puts it top level.
        # Both are served, because the two passes read different shapes.
        thread.payload["pull_request"] = {
            "html_url": f"https://github.test/{self.repo}/pull/{number}",
            "merged_at": merged_at,
        }
        thread.payload["merged_at"] = merged_at
        thread.payload["merged"] = merged_at is not None
        thread.payload["html_url"] = f"https://github.test/{self.repo}/pull/{number}"
        return thread

    def add_comment(
        self,
        thread: FakeThread,
        comment_id: int,
        body: str,
        *,
        author: str = "bob",
        assoc: str = "CONTRIBUTOR",
        created_at: str = "2026-01-01T00:00:00Z",
        # GitHub defines this per comment, so a later value really is an edit -- which is
        # what the cutoff's "text as of T" rule turns on.
        updated_at: str | None = None,
        bot: bool = False,
    ) -> dict[str, Any]:
        comment = {
            "id": comment_id,
            "body": body,
            "user": account(author, bot=bot),
            "author_association": assoc,
            "created_at": created_at,
            "updated_at": updated_at or created_at,
            "html_url": f"https://github.test/c/{comment_id}",
            # What walk_issue_comments reads to find the thread, before the mirror
            # fields are stripped.
            "issue_url": f"{BASE}/repos/{self.repo}/issues/{thread.number}",
        }
        thread.issue_comments.append(comment)
        thread.payload["comments"] = len(thread.issue_comments)
        return comment

    def add_review_comment(
        self,
        thread: FakeThread,
        comment_id: int,
        body: str,
        *,
        path: str = "src/mod.py",
        line: int = 42,
        commit_id: str = "deadbeef",
        author: str = "carol",
        assoc: str = "COLLABORATOR",
        created_at: str = "2026-01-02T00:00:00Z",
        in_reply_to: int | None = None,
        bot: bool = False,
    ) -> dict[str, Any]:
        comment = {
            "id": comment_id,
            "body": body,
            "path": path,
            "line": line,
            "side": "RIGHT",
            "diff_hunk": "@@ -1 +1 @@",
            "commit_id": commit_id,
            "user": account(author, bot=bot),
            "author_association": assoc,
            "created_at": created_at,
            "updated_at": created_at,
            "html_url": (
                f"https://github.test/{self.repo}/pull/{thread.number}#discussion_r{comment_id}"
            ),
            "pull_request_url": f"{BASE}/repos/{self.repo}/pulls/{thread.number}",
        }
        if in_reply_to is not None:
            comment["in_reply_to_id"] = in_reply_to
        thread.review_comments.append(comment)
        return comment

    def add_review(
        self,
        thread: FakeThread,
        review_id: int,
        body: str,
        *,
        state: str = "CHANGES_REQUESTED",
        author: str = "carol",
        assoc: str = "COLLABORATOR",
        submitted_at: str = "2026-01-02T00:00:00Z",
    ) -> dict[str, Any]:
        review = {
            "id": review_id,
            "body": body,
            "state": state,
            "user": account(author),
            "author_association": assoc,
            "submitted_at": submitted_at,
            "created_at": submitted_at,
            "updated_at": submitted_at,
            "commit_id": "deadbeef",
            "html_url": f"https://github.test/r/{review_id}",
        }
        thread.reviews.append(review)
        return review

    # -- serving -----------------------------------------------------------

    def client(self, **kwargs: Any) -> GitHubClient:
        return GitHubClient(
            "fake-token",
            base_url=BASE,
            client=httpx.Client(transport=httpx.MockTransport(self._handle), base_url=BASE),
            sleep=lambda _s: None,
            **kwargs,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        query = dict(request.url.params)
        self.requests.append(path)
        prefix = f"/repos/{self.repo}"

        if path == f"{prefix}/issues":
            return self._list_threads(request, query)
        if path == f"{prefix}/pulls/comments":
            return self._list_review_comments(request, query)
        if path == f"{prefix}/issues/comments":
            return self._list_issue_comments(request, query)
        if path.startswith(f"{prefix}/collaborators/") and path.endswith("/permission"):
            return self._permission(
                path.removeprefix(f"{prefix}/collaborators/")[: -len("/permission")]
            )

        parts = path.removeprefix(prefix).strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("issues", "pulls") and parts[1].isdigit():
            thread = self.threads.get(int(parts[1]))
            if thread is None:
                return _json(404, {"message": "Not Found"})
            tail = parts[2] if len(parts) > 2 else None
            if tail is None:
                return _json(200, thread.payload)
            if tail == "comments" and parts[0] == "issues":
                return self._page(request, thread.issue_comments)
            if tail == "comments" and parts[0] == "pulls":
                return self._page(request, thread.review_comments)
            if tail == "reviews":
                return self._page(request, thread.reviews)
        return _json(404, {"message": f"no route for {path}"})

    def _permission(self, login: str) -> httpx.Response:
        if self.permission_endpoint_forbidden:
            return _json(403, {"message": "Must have push access to view collaborator permission."})
        if login not in self.permissions:
            return _json(404, {"message": "Not Found"})
        return _json(200, {"permission": self.permissions[login], "user": account(login)})

    def _list_threads(self, request: httpx.Request, query: dict[str, str]) -> httpx.Response:
        items = [t.payload for t in self.threads.values()]
        since = query.get("since")
        if since:
            items = [i for i in items if i["updated_at"] >= since]
        items.sort(key=lambda i: (i["updated_at"], i["number"]))
        return self._page(request, items)

    def _list_issue_comments(self, request: httpx.Request, query: dict[str, str]) -> httpx.Response:
        items = [c for t in self.threads.values() for c in t.issue_comments]
        since = query.get("since")
        if since:
            items = [c for c in items if c["created_at"] >= since]
        items.sort(key=lambda c: (c["created_at"], c["id"]))
        return self._page(request, items)

    def _list_review_comments(
        self, request: httpx.Request, query: dict[str, str]
    ) -> httpx.Response:
        items = [c for t in self.threads.values() for c in t.review_comments]
        since = query.get("since")
        if since:
            items = [c for c in items if c["created_at"] >= since]
        items.sort(key=lambda c: (c["created_at"], c["id"]))
        return self._page(request, items)

    def _page(self, request: httpx.Request, items: list[dict[str, Any]]) -> httpx.Response:
        query = dict(request.url.params)
        page = int(query.get("page", 1))
        size = min(int(query.get("per_page", self.page_size)), self.page_size)
        if self.page_cap is not None and page > self.page_cap:
            return _json(
                422,
                {
                    "message": (
                        "Pagination with the page parameter is not supported for large "
                        "datasets, please use cursor based pagination (after/before)"
                    )
                },
            )
        start = (page - 1) * size
        window = items[start : start + size]
        headers = {}
        if start + size < len(items):
            next_url = request.url.copy_merge_params({"page": page + 1, "per_page": size})
            headers["Link"] = f'<{next_url}>; rel="next"'
        return _json(200, window, headers)


class FakeGraphQL:
    """Serves the aliased per-PR query, and lets a test force a rate-limit reply."""

    def __init__(self, fake: FakeGitHub) -> None:
        self.fake = fake
        self.queries: list[str] = []
        self.rate_limit = {"limit": 5000, "cost": 1, "remaining": 4999, "resetAt": None}
        self.fail_once_rate_limited = False

    def client(self, **kwargs: Any) -> GraphQLClient:
        return GraphQLClient(
            "fake-token",
            endpoint="https://api.github.test/graphql",
            client=httpx.Client(transport=httpx.MockTransport(self._handle)),
            **{"sleep": lambda _s: None, **kwargs},
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.queries.append(body["query"])
        if self.fail_once_rate_limited:
            self.fail_once_rate_limited = False
            return _json(200, {"data": None, "errors": [{"type": "RATE_LIMITED"}]})
        numbers = [int(n) for n in re.findall(r"pullRequest\(number: (\d+)\)", body["query"])]
        repository: dict[str, Any] = {}
        for number in numbers:
            thread = self.fake.threads.get(number)
            if thread is None or not thread.is_pr:
                continue  # deleted, or never a PR
            repository[f"pr{number}"] = {
                "number": number,
                "mergedAt": thread.payload.get("merged_at"),
                "additions": 10,
                "deletions": 2,
                "changedFiles": len(thread.files) or 1,
                "mergedBy": {"login": thread.merged_by},
                "reviews": {
                    "totalCount": len(thread.reviews),
                    "nodes": [_as_graphql_review(r) for r in thread.reviews],
                },
                "files": {
                    # `first: 100`, honoured: the query asks for a page and reports the
                    # whole count, so a 323-file refactor stages 100 rows and a
                    # `totalCount` that says so. Serving every node here would hide the
                    # only interesting thing about a large pull request.
                    "totalCount": len(thread.files),
                    "nodes": [
                        {"path": p, "additions": 1, "deletions": 0, "changeType": "MODIFIED"}
                        for p in thread.files[:_GRAPHQL_PAGE]
                    ],
                },
                "commits": {
                    "totalCount": len(thread.commits),
                    "nodes": [
                        {"commit": {"oid": sha, "messageHeadline": head, "messageBody": ""}}
                        for sha, head in thread.commits
                    ],
                },
                "closingIssuesReferences": {
                    "nodes": [
                        {"number": n, "repository": {"nameWithOwner": self.fake.repo}}
                        for n in thread.closing_references
                    ]
                },
            }
        return _json(200, {"data": {"rateLimit": self.rate_limit, "repository": repository}})


def _as_graphql_review(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "databaseId": review["id"],
        "body": review["body"],
        "state": review["state"],
        "submittedAt": review["submitted_at"],
        "url": review["html_url"],
        "authorAssociation": review["author_association"],
        "author": {"login": review["user"]["login"], "__typename": review["user"]["type"]},
        "commit": {"oid": review.get("commit_id")},
    }


def _json(status: int, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        content=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
