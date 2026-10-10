"""The GitHub REST client.

Two shapes of pagination, because section 3 measured two different limits:

* :meth:`GitHubClient.paginate` follows the ``Link: rel="next"`` header **verbatim** and
  never constructs a ``page=`` parameter itself. That is what makes cursor pagination
  automatic: when GitHub hands back ``after=`` cursors instead of page numbers, following
  the header picks them up with no code change, and we cannot walk into the HTTP 422 that
  ``page=101`` returns on a large dataset.
* :meth:`GitHubClient.paginate_since_chained` is for the endpoints hard-capped at 30,000
  items, where ``since`` combined with ``direction=asc`` does **not** lift the cap. It
  takes a slice, reads the last item's timestamp, and uses it as the next ``since``.

Both shapes go through :meth:`GitHubClient.request`, which retries on two different
signals: a status GitHub returned (5xx, or a rate limit), and an exception no status came
back with (a dropped connection, a timeout). The second is not decoration — section 3
sizes a large repository at a restartable day, and over that many requests a transport
failure is routine; unretried it ends the process mid-walk.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

log = logging.getLogger(__name__)

API = "https://api.github.com"
ACCEPT = "application/vnd.github+json"
API_VERSION = "2022-11-28"

# Section 3: the cap is 300 pages x 100 items on the comment endpoints.
SLICE_CAP = 30_000

# Transport failures worth asking again about, because they are not answers about the
# resource: a dropped connection, a timeout, a peer that closed mid-body. A GET is
# idempotent, so a second ask is safe. `LocalProtocolError` and `UnsupportedProtocol` are
# deliberately absent — those describe a request we built wrong, and a retry cannot fix it.
TRANSIENT_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
)


class GitHubError(RuntimeError):
    """A response we cannot use. ``status`` is GitHub's, or None when none came back."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class PaginationCapReached(GitHubError):
    """GitHub refused to page further and told us to use cursors.

    Surfaced rather than swallowed: a walk that silently stops here looks like a
    successful run that happened to find fewer threads, which is the exact failure class
    section 5.2 exists to prevent.
    """


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = API,
        user_agent: str = "relore",
        timeout: float = 30.0,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": ACCEPT,
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": user_agent,
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- one request -------------------------------------------------------

    def request(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries. ``url`` may be a path or an absolute URL from a Link header."""
        target = url if url.startswith("http") else f"{self.base_url}/{url.lstrip('/')}"
        last: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.get(target, params=params)
            except TRANSIENT_TRANSPORT_ERRORS as exc:
                # Section 3 sizes a large repository at a restartable day, and over that
                # many requests a dropped connection is routine rather than exceptional.
                # Left to propagate it ends the process, which costs a re-walk of the
                # current page and turns a healthy walk into a crash loop.
                if attempt == self.max_retries:
                    raise GitHubError(
                        f"{type(exc).__name__} from {target} after "
                        f"{self.max_retries} retries: {exc}"
                    ) from exc
                self._back_off(type(exc).__name__, target, attempt)
                continue
            if response.status_code < 400:
                return response
            last = response
            if response.status_code == 422 and "cursor" in response.text.lower():
                raise PaginationCapReached(
                    f"{target} refused deeper pagination: {response.text[:200]}"
                )
            wait = self._retry_after(response, attempt)
            if wait is None or attempt == self.max_retries:
                break
            self._back_off(response.status_code, target, attempt, wait)
        assert last is not None
        raise GitHubError(
            f"{last.status_code} from {target}: {last.text[:200]}", status=last.status_code
        )

    def _back_off(
        self, reason: object, target: str, attempt: int, wait: float | None = None
    ) -> None:
        """Log and sleep. One schedule and one message, so both retry paths read alike."""
        if wait is None:
            wait = min(2.0**attempt, 60.0)
        log.warning(
            "GitHub %s on %s; sleeping %.0fs (attempt %d/%d)",
            reason,
            target.removeprefix(self.base_url),
            wait,
            attempt + 1,
            self.max_retries,
        )
        self._sleep(wait)

    def _retry_after(self, response: httpx.Response, attempt: int) -> float | None:
        """Seconds to wait, or None if this status is not worth retrying."""
        if response.status_code in (429, 403):
            explicit = response.headers.get("retry-after")
            if explicit:
                return max(1.0, float(explicit))
            # Primary rate limit: remaining==0 means wait for the window, not back off.
            if response.headers.get("x-ratelimit-remaining") == "0":
                reset = response.headers.get("x-ratelimit-reset")
                if reset:
                    return max(1.0, float(reset) - time.time() + 1.0)
            # A 403 that is not a rate limit is a permission problem; retrying is pointless.
            return 60.0 if response.status_code == 429 else None
        if response.status_code >= 500:
            return min(2.0**attempt, 60.0)
        return None

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.request(url, params).json()

    # -- pagination --------------------------------------------------------

    def paginate(
        self, path: str, params: dict[str, Any] | None = None, *, cap: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield items across pages, following Link headers rather than counting pages."""
        url: str | None = path
        query: dict[str, Any] | None = dict(params or {})
        assert query is not None
        query.setdefault("per_page", 100)
        seen = 0
        while url:
            response = self.request(url, query)
            # None, emphatically not {}: httpx *replaces* a URL's query string when it is
            # given a `params` argument, so an empty dict would strip the page or cursor
            # out of the Link URL and walk page 1 forever.
            query = None
            batch = response.json()
            if not isinstance(batch, list):
                raise GitHubError(f"expected a list from {path}, got {type(batch).__name__}")
            for item in batch:
                yield item
                seen += 1
                if cap is not None and seen >= cap:
                    return
            url = _next_link(response)

    def paginate_since_chained(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timestamp_field: str = "created_at",
        slice_cap: int = SLICE_CAP,
    ) -> Iterator[dict[str, Any]]:
        """Walk past the 30,000-item cap by chaining ``since`` on the last item seen.

        Ascending order is mandatory here -- the chain only advances if the last item of a
        slice is the newest one in it. Ids are de-duplicated across slice boundaries
        because ``since`` is inclusive and timestamps repeat within a second.
        """
        query = dict(params or {})
        query["direction"] = "asc"
        emitted: set[Any] = set()
        while True:
            count = 0
            last_stamp: str | None = None
            for item in self.paginate(path, query, cap=slice_cap):
                count += 1
                last_stamp = item.get(timestamp_field) or last_stamp
                key = item.get("id", (item.get("url"), last_stamp))
                if key in emitted:
                    continue
                emitted.add(key)
                yield item
            if count < slice_cap or last_stamp is None:
                return
            if query.get("since") == last_stamp:
                # More than one slice-cap of items share a single timestamp. Chaining
                # cannot advance, and silently looping would be worse than stopping.
                raise GitHubError(
                    f"{path}: cannot advance past since={last_stamp}; {slice_cap} items share it"
                )
            query["since"] = last_stamp


_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def _next_link(response: httpx.Response) -> str | None:
    """The next page's URL, verbatim, whether it carries a page number or a cursor."""
    match = _NEXT_LINK.search(response.headers.get("link", ""))
    return match.group(1) if match else None
