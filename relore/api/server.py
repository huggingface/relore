"""``relored serve`` -- the HTTP JSON API and the page that calls it (section 7, section 8).

Three properties are enforced here rather than documented and hoped for.

**Read-only.** The engine is opened read-only on both dialects (section 11): a
``SELECT``-only role on Postgres, a ``mode=ro`` URI on SQLite. There is no write path for
anyone (section 11's *why*), and the one endpoint that writes -- section 8's relevance
labelling -- writes a JSONL file that is never indexed and never retrieved, so it cannot
become project memory the next agent reads as evidence.

**The untrusted-content envelope is not optional.** Every response goes through
:func:`relore.security.untrusted.scrub_tree` on the way out, applied to the whole payload
rather than to a list of content fields, so adding an endpoint cannot forget it. An
unknown client cannot be assumed to add it.

**Scope is resolved before the query layer sees it.** A token's repositories arrive as
concrete names (:mod:`relore.api.tokens`), and every query carries them.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from sqlalchemy import Engine, select

from relore import __version__
from relore.api import ui
from relore.api.schemas import (
    LabelRequest,
    SearchRequest,
    hit_json,
    inflight_json,
    thread_json,
    why_json,
)
from relore.api.tokens import LABEL_SCOPE, Authenticator, AuthError, RateLimited, Token
from relore.code.blame import blame_line, line_history, origin_history
from relore.code.clone import CloneUnavailable, WorkingClones
from relore.code.defs import definitions
from relore.code.refresh import REFRESH_ENV, start_refresh
from relore.code.refs import references
from relore.code.survey import copies, grep, symbol_body
from relore.code.walk import read
from relore.duration import parse_interval
from relore.render import render_inflight, render_search, render_thread, render_why
from relore.search import QueryError, SearchQuery, expand, open_backend, search_best
from relore.search.queries import HUMAN_TRUST, admissible_trust
from relore.security.untrusted import NOTICE, scrub_tree
from relore.store import schema as s
from relore.store.dialect import is_deployment_grade
from relore.store.migrations import describe
from relore.store.repository import index_summary
from relore.wire import CLIENT_HEADER, SERVER_HEADER, UPGRADE_REQUIRED, explain

log = logging.getLogger(__name__)

LABELS_ENV = "RELORE_LABELS_PATH"
VERDICTS = ("relevant", "not_relevant", "decisive")


@dataclass(frozen=True)
class AsOf:
    """A cutoff pinned to the whole daemon, for a temporal evaluation.

    **Why this is not only a request parameter.** A benchmark condition gives the agent a
    shell and an HTTP API; a cutoff it has to pass is a cutoff it can omit, and one
    forgotten flag silently turns the treatment condition into a system that can read the
    answer. So the honest place for it is the process: `relored serve --as-of T` serves an
    index that *has* no later documents to give, and the agent's own query cannot widen it.

    That is the same shape as section 11's repository scope, for the same reason -- a
    request may narrow this and can never loosen it (:meth:`narrowed`). It is deliberately
    absent from production: an unpinned daemon carries ``AsOf()``, every predicate is
    ``None``, and not one query changes.
    """

    before: dt.datetime | None = None
    exclude: tuple[int, ...] = ()

    @property
    def pinned(self) -> bool:
        return self.before is not None or bool(self.exclude)

    def narrowed(self, before: dt.datetime | None = None, exclude: Sequence[int] = ()) -> AsOf:
        """This pin, tightened by one request. The *earlier* cutoff wins and the two
        exclusion lists union, so neither half of a request can buy back what the pin
        withheld."""
        if before is not None and self.before is not None:
            before = min(before, self.before)
        return AsOf(
            before=before if before is not None else self.before,
            exclude=tuple(dict.fromkeys((*self.exclude, *exclude))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.isoformat() if self.before else None,
            "exclude": list(self.exclude),
        }


class Deps:
    """What the handlers need, built once. Not a framework choice -- the point is that the
    read-only engine and the backend are created at startup, so a request cannot open a
    writable connection by accident."""

    def __init__(
        self,
        engine: Engine,
        *,
        auth: Authenticator | None = None,
        labels_path: Path | None = None,
        clone_root: str | None = None,
        as_of: AsOf | None = None,
    ) -> None:
        self.engine = engine
        self.backend = open_backend(engine)
        self.auth = auth or Authenticator()
        self.labels_path = labels_path
        self.as_of = as_of or AsOf()
        self.clones = WorkingClones(clone_root)
        self.counters: Counter[str] = Counter()

    def indexed_repos(self) -> tuple[str, ...]:
        with self.engine.connect() as conn:
            return tuple(
                str(repo)
                for (repo,) in conn.execute(
                    select(s.threads.c.repo.distinct()).order_by(s.threads.c.repo)
                )
            )

    def status(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            summary = index_summary(conn)
        return {
            "version": __version__,
            "backend": self.backend.info().as_dict(),
            "schema": describe(self.engine),
            # An operator cannot tell a pinned daemon from a live one by reading a result,
            # and a benchmark run that quietly lost its pin looks like a good result. So
            # `status` says so, and says nothing when there is nothing to say.
            **({"as_of": self.as_of.as_dict()} if self.as_of.pinned else {}),
            **summary,
        }


def caller(request: Request, authorization: Annotated[str | None, Header()] = None) -> Token:
    """Authenticate and charge one request. Both limits are per token (section 7).

    Module level, reading its :class:`Deps` off the app rather than off a closure, so the
    annotation below resolves under postponed evaluation -- and so the dependency is one
    function rather than one per app.
    """
    deps: Deps = request.app.state.deps
    try:
        token = deps.auth.authenticate(authorization)
    except AuthError as exc:
        deps.counters["denied"] += 1
        raise HTTPException(status_code=401, detail=str(exc)) from None
    try:
        deps.auth.charge(token)
    except RateLimited as exc:
        deps.counters["rate_limited"] += 1
        raise HTTPException(
            status_code=429,
            detail={"limit": exc.which, "of": exc.limit, "retry_after": exc.retry_after},
            headers={"Retry-After": str(exc.retry_after)},
        ) from None
    deps.counters[f"request:{request.url.path}"] += 1
    return token


#: Every handler takes this, so a route cannot be added without being authenticated and
#: charged: leaving it out is a missing argument, not a silently open endpoint.
Caller = Annotated[Token, Depends(caller)]


def build_app(
    engine: Engine,
    *,
    auth: Authenticator | None = None,
    labels_path: Path | None = None,
    clone_root: str | None = None,
    as_of: AsOf | None = None,
) -> FastAPI:
    deps = Deps(engine, auth=auth, labels_path=labels_path, clone_root=clone_root, as_of=as_of)
    app = FastAPI(title="relore", version=__version__, docs_url="/api/docs")
    app.state.deps = deps

    @app.middleware("http")
    async def version_gate(request: Request, call_next: Callable) -> Response:
        """No ``/api/v1`` request from a client of another version, ever (:mod:`relore.wire`).

        Middleware and not a dependency, for two reasons. It has to stamp
        :data:`~relore.wire.SERVER_HEADER` on *every* response including the refusals --
        that header is how a client whose daemon is too old to enforce this catches the
        same mismatch from its end. And it has to run before authentication, so a caller
        holding a stale client and a stale token is told the actionable thing rather than
        the first thing.

        ``/`` and ``/metrics`` are outside the rule on purpose: the page is how a browser
        *gets* the current client, and ``/metrics`` is the deployment's probe (section
        14.2), which must not fail on a version skew it cannot fix.
        """
        if request.url.path.startswith("/api/v1/"):
            mismatch = explain(request.headers.get(CLIENT_HEADER))
            if mismatch is not None:
                deps.counters["version_refused"] += 1
                return JSONResponse(
                    {"detail": mismatch, "version": __version__},
                    status_code=UPGRADE_REQUIRED,
                    headers={SERVER_HEADER: __version__},
                )
        response = await call_next(request)
        response.headers[SERVER_HEADER] = __version__
        return response

    @app.post("/api/v1/search")
    def search(body: SearchRequest, token: Caller) -> Response:
        repos = token.scope(deps.indexed_repos())
        if body.repos:
            # An INTERSECTION, never a replacement. `repos` is section 11's scope, not a
            # user preference, so the request's list may only remove names from it. A repo
            # the token cannot see is simply absent from the result -- which means asking
            # for one yields an empty page rather than an error, and the caller learns
            # nothing about what exists outside the scope it was given. Written as a filter
            # over `repos` rather than a set intersection so the order stays the scope's.
            wanted = set(body.repos)
            repos = tuple(repo for repo in repos if repo in wanted)
        # The pin first, the request second, and the request can only tighten it.
        as_of = deps.as_of.narrowed(body.before, body.exclude)
        try:
            query = SearchQuery(
                repos=repos,
                text=body.query,
                kind=body.kind,
                trust=body.trust,
                files=tuple(body.files),
                symbols=tuple(body.symbols),
                errors=tuple(body.errors),
                tests=tuple(body.tests),
                labels=tuple(body.labels),
                since=body.since,
                before=as_of.before,
                exclude=as_of.exclude,
                limit=body.limit,
                compact=body.compact,
                sort=body.sort,
            )
        except QueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        legs = expand(query) if body.expand else ()
        hits, widened = search_best(deps.backend, query, expand=body.expand)
        payload = {
            "notice": NOTICE,
            "backend": deps.backend.info().as_dict(),
            "query": {
                "text": query.text,
                "kind": query.kind,
                # Named so an empty result is diagnosable rather than mysterious: a
                # raised floor and a genuinely quiet corpus look identical otherwise.
                "trust_floor": list(admissible_trust(query.trust, query.kind)),
                "limit": query.limit,
                # The same argument as the trust floor above: a cutoff is the one filter
                # that can empty a page for a reason the caller cannot see in their own
                # terms, and a benchmark misconfigured by a day looks exactly like a corpus
                # with nothing to say. Echoed as sent.
                "before": query.before.isoformat() if query.before else None,
                "exclude": list(query.exclude),
                # The *effective* scope, after the request's own repo filter narrowed it.
                # Filtering to a repository the token cannot see comes back as `[]`, which
                # is what makes that empty page diagnosable rather than mysterious.
                "repos": list(repos),
                "sort": query.sort,
                # The same reason as the floor: with expansion on, the question actually
                # asked is not the string the caller sent, and an empty result is only
                # diagnosable if the legs are visible (section 6).
                "legs": [{"leg": leg.name, "term": leg.term} for leg in legs],
                # The filters that were also applied, echoed because a zero page is
                # otherwise indistinguishable from an absence: a *correct* `--error` can
                # zero a query that answers at rank 1 without it (huggingface/relore#47),
                # and the caller reading the page has no evidence which happened.
                "filters": {
                    name: list(values)
                    for name, values in (
                        ("files", query.files),
                        ("symbols", query.symbols),
                        ("errors", query.errors),
                        ("tests", query.tests),
                        ("labels", query.labels),
                    )
                    if values
                },
                "since": query.since.isoformat() if query.since else None,
                # Empty unless the strict conjunction came back with nothing and the call
                # was asked again with the terms disjoined (section 6). Set even when the
                # widened page is *also* empty: that is the answer that saves a turn.
                "widened": widened,
            },
            "count": len(hits),
            # Threads this query otherwise matched that have no collected changed-file
            # list, so `--file` could not test them either way. Absent from the page and
            # not a non-match: saying so is the difference between "nothing else touched
            # this path" and "nothing else we have a file list for".
            "files_untested": deps.backend.files_blind_spot(query),
            "hits": [hit_json(hit, compact=body.compact) for hit in hits],
        }
        return _json(
            payload,
            render=(
                lambda scrubbed: render_search(
                    scrubbed, compact=body.compact, presentation=body.presentation
                )
            )
            if body.render
            else None,
        )

    def _one_repo(token: Caller, repo: str | None) -> str:
        """The repository a bare number refers to.

        Optional only while a token's scope holds exactly one repository -- with several, a
        bare number is ambiguous and guessing would silently answer about the wrong
        project. A repository outside the scope is a 404 rather than a 403: whether it
        exists is not this token's business.

        The message names ``RELORE_REPO`` (issue #36) because refusing once is cheap and
        refusing twenty times is the actual cost: a field run recovered from this in one
        step and then passed ``--repo`` on every one of ~20 subsequent calls. The remedy
        that ends the session's tax belongs in the sentence that first mentions the tax.
        """
        repos = token.scope(deps.indexed_repos())
        if repo is None:
            if len(repos) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"pass repo=: more than one repository is in scope {list(repos)}"
                        " — or set RELORE_REPO to default it for the whole session"
                    ),
                )
            return repos[0]
        if repo not in repos:
            raise HTTPException(status_code=404, detail=f"{repo} not found")
        return repo

    @app.get("/api/v1/thread/{number}")
    def thread(
        number: int,
        token: Caller,
        repo: str | None = None,
        focus: str = "",
        compact: bool = False,
        render: bool = False,
        presentation: bool = False,
        full: bool = False,
        after: str = "",
        outline: bool = False,
        comment: str = "",
        files: bool = False,
        before: dt.datetime | None = None,
        exclude: Annotated[list[int], Query()] = [],  # noqa: B006 - FastAPI reads the default
    ) -> Response:
        """One thread, capped (section 6). See :func:`_one_repo` for ``repo``.

        ``outline`` and ``after`` are the two ways past the cap without raising it,
        ``comment`` serves one addressed document whole, and ``files`` is what the diff's
        path list costs when nobody asked for it -- see
        :meth:`~relore.search.backends.base.SearchBackend.thread` and
        :func:`~relore.render._file_lines` (relore#70).

        A thread the cutoff has not reached, or one the daemon is pinned to withhold, is
        **404** rather than an empty page: "this did not exist yet" and "this exists and
        said nothing" are different answers, and only one of them is true.
        """
        repo = _one_repo(token, repo)
        as_of = deps.as_of.narrowed(before, exclude)
        view = deps.backend.thread(
            repo,
            number,
            focus=focus,
            full=full,
            after=after,
            outline=outline,
            comment=comment,
            before=as_of.before,
            exclude=as_of.exclude,
        )
        if view is None:
            raise HTTPException(status_code=404, detail=f"{repo}#{number} not found")
        payload = {"notice": NOTICE, "thread": thread_json(view, compact=compact)}
        return _json(
            payload,
            render=(
                lambda scrubbed: render_thread(
                    scrubbed, compact=compact, presentation=presentation, files=files
                )
            )
            if render
            else None,
        )

    @app.get("/api/v1/inflight/{number}")
    def inflight(
        number: int,
        token: Caller,
        repo: str | None = None,
        render: bool = False,
        compact: bool = False,
        presentation: bool = False,
        before: dt.datetime | None = None,
        exclude: Annotated[list[int], Query()] = [],  # noqa: B006 - FastAPI reads the default
    ) -> Response:
        """What already claims to close this thread (section 13.3).

        No 404 for a number this index has never seen: "nothing claims to close #48630" is
        a true and useful answer whether or not #48630 is indexed, and refusing it would
        make the *unindexed* case indistinguishable from an error.
        """
        repo = _one_repo(token, repo)
        as_of = deps.as_of.narrowed(before, exclude)
        view = deps.backend.inflight(repo, number, before=as_of.before, exclude=as_of.exclude)
        payload = {"notice": NOTICE, **inflight_json(view)}
        return _json(
            payload,
            render=(
                lambda scrubbed: render_inflight(
                    scrubbed, compact=compact, presentation=presentation
                )
            )
            if render
            else None,
        )

    # -- the code lens (section 1, issue #7) --------------------------------
    #
    # Scoped exactly like the history verbs: `_one_repo` first, so a token cannot read a
    # tree it cannot read threads from. A repository with no clone is a 503 with a sentence
    # -- the index does not depend on a checkout (build plan 14.4b).

    def _clone_root(token: Caller, repo: str | None) -> tuple[str, str]:
        name = _one_repo(token, repo)
        try:
            return name, deps.clones.require(name)
        except CloneUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None

    def _enclosing_span(root: str, path: str, line: int) -> tuple[int, int]:
        """The definition ``line`` sits in, as a range, or the line itself.

        Degrades in one direction only: every failure -- no parser for the language, no
        extents from the provider, a line inside no definition -- returns the bare line,
        which is what this verb did before. A range is an improvement to reach for, never
        a dependency, because `14.4(b)` holds here too: the lens may be absent and the
        answer must still be the old answer rather than an error.
        """
        source = read(str(Path(root) / path))
        if source is None:
            return line, line
        try:
            found = [
                d
                for d in definitions(path, source)
                if d.end_line is not None and d.start_line <= line <= d.end_line
            ]
        except Exception:
            return line, line
        if not found:
            return line, line
        innermost = max(found, key=lambda d: d.start_line)
        return innermost.start_line, int(innermost.end_line or line)

    @app.get("/api/v1/code/defs")
    def code_defs(token: Caller, path: str, repo: str | None = None) -> Response:
        name, root = _clone_root(token, repo)
        source = read(str(Path(root) / path))
        if source is None:
            raise HTTPException(status_code=404, detail=f"{path} is not in {name} at HEAD")
        found = definitions(path, source)
        return _json(
            {
                "repo": name,
                "path": path,
                "head": deps.clones.info(name).head,
                "definitions": [vars(d) for d in found],
            }
        )

    @app.get("/api/v1/code/refs")
    def code_refs(token: Caller, symbol: str, repo: str | None = None) -> Response:
        name, root = _clone_root(token, repo)
        result = references(root, symbol)
        return _json(
            {
                "repo": name,
                "symbol": symbol,
                "head": deps.clones.info(name).head,
                "hits": [vars(hit) for hit in result.hits],
                "by_kind": dict(result.by_kind),
                "searched": result.searched,
                # Section 9 rule 2: a language with no reference tier is named, never
                # answered with an empty list a caller reads as "nothing calls this".
                "unsupported": list(result.unsupported),
            }
        )

    @app.get("/api/v1/code/symbol")
    def code_symbol(token: Caller, qualname: str, repo: str | None = None) -> Response:
        name, root = _clone_root(token, repo)
        found = symbol_body(root, qualname)
        if found is None:
            raise HTTPException(status_code=404, detail=f"{qualname} is not defined in {name}")
        return _json(
            {
                "repo": name,
                "head": deps.clones.info(name).head,
                "path": found.path,
                "qualname": found.definition.qualname,
                "kind": found.definition.kind,
                "start_line": found.definition.start_line,
                "end_line": found.definition.end_line,
                "body": found.body,
                "definitions_total": found.total,
            }
        )

    @app.get("/api/v1/code/grep")
    def code_grep(
        token: Caller, pattern: str, repo: str | None = None, path: str | None = None
    ) -> Response:
        name, root = _clone_root(token, repo)
        try:
            result = grep(root, pattern, path_glob=path)
        except re.error as exc:
            raise HTTPException(status_code=400, detail=f"bad pattern: {exc}") from None
        return _json(
            {
                "repo": name,
                "head": deps.clones.info(name).head,
                "pattern": pattern,
                "path_glob": path,
                "hits": [vars(hit) for hit in result.hits],
                "files_searched": result.files_searched,
                # Both denominators, because one of them was doing duty as the other
                # (issue #37): `files_searched` counts what was read, `files_with_hits`
                # what matched, and a summary naming only one reads as the other.
                "files_with_hits": len({hit.path for hit in result.hits}),
                "truncated": result.truncated,
            }
        )

    @app.get("/api/v1/code/copies")
    def code_copies(
        token: Caller, symbol: str, repo: str | None = None, exact: bool = False
    ) -> Response:
        """Every definition of ``symbol``, grouped by whether the bodies agree (#7 item 4).

        ``exact`` groups on the body's text, which is what this did until issue #35: on
        generated code that made every model its own shape, so it is now the opt-in.
        """
        name, root = _clone_root(token, repo)
        result = copies(root, symbol, exact=exact)
        return _json(
            {
                "repo": name,
                "head": deps.clones.info(name).head,
                "symbol": symbol,
                "total": len(result.copies),
                "exact": exact,
                "truncated": result.truncated,
                "groups": [
                    {
                        "shape_hash": shape_hash,
                        "count": len(group),
                        # How many distinct bodies this shape holds. 1 means the members
                        # are textually identical; more is the measure of what
                        # normalization absorbed, and it is the number a reader doubting
                        # the grouping wants to see.
                        "bodies": len({copy.body_hash for copy in group}),
                        "copies": [vars(c) for c in group],
                    }
                    for shape_hash, group in result.groups.items()
                ],
            }
        )

    @app.get("/api/v1/why")
    def why(
        token: Caller,
        path: str,
        line: int,
        repo: str | None = None,
        render: bool = False,
        compact: bool = False,
        presentation: bool = False,
        before: dt.datetime | None = None,
        exclude: Annotated[list[int], Query()] = [],  # noqa: B006 - FastAPI reads the default
    ) -> Response:
        """Why this line is the way it is (issue #9): blame, then the argument.

        Blame here rather than in the backend: the clone is this layer's, and the query
        layer must stay a database away from a subprocess.
        """
        name, root = _clone_root(token, repo)
        found = blame_line(root, path, line)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"{path}:{line} is not in {name} at HEAD, so blame has nothing to read",
            )
        # The enclosing definition, not the bare line: a line moves inside its function on
        # every reformat, and the chain that tracks a one-line range ends at the first one
        # (relore#57). Falls back to the line where nothing can parse the file.
        start, end = _enclosing_span(root, path, line)
        history = line_history(root, path, start, end)
        as_of = deps.as_of.narrowed(before, exclude)
        view = deps.backend.why(
            name,
            path,
            line,
            sha=found.sha,
            summary=found.summary,
            history=history,
            # The pickaxe the revision chain declines to run. Here rather than in the
            # backend for the same reason blame is: the clone is this layer's, and the
            # query layer stays a database away from a subprocess.
            origin=origin_history(root, path, line, known=history),
            before=as_of.before,
            exclude=as_of.exclude,
        )
        payload = {"notice": NOTICE, **why_json(view, blame=found)}
        return _json(
            payload,
            render=(
                lambda scrubbed: render_why(scrubbed, compact=compact, presentation=presentation)
            )
            if render
            else None,
        )

    @app.post("/api/v1/precedent")
    def precedent(token: Caller) -> Response:
        return _json(
            {
                "detail": "precedent is not implemented yet (planned, milestone 4)",
                "precedents": [],
            },
            status=501,
        )

    @app.get("/api/v1/status")
    def status(token: Caller) -> Response:
        return _json({**deps.status(), "usage": deps.auth.usage(token)})

    @app.post("/api/v1/label")
    def label(body: LabelRequest, token: Caller) -> Response:
        """Section 8's relevance judgement, appended to JSONL.

        This is the only request in the API that writes, and it is not a write to the
        index: section 10's evaluation set is a byproduct of somebody using the UI rather
        than a chore nobody schedules, and it has to land somewhere. It is gated on a
        token scope and on a configured path, it never reaches ``documents``, and nothing
        retrieves it -- so invariant 3 holds: no agent conclusion becomes project memory.
        """
        if deps.labels_path is None:
            raise HTTPException(
                status_code=503, detail=f"labelling is off; start relored serve with {LABELS_ENV}"
            )
        if not token.may(LABEL_SCOPE):
            raise HTTPException(status_code=403, detail="this token may not label")
        if body.verdict not in VERDICTS:
            raise HTTPException(status_code=400, detail=f"verdict must be one of {list(VERDICTS)}")
        row = {
            "labelled_at": _now(),
            "query": body.query,
            "kind": body.kind,
            "repo": body.repo,
            "number": body.number,
            "source_type": body.source_type,
            "filters": {k: v for k, v in body.filters.items() if v},
            "verdict": body.verdict,
            "note": body.note,
            # Section 10 refuses to compare across backends, and section 6.2 warns that a
            # re-resolved repo_authority can legitimately move documents between tiers.
            # Both mean a label is only interpretable next to what produced it.
            "backend": deps.backend.info().as_dict(),
        }
        with deps.labels_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return _json({"labelled": row})

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> Response:
        """Unauthenticated on purpose: it carries counts, never content."""
        return PlainTextResponse(_prometheus(deps), media_type="text/plain; version=0.0.4")

    @app.get("/", response_class=HTMLResponse)
    def home() -> Response:
        # The daemon's own answer, not a guess from a 401: with no tokens configured the
        # page hides every mention of one (huggingface/relore, the Tailscale deployment).
        return HTMLResponse(ui.page(auth_required=not deps.auth.open))

    return app


def _json(
    payload: dict[str, Any],
    *,
    render: Callable[[dict[str, Any]], str] | None = None,
    status: int = 200,
) -> Response:
    """Every response body, scrubbed (section 11), and the envelope applied last.

    The scrub is here so it is structurally impossible for a handler to skip: it walks the
    whole tree, so a field added tomorrow is covered by the code written today.

    That is also why ``render`` is a callback rather than a field the handler fills in.
    Rendering first and scrubbing afterwards strips the envelope's *own* delimiters out of
    the rendered text -- the scrub cannot tell ours from an attacker's, which is the whole
    point of scrubbing them -- and section 8's "view as the model sees it" would then show
    a page with the one layer a person is inspecting quietly removed. So: scrub the
    content, then wrap it. Nothing runs after the envelope.
    """
    body = scrub_tree(payload)
    if render is not None:
        body["rendered"] = render(body)
    return JSONResponse(body, status_code=status)


def _now() -> str:
    from relore.store.dialect import utcnow

    return utcnow().isoformat()


def _prometheus(deps: Deps) -> str:
    state = deps.status()
    lines = [
        "# HELP relore_threads Indexed threads.",
        "# TYPE relore_threads gauge",
        f"relore_threads {state['threads']}",
        "# HELP relore_documents Indexed documents.",
        "# TYPE relore_documents gauge",
        f"relore_documents {state['documents']}",
        "# HELP relore_raw_objects Staged raw API objects.",
        "# TYPE relore_raw_objects gauge",
        f"relore_raw_objects {state['raw_objects']}",
        "# HELP relore_requests_total Requests served, by path.",
        "# TYPE relore_requests_total counter",
    ]
    for key, value in sorted(deps.counters.items()):
        if key.startswith("request:"):
            lines.append(f'relore_requests_total{{path="{key[8:]}"}} {value}')
    lines += [
        "# HELP relore_denied_total Requests rejected by authentication.",
        "# TYPE relore_denied_total counter",
        f"relore_denied_total {deps.counters['denied']}",
        "# HELP relore_rate_limited_total Requests rejected by a rate limit.",
        "# TYPE relore_rate_limited_total counter",
        f"relore_rate_limited_total {deps.counters['rate_limited']}",
        "# HELP relore_pass_high_water_seconds Per-pass high-water mark (section 5.2).",
        "# TYPE relore_pass_high_water_seconds gauge",
    ]
    for row in state["passes"]:
        if row["high_water"]:
            import datetime as dt

            moment = dt.datetime.fromisoformat(row["high_water"])
            lines.append(
                f'relore_pass_high_water_seconds{{repo="{row["repo"]}",pass="{row["pass"]}"}} '
                f"{moment.timestamp():.0f}"
            )
    return "\n".join(lines) + "\n"


def serve(
    url: str,
    *,
    host: str,
    port: int,
    allow_sqlite: bool,
    trust_network: bool = False,
    as_of: AsOf | None = None,
) -> int:
    """Section 4.1's second guardrail, and the loopback rule.

    ``relored serve`` refuses a SQLite URL without ``--allow-sqlite``: a laptop index
    served to a team is how "the ranking is bad" becomes unfalsifiable. And a daemon with
    no tokens configured refuses to bind anywhere but loopback -- a laptop should not have
    to mint a token to read its own index, but an open index on a network is not a
    default anyone chose.

    ``trust_network`` is how that second rule is *answered* rather than removed: it says
    the network in front of this process is the perimeter, which is true when the only
    route to it is a private overlay (the deployment reaches callers over Tailscale, and
    the load balancer is internal). Two properties make that a reasonable trade rather
    than a hole, and both are section 11: every response is read-only, so an unauthorized
    reader cannot change the index, and the corpus is public GitHub history, so the
    confidentiality being traded away is a property the source data never had. What
    remains worth protecting is the *write* -- and section 8's labelling still needs
    :data:`RELORE_LABELS_PATH` to be configured at all.

    It is a flag rather than an inference from "no tokens configured" on purpose: the
    default must stay fail-closed, so that a deployment which *loses* its token
    configuration refuses to come up instead of coming up open. Someone has to have typed
    this.
    """
    import uvicorn

    from relore.store.dialect import make_engine

    engine = make_engine(url, read_only=True)
    if not is_deployment_grade(engine) and not allow_sqlite:
        engine.dispose()
        raise SystemExit(
            "relored serve: refusing to serve a SQLite index. Postgres is the only "
            "supported deployment (the build plan section 4.1). Pass --allow-sqlite "
            "for local development."
        )

    auth = Authenticator.from_env()
    loopback = host in ("127.0.0.1", "::1", "localhost")
    if auth.open and not loopback and not trust_network:
        engine.dispose()
        raise SystemExit(
            f"relored serve: no tokens configured, so refusing to bind {host}. "
            "Set RELORE_API_TOKENS, or bind 127.0.0.1 — or pass --trust-network if the "
            "only route to this port is a private network you control."
        )
    if auth.open:
        # WARNING, not INFO, even when it was asked for: the one line that says an
        # unauthenticated index is reachable should be visible at the level a deployment
        # actually logs at.
        log.warning(
            "no RELORE_API_TOKENS configured: anyone who can reach %s:%s may read this index%s",
            host,
            port,
            " (--trust-network: the network is the perimeter)" if not loopback else "",
        )

    labels = os.environ.get(LABELS_ENV)
    app = build_app(engine, auth=auth, labels_path=Path(labels) if labels else None, as_of=as_of)
    if as_of is not None and as_of.pinned:
        # WARNING for the same reason the open-index line is: a pinned daemon is not a
        # normal one, and serving a benchmark cutoff to real callers would be silently
        # wrong in the direction nobody checks -- an index that is quietly missing the
        # last six months answers every question plausibly.
        log.warning(
            "pinned to a cutoff: serving the index as of %s%s. This is an evaluation "
            "mode, not a deployment.",
            as_of.before.isoformat() if as_of.before else "(no cutoff)",
            f", withholding {len(as_of.exclude)} thread(s)" if as_of.exclude else "",
        )

    # Issue #7's clone ages by itself, and this process is the one that can refresh it:
    # the volume is ReadWriteOnce and held here. Off unless asked for -- a `git fetch` is
    # the one thing `serve` does that reaches the network, and that is a deployment's
    # decision, not a default.
    every = os.environ.get(REFRESH_ENV)
    if every:
        try:
            start_refresh(app.state.deps.clones, parse_interval(every))
        except ValueError as exc:
            raise SystemExit(f"{REFRESH_ENV}: {exc}") from None

    trust_note = ", ".join(HUMAN_TRUST)
    log.info("serving %s on %s:%s (default trust tiers: %s)", url, host, port, trust_note)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
