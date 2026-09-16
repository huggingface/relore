"""``relore`` -- the read-only client.

Two verb families, deliberately in one binary:

* **history verbs** (``search``, ``thread``, ``inflight``, ``precedent``, ``why``,
  ``status``) talk to a
  ``relored`` over HTTP. They need ``RELORE_API``, a daemon of **this same version**
  (:mod:`relore.wire` -- the two ship together and refuse to talk across a difference)
  and, if that daemon requires one, a token in ``RELORE_TOKEN``.
* **code verbs** (``map``, ``defs``, ``refs``) run locally against the working tree and
  never touch the network, a database, or a token. ``defs`` and ``refs`` take ``--repo``
  to ask the daemon's working clone instead, and ``symbol``, ``grep`` and ``copies``
  exist only there (issue #7).

``RELORE_REPO`` is the default for ``--repo`` on every verb that takes one except those
last two -- for them the flag chooses a *tree*, and an environment variable must not make
that choice (issue #36).

This module must not import :mod:`relore.store`, :mod:`relore.github` or
:mod:`relore.api`. That is asserted by ``tests/unit/test_module_boundary.py`` and it is
what makes "an agent cannot write to the index" a property of the build. See AGENTS.md.
It reaches :mod:`relore.render` and :mod:`relore.security.untrusted`, which are pure
functions over dictionaries and text -- one renderer, so what an agent reads here and what
a person inspects in the web UI cannot drift.

**No results is exit 0 with an empty result**, always: an agent must not be able to
mistake "nothing in the index" for "the tool is broken". A *transport* failure is a
different thing and does exit non-zero, with a sentence saying which.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.parse import urlsplit

from relore import __version__, guidance
from relore.code.api import MissingParser
from relore.render import (
    COMPACT_CHARS,
    render_inflight,
    render_search,
    render_status,
    render_thread,
    render_why,
    trim,
)
from relore.wire import CLIENT_HEADER, SERVER_HEADER, UPGRADE_REQUIRED, explain

API_ENV = "RELORE_API"
TOKEN_ENVS = ("RELORE_TOKEN", "RELORE_API_TOKEN")
# The default for `--repo` (issue #36). A daemon serving more than one repository makes
# `--repo` mandatory on every bare number, and a single-repo session then pays that on
# every call for its whole length -- a field run passed it on ~20 consecutive calls. The
# mechanism and the precedent already existed one line above: `RELORE_API` and
# `RELORE_TOKEN` are read from the environment, and this is the third thing that is
# constant for a session.
#
# It matters most for the reader who never sees the rule. The precondition is documented
# and the error message states it, and the way that run still met it was a summarizing
# fetch of the landing page that dropped the flag off every example -- content present,
# lost in the reader's transform. An environment default is the only fix that survives
# that, because it does not depend on having read anything.
REPO_ENV = "RELORE_REPO"
# The deployment. This was `http://localhost:8080` for a day and the reason is worth
# keeping, because it is the condition this default is really coupled to: the name existed
# and did not *resolve* -- no certificate, so the ALB controller built no load balancer, so
# no DNS record -- and a default nobody can reach is worse than one that is merely often
# wrong. A connection refused on localhost tells you to start a daemon; a DNS failure on a
# hostname you never typed tells you nothing you can act on.
#
# It resolves now (2026-09-11): the certificate landed, the controller built the load
# balancer and auto-discovered the cert, and `external-dns` created the alias once its
# domain filter included the name. The load balancer is **internal**, so the failure mode
# off the VPN is a timeout against a private address rather than a DNS error -- which is
# actionable only if the message says so, and :func:`_call` does.
#
# Still `ghlore.huggingface.tech` after the 2026-09-14 rename to relore, deliberately: the
# name is deployed, it is in `external-dns`'s domain filter and on the certificate, and a
# rename would be a DNS + cert change that buys nothing a caller can see.
DEFAULT_API = "https://ghlore.huggingface.tech"

_MILESTONE = {"precedent": 4, "map": 2, "defs": 2, "refs": 2}


#: The flags that mean the same thing whatever verb they follow, spelled once because
#: spelling them per verb is what produced the matrix in huggingface/relore#55: `--json`
#: reached all twelve, `--compact` three, `--plain` none. A flag that works after `thread`
#: and not after `grep` has no rule a caller can learn, and the error calls it
#: *unrecognized* rather than misplaced -- so the repair an agent reaches for is to delete
#: the flag, which on `grep --compact` is the opposite of what it wanted.
#:
#: Adding one here puts it on the top-level parser and on every subparser at once. The
#: help text is the top-level one; subparsers take the same flag hidden (see
#: :func:`_also_after_the_verb`).
GLOBAL_FLAGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--json", {"action": "store_true", "help": "machine-readable output"}),
    (
        "--compact",
        {
            "action": "store_true",
            "help": "trim snippets, and a truncated changed-file list to its shape, for a "
            "tight context budget",
        },
    ),
    (
        "--no-compact",
        {
            "action": "store_true",
            "help": "the whole page when piped, undoing the default below",
        },
    ),
    (
        "--plain",
        {
            "action": "store_true",
            "help": "the piped form even on a terminal: facts only, no suggestions (#13)",
        },
    ),
    ("--api", {"help": f"the relored base URL (default: {API_ENV}, else {DEFAULT_API})"}),
)


def _also_after_the_verb(parser: argparse.ArgumentParser) -> None:
    """Accept every global flag after the subcommand as well as before it.

    ``relore search ... --json`` is what anyone composing a command by analogy with
    ``git`` and ``gh`` writes, and argparse's answer to it was ``unrecognized arguments:
    --json`` -- which names the flag as unknown rather than misplaced, so the reader looks
    for a typo instead of moving it. Accepting it in both positions is cheaper than
    teaching every caller our own convention.

    **Every global, not a list per verb.** The first version of this took the flags to
    alias as arguments, and what it aliased drifted with whoever was adding a verb:
    `--compact` reached `search`, `thread` and `precedent` -- the three verbs that existed
    when huggingface/relore#6 was closed, one of them still a stub -- and not `grep`,
    `symbol` or `copies`, whose output is the output most worth trimming (#55). So the
    call site no longer chooses, and it is applied by walking ``sub.choices`` after the
    verbs are built, which is what makes a verb added tomorrow arrive with all four.

    ``SUPPRESS`` is what makes the alias harmless: without it the subparser would write its
    own default over a flag given *before* the verb, silently turning ``relore --json
    search`` back off. Hidden from the subcommand's help, because it is already in the
    program's.
    """
    for flag, spec in GLOBAL_FLAGS:
        parser.add_argument(
            flag,
            **{key: value for key, value in spec.items() if key != "help"},
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


#: What ``--repo`` says on the six verbs that refuse a bare argument without it. ``defs``
#: and ``refs`` spell their own, because for them the flag means something else.
_REPO_HELP = f"OWNER/NAME; needed when the daemon serves several (default: ${REPO_ENV})"


def build_parser() -> argparse.ArgumentParser:
    # `--help` is the only surface that arrives byte-exact, needs no network and is
    # already installed, so issue #38 asks it to be sufficient on its own: the epilog is
    # where the *order* to reach for the verbs in lives. It comes from
    # :mod:`relore.guidance` rather than being written here, because the landing page and
    # the docs render the same text and the four surfaces drifted once already (#40).
    p = argparse.ArgumentParser(
        prog="relore",
        description=__doc__.splitlines()[0],
        epilog=guidance.epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"relore {__version__}")
    for flag, spec in GLOBAL_FLAGS:
        p.add_argument(flag, **spec)
    sub = p.add_subparsers(dest="verb", required=True)

    s = sub.add_parser("search", help="search the indexed issue/PR history")
    s.add_argument("query", nargs="?", default="")
    # Repeatable, because the API takes a list of each: a traceback names more than one
    # file, and section 6's expansion will fan a single call out over all of them.
    for flag, help_text in (
        ("--error", "an exception line or normalized message"),
        ("--test", "a test id"),
        ("--file", "a repository path"),
        ("--symbol", "a function or class name"),
        ("--label", "a GitHub label"),
    ):
        s.add_argument(flag, action="append", default=[], help=f"{help_text} (repeatable)")
    s.add_argument("--kind", help="failure | precedent | rationale")
    s.add_argument(
        "--trust",
        help="raise the floor to 'authoritative', or 'machine' to ask what our own bots said",
    )
    s.add_argument("--since", help="ISO timestamp; only documents written after it")
    # Narrows the token's scope, never widens it -- see the server. Repeatable, so it
    # reads like the other filters even though one name is the usual case.
    s.add_argument(
        "--repo",
        action="append",
        default=[],
        dest="repos",
        help=f"OWNER/NAME; narrow to this repository (repeatable; default: ${REPO_ENV})",
    )
    # Spelled out rather than imported from `search.queries`, like `--kind` above: reaching
    # that module executes `relore/search/__init__.py`, which imports SQLAlchemy, and
    # `test_module_boundary` exists to catch exactly that. The server validates the value.
    s.add_argument(
        "--sort",
        choices=("relevance", "newest"),
        default="relevance",
        help="'newest' orders the same hits by date instead of score",
    )
    s.add_argument("--limit", type=int, default=10)
    s.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="ask exactly one question instead of fanning out over the query's parts",
    )

    t = sub.add_parser("thread", help="one thread, comments ranked by relevance")
    t.add_argument("number", type=int)
    t.add_argument(
        "--focus",
        default="",
        help="rank the comments by this, best first — or, with --outline, keep only the "
        "lines carrying one of these words",
    )
    t.add_argument("--repo", help=_REPO_HELP)
    t.add_argument(
        "--full",
        action="store_true",
        help="serve the opening post whole instead of its first 800 characters",
    )
    t.add_argument(
        "--outline",
        action="store_true",
        help="one line per comment for the WHOLE thread, instead of a page of ten: "
        "the shape first, then ask for the parts; --focus narrows it to the lines "
        "carrying a word",
    )
    t.add_argument(
        "--after",
        default="",
        metavar="ID",
        help="start after this comment id (from an outline line or a URL), to sweep a "
        "long thread instead of re-rolling the same page",
    )
    t.add_argument(
        "--comment",
        default="",
        metavar="ID",
        help="serve one comment whole, by the id in its line on a page or an outline "
        "(a page cuts a long comment to a window)",
    )
    t.add_argument(
        "--files",
        action="store_true",
        help="also list the changed-file paths; the count and the truncation notice are "
        "always shown",
    )

    inflight = sub.add_parser(
        "inflight",
        help="open pull requests that already claim to close this issue",
    )
    inflight.add_argument("number", type=int)
    inflight.add_argument("--repo", help=_REPO_HELP)

    pr = sub.add_parser(
        "precedent",
        help="(NOT IMPLEMENTED YET) completed units of work and what they consisted of",
    )
    pr.add_argument("--kind")
    pr.add_argument("--file")
    # Every other listing verb is limitable, so this one refusing `--limit` is a papercut
    # rather than a decision.
    pr.add_argument("--limit", type=int, default=5)

    w = sub.add_parser(
        "why",
        help="the pull request that last changed this line, and what reviewers said on it",
    )
    w.add_argument("location", metavar="PATH:LINE")
    w.add_argument("--repo", help=_REPO_HELP)

    sub.add_parser("status", help="index coverage and collector telemetry")

    m = sub.add_parser(
        "map",
        help=(
            "ranked repo map of the local checkout: name matches per definition, dunders "
            "excluded (no server)"
        ),
    )
    m.add_argument("--limit", type=int, default=40)
    m.add_argument("path", nargs="?", default=".")

    d = sub.add_parser("defs", help="definitions in a file, locally or with --repo")
    d.add_argument("path")
    d.add_argument(
        "--repo", help=f"OWNER/NAME; ask the daemon's working clone instead (default: ${REPO_ENV})"
    )

    r = sub.add_parser(
        "refs",
        help=(
            "every occurrence of a symbol, by kind: call, definition, attribute, name. "
            "Local unless --repo"
        ),
    )
    r.add_argument("symbol")
    r.add_argument(
        "--repo", help=f"OWNER/NAME; ask the daemon's working clone instead (default: ${REPO_ENV})"
    )

    sym = sub.add_parser("symbol", help="the source of one definition, from the daemon's clone")
    sym.add_argument("qualname")
    sym.add_argument("--repo", help=_REPO_HELP)

    g = sub.add_parser("grep", help="a regular expression over the daemon's working clone")
    g.add_argument("pattern")
    g.add_argument("--repo", help=_REPO_HELP)
    g.add_argument("--path", help="glob the paths must match, e.g. 'src/**/modeling_*.py'")

    c = sub.add_parser(
        "copies",
        help="every definition of a symbol, grouped by whether the bodies agree",
    )
    c.add_argument("symbol")
    c.add_argument("--repo", help=_REPO_HELP)
    c.add_argument(
        "--exact",
        action="store_true",
        help="group by the body's exact text, annotations and docstrings included",
    )

    # Applied here, not at each `add_parser`: a verb cannot then be born missing a global,
    # which is precisely how #55's matrix formed one `add_parser` at a time.
    for verb_parser in sub.choices.values():
        _also_after_the_verb(verb_parser)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "search": _search,
        "thread": _thread,
        "inflight": _inflight,
        "why": _why,
        "status": _status,
        "map": _map,
        "defs": _defs,
        "refs": _refs,
        "symbol": _symbol,
        "grep": _grep,
        "copies": _copies,
    }.get(args.verb)
    if handler is None:
        raise SystemExit(
            f"relore {args.verb}: not implemented yet. It is planned "
            f"(milestone {_MILESTONE[args.verb]}) and `--help` marks it, so nothing else "
            "in your plan depends on it."
        )
    try:
        return handler(args)
    except MissingParser as exc:
        # An install, not a bug -- so a sentence, never an ImportError traceback (AGENTS.md,
        # section 2). Raised by the registry rather than by an eager `import tree_sitter`
        # check here, because a machine with universal-ctags and no grammar can still
        # answer, and refusing it would be wrong.
        raise SystemExit(f"relore: {exc}") from None


# -- history verbs ---------------------------------------------------------


def _repo(args: argparse.Namespace) -> str | None:
    """Which repository this call is about: ``--repo``, else ``RELORE_REPO`` (issue #36).

    ``None`` is not a failure and is not a guess -- it hands the question to the daemon,
    which answers it if exactly one repository is in scope and otherwise refuses with the
    list. That order matters: the flag has to win, so a session with a default set can
    still ask about another repository without unsetting anything, and an empty variable
    reads as unset rather than as a repository named ``""``.

    Read here rather than as an ``argparse`` default so that the value is the environment
    at *call* time, which is what a test that sets it and a shell that exports it after
    import both expect.
    """
    return getattr(args, "repo", None) or os.environ.get(REPO_ENV) or None


def _repos(args: argparse.Namespace) -> list[str]:
    """The same default for ``search``, whose ``--repo`` is repeatable.

    ``search`` spans the whole scope by design, so this is the one place the variable
    *narrows* an answer that would otherwise have been wider. That is what setting it
    asks for -- it names the repository the session is about -- and any ``--repo`` on the
    call replaces it wholesale rather than adding to it, so the flag stays the last word.
    """
    if args.repos:
        return args.repos
    default = os.environ.get(REPO_ENV)
    return [default] if default else []


def _search(args: argparse.Namespace) -> int:
    payload = _call(
        args,
        "POST",
        "/api/v1/search",
        body={
            "query": args.query,
            "kind": args.kind,
            "trust": args.trust,
            "files": args.file,
            "symbols": args.symbol,
            "errors": args.error,
            "tests": args.test,
            "labels": args.label,
            "repos": _repos(args),
            "since": args.since,
            "limit": args.limit,
            "compact": _compact(args),
            "sort": args.sort,
            "expand": args.expand,
            # The server renders it, so the envelope a person inspects in the web UI and
            # the one an agent reads here are the same string from the same code.
            "render": not args.json,
            "presentation": _presentation(args),
        },
    )
    return _emit(
        args,
        payload,
        lambda: (
            payload.get("rendered")
            or render_search(payload, compact=_compact(args), presentation=_presentation(args))
        ),
    )


def _thread(args: argparse.Namespace) -> int:
    query = {
        "focus": args.focus,
        "compact": str(_compact(args)).lower(),
        "render": str(not args.json).lower(),
        "presentation": str(_presentation(args)).lower(),
        "full": str(args.full).lower(),
        "outline": str(args.outline).lower(),
        "after": args.after,
        "comment": args.comment,
        "files": str(args.files).lower(),
    }
    if repo := _repo(args):
        query["repo"] = repo
    payload = _call(args, "GET", f"/api/v1/thread/{args.number}", params=query)
    return _emit(
        args,
        payload,
        lambda: (
            payload.get("rendered")
            or render_thread(
                payload,
                compact=_compact(args),
                presentation=_presentation(args),
                files=args.files,
            )
        ),
    )


def _inflight(args: argparse.Namespace) -> int:
    """Before diagnosing, ask whether somebody is already fixing it.

    Cheap, one hop, and it prevents the most expensive mistake an agent makes -- see
    :mod:`relore.ingest.relationships`.
    """
    query = {
        "render": str(not args.json).lower(),
        "compact": str(_compact(args)).lower(),
        "presentation": str(_presentation(args)).lower(),
    }
    if repo := _repo(args):
        query["repo"] = repo
    payload = _call(args, "GET", f"/api/v1/inflight/{args.number}", params=query)
    return _emit(
        args,
        payload,
        lambda: (
            payload.get("rendered")
            or render_inflight(payload, compact=_compact(args), presentation=_presentation(args))
        ),
    )


def _why(args: argparse.Namespace) -> int:
    """`git blame` gives the commit; this gives the argument (issue #9)."""
    path, _, line = args.location.rpartition(":")
    if not path or not line.isdigit():
        raise SystemExit(f"relore why: expected PATH:LINE, got {args.location!r}")
    params = {
        "path": path,
        "line": line,
        "render": str(not args.json).lower(),
        "compact": str(_compact(args)).lower(),
        "presentation": str(_presentation(args)).lower(),
    }
    if repo := _repo(args):
        params["repo"] = repo
    payload = _call(args, "GET", "/api/v1/why", params=params)
    return _emit(
        args,
        payload,
        lambda: (
            payload.get("rendered")
            or render_why(payload, compact=_compact(args), presentation=_presentation(args))
        ),
    )


def _status(args: argparse.Namespace) -> int:
    payload = _call(args, "GET", "/api/v1/status")
    return _emit(args, payload, lambda: render_status(payload))


def _presentation(args: argparse.Namespace) -> bool:
    """Whether a person is reading this (#13).

    The `git`/`gh` convention: a terminal gets the flags worth trying next, a pipe gets the
    documented grammar and nothing else. `--plain` forces the pipe form for a script that
    happens to own a TTY.
    """
    return sys.stdout.isatty() and not getattr(args, "plain", False)


def _compact(args: argparse.Namespace) -> bool:
    """Whether to trim quoted prose -- **on by default for a pipe**.

    A tool result is not paid for once. It is re-sent with every later turn of the session
    that read it, so its cost is its size times the turns remaining: measured on one agent
    run, relore's own output came to 32,349 tokens and **892,197** once re-billing is
    counted, 47% of that run. A 12,000-character thread page read at turn 13 is thirty
    copies of itself by the end.

    `--compact` existed for this and an agent used it twice in twenty-three calls. A saving
    that depends on the caller remembering a flag does not happen, so the flag becomes the
    default for the caller that pays for it -- and `--no-compact` is there for the pipe
    that genuinely wants the whole page. A terminal is unchanged: a person is reading it
    once, and pays for it once.
    """
    if getattr(args, "compact", False):
        return True
    if getattr(args, "json", False):
        # `--json` is the machine surface and asked for the structure. Trimming it by
        # default would drop fields from a program's input to save a model's context --
        # the wrong caller paying. Explicit `--compact --json` still trims, because then
        # somebody asked.
        return False
    return not sys.stdout.isatty() and not getattr(args, "no_compact", False)


def _emit(args: argparse.Namespace, payload: dict[str, Any], text) -> int:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(text())
    return 0


# -- code verbs: local, offline, no token ----------------------------------
#
# Deliberately local (section 1). The whole justification for a daemon is that
# `/search/issues` is 30 requests/minute shared across the token; that constraint does not
# exist for code -- a parser over the working tree costs nothing, needs no token, and
# describes *the tree the caller is actually on*, dirty files included. A server-side code
# index knows `main`; the agent is on a feature branch mid-edit.


def _map(args: argparse.Namespace) -> int:
    from relore.code.repomap import repo_map

    result = repo_map(args.path, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                {
                    "files": result.files,
                    "definitions": result.definitions,
                    "ranked_by": result.ranked_by,
                    "excluded_dunders": result.excluded_dunders,
                    "entries": [vars(e) for e in result.entries],
                },
                indent=2,
            )
        )
        return 0
    print(
        f"{result.definitions} definitions in {result.files} files, "
        f"top {len(result.entries)} by {result.ranked_by}"
        + (f" ({result.excluded_dunders} dunders excluded)" if result.excluded_dunders else "")
    )
    print(
        "  (a count of the written *name*: same-named definitions share it, so the "
        "definition count is how much this row overstates)"
    )
    for entry in result.entries:
        shared = f"{entry.shared_by} definition" + ("s" if entry.shared_by > 1 else "")
        weight = f"{entry.matches} matches / {shared}" if entry.matches else "-"
        print(f"  {entry.qualname:44} {entry.kind:9} {weight:28} {entry.path}:{entry.line}")
    return 0


def _code_params(args: argparse.Namespace, **extra: Any) -> dict[str, str]:
    params = {key: value for key, value in extra.items() if value}
    if repo := _repo(args):
        params["repo"] = repo
    return params


def _defs(args: argparse.Namespace) -> int:
    from relore.code.defs import definitions
    from relore.code.walk import read

    # `args.repo`, not `_repo(args)`: for `defs` and `refs` the flag does not merely name a
    # repository, it *switches the verb* from the tree you are standing in to the daemon's
    # clone of HEAD. `RELORE_REPO` says which repository a session is about; it must not
    # silently move these two off the working copy, uncommitted edits and all, which is the
    # only thing they offer that nothing else does. Where the flag is given, `_code_params`
    # reads the same value back out of it.
    if args.repo:
        payload = _call(args, "GET", "/api/v1/code/defs", params=_code_params(args, path=args.path))
        return _emit(args, payload, lambda: _defs_text(payload["definitions"]))

    source = read(args.path)
    if source is None:
        raise SystemExit(f"relore: cannot read {args.path}")
    found = definitions(args.path, source)
    if args.json:
        print(json.dumps([vars(d) for d in found], indent=2))
        return 0
    print(_defs_text([vars(d) for d in found]))
    return 0


def _defs_text(found: list[dict[str, Any]]) -> str:
    lines = []
    for definition in found:
        end = definition.get("end_line")
        extent = f"{definition['start_line']}-{end}" if end else str(definition["start_line"])
        lines.append(f"{extent:12} {definition['kind']:9} {definition['qualname']}")
    return "\n".join(lines)


def _refs(args: argparse.Namespace) -> int:
    from relore.code.refs import references

    if args.repo:
        payload = _call(
            args, "GET", "/api/v1/code/refs", params=_code_params(args, symbol=args.symbol)
        )
        return _emit(args, payload, lambda: _refs_text(payload))

    result = references(".", args.symbol)
    if args.json:
        print(
            json.dumps(
                {
                    "searched": result.searched,
                    "unsupported": list(result.unsupported),
                    "by_kind": result.by_kind,
                    "hits": [vars(h) for h in result.hits],
                },
                indent=2,
            )
        )
        return 0
    for hit in result.hits:
        print(f"{hit.path}:{hit.line}  {hit.kind}")
    counts = ", ".join(f"{count} {kind}" for kind, count in result.by_kind.items())
    print(
        f"-- {len(result.hits)} references in {result.searched} files"
        + (f" ({counts})" if counts else "")
    )
    if result.unsupported:
        # Section 9's rule 2: say so and exit 0, rather than return an empty list a caller
        # would read as "nothing calls this".
        print(
            f"   ({', '.join(result.unsupported)} offers no reference tier, so files it "
            "claims were not searched)"
        )
    return 0


def _refs_text(payload: dict[str, Any]) -> str:
    hits = payload.get("hits") or []
    lines = [f"{hit['path']}:{hit['line']}  {hit['kind']}" for hit in hits]
    counts = ", ".join(f"{count} {kind}" for kind, count in (payload.get("by_kind") or {}).items())
    lines.append(
        f"-- {len(hits)} references at {payload.get('head', '')[:8]}"
        + (f" ({counts})" if counts else "")
    )
    return "\n".join(lines)


def _symbol(args: argparse.Namespace) -> int:
    """The 18 lines that matter, without a clone (#7 item 2)."""
    payload = _call(
        args, "GET", "/api/v1/code/symbol", params=_code_params(args, qualname=args.qualname)
    )
    return _emit(args, payload, lambda: _symbol_text(payload, presentation=_presentation(args)))


def _symbol_text(payload: dict[str, Any], *, presentation: bool = False) -> str:
    head = f"{payload['path']}:{payload['start_line']}  {payload['kind']}  {payload['qualname']}"
    total = payload.get("definitions_total", 1)
    if total > 1:
        # Serving one of many as *the* body is a wrong answer a caller cannot see. The
        # count is the fact and is printed either way; the verb that groups them is advice.
        head += f"\n({total} definitions of this name" + (
            "; `relore copies` groups them)" if presentation else ")"
        )
    # NOT trimmed by `--compact`: the body is the answer this verb exists to give, and a
    # silently shortened one is a wrong answer rather than a cheaper one.
    return f"{head}\n{payload['body']}"


def _grep(args: argparse.Namespace) -> int:
    payload = _call(
        args,
        "GET",
        "/api/v1/code/grep",
        params=_code_params(args, pattern=args.pattern, path=args.path),
    )
    return _emit(
        args,
        payload,
        lambda: _grep_text(payload, compact=_compact(args), presentation=_presentation(args)),
    )


def _grep_text(
    payload: dict[str, Any], *, compact: bool = False, presentation: bool = False
) -> str:
    """Both numbers named, because one of them used to mean the other thing (issue #37).

    The summary read ``-- 3 in 4 files``, where 4 was the count of files *searched*. The
    natural English reading is "appears in four files", and on the audit this verb is for --
    *which models are affected* -- a file count that over-reports is a wrong answer in the
    direction that looks like diligence. Three hits in two files across four searched is
    three numbers, and saying only two of them was what made them ambiguous.

    ``files_with_hits`` is counted from the hits rather than taken on faith, so it stays
    right for the capped answer too: it describes the rows printed below it.

    The matched line is the one piece of unbounded text this verb prints -- a minified
    bundle or a long tensor expression is one hit and 300 characters -- so it is what
    ``--compact`` shortens, and the page says how many lines it shortened
    (huggingface/relore#55). The cap is a fact and is printed either way; *how to narrow
    it* is advice, so it belongs to the terminal form (#13).
    """
    hits = payload.get("hits") or []
    lines, trimmed = [], 0
    for hit in hits:
        text = hit["text"]
        if compact:
            text, was_trimmed = trim(text)
            trimmed += was_trimmed
        lines.append(f"{hit['path']}:{hit['line']}  {text}")
    with_hits = len({hit["path"] for hit in hits})
    searched = payload.get("files_searched", 0)
    lines.append(
        f"-- {len(hits)} hit{'' if len(hits) == 1 else 's'} in {with_hits} of "
        f"{searched} file{'' if searched == 1 else 's'} searched"
    )
    if trimmed:
        lines.append(
            f"   ({trimmed} line{'' if trimmed == 1 else 's'} shortened to {COMPACT_CHARS} "
            "characters by --compact"
            + ("; `--json` serves them whole" if presentation else "")
            + ")"
        )
    if payload.get("truncated"):
        lines.append("   (capped" + (": narrow it with --path" if presentation else "") + ")")
    return "\n".join(lines)


def _copies(args: argparse.Namespace) -> int:
    """Which copies diverge -- the question a repository that duplicates code on purpose
    actually asks (#7 item 4)."""
    payload = _call(
        args,
        "GET",
        "/api/v1/code/copies",
        params=_code_params(args, symbol=args.symbol, exact=str(args.exact).lower()),
    )
    return _emit(args, payload, lambda: _copies_text(payload, presentation=_presentation(args)))


def _copies_text(payload: dict[str, Any], *, presentation: bool = False) -> str:
    """The grouping, and what was normalized away to reach it (issue #35).

    Two lines of disclosure rather than none, because the answer changed shape: a group is
    now "these bodies do the same thing" rather than "these bodies are the same text", and
    a reader who is not told that will read a shape of one as a textual difference and go
    looking for a typo.

    The singleton callout is the other half. On a symbol with 186 definitions the row worth
    reading is almost always the shape with one member -- that is what *diverged* -- and it
    sorts last, under everything else.
    """
    groups = payload.get("groups") or []
    lines = [
        f"{payload.get('total', 0)} definitions of {payload.get('symbol')} "
        f"in {len(groups)} shape{'' if len(groups) == 1 else 's'}"
    ]
    if payload.get("exact"):
        lines.append("(grouped by the exact text of the body)")
    else:
        lines.append(
            "(grouped by what the body does: type annotations, docstrings and comments "
            "are normalized away first"
            + (" — `--exact` groups by the text instead" if presentation else "")
            + ")"
        )
    for index, group in enumerate(groups, start=1):
        count = group["count"]
        if count == 1 and len(groups) > 1:
            note = "  <- the only one of its shape"
        elif index == 1 and len(groups) > 1:
            note = " (the majority shape)"
        else:
            note = ""
        lines.append(f"\n-- shape {index}: {count} cop{'y' if count == 1 else 'ies'}{note}")
        lines += [f"   {copy['path']}:{copy['start_line']}" for copy in group["copies"]]
    if payload.get("truncated"):
        lines.append("\n(capped.)")
    return "\n".join(lines)


# -- transport -------------------------------------------------------------


def _call(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One request, with the failure modes told apart.

    A 404 or an empty index is data; a refused connection, a 401, a 426 and a 429 are not,
    and each gets its own sentence. Blurring them is how an agent decides the project has
    no history when the daemon is simply down.
    """
    import httpx

    base = args.api or os.environ.get(API_ENV) or DEFAULT_API
    headers = {"accept": "application/json", CLIENT_HEADER: __version__}
    # The *name* too, not just the value: a 401 has to say which variable was rejected,
    # and there are two it could have come from.
    sent_from = next((e for e in TOKEN_ENVS if os.environ.get(e)), None)
    if sent_from:
        headers["authorization"] = f"Bearer {os.environ[sent_from]}"

    try:
        response = httpx.request(
            method,
            f"{base.rstrip('/')}{path}",
            json=body,
            params=params,
            headers=headers,
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise SystemExit(f"relore: {_unreachable(base, exc)}") from None

    if response.status_code == UPGRADE_REQUIRED:
        # The daemon caught the version difference. It composed the sentence, because it
        # knows both numbers -- print that rather than a second opinion.
        raise SystemExit(f"relore: {_detail(response)}")

    # The other direction: a daemon older than the handshake enforces nothing, so the
    # client is the only end that can catch "the deployment is behind". Checked before
    # the 401 and the 429 so a version problem is never reported as a credential problem,
    # which is the same ordering the daemon's own gate uses.
    served_by = response.headers.get(SERVER_HEADER)
    if served_by is None:
        raise SystemExit(
            f"relore: {base} answered without a {SERVER_HEADER} header, so it is not a "
            f"relore daemon of this generation — it predates the version handshake, or "
            f"something else is answering on that address. This client is {__version__}."
        )
    mismatch = explain(__version__, served_by)
    if mismatch is not None:
        raise SystemExit(f"relore: {mismatch}")

    if response.status_code == 401:
        # Two failures, not one. Telling an operator who has already exported a token to
        # export a token sends them looking for a typo in the value, when the usual cause
        # is that the value is fine and belongs to a different daemon.
        if sent_from:
            raise SystemExit(
                f"relore: {base} rejected the token in {sent_from}. A token is only valid "
                f"on the daemon whose RELORE_API_TOKENS lists it, so one minted for "
                f"another deployment will not work here."
            )
        raise SystemExit(
            f"relore: {base} requires a token. Set {TOKEN_ENVS[0]} to one scoped to your "
            "repositories."
        )
    if response.status_code == 429:
        detail = _detail(response)
        raise SystemExit(f"relore: rate limited ({detail}). Retry after the header says to.")
    if response.status_code >= 400:
        raise SystemExit(
            f"relore: {base}{path} returned {response.status_code}: {_detail(response)}"
        )
    return dict(response.json())


def _unreachable(base: str, exc: Exception) -> str:
    """Why nothing answered, told apart by *which* address did not answer.

    Two situations wear the same symptom and take opposite next actions. The deployment
    sits behind an **internal** load balancer, so from outside the VPN it is a timeout
    against a private address and looks exactly like a daemon that is down. A loopback
    address is the other one: nothing is listening because the caller pointed
    ``RELORE_API`` at a daemon they have not started -- and a client-only install cannot
    start one, which is what turned a good error message into a dead end
    (huggingface/relore#19). So the remedy offered names the extra, and names the way out
    that needs no daemon at all.
    """
    why = f"cannot reach {base} ({exc.__class__.__name__})"
    if _is_loopback(base):
        return (
            f"{why}. That is a local address, so nothing is listening on it: {API_ENV} "
            f"points at a daemon you have not started. Start one -- `relored serve` needs "
            f"`pip install 'relore[server]'`, which the client install does not carry -- "
            f"or unset {API_ENV} to use the deployment at {DEFAULT_API}."
        )
    return (
        f"{why}. The deployment is VPN-internal — check the VPN first. Otherwise point "
        f"{API_ENV} at your own `relored serve`, or port-forward the deployment: "
        f"`kubectl -n ghlore port-forward deploy/ghlore 8080:8080`."
    )


def _is_loopback(base: str) -> bool:
    host = urlsplit(base).hostname or ""
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return detail if isinstance(detail, str) else json.dumps(detail)


if __name__ == "__main__":
    sys.exit(main())
