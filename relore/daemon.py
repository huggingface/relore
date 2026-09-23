"""``relored`` -- index and serve. Needs a database and a GitHub token.

``fetch`` and ``derive`` are separate on purpose: raw API payloads are staged in
``raw_objects``, so improving an extractor and re-deriving costs minutes of local CPU
instead of another day of API budget. ``poll`` does both for the few threads that moved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, select

from relore import __version__

if TYPE_CHECKING:
    from pathlib import Path

    from relore.github.client import GitHubClient
    from relore.github.graphql import GraphQLClient

# Kept as literals so `--help` needs no import of the ingest package.
PASS_NAMES = ("threads", "issue_comments", "pr_comments", "files", "derive", "authority")
SLICE_NAMES = ("rationale", "failure", "precedent")
SYSTEM_NAMES = ("index", "index+expand", "github", "grep")

DB_URL_ENV = "RELORE_DATABASE_URL"
TOKEN_ENVS = ("GITHUB_TOKEN", "RELORE_GITHUB_TOKEN")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relored", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"relored {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="verb", required=True)

    sub.add_parser("migrate", help="create or upgrade the schema")

    q = sub.add_parser("fetch", help="stage raw payloads only (network, resumable)")
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument("--only", action="append", choices=PASS_NAMES, help="repeatable")

    q = sub.add_parser("backfill", help="full history: fetch, derive, then the per-PR pass")
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument("--only", action="append", choices=PASS_NAMES, help="repeatable")
    q.add_argument(
        "--no-graphql",
        action="store_true",
        help="skip the per-PR pass; on this path it is the only source of review bodies",
    )
    q.add_argument(
        "--no-authority",
        action="store_true",
        help="skip resolving MEMBER authors; every one of them stays `reported`",
    )
    q.add_argument(
        "--refresh-details",
        action="store_true",
        help="re-fetch per-PR detail already staged, instead of only PRs that lack it",
    )
    q.add_argument(
        "--merged-only",
        action="store_true",
        help="per-PR detail for merged PRs only; open ones keep no changed-file list, "
        "so `search --file` cannot see the work currently in flight",
    )

    q = sub.add_parser(
        "sample",
        help="index a bounded window of a repository, each thread fetched whole",
    )
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument(
        "--since", required=True, metavar="YYYY-MM-DD", help="threads updated on or after this"
    )
    q.add_argument("--kind", default="issue", choices=("issue", "pr"), help="default: issue")
    q.add_argument(
        "--merged",
        action="store_true",
        help="pull requests only: skip anything that was never merged",
    )
    q.add_argument("--limit", type=int, help="stop after this many threads (a smoke test)")

    q = sub.add_parser("sweep", help="reconcile deleted comments (a full walk; weekly is ample)")
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")

    q = sub.add_parser(
        "authority",
        help="resolve MEMBER authors to write access, then re-derive what moved tier",
    )
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument(
        "--offline",
        action="store_true",
        help="skip the permission API and use the merged_by fallback alone",
    )
    q.add_argument(
        "--refresh",
        action="store_true",
        help="re-resolve every author, ignoring how recently it was last checked",
    )

    q = sub.add_parser(
        "derive", help="rebuild threads/documents from raw_objects (local, no network)"
    )
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument("--number", type=int, action="append", help="repeatable; default all")

    q = sub.add_parser("poll", help="keep an indexed repo current (fetch + derive on the delta)")
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument("--interval", default="5m", help="30s, 5m, 2h (default: 5m)")
    q.add_argument("--once", action="store_true", help="one pass, then exit")

    q = sub.add_parser(
        "clone",
        help="create or refresh the working clone the code verbs read (issue #7)",
    )
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument("--root", help="where clones live (default: $RELORE_CLONE_ROOT)")
    q.add_argument("--url", help="clone URL (default: https://github.com/OWNER/NAME.git)")

    q = sub.add_parser("serve", help="HTTP JSON API + web search UI")
    q.add_argument("--port", type=int, default=8080)
    q.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "loopback by default; binding wider needs RELORE_API_TOKENS or "
            "--trust-network (default: 127.0.0.1)"
        ),
    )
    # SQLite is a development affordance, not a deployment: it has no trigram or vector
    # tier and scores with bm25 rather than ts_rank_cd, so an index served from one makes
    # "the ranking is bad" unfalsifiable. build-plan section 4.1.
    q.add_argument(
        "--allow-sqlite",
        action="store_true",
        help="serve from a SQLite database anyway (development only, never production)",
    )
    # The loopback rule, answered rather than removed. Read-only and a public corpus is
    # what makes it a reasonable trade; see `api.server.serve`.
    q.add_argument(
        "--trust-network",
        action="store_true",
        help="bind wider with no tokens, because the only route here is a private network",
    )
    # A cutoff a caller has to pass is a cutoff a caller can omit. For a temporal
    # evaluation the pin belongs to the process, so the agent's own query cannot widen it.
    q.add_argument(
        "--as-of",
        help=(
            "EVALUATION MODE: serve the index as it stood at this ISO timestamp. Every "
            "read is bounded, and a request may only narrow it further"
        ),
    )
    q.add_argument(
        "--exclude-thread",
        type=int,
        action="append",
        default=[],
        dest="exclude_threads",
        metavar="N",
        help="EVALUATION MODE: withhold this thread from every read (repeatable)",
    )

    q = sub.add_parser(
        "mine",
        help="section 10: mine benchmark candidates out of the index (local, no network)",
    )
    q.add_argument("--repo", required=True, metavar="OWNER/NAME")
    q.add_argument(
        "--slice",
        dest="slices",
        action="append",
        choices=SLICE_NAMES,
        help="repeatable; default all three",
    )
    q.add_argument("--out", required=True, metavar="PATH", help="JSONL evaluation set to grow")
    q.add_argument("--limit", type=int, default=60, help="candidates per slice (default: 60)")

    q = sub.add_parser(
        "export-corpus",
        help="section 13: the pre-cutoff documents, as JSONL, for the lexical/dense baselines",
    )
    q.add_argument(
        "--repo", required=True, metavar="OWNER/NAME", help="repeatable", action="append"
    )
    q.add_argument(
        "--before",
        metavar="TIMESTAMP",
        help="the cutoff, ISO and zoned (2026-03-01T00:00:00Z). Omit to export the whole "
        "index, which is recorded as a corpus with no cutoff",
    )
    q.add_argument(
        "--exclude-thread",
        type=int,
        action="append",
        default=[],
        dest="exclude_threads",
        metavar="N",
        help="withhold this thread from the export (repeatable), for a thread that "
        "predates the cutoff and is still the answer",
    )
    q.add_argument("--out", required=True, metavar="PATH", help="JSONL to write")

    q = sub.add_parser(
        "judge",
        help="section 10: fold the UI's relevance labels into the evaluation set",
    )
    q.add_argument("--set", dest="eval_set", required=True, metavar="PATH")
    q.add_argument(
        "--labels",
        metavar="PATH",
        help="the JSONL `serve` appends to; omit to fold nothing, which is how a set "
        "that is already judged gets frozen",
    )
    q.add_argument(
        "--as",
        dest="judge",
        default="human",
        choices=("human", "model"),
        help="who judged (default: human)",
    )
    q.add_argument("--freeze", action="store_true", help="freeze the set afterwards")

    q = sub.add_parser(
        "bench", help="section 10: score an evaluation set against relore and both baselines"
    )
    q.add_argument(
        "--repo", required=True, metavar="OWNER/NAME", help="repeatable", action="append"
    )
    q.add_argument("--set", dest="eval_set", required=True, metavar="PATH", help="the frozen JSONL")
    q.add_argument(
        "--system",
        dest="systems",
        action="append",
        choices=SYSTEM_NAMES,
        help="repeatable; default index + github",
    )
    q.add_argument(
        "--clone",
        metavar="[OWNER/NAME=]PATH",
        action="append",
        help="a checkout, required by the grep baseline; repeatable, one per --repo. "
        "The prefix may be omitted only when a single repository is being scored",
    )
    q.add_argument("--json", action="store_true", help="the whole report, machine-readable")
    q.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="score a set that is not frozen (section 10 says freeze it first)",
    )
    q.add_argument(
        "--before",
        metavar="TIMESTAMP",
        help="EVALUATION MODE: score the index as it stood at this instant. The report "
        "says what the post-filter does not fix",
    )
    q.add_argument(
        "--exclude-thread",
        type=int,
        action="append",
        default=[],
        dest="exclude_threads",
        metavar="N",
        help="EVALUATION MODE: withhold this thread from every query (repeatable)",
    )

    q = sub.add_parser("status", help="live sync cursors, per repo and pass")
    q.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    handler = {
        "migrate": _migrate,
        "fetch": _fetch,
        "backfill": _backfill,
        "derive": _derive,
        "sample": _sample,
        "poll": _poll,
        "sweep": _sweep,
        "authority": _authority,
        "mine": _mine,
        "export-corpus": _export_corpus,
        "bench": _bench,
        "judge": _judge,
        "clone": _clone,
        "serve": _serve,
        "status": _status,
    }[args.verb]
    return handler(args)


# -- verbs -----------------------------------------------------------------


def _migrate(_args: argparse.Namespace) -> int:
    from relore.store.migrations import describe, migrate

    engine = _engine()
    applied = migrate(engine)
    state = describe(engine)
    if applied:
        print(f"applied {applied} on {state['dialect']}")
    else:
        print(f"already current on {state['dialect']} (applied {state['applied']})")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    """Stage only. Deriving is local and free, so it is a separate step by design."""
    return _run_backfill(args, derive=False, graphql=False, authority=False)


def _backfill(args: argparse.Namespace) -> int:
    return _run_backfill(
        args,
        derive=True,
        graphql=not args.no_graphql,
        authority=not args.no_authority,
        refresh_details=args.refresh_details,
        include_open=not args.merged_only,
    )


def _run_backfill(
    args: argparse.Namespace,
    *,
    derive: bool,
    graphql: bool,
    authority: bool,
    refresh_details: bool = False,
    include_open: bool = True,
) -> int:
    from relore.ingest.backfill import backfill

    engine = _engine()
    only = tuple(args.only) if args.only else None
    with _client() as client:
        gql = _graphql_client() if graphql else None
        try:
            result = backfill(
                engine,
                client,
                args.repo,
                only=only,
                derive=derive,
                gql=gql,
                authority=authority,
                refresh_details=refresh_details,
                include_open=include_open,
            )
        finally:
            if gql is not None:
                gql.close()
    for outcome in result.passes:
        state = "skipped" if outcome.skipped else f"{outcome.staged} objects"
        print(f"{outcome.name:16} {state}")
    if result.authority is not None:
        a = result.authority
        print(
            f"{'authority':16} {a.authoritative}/{a.candidates} MEMBER authors have write "
            f"access ({a.resolved} resolved this pass, {a.rederived} threads re-derived)"
        )
    print(f"derived {result.derived} threads, {result.documents_written} document writes")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    from relore.ingest.sweep import sweep

    engine = _engine()
    with _client() as client:
        result = sweep(engine, client, args.repo)
    print(f"live      {result.live}")
    print(f"removed   {result.removed}")
    print(f"rederived {result.rederived} threads, {result.documents_written} writes")
    return 0


def _authority(args: argparse.Namespace) -> int:
    """Section 6.2. Network unless --offline, so it is its own verb rather than part of
    derive, which section 5.0 requires to stay local."""
    import datetime as dt

    from relore.ingest.authority import resolve_authority

    engine = _engine()
    refresh = dt.timedelta(0) if args.refresh else None
    if args.offline:
        result = resolve_authority(engine, args.repo, client=None, **_refresh_kw(refresh))
    else:
        with _client() as client:
            result = resolve_authority(engine, args.repo, client=client, **_refresh_kw(refresh))

    print(f"candidates  {result.candidates} MEMBER authors")
    print(f"resolved    {result.resolved} ({result.skipped_fresh} still fresh)")
    for source, count in sorted(result.sources.items()):
        print(f"  via {source:16} {count}")
    if not result.endpoint_available:
        print("  note: the permission endpoint was unavailable; merged_by carried the pass")
    print(f"authoritative {result.authoritative} of {result.candidates} MEMBER authors")
    if result.promoted:
        print(f"promoted    {', '.join(sorted(result.promoted))}")
    print(f"re-derived  {result.rederived} threads, {result.documents_written} document writes")
    return 0


def _refresh_kw(refresh: object) -> dict[str, Any]:
    return {} if refresh is None else {"refresh_after": refresh}


def _derive(args: argparse.Namespace) -> int:
    """Re-derive a repository, or the given numbers.

    **The numbers to derive are the ones with a staged thread, not every number staged.**
    A comment walk stages a comment under its thread number, and a thread can be deleted
    or transferred upstream between that walk and the thread walk -- so a number can hold
    comments and no issue. Collecting every staged number instead made a full re-derive
    die on the first of them with `not staged; fetch it first`, and because nothing about
    `derive` keeps a cursor, the retry died at the same number for ever: a Job that
    restarts and never progresses, which is the shape section 5.2 warns about. Measured on
    production: 20 such numbers out of 47,928, and one of them (`#1998`) is 40 threads
    into the list.

    `sweep` already treats this as expected and skips with a warning; this is the same
    decision, plus the count, because a silent skip is the pattern section 13.3 #10 exists
    to stop.
    """
    from relore.ingest.index_thread import THREAD_HEAD_TYPES, derive_thread
    from relore.store.schema import raw_objects

    engine = _engine()
    numbers, orphans = args.number, 0
    if not numbers:
        with engine.connect() as conn:
            staged = _staged_numbers(conn, args.repo, raw_objects)
            heads = _staged_numbers(conn, args.repo, raw_objects, types=THREAD_HEAD_TYPES)
        numbers = sorted(heads)
        orphans = len(staged - heads)
    if not numbers:
        print(f"nothing staged for {args.repo}")
        return 0
    wrote = signals = skipped = 0
    for number in numbers:
        with engine.begin() as conn:
            try:
                result = derive_thread(conn, args.repo, number)
            except LookupError:
                # Only reachable when a head is pruned while this runs; the pre-filter
                # above covers the ordinary case.
                skipped += 1
                continue
        wrote += result.wrote
        signals += result.signals.wrote
    # Two counters, because they answer different questions: a re-derive of unchanged
    # threads must write neither, and an extractor that improved writes only the second.
    print(
        f"re-derived {len(numbers) - skipped} threads, {wrote} document writes, "
        f"{signals} signal writes"
    )
    if orphans or skipped:
        print(
            f"skipped {orphans + skipped} numbers with staged comments but no staged "
            "issue or pull request (deleted or transferred upstream after the comment "
            "walk saw them; `relored sweep` prunes them)"
        )
    return 0


def _staged_numbers(
    conn: Any, repo: str, raw_objects: Any, *, types: tuple[str, ...] = ()
) -> set[int]:
    stmt = select(raw_objects.c.thread_number.distinct()).where(
        raw_objects.c.repo == repo, raw_objects.c.thread_number.isnot(None)
    )
    if types:
        stmt = stmt.where(raw_objects.c.object_type.in_(types))
    return {int(n) for (n,) in conn.execute(stmt)}


def _poll(args: argparse.Namespace) -> int:
    from relore.duration import parse_interval
    from relore.ingest.poll import poll_forever, poll_once

    engine = _engine()
    with _client() as client:
        if args.once:
            result = poll_once(engine, client, args.repo)
            print(
                f"{result.moved} moved, {result.indexed} indexed, "
                f"{result.documents_written} document writes"
            )
            return 0 if result.clean else 1
        poll_forever(engine, client, args.repo, parse_interval(args.interval))
    return 0


def _judge(args: argparse.Namespace) -> int:
    """Section 10. The labels file is what somebody clicked; the set is what a number is
    measured against. This is the join, and it is deliberately not automatic."""
    from pathlib import Path

    from relore.bench import dataset as ds
    from relore.bench.judge import fold, read_labels

    data = ds.load(Path(args.eval_set))
    if data.frozen:
        print(
            f"{args.eval_set} was frozen at {data.frozen_at}; fold labels into a new file",
            file=sys.stderr,
        )
        return 2
    if args.labels is None and not args.freeze:
        print("nothing to do: pass --labels to fold, --freeze to freeze, or both", file=sys.stderr)
        return 2
    rows = read_labels(Path(args.labels)) if args.labels else []
    folded, stats = fold(data, rows, judge=args.judge)

    print(f"labels     {stats.labels}")
    print(f"judged     {stats.judged} existing candidates")
    print(f"minted     {stats.minted} examples the UI asked that mining had not")
    if stats.dropped:
        # Not a zero: it means the corpus does not hold the answer, and averaging that in
        # would move every system's recall for a reason that is not about retrieval.
        print(f"dropped    {len(stats.dropped)} questions with no relevant thread")
        for line in stats.dropped[:5]:
            print(f"  {line}")
    if stats.ambiguous:
        print(f"ambiguous  {len(stats.ambiguous)} labels match more than one example")
        for line in stats.ambiguous[:5]:
            print(f"  {line}")
    if len(stats.backends) > 1:
        print(f"warning    labels span {sorted(stats.backends)}; section 10 refuses to mix them")

    if args.freeze:
        folded = folded.freeze()
    ds.save(folded, Path(args.eval_set))
    for kind, counts in folded.counts().items():
        print(f"  {kind:11} {counts['examples']:4} examples, {counts['judged']:4} judged")
    return 0


def _export_corpus(args: argparse.Namespace) -> int:
    """Section 10's baseline corpus, as it stood at the cutoff. Local and read-only.

    Selects with ``search.backends.base.corpus_scope``, the same predicate ``_filters``
    applies, so a baseline cannot end up reading a different corpus than the index.
    """
    from pathlib import Path

    from relore.bench.corpus import export

    before = _instant(args.before, "--before") if args.before else None
    stats = export(
        _engine(),
        Path(args.out),
        repos=tuple(args.repo),
        before=before,
        exclude=tuple(args.exclude_threads),
    )
    window = before.isoformat() if before else "no cutoff"
    print(f"{stats.documents} documents from {stats.threads} threads ({window})")
    if stats.exclude:
        print(f"withheld  {', '.join(f'#{n}' for n in stats.exclude)}")
    print(f"sha256    {stats.digest}")
    print(f"wrote     {stats.path}")
    if before is None:
        print(
            "\nno --before: this is the whole index, a valid control but not a pre-cutoff "
            "corpus. The header says so.",
            file=sys.stderr,
        )
    return 0


def _bench(args: argparse.Namespace) -> int:
    """Section 10. Two rules are enforced by the code rather than by the reader: one
    backend per report, and every baseline restricted to the corpus window."""
    from pathlib import Path

    from relore.bench import dataset as ds
    from relore.bench.run import (
        GitHubSearchSystem,
        GrepSystem,
        IndexSystem,
        Report,
        run_system,
        threads_by_path,
    )

    wanted = tuple(args.systems) if args.systems else ("index", "github")
    # Section 13. `github` and `grep` answer from outside this index and cannot honour a
    # cutoff, so asking for both is refused rather than reported with a baseline that saw
    # the future. Checked here because it is a question about the flags alone.
    before = _instant(args.before, "--before") if args.before else None
    exclude = tuple(args.exclude_threads)
    if (before is not None or exclude) and ({"github", "grep"} & set(wanted)):
        print(
            "--before/--exclude-thread cannot be honoured by the `github` or `grep` "
            "baselines: neither answers from this index, so neither can be held to its "
            "cutoff. Score them in a separate, untimed run.",
            file=sys.stderr,
        )
        return 2

    data = ds.load(Path(args.eval_set))
    if not data.frozen and not args.allow_unfrozen:
        print(
            f"{args.eval_set} is not frozen. Section 10 freezes the set before tuning, "
            "because a set that grows while the weights move is not a benchmark.\n"
            "  relored bench --allow-unfrozen ...   # to score it anyway",
            file=sys.stderr,
        )
        return 2
    scoreable = data.judged()
    if not scoreable.examples:
        print(f"{args.eval_set} holds no judged examples yet; label some first", file=sys.stderr)
        return 2

    repos = tuple(args.repo)
    windows = {repo: data.since(repo) for repo in repos}
    # One floor for the whole run: a baseline restricted more narrowly than the index was
    # sampled is not the same comparison, and neither is one restricted less.
    floors = {value for value in windows.values() if value}
    window = min(floors) if floors else None
    # One qualifier for the whole run. With two repositories sampled from different dates
    # the widest floor under-restricts the narrower one, so the run is not the comparison
    # section 10 asks for and says so rather than printing a number that looks like it is.
    same_window = len(floors) <= 1 and not (floors and any(v is None for v in windows.values()))

    engine = _engine()
    index = IndexSystem(engine, repos, before=before, exclude=exclude)
    report = Report(
        backend=index.backend.info().as_dict(),
        corpus=[c.__dict__ for c in data.corpus],
        cutoff=(
            {"before": before.isoformat() if before else None, "exclude": list(exclude)}
            if (before is not None or exclude)
            else None
        ),
    )

    for name in wanted:
        if name == "index":
            report.add(run_system(index, scoreable, window=window))
        elif name == "index+expand":
            report.add(
                run_system(
                    IndexSystem(engine, repos, expand=True, before=before, exclude=exclude),
                    scoreable,
                    window=window,
                )
            )
        elif name == "github":
            with _client() as client:
                run = run_system(
                    GitHubSearchSystem(client, repos, window), scoreable, window=window
                )
            run.window_applied = window is not None and same_window
            report.add(run)
        elif name == "grep":
            try:
                clones = _clones(args.clone, repos)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            with engine.connect() as conn:
                paths = {repo: threads_by_path(conn, repo) for repo in clones}
            run = run_system(GrepSystem(clones, paths), scoreable, window=window)
            # grep reads a working tree, which has no notion of a window at all.
            run.window_applied = False
            report.add(run)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    print(f"backend  {report.backend['name']} / {report.backend['ranking']}")
    print(f"window   {window or 'none declared (full history)'}")
    if report.cutoff is not None:
        print(f"cutoff   {report.cutoff['before'] or 'none'}")
        if exclude:
            print(f"withheld {', '.join(f'#{n}' for n in exclude)}")
        print(f"dropped  {index.post_filtered} hits post-filtered (signal-table leak)")
    counts = scoreable.counts()
    for kind in SLICE_NAMES:
        print(
            f"  {kind:11} {counts[kind]['judged']:3} judged "
            f"({counts[kind]['human']} human, {counts[kind]['model']} model)"
        )
    print()
    print(f"{'system':8} {'slice':11} {'n':>4} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'empty':>6}")
    for row in report.table():
        print(
            f"{row['system']:8} {row['slice']:11} {row['n']:4} {row['recall@5']:7.3f} "
            f"{row['recall@10']:7.3f} {row['mrr']:7.3f} {row['empty']:6}"
        )
        if "leak_free" in row:
            free = row["leak_free"]
            print(
                f"{'':8} {'  leak-free':11} {free['n']:4} {free['recall@5']:7.3f} "
                f"{free['recall@10']:7.3f} {free['mrr']:7.3f} {free['empty']:6}"
            )
    for run in report.runs:
        if not run.window_applied:
            print(
                f"\nnote: {run.system} could not be restricted to the window; "
                "section 10 says that is not a comparison"
            )
    if report.cutoff is not None:
        # Under the table: it is a caveat about the numbers above it, not a preamble.
        from relore.bench.run import RANKING_CAVEAT

        print(f"\nnote: {RANKING_CAVEAT}")
    return 0


def _mine(args: argparse.Namespace) -> int:
    """Section 10, and only the mechanical half of it.

    It appends *candidates* -- ``judged_by`` is null until somebody judges them, and the
    runner drops every unjudged row rather than counting it. Re-running is safe: an id
    already in the file is not added twice, which is what makes mining and labelling one
    loop instead of two files that drift.
    """
    from pathlib import Path

    from relore.bench import dataset as ds
    from relore.bench.mine import MINERS, corpus_of

    engine = _engine()
    out = Path(args.out)
    current = ds.load(out) if out.exists() else ds.Dataset()
    if current.frozen:
        print(f"{out} was frozen at {current.frozen_at}; mine into a new file", file=sys.stderr)
        return 2

    wanted = tuple(args.slices) if args.slices else SLICE_NAMES
    found: list[ds.Example] = []
    with engine.connect() as conn:
        corpus = corpus_of(conn, args.repo)
        for name in wanted:
            rows = MINERS[name](conn, args.repo, limit=args.limit)
            found.extend(rows)
            print(f"{name:11} {len(rows)} candidates")

    known = {ex.id for ex in current.examples}
    grown = current.add(found)
    merged = {c.repo + c.thread_type: c for c in (*current.corpus, *corpus)}
    grown = ds.Dataset(examples=grown.examples, corpus=tuple(merged.values()), frozen_at=None)
    ds.save(grown, out)
    print(f"{sum(1 for ex in found if ex.id not in known)} new, {len(grown.examples)} in {out}")
    for kind, counts in grown.counts().items():
        print(f"  {kind:11} {counts['examples']:4} examples, {counts['judged']:4} judged")
    return 0


def _clone(args: argparse.Namespace) -> int:
    """The clone the code verbs read. Run it where `serve` runs -- that process is the one
    that answers them (issue #7)."""
    from relore.code.clone import CloneUnavailable, WorkingClones

    try:
        info = WorkingClones(args.root).ensure(args.repo, url=args.url)
    except CloneUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{info.repo}  {info.path}\nHEAD {info.head[:12]}  committed {info.fetched_at}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Needs the ``[server]`` extra. Imported here so every other verb runs without it."""
    try:
        from relore.api.server import AsOf, serve
    except ImportError:
        raise SystemExit(
            "relored serve needs the server extra.\n"
            "  pip install 'relore[postgres]'   # or relore[server] for a SQLite laptop index"
        ) from None

    url = os.environ.get(DB_URL_ENV)
    if not url:
        raise SystemExit(f"set {DB_URL_ENV}")
    before = None
    if args.as_of:
        before = _instant(args.as_of, "--as-of")
    return serve(
        url,
        host=args.host,
        port=args.port,
        allow_sqlite=args.allow_sqlite,
        trust_network=args.trust_network,
        as_of=AsOf(before=before, exclude=tuple(args.exclude_threads)),
    )


def _instant(raw: str, flag: str) -> dt.datetime:
    """An ISO timestamp, and it must carry a zone.

    A naive cutoff is the bug that does not look like one: it parses, it filters, and it
    is wrong by however many hours the reader's machine is from UTC -- which on a
    leakage boundary is a document either side of it.
    """
    try:
        moment = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"{flag}: {raw!r} is not an ISO timestamp") from None
    if moment.tzinfo is None:
        raise SystemExit(f"{flag}: {raw!r} has no timezone; write it as ...Z or ...+00:00")
    return moment


def _sample(args: argparse.Namespace) -> int:
    """Section 10's corpus: bounded, and every thread complete. Not a small backfill --
    see relore/ingest/sample.py for why a `since` on the bulk passes is not the same
    thing."""
    from relore.ingest.sample import sample
    from relore.ingest.timestamps import parse_timestamp

    since = parse_timestamp(args.since if "T" in args.since else f"{args.since}T00:00:00Z")
    if since is None:
        print(f"could not read --since {args.since!r}; use YYYY-MM-DD", file=sys.stderr)
        return 2
    if args.merged and args.kind != "pr":
        print("--merged only applies to --kind pr", file=sys.stderr)
        return 2
    engine = _engine()
    with _client() as client:
        result = sample(
            engine,
            client,
            args.repo,
            since=since,
            thread_type=args.kind,
            merged_only=args.merged,
            limit=args.limit,
        )
    print(
        f"sampled {result.indexed}/{result.selected} {result.thread_type} threads "
        f"(of {result.seen} seen, {result.skipped} already current), "
        f"{result.documents_written} document writes, "
        f"{result.signals_written} signal writes"
    )
    if result.failed:
        print(f"failed: {result.failed}", file=sys.stderr)
    return 0 if result.clean else 1


def _status(args: argparse.Namespace) -> int:
    from relore.search import open_backend
    from relore.store.migrations import describe
    from relore.store.repository import index_summary

    engine = _engine()
    state = describe(engine)
    with engine.connect() as conn:
        summary = index_summary(conn)
    # Section 4.1: print the search backend and what it can answer, so a surprising result
    # set is diagnosable rather than mysterious.
    payload = {**state, **summary, "search_backend": open_backend(engine).info().as_dict()}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    backend = payload["search_backend"]
    print(
        f"backend   {payload['dialect']} · search {backend['ranking']} "
        f"· capabilities {', '.join(backend['capabilities']) or 'none'}"
    )
    print(f"schema    applied {payload['applied']}, pending {payload['pending']}")
    print(
        f"index     {payload['threads']} threads, {payload['documents']} documents, "
        f"{payload['raw_objects']} raw objects"
    )
    # Section 14.1: nothing searches the history, so this count is the only way to see the
    # capture is working. Zero after a long-running poll is a symptom, not a quiet corpus.
    print(f"history   {payload['superseded_documents']} superseded documents kept")
    print(payload["passes_note"])
    for row in payload["passes"]:
        note = f" ({row['note']})" if row["note"] else ""
        print(
            f"  {row['repo']} [{row['pass']}] high-water {row['high_water']} "
            f"last-ok {row['last_ok_at']}{note}"
        )
    # Loud, and above the pass list on purpose: a sampled corpus that reads as a full
    # history is how a recall number ends up compared with a baseline that saw more.
    for row in payload["samples"]:
        print(
            f"  SAMPLE  {row['repo']} {row['thread_type']}s from "
            f"{row['indexed_from'][:10]} only ({row['selector']}) -- NOT full history"
        )
    return 0


# -- wiring ----------------------------------------------------------------


def _clones(values: list[str] | None, repos: tuple[str, ...]) -> dict[str, Path]:
    """``--clone`` arguments as one checkout per repository.

    Section 10's grep baseline greps every clone and pools the matches (see
    :class:`~relore.bench.run.GrepSystem`), so it needs to know which repository a path
    belongs to. The bare form is allowed for a single repository because that is the
    common case and there is nothing to be ambiguous about; with two, an unlabelled path
    would silently attribute one repository's files to the other.
    """
    from pathlib import Path

    if not values:
        raise ValueError("the grep baseline needs --clone [owner/name=]<checkout>")
    out: dict[str, Path] = {}
    for value in values:
        repo, sep, path = value.partition("=")
        if not sep:
            if len(repos) > 1:
                raise ValueError(
                    f"--clone {value}: scoring {len(repos)} repositories, so each clone "
                    "needs its repository: --clone owner/name=<checkout>"
                )
            repo, path = repos[0], value
        if repo not in repos:
            raise ValueError(f"--clone {value}: {repo} is not among --repo {list(repos)}")
        out[repo] = Path(path)
    return out


def _engine() -> Engine:
    from relore.store.dialect import make_engine

    url = os.environ.get(DB_URL_ENV)
    if not url:
        raise SystemExit(
            f"set {DB_URL_ENV} (postgresql://... for production, "
            "sqlite:///relore.db to try it locally)"
        )
    return make_engine(url)


def _client() -> GitHubClient:
    # Imported inside the function rather than at module scope so `relored migrate`,
    # `derive` and `status` work with no token, and no httpx, in the picture.
    from relore.github.client import GitHubClient

    for env in TOKEN_ENVS:
        token = os.environ.get(env)
        if token:
            return GitHubClient(token)
    raise SystemExit(f"set {TOKEN_ENVS[0]} (needs issues:read + pull_requests:read)")


def _graphql_client() -> GraphQLClient:
    from relore.github.graphql import GraphQLClient

    for env in TOKEN_ENVS:
        token = os.environ.get(env)
        if token:
            return GraphQLClient(token)
    raise SystemExit(f"set {TOKEN_ENVS[0]}")


if __name__ == "__main__":
    sys.exit(main())
