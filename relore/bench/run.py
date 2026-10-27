"""Section 10's runner: score the evaluation set against `relore` and both baselines.

Three systems, because two of them are the argument:

* ``index`` -- `relore` itself, through the same backend the API serves from.
* ``index+expand`` -- the same backend with section 6's query expansion on. Scored beside
  ``index`` rather than replacing it, so expansion is a measured delta and not a claim.
* ``github`` -- ``/search/issues``, scored identically.
* ``grep`` -- the agent's own repo-wide grep, which is the harder control and the one
  that actually competes. An example grep also answers is not evidence for an index.

Two rules from section 10 are enforced here rather than left to whoever reads the table:

**Never compare across backends.** ``ts_rank_cd`` and ``bm25`` are different engines, so
a report holds exactly one `relore` backend and refuses a run from another.

**Restrict every baseline to the corpus window.** Section 5.5's sample indexes a bounded
window; scoring `relore` against a GitHub search over eight years measures the sample, not
the retrieval. The window comes off the dataset header and is applied to the GitHub query
as ``updated:>=``. A run whose window could not be applied says so in the report instead
of reporting a number.

The `grep` baseline needs an operationalization, because grep returns *files* and the
question is about *threads*. It gets the one an agent would actually use: grep the terms
over a checkout, then take the threads that touched the matching paths, ranked by how many
of them each thread touched. When grep comes back empty the example scores zero, which is
the outcome use-cases section 7d says is the whole point -- so ``grep_empty`` is reported
next to the recall rather than buried in it.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Connection, Engine, select

from relore.bench.dataset import Dataset, Example
from relore.search import MAX_HITS, SearchQuery, open_backend, search_best
from relore.store import schema as s

#: Section 10 asks for Recall@5, Recall@10 and MRR. Ten is also ``MAX_HITS`` -- the cap is
#: the contract, so a system that could return more is not measured as if it did.
CUTOFFS = (5, 10)

#: GitHub's search API is 30 requests a minute for an authenticated caller, and the
#: secondary limit punishes bursts harder than the documented one.
GITHUB_SEARCH_PAUSE = 2.1

#: ``/search/issues`` rejects a longer ``q`` with a 422, and a normalized error message is
#: routinely longer than this on its own (section 3's constraints reach the baseline too).
MAX_GITHUB_QUERY_CHARS = 256


ThreadRef = tuple[str, int]


def error_literal(error: str, *, minimum: int = 12) -> str:
    """The longest run of a normalized error that is still literal text.

    Section 5.3 replaces numbers, addresses and paths with placeholders, which is what
    makes the term match across runs -- and what makes it match *nothing* outside this
    index. A baseline handed ``<n>`` is not being asked the same question badly, it is
    being asked an impossible one, so it gets the fragment a person would have pasted.
    """
    runs = [part.strip(" :,.;()[]{}'\"") for part in re.split(r"<[a-z_]+>", error)]
    best = max(runs, key=len) if runs else ""
    return best if len(best) >= minimum else ""


@dataclass(frozen=True)
class Result:
    """What one system returned for one example, in order."""

    example_id: str
    kind: str
    ranked: tuple[ThreadRef, ...]
    relevant: tuple[ThreadRef, ...]
    leaks: bool = False
    note: str = ""

    def hit_rank(self) -> int | None:
        for position, ref in enumerate(self.ranked[: max(CUTOFFS)], start=1):
            if ref in self.relevant:
                return position
        return None

    def recall_at(self, k: int) -> float:
        if not self.relevant:
            return 0.0
        found = len(set(self.ranked[:k]) & set(self.relevant))
        return found / len(self.relevant)

    def reciprocal_rank(self) -> float:
        rank = self.hit_rank()
        return 0.0 if rank is None else 1.0 / rank


@dataclass(frozen=True)
class Score:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    empty: int = 0

    @classmethod
    def over(cls, results: Sequence[Result]) -> Score:
        if not results:
            return cls()
        return cls(
            n=len(results),
            recall={k: _mean(r.recall_at(k) for r in results) for k in CUTOFFS},
            mrr=_mean(r.reciprocal_rank() for r in results),
            empty=sum(1 for r in results if not r.ranked),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            **{f"recall@{k}": round(v, 3) for k, v in self.recall.items()},
            "mrr": round(self.mrr, 3),
            "empty": self.empty,
        }


class System(Protocol):
    name: str

    def search(self, example: Example) -> tuple[tuple[ThreadRef, ...], str]:
        """Ranked thread references, best first, and a note for the report."""


@dataclass
class Run:
    system: str
    results: list[Result] = field(default_factory=list)
    backend: dict[str, Any] | None = None
    window: str | None = None
    window_applied: bool = True


class BackendMismatch(RuntimeError):
    """Section 10: a recall figure measured on ``bm25`` describes a different engine."""


#: Printed and carried in the JSON of every report produced under a cutoff, so it travels
#: with the number rather than with a reader's memory.
RANKING_CAVEAT = (
    "ordering is still inflated: the thread_* signal tables carry no timestamp, so "
    "`_overlap` scored rows created after the cutoff. Filtering is corrected by the "
    "post-filter; ranking is not, until those tables carry a first_seen_at"
)


@dataclass
class Report:
    backend: dict[str, Any]
    corpus: tuple[dict[str, Any], ...] = ()
    runs: list[Run] = field(default_factory=list)
    #: ``{"before": …, "exclude": [...]}`` when this is a temporal run, else ``None``.
    cutoff: dict[str, Any] | None = None

    def add(self, run: Run) -> None:
        if run.backend is not None and run.backend != self.backend:
            raise BackendMismatch(
                f"{run.system} ran on {run.backend.get('ranking')}, this report holds "
                f"{self.backend.get('ranking')}; section 10 refuses to merge them"
            )
        self.runs.append(run)

    def table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            by_kind: dict[str, list[Result]] = {}
            for result in run.results:
                by_kind.setdefault(result.kind, []).append(result)
            for kind, results in sorted(by_kind.items()):
                row = {"system": run.system, "slice": kind, **Score.over(results).as_dict()}
                if kind == "rationale":
                    # The leak declaration (see mine.py): a human finding is itself a
                    # retrievable document in the answer's thread, so every lexical system
                    # can hit it without retrieving anything. Pooled with the bot subset
                    # the number describes neither.
                    clean = [r for r in results if not r.leaks]
                    if clean:
                        row["leak_free"] = Score.over(clean).as_dict()
                rows.append(row)
        return rows

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.backend,
            "corpus": list(self.corpus),
            "windows": {run.system: run.window for run in self.runs},
            "window_applied": {run.system: run.window_applied for run in self.runs},
            "rows": self.table(),
        }
        if self.cutoff is not None:
            out["cutoff"] = self.cutoff
            out["caveats"] = [RANKING_CAVEAT]
        return out


# -- the systems ------------------------------------------------------------


class IndexSystem:
    """`relore`, through the backend the API serves from -- not a reimplementation.

    ``expand`` selects section 6's query expansion, and it is a *separate system* rather
    than a flag folded into this one: expansion has to earn its place against the same
    examples the unexpanded index was measured on, and a report that quietly changed what
    ``index`` means would make every earlier number incomparable.

    ``before`` makes it a temporal run (section 13): the cutoff goes on every query and the
    page then goes through :class:`~relore.bench.corpus.SignalPostFilter`. Unpinned, both
    are the identity, so a timed and an untimed run come off the same code path.
    """

    def __init__(
        self,
        engine: Engine,
        repos: Sequence[str],
        *,
        kind_aware: bool = True,
        expand: bool = False,
        before: dt.datetime | None = None,
        exclude: Sequence[int] = (),
    ) -> None:
        self.backend = open_backend(engine)
        self.repos = tuple(repos)
        self.kind_aware = kind_aware
        self.expand = expand
        self.before = before
        self.exclude = tuple(exclude)
        self.engine = engine
        #: How many hits the post-filter took off this run's pages -- reported rather than
        #: inferred from a recall that moved. Zero means the leak did not touch the number.
        self.post_filtered = 0
        self.name = "index+expand" if expand else "index"

    def window(self, example: Example) -> tuple[dt.datetime | None, tuple[int, ...]]:
        """This example's cutoff and withheld threads, narrowed against the run's.

        An example may narrow the run's window and can never widen it -- the same rule, and
        the same argument, as the ``AsOf`` daemon pin: a bound the run was started with is
        not something the data it reads may lift. So the cutoff is the earlier of the two
        and the exclusions are the union.
        """
        mine = example.window()
        before = min(filter(None, (self.before, mine)), default=None)
        return before, tuple(sorted(set(self.exclude) | set(example.exclude)))

    def search(self, example: Example) -> tuple[tuple[ThreadRef, ...], str]:
        from relore.bench.corpus import SignalPostFilter

        before, exclude = self.window(example)
        query = SearchQuery(
            repos=self.repos,
            text=example.query,
            kind=example.kind if self.kind_aware else None,
            files=example.files,
            symbols=example.symbols,
            errors=example.errors,
            tests=example.tests,
            before=before,
            exclude=exclude,
            limit=MAX_HITS,
        )
        # `search_best`, not `search_expanded`: the system scored here has to be the system
        # the API serves, or the number describes something nobody runs. The widening it
        # adds is reachable only from an empty page, and neither frozen set has one (both
        # report `empty 0`), so it cannot move either score -- which is the point. What it
        # would catch is a future set that *does* contain an empty row.
        hits, _widened = search_best(self.backend, query, expand=self.expand)
        # The post-filter is pinned per example, not per run: it re-derives each hit
        # thread's signals as of *this* cutoff, so a run-level one would size the leak
        # against the wrong instant for every example that carries its own.
        kept = SignalPostFilter(self.engine, before).keep(hits, query) if before else hits
        self.post_filtered += len(hits) - len(kept)
        return _dedupe((hit.repo, hit.number) for hit in kept), ""


class GitHubSearchSystem:
    """``/search/issues``, restricted to the same repositories and the same window."""

    name = "github"

    def __init__(
        self, client, repos: Sequence[str], since: str | None, *, pause: float = GITHUB_SEARCH_PAUSE
    ) -> None:
        self.client = client
        self.repos = tuple(repos)
        self.since = since
        self.pause = pause

    def query_string(self, example: Example) -> str:
        scope = [f"repo:{repo}" for repo in self.repos]
        if self.since:
            scope.append(f"updated:>={self.since}")
        terms = [example.query] if example.query else []
        terms += list(example.symbols)
        # A caller with a traceback pastes the error; with a node id, the node id. Both
        # are literals to GitHub, so quote them rather than letting them tokenize.
        terms += [f'"{value}"' for value in filter(None, map(error_literal, example.errors))]
        terms += [f'"{value}"' for value in example.tests]
        # A path is a quoted *term*, never `path:`. That qualifier belongs to code search;
        # `/search/issues` accepts it, returns zero for everything, and reports no error --
        # measured, and it silently zeroed all 51 rationale examples on the first run.
        terms += [f'"{path}"' for path in example.files[:1]]
        return _fit(terms, scope, MAX_GITHUB_QUERY_CHARS)

    def search(self, example: Example) -> tuple[tuple[ThreadRef, ...], str]:
        query = self.query_string(example)
        # A scope with nothing in it matches the whole repository, and the top ten of that
        # would score as if the baseline had answered. Refuse rather than send it.
        if not any(part for part in query.split() if not part.startswith(("repo:", "updated:"))):
            return (), "no expressible query"
        payload = self.client.get_json(
            "https://api.github.com/search/issues",
            params={"q": query, "per_page": max(CUTOFFS), "sort": "best-match"},
        )
        if self.pause:
            time.sleep(self.pause)
        refs = []
        for item in payload.get("items", []):
            url = item.get("repository_url", "")
            repo = "/".join(url.rsplit("/", 2)[-2:]) if url else self.repos[0]
            refs.append((repo, int(item["number"])))
        return _dedupe(refs), query


class GrepSystem:
    """The agent's own repo-wide grep, mapped from files onto threads.

    Section 10 calls this the harder control. It is also the one whose *empty* result is
    the finding, so an example it cannot answer is scored zero rather than skipped.

    **One clone per indexed repository, and every one is grepped.** A corpus can span
    repositories -- section 10's rationale slice needs the ones where a bot is actually
    argued with -- and an agent sitting in front of two checkouts greps both. Scoping the
    grep to the repository holding the answer would be reading the ground truth, so the
    matches are pooled and a path is keyed by its repository throughout.
    """

    name = "grep"

    def __init__(
        self,
        clones: Mapping[str, Path],
        threads_by_path: Mapping[str, Mapping[str, Sequence[ThreadRef]]],
        *,
        timeout: float = 30.0,
    ) -> None:
        self.clones = dict(clones)
        self.threads_by_path = {repo: dict(paths) for repo, paths in threads_by_path.items()}
        self.timeout = timeout

    #: An agent greps three or four things before it reads, not thirty.
    MAX_TERMS = 3

    def terms(self, example: Example) -> list[str]:
        """The most specific terms first, because that is the order an agent greps in.

        use-cases section 7d has the real traces: ``grep original_sizes src/transformers``,
        ``grep 'class EdgeTamModel|def forward'`` -- identifiers and literals, never prose.
        Taking the query's words first and truncating put ``name``, ``object`` and
        ``module`` in front of the error literal that carries the question, which is the
        baseline being unable to express the query rather than being unable to answer it
        (section 10.2). Measured: on the failure slice the discarded literal was the whole
        term set for 42 of 48 examples.
        """
        out = [literal for literal in map(error_literal, example.errors) if literal]
        out += list(example.tests)
        out += list(example.symbols)
        out += [t for t in example.query.split() if len(t) > 3]
        seen: list[str] = []
        for term in out:
            if term not in seen:
                seen.append(term)
        return seen[: self.MAX_TERMS]

    def search(self, example: Example) -> tuple[tuple[ThreadRef, ...], str]:
        terms = self.terms(example)
        if not terms:
            return (), "no expressible query"
        paths: dict[tuple[str, str], int] = {}
        for repo, clone in self.clones.items():
            for term in terms:
                for path in self._grep(clone, term):
                    paths[(repo, path)] = paths.get((repo, path), 0) + 1
        if not paths:
            return (), "grep found nothing"
        weighted: dict[ThreadRef, int] = {}
        for (repo, path), weight in paths.items():
            for ref in self.threads_by_path.get(repo, {}).get(path, ()):
                weighted[ref] = weighted.get(ref, 0) + weight
        ranked = sorted(weighted, key=lambda ref: (-weighted[ref], ref))
        return tuple(ranked[: max(CUTOFFS)]), f"{len(paths)} files matched"

    def _grep(self, clone: Path, term: str) -> list[str]:
        try:
            done = subprocess.run(
                ["git", "grep", "-l", "-F", "-I", "--", term],
                cwd=clone,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [line for line in done.stdout.splitlines() if line]


def threads_by_path(conn: Connection, repo: str) -> dict[str, list[ThreadRef]]:
    """``thread_files`` inverted: the grep baseline's only bridge from a path to a thread."""
    rows = conn.execute(
        select(s.thread_files.c.path, s.threads.c.github_number)
        .select_from(s.thread_files.join(s.threads, s.threads.c.id == s.thread_files.c.thread_id))
        .where(s.threads.c.repo == repo)
    ).all()
    out: dict[str, list[ThreadRef]] = {}
    for path, number in rows:
        out.setdefault(path, []).append((repo, int(number)))
    return out


# -- running ----------------------------------------------------------------


def run_system(system: System, dataset: Dataset, *, window: str | None = None) -> Run:
    run = Run(system=system.name, window=window)
    if isinstance(system, IndexSystem):
        run.backend = system.backend.info().as_dict()
    for example in dataset.judged().examples:
        ranked, note = system.search(example)
        run.results.append(
            Result(
                example_id=example.id,
                kind=example.kind,
                ranked=ranked,
                relevant=tuple((r.repo, r.number) for r in example.relevant),
                leaks=bool(example.source.get("leaks")),
                note=note,
            )
        )
    return run


def _fit(terms: list[str], scope: list[str], budget: int) -> str:
    """Every scope qualifier, then as many terms as the budget leaves.

    Dropping a scope qualifier would silently widen the search past the corpus window,
    which is the one thing section 10 will not have; dropping a term only makes the query
    less specific, which is a fair thing to happen to an over-long paste.
    """
    query = " ".join(scope)
    for term in terms:
        candidate = f"{term} {query}" if query else term
        if len(candidate) <= budget:
            query = candidate
    return query.strip()


def _dedupe(refs: Iterable[ThreadRef]) -> tuple[ThreadRef, ...]:
    """One thread is one result. The API caps at three documents per thread (section 6),
    and counting the same thread twice would make a chatty thread look like two answers."""
    seen: list[ThreadRef] = []
    for ref in refs:
        if ref not in seen:
            seen.append(ref)
    return tuple(seen)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
