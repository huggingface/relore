"""What the code lens costs, and whether it is still telling the truth (issue #83).

The client-side counterpart to ``page_cost.py``. No database, no network, no model: point it
at a checkout and it times ``relore``'s local verbs against the two baselines an agent
actually reaches for, and then checks the answer against the slower one.

Both halves are the point, and the second is why this exists rather than a shell loop with
``time``. Issue #83 made the lens 20-35x faster by not parsing files that cannot hold the
answer and by memoising the ones that do, and **either change can be made arbitrarily fast
by returning less**. So speed is reported beside a strict-subset check, and a run that gets
quicker while dropping a location fails loudly instead of looking like a win.

    python benchmarks/probes/code_lens.py --root ~/src/transformers
    python benchmarks/probes/code_lens.py --root . --symbol parse --skip-naive

Three things it reports.

**Speed**, per verb and per symbol class -- rare, mid, ubiquitous -- because one number for
``refs`` is the number for whichever symbol was picked. The prefilter is only as selective
as the name is rare, and a table with ``use_kernels`` in it and not ``config`` overstates the
change by an order of magnitude.

**Accuracy**, as a subset relation rather than a count. ``relore refs`` should return a
*strict subset* of what ``git grep`` finds in ``.py`` files, and every extra grep line should
be explicable. Three explanations are accepted and each is established independently of the
thing being measured:

* the line is in a file the walker never visits -- ``.circleci/``, ``build/``, anything under
  a dot directory or over ``MAX_FILE_BYTES``. The walked set is taken from
  :func:`relore.code.walk.source_files` itself, because a comparison over two different file
  sets is not a comparison;
* the match is inside a *longer identifier* -- ``set_use_kernels``, ``test_getter_use_kernels``
  -- so it is a different name that happens to contain these characters;
* the line is **prose**: a comment or a string. That is decided by :mod:`tokenize`, the
  stdlib's own Python lexer, and not by a regular expression. A per-line heuristic cannot see
  that a docstring's fourth line is inside a docstring, and the first draft of this script
  reported 867 such lines as losses; an independent tokenizer settles it exactly, and using a
  *different* implementation from the one under test is the point.

Whatever survives all three is printed in full and fails the run. So is a relore location
grep does not have, which would mean the lens invented a line.

**Identity**, which is issue #83's acceptance criterion in executable form: the same verb run
with and without ``RELORE_NO_CACHE`` must produce identical bytes. The speed columns are
worth nothing without it.

The naive ``grep -rn`` baseline is kept even though it is the slowest thing here, because it
is the one an agent reaches for by reflex. On ``transformers`` it was both slowest and
noisiest -- 486 of its 606 hits came from ``.claude/worktrees`` and 29 from ``build/lib``,
neither of which is source. ``--skip-naive`` drops it when you are iterating.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relore.code.cache import CACHE_DIR  # noqa: E402
from relore.code.walk import source_files  # noqa: E402

#: The three symbol classes, as measured on `huggingface/transformers`. A benchmark that
#: reported only the first would claim a 20x that no ubiquitous name ever sees.
DEFAULT_SYMBOLS = ("use_kernels", "compute_default_rope_parameters", "forward", "config")

#: What the baselines are allowed to look at. `git grep` with no pathspec searches every
#: tracked file including notebooks and JSON fixtures, which is not what `refs` claims to
#: answer -- the comparison has to be between two answers to the same question.
BASELINE_GLOBS = ("*.py", "*.md")

#: The one extension the subset check runs on. `refs` also answers for `.pyi` and for
#: whatever ctags claims, and grep is being asked about `.py`, so anything else would be
#: compared against a baseline that was never looking.
COMPARED = ".py"

IDENTIFIER = "[A-Za-z0-9_]"


@dataclass
class Run:
    label: str
    seconds: float
    stdout: str
    rows: int

    @property
    def size(self) -> int:
        return len(self.stdout.encode())


@dataclass
class Accuracy:
    symbol: str
    lens: set[tuple[str, int]] = field(default_factory=set)
    grep: set[tuple[str, int]] = field(default_factory=set)
    #: A grep line whose match is inside a longer identifier -- `set_use_kernels` for
    #: `use_kernels`. Not a reference to the symbol, and the lens is right to omit it.
    wider_identifier: list[str] = field(default_factory=list)
    #: A grep line where the symbol is written outside code: a docstring, a comment, an
    #: error string. Also not a reference.
    prose: list[str] = field(default_factory=list)
    #: A grep line in a file the walker never visits -- a dot directory, `build/`, a
    #: vendored tree. Not a disagreement: the two tools were asked about different files.
    outside_walk: list[str] = field(default_factory=list)
    #: A grep line that is none of the above, which the lens still did not report. Every one
    #: of these is a location the lens may be losing, and the reason this is not a count.
    unexplained: list[str] = field(default_factory=list)

    @property
    def subset(self) -> bool:
        return self.lens <= self.grep

    @property
    def invented(self) -> set[tuple[str, int]]:
        """Locations the lens reports and grep does not. Should always be empty: a parser
        cannot see a name in bytes that do not contain it."""
        return self.lens - self.grep


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    root = os.path.realpath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"no such tree: {root}", file=sys.stderr)
        return 2
    if shutil.which(args.relore) is None:
        print(f"{args.relore!r} is not on PATH", file=sys.stderr)
        return 2

    print(f"tree     {root}")
    print(f"head     {_git(root, 'rev-parse', '--short', 'HEAD') or '(not a git repository)'}")
    print(f"dirty    {len((_git(root, 'diff-files', '--name-only') or '').split())} file(s)")
    print(f"relore   {_run([args.relore, '--version'], root).stdout.strip()}")
    print()

    speed = _speed(args, root)
    accuracy = [_accuracy(args, root, symbol) for symbol in args.symbol]
    identity = _identity(args, root)

    _report(speed, accuracy, identity)
    if args.json:
        _write_json(args.json, root, speed, accuracy, identity)
    # A benchmark that cannot fail is a report. Losing a location or drifting between the
    # cached and uncached paths is a defect, so it is an exit code.
    broken = [a for a in accuracy if not a.subset or a.unexplained]
    return 1 if broken or not all(identity.values()) else 0


# -- speed -----------------------------------------------------------------


def _speed(args: argparse.Namespace, root: str) -> list[Run]:
    """Every verb and baseline, timed once each.

    Once, not best-of-three, and the reason is the subject: these calls are 0.1 s to 20 s and
    the differences being measured are multiples, not percentages. A repeat loop would buy
    noise reduction nobody needs and hide the number that actually varies -- a cold ``map``
    against a warm one.
    """
    runs: list[Run] = []
    for symbol in args.symbol:
        runs.append(_time(f"relore refs {symbol}", [args.relore, "refs", symbol], root))
        runs.append(
            _time(
                f"git grep -n {symbol}",
                ["git", "grep", "-n", symbol, "--", *BASELINE_GLOBS],
                root,
            )
        )
        if not args.skip_naive:
            includes = [f"--include={glob}" for glob in BASELINE_GLOBS]
            runs.append(_time(f"grep -rn {symbol}", ["grep", "-rn", symbol, *includes, "."], root))
    if args.map:
        _drop_cache(root)
        runs.append(_time("relore map (cold)", [args.relore, "map"], root))
        runs.append(_time("relore map (warm)", [args.relore, "map"], root))
        runs.append(
            _time(
                "relore map (no cache)",
                [args.relore, "map"],
                root,
                env={"RELORE_NO_CACHE": "1"},
            )
        )
    return runs


def _time(label: str, command: list[str], root: str, env: dict[str, str] | None = None) -> Run:
    started = time.perf_counter()
    done = _run(command, root, env=env)
    elapsed = time.perf_counter() - started
    rows = len([line for line in done.stdout.splitlines() if line.strip()])
    return Run(label=label, seconds=elapsed, stdout=done.stdout, rows=rows)


# -- accuracy --------------------------------------------------------------


def _accuracy(args: argparse.Namespace, root: str, symbol: str) -> Accuracy:
    """Where the lens and grep disagree, and whether each disagreement is defensible.

    The lens is *supposed* to return less than grep -- that is the product. What it may not
    do is return something grep cannot see, or drop a plain occurrence of the name in code
    that it did look at.
    """
    result = Accuracy(symbol=symbol)
    walked = _walked(root)
    result.lens = {
        location
        for location in _lens_locations(_run([args.relore, "refs", symbol], root).stdout)
        if location[0].endswith(COMPARED)
    }
    # A line is excused as a longer identifier only when it holds *no* standalone
    # occurrence. `self.config = config_kwargs` contains both, and excusing it on the
    # strength of `config_kwargs` would hide the bare `config` the lens did not report --
    # on `transformers` that bucket is 36,592 lines, so it can hide a great deal.
    standalone = re.compile(f"(?<!{IDENTIFIER}){re.escape(symbol)}(?!{IDENTIFIER})")
    grep = _run(["git", "grep", "-n", symbol, "--", f"*{COMPARED}"], root).stdout
    prose_lines: dict[str, set[int]] = {}
    for line in grep.splitlines():
        location, text = _grep_line(line)
        if location is None:
            continue
        path, number = location
        if path not in walked:
            result.outside_walk.append(line)
            continue
        result.grep.add(location)
        if location in result.lens:
            continue
        if standalone.search(text) is None:
            result.wider_identifier.append(line)
            continue
        if path not in prose_lines:
            prose_lines[path] = _prose_lines(os.path.join(root, path))
        if number in prose_lines[path]:
            result.prose.append(line)
        else:
            result.unexplained.append(line)
    return result


def _walked(root: str) -> set[str]:
    """Exactly the files ``relore`` would parse, asked of the walker rather than guessed.

    ``git grep`` searches every tracked file; the lens skips dot directories, ``build/``,
    vendored trees and anything over ``MAX_FILE_BYTES``. Counting a hit in ``.circleci/`` as
    a location the lens lost would be scoring it against a question it was never asked.
    """
    return {
        os.path.normpath(os.path.relpath(path, root))
        for path in source_files(root)
        if path.endswith(COMPARED)
    }


def _prose_lines(path: str) -> set[int]:
    """Every line of a Python file that lies inside a comment or a string literal.

    :mod:`tokenize` rather than a regular expression, and a *different* implementation from
    the tree-sitter grammar under test, so "the lens skipped this because it is prose" is
    established by something with no stake in the answer. A file the stdlib cannot tokenize
    contributes nothing: an unexplained line is the safe direction to be wrong in.
    """
    import tokenize

    covered: set[int] = set()
    try:
        with open(path, "rb") as handle:
            for token in tokenize.tokenize(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    covered.update(range(token.start[0], token.end[0] + 1))
    except (OSError, SyntaxError, tokenize.TokenError, UnicodeDecodeError, IndentationError):
        return set()
    return covered


def _lens_locations(stdout: str) -> set[tuple[str, int]]:
    """``./path/to/file.py:214  name`` -> ``("path/to/file.py", 214)``.

    The ``./`` comes from the verb being handed ``.`` as its root, and grep does not write
    it; normalising here rather than changing the verb keeps the comparison a property of
    this script.
    """
    out: set[tuple[str, int]] = set()
    for line in stdout.splitlines():
        if line.startswith("--") or not line.strip():
            continue
        path, _, rest = line.partition(":")
        number, _, _ = rest.partition(" ")
        if number.strip().isdigit():
            out.add((os.path.normpath(path), int(number)))
    return out


def _grep_line(line: str) -> tuple[tuple[str, int] | None, str]:
    path, _, rest = line.partition(":")
    number, _, text = rest.partition(":")
    if not number.isdigit():
        return None, line
    return (os.path.normpath(path), int(number)), text


# -- identity --------------------------------------------------------------


def _identity(args: argparse.Namespace, root: str) -> dict[str, bool]:
    """Issue #83's acceptance criterion: cached and uncached must be the same bytes.

    ``map`` is run against a warm cache and then with ``RELORE_NO_CACHE``; ``refs`` is run
    both ways too, although it consults no cache today, because "``refs`` does not use the
    cache" is a decision someone may reverse and this is where it would be caught.
    """
    checks: dict[str, bool] = {}
    verbs = [(f"refs {symbol}", [args.relore, "refs", symbol]) for symbol in args.symbol]
    if args.map:
        verbs.append(("map", [args.relore, "map"]))
    for label, command in verbs:
        cached = _run(command, root)
        plain = _run(command, root, env={"RELORE_NO_CACHE": "1"})
        # Both failing the same way -- a missing grammar, a bad path -- gives two empty
        # stdouts, which compare equal. That is the check passing by asserting nothing, so
        # the exit codes are part of it.
        checks[label] = (
            cached.returncode == 0 and plain.returncode == 0 and cached.stdout == plain.stdout
        )
    return checks


# -- reporting -------------------------------------------------------------


def _report(speed: list[Run], accuracy: list[Accuracy], identity: dict[str, bool]) -> None:
    print("speed")
    print(f"  {'verb':<40}{'wall':>9}{'output':>11}{'rows':>8}")
    for run in speed:
        print(f"  {run.label:<40}{run.seconds:>8.2f}s{_bytes(run.size):>11}{run.rows:>8}")
    if any(run.label.startswith("relore map") for run in speed):
        # Said rather than left blank. `map` is the slowest verb here and the absence of a
        # baseline beside it reads as an omission; it is not one, and the reason is the
        # point of the verb: grep answers "where is this string", and a ranked repo map is
        # not that question asked faster. The cold/warm pair is its own baseline.
        print("  (map has no grep baseline: grep answers no equivalent question)")
    print()

    print("accuracy -- relore's .py locations against git grep's")
    for item in accuracy:
        mark = "subset" if item.subset else "NOT A SUBSET"
        print(
            f"  {item.symbol:<34}{len(item.lens):>6} of {len(item.grep):<6} {mark}"
            f"   ({len(item.wider_identifier)} wider identifier,"
            f" {len(item.prose)} prose,"
            f" {len(item.unexplained)} unexplained;"
            f" {len(item.outside_walk)} outside the walk)"
        )
        for line in item.unexplained[:10]:
            print(f"      unexplained  {line}")
        for path, number in sorted(item.invented)[:10]:
            print(f"      INVENTED     {path}:{number} -- grep cannot see this")
    print()

    print("identity -- cached output against RELORE_NO_CACHE")
    for label, same in identity.items():
        print(f"  {label:<40}{'identical' if same else 'DIFFERS'}")


def _write_json(path: str, root: str, speed, accuracy, identity) -> None:
    payload = {
        "root": root,
        "speed": [
            {"label": r.label, "seconds": round(r.seconds, 3), "bytes": r.size, "rows": r.rows}
            for r in speed
        ],
        "accuracy": [
            {
                "symbol": a.symbol,
                "lens": len(a.lens),
                "grep": len(a.grep),
                "subset": a.subset,
                "wider_identifier": len(a.wider_identifier),
                "prose": len(a.prose),
                "outside_walk": len(a.outside_walk),
                "unexplained": a.unexplained,
                "invented": sorted(f"{p}:{n}" for p, n in a.invented),
            }
            for a in accuracy
        ],
        "identity": identity,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {path}")


def _bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


# -- plumbing --------------------------------------------------------------


def _run(
    command: list[str], root: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a command in ``root`` with ``RELORE_NO_CACHE`` under this script's control.

    Inherited from the shell it would silently make every "cached" row an uncached one, and
    the identity check would then compare the uncached path against itself and pass.
    """
    environment = dict(os.environ)
    environment.pop("RELORE_NO_CACHE", None)
    environment.update(env or {})
    return subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False, env=environment
    )


def _git(root: str, *args: str) -> str | None:
    done = _run(["git", *args], root)
    return done.stdout.strip() if done.returncode == 0 else None


def _drop_cache(root: str) -> None:
    """Remove the cache the verb would actually read.

    It lives at the *repository* root, not at ``--root``, so a run pointed at a subdirectory
    would otherwise leave it in place and time a "cold" map against a warm cache -- the one
    row this probe exists to show, quietly reading as noise.
    """
    anchor = _git(root, "rev-parse", "--show-toplevel") or root
    shutil.rmtree(os.path.join(anchor, CACHE_DIR), ignore_errors=True)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the checkout to measure")
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help=f"repeatable; defaults to {', '.join(DEFAULT_SYMBOLS)} (rare -> ubiquitous)",
    )
    parser.add_argument("--relore", default="relore", help="the client to time")
    parser.add_argument(
        "--no-map", dest="map", action="store_false", help="skip the map verb (the slow one)"
    )
    parser.add_argument("--skip-naive", action="store_true", help="drop the `grep -rn` baseline")
    parser.add_argument("--json", help="also write the rows here, so two runs can be diffed")
    args = parser.parse_args(argv)
    args.symbol = args.symbol or list(DEFAULT_SYMBOLS)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
