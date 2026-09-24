"""Section 5.3's extraction pass: text and payloads in, signal rows out.

**Precision over exhaustiveness**, because the two mistakes do not cost the same: a wrong
symbol costs ranking quality on every future query, a missed one costs one query. Every
pattern here is anchored on a structural marker -- a stack frame, a diff line, a code
span, a node id -- never on "looks like an identifier".

Everything is a pure function of already-**redacted** text, and never of ``raw_objects``:
a secret that section 11.3 kept out of ``documents.body_text`` must not reappear in
``thread_symbols.symbol``. The one input that does not arrive redacted is a review
comment's ``diff_hunk``, stored verbatim in the document's metadata, so this module
redacts it on the way in.

The input is the **unchunked** redacted text of each source object, not the document rows.
A pasted traceback is one paragraph with no blank line in it, so past section 5.4's
``MAX_CHUNK_CHARS`` the chunker has no natural boundary and cuts on length, through the
middle of a line. An error line, a node id or a path can therefore land half in one chunk
and half in the next and be extracted from neither -- while the person pasting that
traceback into ``--error`` has all of it. The documents are the chunks; the signals are
the thread's.

The signal tables are separate from ``documents`` so an exact match can be weighted
independently of prose (section 4). Nothing here scores anything -- the section 6 weights
are the next piece of milestone 3 and are gated on section 10's evaluation set.

**Two declared ceilings**, so a later reader measures them rather than rediscovering them:

* An **absolute** path cannot be mapped to a repository-relative one without the working
  clone, which is milestone 4 (``path_aliases``, section 1). Only the ``site-packages``
  case is recoverable here, so the rest are skipped rather than guessed at.
* URLs are removed before prose is read, which also discards the path inside a GitHub
  permalink. That is a real signal given up for a large class of false positives
  (``github.com/owner/repo/blob/main/src/x.py`` is otherwise a "path"); recovering it
  wants a pattern for GitHub's own URL shapes, not a looser path rule.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from relore.ingest.normalize import redact
from relore.ingest.timestamps import first_seen_at

# -- where a path came from (huggingface/relore#17) ------------------------

#: The per-PR pass's diff. Definitive, and the only source counted against
#: ``threads.metadata.changed_files``.
CHANGED = "changed"
#: The file an inline review comment is attached to. GitHub asserts this, not us -- so it
#: is a fact about the thread, but it is not the diff and it has no ``change_type``.
FROM_COMMENT = "comment"
#: A path-shaped token read out of prose. May be a bare basename, may be a file this
#: thread never touched, and may be both at once: ``huggingface/transformers#39847``
#: carries `modeling_rope_utils.py` from a comment *and* `src/transformers/…` from the
#: diff, and `config.json`, which that pull request does not touch.
MENTIONED = "mentioned"

#: Strongest first. A path with several provenances is stored as the strongest one.
FILE_SOURCES = (CHANGED, FROM_COMMENT, MENTIONED)


# -- what a signal row looks like ------------------------------------------


@dataclass
class Signals:
    """One thread's signal rows, keyed the way ``reconcile_signals`` writes them.

    Deduplicated on the way in: the same path named in prose and in the changed-file list
    is one row, and the row that knows its ``change_type`` wins. A duplicate would not be
    wrong so much as double-counted, and section 6's overlap terms count rows.
    """

    files: list[dict[str, Any]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    commits: list[dict[str, Any]] = field(default_factory=list)

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        """The rows keyed by ``store.repository.SIGNAL_TABLES``' names. The store layer
        takes a mapping rather than this class, so it does not import the extractor."""
        return {
            "files": self.files,
            "symbols": self.symbols,
            "errors": self.errors,
            "tests": self.tests,
            "commits": self.commits,
        }

    @property
    def total(self) -> int:
        return (
            len(self.files)
            + len(self.symbols)
            + len(self.errors)
            + len(self.tests)
            + len(self.commits)
        )


#: Extensions that make a bare filename (no directory) a plausible repository path.
#: A token with a ``/`` needs no such list; one without it is otherwise indistinguishable
#: from an abbreviation -- ``e.g.`` would be a file named ``e`` with extension ``g``.
SOURCE_EXTENSIONS = frozenset(
    # fmt: off
    (
        "py",
        "pyi",
        "pyx",
        "c",
        "h",
        "cc",
        "cpp",
        "cxx",
        "cu",
        "cuh",
        "js",
        "jsx",
        "ts",
        "tsx",
        "go",
        "rs",
        "java",
        "kt",
        "kts",
        "rb",
        "php",
        "swift",
        "m",
        "mm",
        "scala",
        "sh",
        "bash",
        "zsh",
        "sql",
        "yaml",
        "yml",
        "toml",
        "cfg",
        "ini",
        "json",
        "md",
        "rst",
        "txt",
        "lock",
        "mk",
        "make",
        "cmake",
        "dockerfile",
        "gradle",
        "proto",
        "graphql",
        "css",
        "scss",
        "html",
        "xml",
        "csv",
        "tsv",
        "ipynb",
    )
    # fmt: on
)

#: Words that pass every identifier test and mean nothing as symbols.
_SYMBOL_STOPWORDS = frozenset(
    {"true", "false", "none", "null", "nan", "self", "cls", "todo", "fixme", "n/a"}
)

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_CODE_SPAN = re.compile(r"`{1,3}([^`\n]+)`{1,3}")
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

#: Section 5.3's error shape, plus the runner gutters it appears behind. ``E   `` is
#: pytest's; ``- `` is the ``FAILED tests/x.py::test_y - ValueError: boom`` summary line.
_ERROR = re.compile(
    r"(?:^|(?<=\s))(?P<type>[A-Za-z_][\w.]*(?:Error|Exception))\s*:[ \t]*(?P<message>\S[^\n]*)",
    re.MULTILINE,
)

#: ``File "path/x.py", line 412, in forward`` -- a CPython frame. The most reliable
#: symbol *and* path source in the corpus, because both come from the interpreter.
_PY_FRAME = re.compile(r'File "(?P<path>[^"\n]+)", line \d+, in (?P<symbol>[A-Za-z_]\w*)')
#: pytest's own frame rendering: ``tests/test_x.py:412: in test_y``.
_PYTEST_FRAME = re.compile(
    r"^(?P<path>[\w./-]+\.py):\d+: in (?P<symbol>[A-Za-z_]\w*)", re.MULTILINE
)

#: A definition on a diff line. The leading ``[+-]`` is optional so context lines and
#: plain fenced code count too.
_DEFINITION = re.compile(
    r"^[+\- ]?\s*(?:async\s+)?(?P<kind>def|class|function)\s+(?P<symbol>[A-Za-z_]\w*)",
    re.MULTILINE,
)

#: A pytest node id. The file part carries the path, so this feeds ``thread_files`` too.
#: A parameter list may contain spaces (``test_x[a, b]``); the rest of the id may not,
#: so the space is admitted only inside the brackets. Allowing it anywhere makes the
#: id swallow the next word of the sentence -- and a node id is matched exactly.
_NODE_ID = re.compile(r"(?P<file>[\w./-]+\.py)::(?P<rest>[\w:.-]*\w(?:\[[^\]\n]*\])?)")
#: A unittest/dotted id: ``tests.models.test_gemma.GemmaTest.test_generate``.
_DOTTED_TEST = re.compile(r"\b(?:[A-Za-z_]\w*\.){2,}(?P<function>test_\w+)\b")

#: A leading ``/`` is part of the token so that :func:`_repo_path` gets to decide what
#: to do with an absolute path. Excluding it here instead would make an absolute path
#: match from its second segment, which is a *different* path and a plausible one.
_PATH_TOKEN = re.compile(
    r"(?<![\w.-])(/?(?:[\w.+-]+/)*[\w.+-]+\.[A-Za-z][A-Za-z0-9]{0,4})(?![\w/])"
)

_PARAMS = re.compile(r"\[[^\]]*\]")
_HEX_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUMBER_RUN = re.compile(r"(?:[-+]?\d[\d.eE+-]*\s*,\s*){2,}[-+]?\d[\d.eE+-]*")
_DIGITS = re.compile(r"\b\d+(?:\.\d+)?\b")
_ABS_PATH = re.compile(r"(?<![\w])/(?:[\w.+-]+/)+[\w.+-]+")
_WHITESPACE = re.compile(r"\s+")
#: GitHub autolinks a bare host inside an error message, so the stored text picks up
#: markdown a terminal never printed -- and the caller's paste then does not contain it.
#: Nested, because autolinking an already-linked host is exactly what produces
#: ``[[host](url)](url)``.
_MD_LINK = re.compile(r"\[([^\][]*)\]\([^()\s]*\)")

#: Long enough to stay discriminating, short enough that a tensor dump cannot become the
#: row. Section 6's error term matches on containment, not on equality.
MAX_MESSAGE_CHARS = 300


# -- the pass --------------------------------------------------------------


def extract_signals(
    documents: Iterable[Mapping[str, Any]],
    detail: Mapping[str, Any] | None = None,
    merged_at: dt.datetime | None = None,
) -> Signals:
    """Every signal row for one thread, each dated with when the thread first carried it.

    ``documents`` are the rows about to be written, so extraction sees exactly the text
    retrieval will. ``detail`` is the per-PR GraphQL node (section 3), the only source of
    a merged PR's changed-file list and commits.

    ``merged_at`` dates those two. Nothing in ``detail`` carries a date of its own --
    ``files.nodes`` is ``path additions deletions changeType`` and ``commits.nodes`` is
    ``oid messageHeadline messageBody`` (``github/graphql.py``) -- and asking GitHub for
    one would make a re-derive a network operation, which is the property that makes
    ``relored derive`` rebuildable offline. So the date used is the one instant the list
    is known to have been final, which is the same rule
    :func:`relore.bench.corpus._merged_diff_paths` already applies to the same rows: a
    pull request still open at the cutoff contributes none of its diff, because the prefix
    it had then cannot be reconstructed. ``None`` for an unmerged thread, which fails
    closed.

    Each document is scanned into its own dictionaries and merged, rather than scanned
    into shared ones. The merge rules are the ones the shared dictionaries had -- first
    non-null wins, everywhere -- so the rows are unchanged; what the split buys is knowing
    *which* document produced a value, without which the date is a guess.
    """
    files: dict[str, str | None] = {}
    #: Where each path came from, strongest wins (see :data:`FILE_SOURCES`). Kept beside
    #: `files` rather than folded into it because `change_type` is null for two of the
    #: three sources, which is what let a filename somebody typed in a comment sit
    #: indistinguishably among a pull request's real diff (huggingface/relore#17).
    sources: dict[str, str] = {}
    symbols: dict[tuple[str, str | None], str | None] = {}
    errors: dict[tuple[str | None, str], None] = {}
    tests: dict[tuple[str, str | None], None] = {}
    #: ``(table, row key) -> the earliest document that produced it``. Keyed by table too
    #: because a path and a test id are both strings and both live in here.
    dates: dict[tuple[str, Any], list[dt.datetime | None]] = {}

    def note(table: str, keys: Iterable[Any], when: dt.datetime | None) -> None:
        for key in keys:
            dates.setdefault((table, key), []).append(when)

    for document in documents:
        when = document.get("existed_at")
        seen_files: dict[str, str | None] = {}
        seen_symbols: dict[tuple[str, str | None], str | None] = {}
        seen_errors: dict[tuple[str | None, str], None] = {}
        seen_tests: dict[tuple[str, str | None], None] = {}
        _scan_text(
            _readable(document),
            files=seen_files,
            symbols=seen_symbols,
            errors=seen_errors,
            tests=seen_tests,
        )
        # An inline review comment's path is definitive: GitHub attached it to that file.
        definitive = _definitive_path(str((document.get("metadata") or {}).get("path") or ""))
        if definitive:
            seen_files.setdefault(definitive, None)

        for path, change_type in seen_files.items():
            files.setdefault(path, change_type)
        for key, kind in seen_symbols.items():
            if symbols.get(key) is None:
                symbols[key] = kind
        errors.update({key: None for key in seen_errors if key not in errors})
        tests.update({key: None for key in seen_tests if key not in tests})

        note("files", seen_files, when)
        note("symbols", seen_symbols, when)
        note("errors", seen_errors, when)
        note("tests", seen_tests, when)

        # Everything `_scan_text` adds is read out of prose: a traceback frame, a node id,
        # a path-shaped token. Inference, and sometimes a bare basename.
        for path in files:
            sources.setdefault(path, MENTIONED)
        if definitive:
            sources[definitive] = FROM_COMMENT

    for path, change_type in _changed_files(detail):
        files[path] = change_type
        sources[path] = CHANGED
        note("files", (path,), merged_at)

    commits = _commits(detail)
    note("commits", (row["sha"] for row in commits), merged_at)

    def dated(table: str, key: Any) -> dt.datetime | None:
        return first_seen_at(dates.get((table, key), ()))

    return Signals(
        files=[
            {
                "path": path,
                "change_type": change,
                "source": sources.get(path, MENTIONED),
                "first_seen_at": dated("files", path),
            }
            for path, change in sorted(files.items())
        ],
        symbols=[
            {
                "symbol": symbol,
                "path": path,
                "symbol_type": kind,
                "first_seen_at": dated("symbols", (symbol, path)),
            }
            for (symbol, path), kind in sorted(symbols.items(), key=_symbol_sort)
        ],
        errors=[
            {
                "exception_type": exception,
                "message_norm": message,
                "first_seen_at": dated("errors", (exception, message)),
            }
            for exception, message in sorted(errors, key=lambda k: (k[0] or "", k[1]))
        ],
        tests=[
            {
                "test_id": test_id,
                "test_function": function,
                "first_seen_at": dated("tests", (test_id, function)),
            }
            for test_id, function in sorted(tests, key=lambda k: (k[0], k[1] or ""))
        ],
        commits=[{**row, "first_seen_at": dated("commits", row["sha"])} for row in commits],
    )


def _symbol_sort(item: tuple[tuple[str, str | None], str | None]) -> tuple[str, str]:
    (symbol, path), _kind = item
    return symbol, path or ""


def _readable(document: Mapping[str, Any]) -> str:
    """The text of one document, plus its diff hunk -- redacted.

    ``body_markdown`` rather than ``body_text``: code spans and fences are where the
    discriminating tokens live and ``normalize`` rewrites markdown links, which is right
    for full text and wrong for a path. The hunk is redacted here because it is stored
    verbatim (it is metadata, not body), and a signal table is not a way around
    section 11.3.
    """
    parts = [str(document.get("body_markdown") or "")]
    hunk = (document.get("metadata") or {}).get("diff_hunk")
    if isinstance(hunk, str) and hunk:
        clean, _found = redact(hunk)
        parts.append(clean)
    return "\n".join(parts)


def _scan_text(
    text: str,
    *,
    files: dict[str, str | None],
    symbols: dict[tuple[str, str | None], str | None],
    errors: dict[tuple[str | None, str], None],
    tests: dict[tuple[str, str | None], None],
) -> None:
    for match in _ERROR.finditer(text):
        message = normalize_error_message(match.group("message"))
        if message:
            errors.setdefault((match.group("type").rsplit(".", 1)[-1], message), None)

    for match in _NODE_ID.finditer(text):
        _add_test(tests, f"{match.group('file')}::{match.group('rest')}")
        _add_path(files, match.group("file"))
    for match in _DOTTED_TEST.finditer(text):
        _add_test(tests, match.group(0))

    for pattern in (_PY_FRAME, _PYTEST_FRAME):
        for match in pattern.finditer(text):
            path = _repo_path(match.group("path"))
            _remember(symbols, match.group("symbol"), path, "frame")
            if path:
                files.setdefault(path, None)

    for match in _DEFINITION.finditer(text):
        kind = match.group("kind")
        _remember(symbols, match.group("symbol"), None, "class" if kind == "class" else "function")

    prose = _URL.sub(" ", text)
    for match in _PATH_TOKEN.finditer(prose):
        _add_path(files, match.group(1))
    for span in _spans(text):
        symbol = span.removesuffix("()")
        if is_symbol(symbol):
            _remember(symbols, symbol, None, "call" if span.endswith("()") else None)


def _spans(text: str) -> list[str]:
    """Inline code spans, and the first line of nothing else.

    Fenced blocks are read by the definition and frame patterns above, not as spans: a
    whole block is not one identifier.
    """
    return [span.strip() for span in _CODE_SPAN.findall(_FENCE.sub(" ", text))]


def is_symbol(candidate: str) -> bool:
    """Whether a code span is an identifier rather than a word or a fragment.

    The shape test alone admits ``true`` and ``main``; requiring a *marker* of code --
    CamelCase, an underscore, or a dotted path -- is what keeps prose out. A single
    lowercase word in backticks is usually a value, a flag or emphasis.
    """
    if len(candidate) < 3 or candidate.lower() in _SYMBOL_STOPWORDS:
        return False
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", candidate):
        return False
    # `webapp.py` passes every test above -- it has a dot. It is a *file*, and it is
    # already indexed as one; section 6 weights the file and symbol overlaps separately,
    # so counting a path in both double-counts the same evidence.
    if candidate.rpartition(".")[2].lower() in SOURCE_EXTENSIONS:
        return False
    return bool(re.search(r"[a-z][A-Z]", candidate) or "_" in candidate or "." in candidate)


def _remember(
    symbols: dict[tuple[str, str | None], str | None],
    symbol: str,
    path: str | None,
    kind: str | None,
) -> None:
    """Keep the most specific reading of a symbol: a typed one beats an untyped one."""
    key = (symbol, path)
    if symbols.get(key) is None:
        symbols[key] = kind


def _add_test(tests: dict[tuple[str, str | None], None], node_id: str) -> None:
    """Both the written id and its parametrization-stripped form (section 5.3).

    Storing only the stripped form loses an exact match on ``test_x[cuda]``; storing only
    the written one loses the unification the plan asks for. Both are one row each and
    the filter is an ``IN``, so the caller chooses which question they are asking.
    """
    function = _PARAMS.sub("", node_id.rsplit("::", 1)[-1].rsplit(".", 1)[-1])
    # `tests/t.py::SomeTests` is a valid node id that names a *class*. Its last component
    # is not a function, and recording it as one would put a class name in the column a
    # query filters functions by.
    function = function if function.startswith("test") else None
    tests.setdefault((node_id, function), None)
    stripped = _PARAMS.sub("", node_id)
    if stripped != node_id:
        tests.setdefault((stripped, function), None)


def _add_path(files: dict[str, str | None], candidate: str) -> None:
    path = _repo_path(candidate)
    if path:
        files.setdefault(path, None)


def _definitive_path(candidate: str) -> str | None:
    """A path GitHub itself attached to something: the changed-file list, or an inline
    review comment's own file.

    The prose heuristic below must **not** be applied to one of these. It exists to decide
    whether an ambiguous token is a path at all, and it answers no for ``Dockerfile``,
    ``Makefile`` and ``.gitignore`` -- which have no extension to test and are perfectly
    good repository paths. Filtering a definitive source through a guess loses data that
    was never in question.
    """
    path = candidate.strip().removeprefix("./")
    if not path or path.startswith(("/", "~", "\\")) or "://" in path:
        return None
    if ".." in path.split("/"):
        return None
    return path


def _repo_path(candidate: str) -> str | None:
    """A repository-relative path *inferred from text*, or None when the token is not one.

    An absolute path is a machine's, not a repository's; the only one recoverable without
    milestone 4's working clone is an installed package, where everything after
    ``site-packages/`` is the path inside that distribution.
    """
    path = candidate.strip().strip("\"'`,;:()[]<>")
    if "site-packages/" in path:
        path = path.split("site-packages/", 1)[1]
    if not path or path.startswith(("/", "~", "\\")) or "://" in path:
        return None
    path = path.removeprefix("./")
    if path.startswith(("a/", "b/")) and path.count("/") > 1:  # a diff header's prefixes
        path = path.split("/", 1)[1]
    if ".." in path.split("/") or not re.fullmatch(r"[\w.+/-]+", path):
        return None
    head, _dot, extension = path.rpartition(".")
    if not head or not extension.isalnum() or not extension[0].isalpha():
        return None
    if "/" not in path and extension.lower() not in SOURCE_EXTENSIONS:
        return None
    return path


def normalize_error_message(message: str) -> str:
    """Section 5.3's normalized error text -- and the form a query must be put in.

    **Both sides have to agree.** The stored form is what section 6's error term matches
    against by containment, so a caller pasting a raw traceback matches nothing unless
    the same normalization runs on the way in. :class:`relore.search.queries.SearchQuery`
    calls this on ``errors`` for exactly that reason, and section 6's query expansion will
    call it again on the error line it pulls out of a traceback.

    Everything removed is something that differs between two occurrences of the *same*
    failure: addresses, absolute paths, counts, and the tensor dumps that make a torch
    error a kilobyte long.
    """
    text = message
    for _ in range(3):  # bounded: each pass unwraps one level of nesting
        unwrapped = _MD_LINK.sub(lambda m: m.group(1), text)
        if unwrapped == text:
            break
        text = unwrapped
    text = _HEX_ADDRESS.sub("0x?", text)
    text = _NUMBER_RUN.sub("<values>", text)
    text = _ABS_PATH.sub(lambda m: m.group(0).rsplit("/", 1)[-1], text)
    text = _DIGITS.sub("<n>", text)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    return text[:MAX_MESSAGE_CHARS].strip()


def error_query_form(text: str) -> str:
    """Put a pasted error -- one line or a whole traceback -- into the stored form.

    A caller pastes what their terminal printed. If that contains a recognizable error
    line, the message on it is what was indexed; otherwise normalize the paste itself, so
    a partial phrase still matches by containment rather than silently matching nothing.
    """
    match = None
    for match in _ERROR.finditer(text):  # noqa: B007 -- the last frame is the raised one
        pass
    if match:
        return normalize_error_message(match.group("message"))
    # No recognizable error line. If this is multi-line it is a traceback whose tail is
    # the failure -- possibly cut off by the terminal, possibly an exception type section
    # 5.3 does not recognize. Either way, normalizing the *whole* block produces a string
    # longer than any stored message, so it could not match by containment even in
    # principle; the last line at least can. A single line is a fragment and is used
    # whole.
    lines = [line for line in text.splitlines() if line.strip()]
    return normalize_error_message(lines[-1] if len(lines) > 1 else text)


def _changed_files(detail: Mapping[str, Any] | None) -> list[tuple[str, str | None]]:
    """The merged PR's changed-file list (section 3's per-PR GraphQL pass).

    Definitive where prose is inference, which is why it is applied last and overwrites.
    A PR whose file list was truncated at 100 (``graphql_truncated``) contributes what it
    has; the alternative is contributing nothing.
    """
    nodes = ((detail or {}).get("files") or {}).get("nodes") or []
    out = []
    for node in nodes:
        path = _definitive_path(str(node.get("path") or ""))
        if path:
            out.append((path, node.get("changeType")))
    return out


def _commits(detail: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The PR's own commits. A sha in prose is deliberately **not** extracted: a bare
    7-hex token is indistinguishable from any other hash, and a wrong commit is the
    expensive kind of wrong."""
    nodes = ((detail or {}).get("commits") or {}).get("nodes") or []
    out: dict[str, str | None] = {}
    for node in nodes:
        commit = node.get("commit") or {}
        sha = str(commit.get("oid") or "")
        if sha:
            out.setdefault(sha, commit.get("messageHeadline"))
    return [{"sha": sha, "message": message} for sha, message in sorted(out.items())]
