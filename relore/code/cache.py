"""A content-keyed parse cache for the verbs that walk a whole tree (issue #83, lever 2).

``walk.py`` used to say *nothing is cached*, and gave two reasons. Issue #83 re-opened both
rather than working around them, and this module is the answer to each:

* **"fast enough"** was measured and is not. On ``huggingface/transformers`` @ ``d9890f6``
  -- 4,883 claimed files -- ``relore map`` is 18.2 s and repeats in full on the next
  identical call. Reading every file is 0.59 s of that; *parsing* them is 16.4 s. The parse
  is the whole cost, so the thing to skip is the parse -- and only the parse.
* **"a staleness hazard aimed at exactly the property that justifies doing this locally"**
  is real, and it is the reason for every design choice below. A cache keyed on *content*
  cannot describe a tree other than the one on disk. It is the key, not the caching, that
  was the hazard.

The invariant this module exists to keep: **a cached answer is byte-identical to the
uncached one.** Four things enforce it.

1. **The key is the content, and the content is read.** Every file is read and hashed on
   every call; an entry is fresh when its key equals the sha of the bytes on disk right now.

   Issue #83 proposed a faster key -- ``git ls-files -s`` returns the blob sha of 4,885
   tracked files in 0.00 s and ``git diff-files`` names the dirty ones in 0.01 s, so a clean
   file need never be opened -- and it was built, measured at 0.6 s against 1.0 s, and taken
   out again. **git's "clean" is a stat comparison, not a content one.** It is the mtime
   mechanism the rest of this list exists to avoid, wearing git's name, and it has
   documented opt-outs: ``core.checkStat minimal``, ``core.trustctime false``, a filesystem
   monitor, and above all ``git update-index --assume-unchanged``, which tells git to stop
   looking at a file altogether. Measured: with ``--assume-unchanged`` set, a cached ``map``
   reported the previous definition while the file on disk held a new one, and
   ``RELORE_NO_CACHE`` disagreed with it. That is exactly the failure this module is not
   allowed to have, so the 0.4 s is spent and ``map`` settles at 1.0 s rather than 0.6 s.

   Reading is affordable precisely because it was never the cost: 0.59 s to read all 4,883
   files against 16.4 s to parse them. Hashing what we read adds ~0.2 s. git is still used,
   for one thing that cannot be wrong -- deciding *where* the cache file lives.
2. **The provider is part of the key.** A file parsed by ``python`` and a file parsed by the
   ctags fallback do not produce the same definitions, and which one reads a file depends on
   what is installed -- ``pip uninstall tree-sitter-python`` silently changes every answer.
   An entry built by a different provider is a miss, not a hit.
3. **Reference positions are never stored.** An entry holds per-name reference *counts* and
   nothing else about a reference -- 3.1 M rows on ``transformers`` is a real index, and it
   is the one thing here that could put a wrong line number on a page. ``map`` ranks on the
   counts and prints no reference, so it needs no more. Definition positions *are* stored,
   because they are ``map``'s answer rather than a hint toward it and there is no cheaper
   form of "every definition in the tree" than every definition in the tree; they are safe
   for the same reason the counts are, which is that the key is the exact content they were
   read from.
4. **``defs`` is exempt, and so is the daemon.** ``relore defs`` is 0.09 s on the largest
   file in ``transformers`` and has nothing to gain, and the same
   :func:`~relore.code.defs.definitions` runs inside ``relored`` against a *historical blob*
   at a document's commit -- where a working-tree cache is not merely stale but actively
   wrong. Nothing in this module is reachable from :mod:`relore.code.defs`.

**Only ``map`` uses this at all**, which is less than issue #83 proposed. Writing had to land
here regardless: an entry carries definitions *and* reference counts, so ``refs`` -- which
parses references only, on a prefiltered handful of files -- would do strictly more work to
write one than to answer the question. *Reading* was built for ``refs`` too, benchmarked, and
removed; once rule 1 took away the ability to skip a read, all a cache can save anybody is a
parse, and skipping the parse for a single named symbol is precisely what
:func:`~relore.code.walk.needle` already does. ``copies`` and ``symbol`` carry that same
prefilter, so the same reasoning covers them.

The module boundary holds: this imports ``json``, ``gzip``, ``hashlib``, ``os`` and
``subprocess``, and nothing from :mod:`relore.store`, :mod:`relore.github` or
:mod:`relore.api`. It is also not a provider -- section 9's rule 4 forbids a *provider* from
walking a filesystem or shelling out to git, which is why the git call lives here beside the
walker rather than behind the provider interface.

Set ``RELORE_NO_CACHE`` to turn the whole thing off; every verb then behaves exactly as it
did before this module existed, which is what makes the equivalence testable.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from relore.code.api import REFS
from relore.code.walk import read

log = logging.getLogger(__name__)

#: Bumped whenever :class:`Entry` changes shape. An older file is discarded rather than
#: migrated: it is a cache, so the cost of throwing it away is one slow call.
FORMAT = 1

DISABLE_ENV = "RELORE_NO_CACHE"

CACHE_DIR = ".relore"
CACHE_FILE = "code.json.gz"

#: Written into the cache directory the first time it is created. ``relore map`` runs in
#: whatever repository the caller is sitting in, and leaving 4 MB of untracked noise in
#: someone else's ``git status`` is not a thing a read-only tool should do. A directory that
#: ignores itself needs no cooperation from the repository it lands in.
SELF_IGNORE = "*\n"

#: How long git is given to answer. A pathological repository (a cold NFS mount, a huge
#: index) must make the cache unavailable, never make the verb hang.
GIT_TIMEOUT = 30


@dataclass(frozen=True)
class Entry:
    """One file's parse, keyed by the content it was read from.

    Exactly what :func:`~relore.code.repomap.repo_map` consumes and not one field more.
    ``definitions`` holds the four it prints; ``calls`` holds the occurrences whose kind is
    not ``definition``, counted, because the ranking is a count.

    An earlier draft also carried the full set of names each file mentions, as the "which
    files" hint issue #83 designed for ``refs``. It is gone: ``refs`` measured faster
    without it, so the field was 7 MB of cache nobody read.
    """

    provider: str
    sha: str
    definitions: tuple[tuple[str, str, str, int], ...]
    calls: tuple[tuple[str, int], ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "p": self.provider,
            "s": self.sha,
            "d": [list(d) for d in self.definitions],
            "c": dict(self.calls),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Entry:
        return cls(
            provider=raw["p"],
            sha=raw["s"],
            definitions=tuple((d[0], d[1], d[2], int(d[3])) for d in raw["d"]),
            calls=tuple((name, int(count)) for name, count in raw["c"].items()),
        )


def blob_sha(source: bytes) -> str:
    """Git's own object name for these bytes.

    Nothing requires it to be git's function -- any content hash would do -- but using the
    one every developer already has a tool for means a suspicious entry can be checked by
    hand with ``git hash-object``, which is worth more than the alternative's novelty.
    """
    header = b"blob %d\0" % len(source)
    return hashlib.sha1(header + source).hexdigest()  # noqa: S324 -- git's name, not a MAC


def open_session(root: str) -> Session:
    """The cache for the tree containing ``root``, or a disabled one.

    Never raises. A tree with no git, no cache file, a corrupt cache file or a read-only
    checkout all produce a session that answers "miss" to everything, which is precisely the
    behaviour of the code before this module existed.
    """
    if os.environ.get(DISABLE_ENV):
        return Session(root, anchor=None, entries={})
    anchor = _anchor(root)
    if anchor is None:
        return Session(root, anchor=None, entries={})
    return Session(root, anchor=anchor, entries=_load(anchor))


class Session:
    """A cache open for the duration of one verb.

    Use it as a context manager: the file is written once, on exit, and only if something
    was added. Writing per entry would turn a 1 s call into thousands of rewrites of a
    4 MB file.
    """

    def __init__(self, root: str, anchor: str | None, entries: dict[str, Entry]):
        self.root = root
        self.anchor = anchor
        self._entries = entries
        self._added = False
        #: Paths this session actually visited. Pruning is keyed on it so a cache cannot
        #: grow without bound across renames -- but only a walk that visited the whole tree
        #: is allowed to prune, which is why :meth:`close` takes the decision.
        self._seen: set[str] = set()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def lookup(self, path: str, provider_name: str) -> tuple[Entry | None, bytes | None]:
        """The fresh entry for ``path``, or the bytes to parse it from. Never both.

        ``(None, None)`` means the file is unreadable or over
        :data:`~relore.code.walk.MAX_FILE_BYTES`, which every caller already treats as "skip
        it" -- the cache does not change what is skipped, only what is parsed.

        The file is always read, and the key is always the sha of what was read. There is no
        arrangement under which this returns an entry for bytes it has not seen: that is what
        makes the answer describe the tree the caller is standing on, and the module
        docstring's rule 1 records what it cost and which faster design was rejected for it.

        ``provider_name`` is the other half of the key and is checked here rather than by the
        caller, because forgetting it is not a slow answer but a wrong one: which provider
        reads a file depends on what is installed, and an entry that survived a ``pip
        uninstall tree-sitter-python`` would describe a parse nothing on this machine would
        produce.
        """
        if self.anchor is None:
            return None, read(path)
        key = self._key(path)
        self._seen.add(key)
        source = read(path)
        if source is None:
            return None, None
        entry = self._entries.get(key)
        if entry is not None and entry.provider == provider_name and entry.sha == blob_sha(source):
            return entry, None
        return None, source

    def store(self, path: str, provider: Any, source: bytes) -> Entry:
        """Parse ``source`` into an entry and remember it.

        This is the only place an entry is built, so the two consumers cannot disagree about
        what one contains. It returns the entry whether or not the cache is enabled -- with
        ``RELORE_NO_CACHE`` set, callers still get their answer from here, they just get it
        again next time.
        """
        definitions = tuple(
            (d.qualname, d.name, d.kind, d.start_line) for d in provider.defs(path, source)
        )
        calls: dict[str, int] = {}
        if REFS in provider.capabilities:
            for reference in provider.refs(path, source):
                if reference.kind != "definition":
                    calls[reference.name] = calls.get(reference.name, 0) + 1
        entry = Entry(
            provider=provider.name,
            sha=blob_sha(source),
            definitions=definitions,
            calls=tuple(sorted(calls.items())),
        )
        if self.anchor is not None:
            key = self._key(path)
            self._seen.add(key)
            self._entries[key] = entry
            self._added = True
        return entry

    def close(self) -> None:
        """Write the cache back, if this session changed it.

        Entries this session did not visit are dropped **only** when the walk covered the
        whole anchor, or the cache would grow by one dead entry per rename for ever. The
        condition is the reason the session remembers its root: ``relore map`` walks the
        repository and may prune, ``relore map src/transformers`` walks a subtree and must
        not -- one subdirectory run would otherwise throw away the rest of the tree's work.

        Deleting a file is a change with nothing added, so "did this session parse anything"
        is the wrong question to gate the write on: a tree that only ever loses files would
        keep every entry it ever had. The pruning is computed first and counts as a change.

        Nothing raised from here reaches the caller. The answer was printed before the write
        was attempted, so a read-only checkout, a full disk or a race on the temporary file
        can cost the next call's speed and must not cost this one's exit code.
        """
        if self.anchor is None:
            return
        if os.path.realpath(self.root) == self.anchor:
            kept = {k: v for k, v in self._entries.items() if k in self._seen}
            if len(kept) != len(self._entries):
                self._entries = kept
                self._added = True
        if not self._added:
            return
        try:
            _save(self.anchor, self._entries)
        except Exception as exc:  # noqa: BLE001 -- a cache write is never worth a traceback
            log.debug("could not write the code cache at %s (%s)", self.anchor, exc)
        self._added = False

    def _key(self, path: str) -> str:
        """This file's path relative to the anchor, which is what git's listing is keyed on.

        ``realpath`` on both sides, because the anchor came from ``rev-parse --show-toplevel``
        and is already resolved: on macOS a walk rooted at ``/tmp`` against an anchor of
        ``/private/tmp`` would otherwise produce ``../../tmp/m.py`` and match nothing git
        said, leaving a cache that writes on every run and hits on none.
        """
        assert self.anchor is not None
        return os.path.relpath(os.path.realpath(path), self.anchor)


def _anchor(root: str) -> str | None:
    """Where the cache lives: the repository root, else the walked root.

    Anchoring at the repository rather than at ``root`` is what lets ``relore map`` and
    ``relore map src/transformers`` share one cache instead of scattering a ``.relore/`` into
    every subdirectory anyone ever pointed the verb at.
    """
    top = _git(root, "rev-parse", "--show-toplevel")
    if top:
        return os.path.realpath(top.strip())
    if os.path.isdir(root):
        return os.path.realpath(root)
    parent = os.path.dirname(os.path.realpath(root))
    return parent or None


def _git(root: str, *args: str) -> str | None:
    """A git plumbing call, or ``None`` for any reason at all.

    Used for one question -- where the repository root is -- so every failure mode (no git on
    PATH, not a repository, a timeout) has the same harmless consequence: the cache lands at
    the walked root instead of the repository root. Nothing about *freshness* is asked of
    git, deliberately; see rule 1.
    """
    try:
        done = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", "replace")


def _path(anchor: str) -> str:
    return os.path.join(anchor, CACHE_DIR, CACHE_FILE)


def _load(anchor: str) -> dict[str, Entry]:
    try:
        with gzip.open(_path(anchor), "rb") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, EOFError):
        return {}
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        return {}
    try:
        return {path: Entry.from_json(raw) for path, raw in payload["entries"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        # A truncated or hand-edited cache is a cache miss, not a traceback. It is the one
        # file in this design a caller is invited to delete.
        log.debug("discarding an unreadable code cache at %s", _path(anchor))
        return {}


def _save(anchor: str, entries: dict[str, Entry]) -> None:
    """Write atomically, and never fail the verb.

    ``os.replace`` because two ``relore map`` runs in one tree is an ordinary thing to do and
    a half-written cache that still parses as JSON is the one failure this design cannot
    detect. A read-only checkout, a full disk or a race on the temporary file all end the
    same way: the answer was already printed, so the only thing lost is the next call's
    speed.
    """
    directory = os.path.join(anchor, CACHE_DIR)
    payload = {
        "format": FORMAT,
        "entries": {path: entry.as_json() for path, entry in entries.items()},
    }
    temporary = _path(anchor) + f".{os.getpid()}.tmp"
    try:
        os.makedirs(directory, exist_ok=True)
        ignore = os.path.join(directory, ".gitignore")
        if not os.path.exists(ignore):
            with open(ignore, "w", encoding="utf-8") as handle:
                handle.write(SELF_IGNORE)
        with gzip.open(temporary, "wb", compresslevel=1) as handle:
            handle.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        os.replace(temporary, _path(anchor))
    except OSError as exc:
        log.debug("could not write the code cache at %s (%s)", directory, exc)
        with contextlib.suppress(OSError):
            os.unlink(temporary)
