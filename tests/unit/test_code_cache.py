"""The parse cache, and the one property it is allowed to have (issue #83).

``walk.py`` used to refuse a cache outright, on the grounds that it is *a staleness hazard
aimed at exactly the property that justifies doing this locally* -- that the answer
describes the tree the caller is actually on, dirty files included. The cache exists now,
and that sentence is still the specification: every test here is a way for a cached answer
to differ from an uncached one, asserted not to.

So the assertions are deliberately never about hit rates. A cache that quietly never hits is
slow; a cache that hits when it should not is a wrong answer with a line number on it, and
only the second kind is worth a test.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess

import pytest
from toy_language import ToyProvider

from relore.code import cache, registry
from relore.code.repomap import repo_map

BEFORE = b"def before():\n    pass\n"
AFTER = b"def after():\n    pass\n"


@pytest.fixture(autouse=True)
def only_python(monkeypatch):
    """Provider discovery reset around every test, and no inherited ``RELORE_NO_CACHE``.

    The environment matters more here than anywhere else in the suite: with the variable set
    in a developer's shell, every assertion below would pass by measuring the uncached path
    twice and proving nothing.
    """
    monkeypatch.delenv(cache.DISABLE_ENV, raising=False)
    registry.reset()
    yield
    monkeypatch.undo()
    registry.reset()


@pytest.fixture
def python():
    try:
        from relore.code.providers.python import PythonProvider
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tree-sitter-python: {exc}")
    return PythonProvider()


@pytest.fixture
def repo(tmp_path):
    """A real git repository, because the fast path *is* git's index.

    Two things only a real repository exercises: ``ls-files -s`` handing back a blob sha
    without anyone reading the file, and ``diff-files`` withdrawing it the moment the
    worktree diverges. A fake would be asserting that the fake works.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.org")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def names(root) -> set[str]:
    return {entry.qualname for entry in repo_map(str(root), limit=50).entries}


# -- the invariant: cached == uncached -------------------------------------


def test_a_warm_run_agrees_with_a_cold_one(python, monkeypatch, repo) -> None:
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(b"def busy():\n    pass\n\ndef quiet():\n    pass\n")
    (repo / "user.py").write_bytes(b"def go():\n    busy()\n    busy()\n")
    _commit(repo)

    cold = repo_map(str(repo), limit=50)
    assert (repo / cache.CACHE_DIR / cache.CACHE_FILE).exists()
    warm = repo_map(str(repo), limit=50)

    assert cold == warm
    assert cold.entries[0].matches == 2


def test_disabling_the_cache_changes_nothing_but_the_speed(python, monkeypatch, repo) -> None:
    """``RELORE_NO_CACHE`` is what makes the equivalence testable at all, so it is itself
    load-bearing: an escape hatch that took a different code path to a different answer
    would hide the bug it exists to expose."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(
        b"class A:\n    def to(self):\n        pass\n\ndef f():\n    pass\n"
    )
    _commit(repo)
    warm = repo_map(str(repo), limit=50)

    monkeypatch.setenv(cache.DISABLE_ENV, "1")
    disabled = repo_map(str(repo), limit=50)

    assert warm == disabled


def test_the_cache_is_not_written_when_it_is_disabled(python, monkeypatch, repo) -> None:
    _use(monkeypatch, python)
    monkeypatch.setenv(cache.DISABLE_ENV, "1")
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)

    repo_map(str(repo))

    assert not (repo / cache.CACHE_DIR).exists()


# -- freshness -------------------------------------------------------------


def test_an_edit_that_does_not_move_mtime_is_still_seen(python, monkeypatch, tmp_path) -> None:
    """The whole argument for keying on content rather than on a timestamp.

    ``touch -r``, ``rsync -t`` and a coarse filesystem clock all change a file's bytes while
    leaving its mtime where it was. An mtime-keyed cache answers such a tree with the
    previous parse and no way to tell; a content-keyed one cannot, because it never consults
    a timestamp for anything.
    """
    _use(monkeypatch, python)
    path = tmp_path / "m.py"
    path.write_bytes(BEFORE)
    assert names(tmp_path) == {"before"}
    stamped = os.stat(path)

    path.write_bytes(AFTER)
    os.utime(path, (stamped.st_atime, stamped.st_mtime))

    assert os.stat(path).st_mtime == stamped.st_mtime, "the test needs the mtime pinned"
    assert names(tmp_path) == {"after"}


def test_a_dirty_tracked_file_is_reparsed(python, monkeypatch, repo) -> None:
    """A committed file's index sha describes the *index*. The moment the worktree diverges
    the key has to be withdrawn, or ``map`` starts describing a tree nobody is on."""
    _use(monkeypatch, python)
    path = repo / "m.py"
    path.write_bytes(BEFORE)
    _commit(repo)
    assert names(repo) == {"before"}

    path.write_bytes(AFTER)

    assert names(repo) == {"after"}


def test_committing_the_edit_is_seen_too(python, monkeypatch, repo) -> None:
    """The other half: a new commit gives the file a new blob sha, so the entry keyed on the
    old one is a miss. Without this, the cache would go stale exactly when work is saved."""
    _use(monkeypatch, python)
    path = repo / "m.py"
    path.write_bytes(BEFORE)
    _commit(repo)
    assert names(repo) == {"before"}

    path.write_bytes(AFTER)
    _commit(repo)

    assert names(repo) == {"after"}


def test_a_new_file_joins_a_warm_cache(python, monkeypatch, repo) -> None:
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    assert names(repo) == {"before"}

    (repo / "extra.py").write_bytes(AFTER)

    assert names(repo) == {"before", "after"}


def test_a_deleted_file_leaves_the_answer(python, monkeypatch, repo) -> None:
    """And leaves the cache: an entry nobody visits is dropped on the next full walk, so a
    long-lived checkout cannot accumulate one dead entry per rename for ever."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    (repo / "gone.py").write_bytes(AFTER)
    _commit(repo)
    assert names(repo) == {"before", "after"}

    (repo / "gone.py").unlink()

    assert names(repo) == {"before"}
    assert "gone.py" not in _entries(repo)


# -- the half of the key that is not the content ---------------------------


def test_another_provider_does_not_inherit_the_entry(python, monkeypatch, repo) -> None:
    """Which provider reads a file depends on what is installed, and ``pip uninstall
    tree-sitter-python`` silently moves every ``*.py`` to the ctags fallback -- a different
    parse of identical bytes. Content alone is not the key; content *and* provider is.
    """
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(b"def before():\n    pass\n")
    _commit(repo)
    assert names(repo) == {"before"}

    other = ToyProvider()
    other.patterns = ("*.py",)
    _use(monkeypatch, other)

    assert names(repo) != {"before"}


def test_a_format_bump_discards_the_old_file(python, monkeypatch, repo) -> None:
    """An entry shape that changed is thrown away rather than migrated: it is a cache, so
    the cost of being wrong about the old layout is one slow call."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    repo_map(str(repo))

    monkeypatch.setattr(cache, "FORMAT", cache.FORMAT + 1)

    assert names(repo) == {"before"}
    assert _payload(repo)["format"] == cache.FORMAT


def test_a_file_git_was_told_to_stop_watching_is_still_reparsed(python, monkeypatch, repo) -> None:
    """The measurement that removed the index fast path, kept as a test.

    An earlier draft took each file's key from ``git ls-files -s`` and reparsed only what
    ``git diff-files`` called dirty -- 0.5 s instead of 1.2 s, because a clean file never had
    to be opened. But git's "clean" is a comparison of **stat data**, not of content, and it
    has documented opt-outs. ``--assume-unchanged`` is the bluntest: it tells git to stop
    looking at the file at all, and the cached ``map`` then reported the previous definition
    while the file on disk held a new one.
    """
    _use(monkeypatch, python)
    path = repo / "m.py"
    path.write_bytes(BEFORE)
    _commit(repo)
    assert names(repo) == {"before"}

    _git(repo, "update-index", "--assume-unchanged", "m.py")
    path.write_bytes(AFTER)
    assert _git(repo, "diff-files", "--name-only") == "", "git must consider the file clean"

    assert names(repo) == {"after"}


def test_an_entry_is_keyed_on_the_bytes_that_were_parsed(python, monkeypatch, repo) -> None:
    """The key is the sha of what was read, so an entry can never describe other bytes."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)

    repo_map(str(repo))

    assert _entries(repo)["m.py"]["s"] == cache.blob_sha(BEFORE)
    assert cache.blob_sha(BEFORE) == _git(repo, "rev-parse", "HEAD:m.py")


def test_a_symlinked_root_keys_the_same_as_the_real_one(
    python, monkeypatch, repo, tmp_path
) -> None:
    """``/tmp`` is ``/private/tmp`` on macOS, and git reports paths against the resolved
    toplevel. An unresolved key would read ``../../tmp/m.py``, match nothing git said, and
    leave a cache that writes every run and hits never."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    link = repo.parent / "link"  # beside the repository, so the walk cannot descend into it
    link.symlink_to(repo, target_is_directory=True)

    assert names(link) == {"before"}

    assert set(_entries(repo)) == {"m.py"}


# -- never break the verb --------------------------------------------------


def test_a_corrupt_cache_is_a_miss_rather_than_a_traceback(python, monkeypatch, repo) -> None:
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    repo_map(str(repo))
    (repo / cache.CACHE_DIR / cache.CACHE_FILE).write_bytes(b"not a gzip stream at all")

    assert names(repo) == {"before"}


def test_a_tree_it_cannot_write_to_still_answers(python, monkeypatch, repo) -> None:
    """A read-only checkout, a full disk and a sandbox all land here. The answer was already
    computed by the time the write is attempted, so the only thing a failed write may cost
    is the next call's speed."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    monkeypatch.setattr(cache, "_save", _explode)

    assert names(repo) == {"before"}


def test_a_tree_with_no_git_still_caches(python, monkeypatch, tmp_path) -> None:
    """git buys the right to skip the *read*, not the right to cache. Without it the key is
    the sha of the bytes, which costs a read and keeps every guarantee."""
    _use(monkeypatch, python)
    (tmp_path / "m.py").write_bytes(BEFORE)

    assert names(tmp_path) == {"before"}
    assert (tmp_path / cache.CACHE_DIR / cache.CACHE_FILE).exists()
    assert names(tmp_path) == {"before"}


def test_git_falling_over_only_moves_where_the_cache_lands(python, monkeypatch, repo) -> None:
    """git is asked one question -- where the repository root is -- so losing it costs the
    shared anchor and nothing else. Freshness never depended on it."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    monkeypatch.setattr(cache, "_git", lambda *args: None)

    assert names(repo) == {"before"}
    assert names(repo) == {"before"}


# -- where it lands --------------------------------------------------------


def test_the_cache_directory_ignores_itself(python, monkeypatch, repo) -> None:
    """``relore map`` runs in whatever repository the caller is sitting in, and a read-only
    tool does not get to leave 4 MB of untracked noise in someone else's ``git status``."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)

    repo_map(str(repo))

    assert (repo / cache.CACHE_DIR / ".gitignore").read_text() == cache.SELF_IGNORE
    assert _git(repo, "status", "--porcelain") == ""


def test_a_subtree_run_shares_the_repository_cache_and_prunes_nothing(
    python, monkeypatch, repo
) -> None:
    """``relore map src/x`` must not throw away the rest of the repository's work, and must
    not scatter a ``.relore/`` into every directory anyone ever pointed the verb at."""
    _use(monkeypatch, python)
    (repo / "top.py").write_bytes(BEFORE)
    package = repo / "package"
    package.mkdir()
    (package / "inner.py").write_bytes(AFTER)
    _commit(repo)
    repo_map(str(repo))
    assert set(_entries(repo)) == {"top.py", "package/inner.py"}

    (package / "second.py").write_bytes(b"def second():\n    pass\n")
    assert names(package) == {"after", "second"}

    assert not (package / cache.CACHE_DIR).exists()
    assert set(_entries(repo)) == {"top.py", "package/inner.py", "package/second.py"}


def test_the_cache_does_not_index_itself(python, monkeypatch, repo) -> None:
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)

    repo_map(str(repo))

    assert all(not path.startswith(cache.CACHE_DIR) for path in _entries(repo))


def test_one_file_is_answered_without_wiping_the_tree(python, monkeypatch, repo) -> None:
    """``source_files`` accepts a file path, so ``map`` does too -- and a one-file walk that
    pruned on the way out would delete every other entry in the repository."""
    _use(monkeypatch, python)
    (repo / "m.py").write_bytes(BEFORE)
    (repo / "other.py").write_bytes(AFTER)
    _commit(repo)
    repo_map(str(repo))

    assert names(repo / "m.py") == {"before"}
    assert set(_entries(repo)) == {"m.py", "other.py"}


# -- the key itself --------------------------------------------------------


def test_the_content_key_is_gits_own_object_name(repo) -> None:
    """Not an implementation detail: a file we hashed ourselves and a file git told us about
    have to land in one key space, or every entry written by one path would miss on the
    other and the cache would look like it worked while never hitting.
    """
    (repo / "m.py").write_bytes(BEFORE)
    _commit(repo)
    expected = _git(repo, "rev-parse", "HEAD:m.py")

    assert cache.blob_sha(BEFORE) == expected


def _entries(root) -> dict:
    return _payload(root)["entries"]


def _payload(root) -> dict:
    with gzip.open(root / cache.CACHE_DIR / cache.CACHE_FILE, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _explode(*args, **kwargs) -> None:
    raise OSError("read-only file system")


def _use(monkeypatch: pytest.MonkeyPatch, *providers) -> None:
    monkeypatch.setattr(registry, "providers", lambda: tuple(providers))
    monkeypatch.setattr(registry, "ctags_provider", lambda: None)


def _git(root, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True, text=True
    )
    return done.stdout.strip()


def _commit(root) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "x")
