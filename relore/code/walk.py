"""Finding the files to parse in a working tree.

This is the one part of :mod:`relore.code` that touches a filesystem, and it is separate
from the providers on purpose: section 9's rule 4 makes a provider a pure function of
``(path, bytes)``, which is what lets the same provider run here against a dirty working
tree and inside ``relored`` against a historical blob. Give a provider a walker and that
stops being true.

**Nothing here is cached, and the walk is not the cost.** Reading all 4,883 claimed files
in ``huggingface/transformers`` is 0.59 s; parsing them is 16.4 s. So this module stays a
plain walk, and the two things that made the verbs slow are handled where the parse is:
:func:`needle`, the byte prefilter, and :mod:`relore.code.cache`.

That module replaces a claim this file used to make -- that a cached repo map is *a
staleness hazard aimed at exactly the property that justifies doing this locally*, and that
parsing a Python repository is fast enough not to need one (section 1). Issue #83 measured
the second half and it was false: ``map`` is 18.2 s and repeats in full on every call. The
first half was true and survives as a constraint rather than a prohibition -- the hazard is
a cache keyed on a *timestamp*, and one keyed on content cannot describe a tree other than
the one on disk. See :mod:`relore.code.cache` for the four rules that keep a cached answer
byte-identical to an uncached one.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

#: Directories never worth walking. Not a gitignore parser: this runs in whatever
#: directory the caller is sitting in, and the cost of being wrong is a slower map rather
#: than a wrong one.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
        "site-packages",
        ".tox",
        ".eggs",
    }
)

#: A file bigger than this is a vendored bundle, a fixture or a generated blob, and parsing
#: it costs more than the one symbol it might contribute.
MAX_FILE_BYTES = 2_000_000

#: The last run of identifier characters in a name -- see :func:`needle`.
_TRAILING_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+$")


def source_files(root: str) -> Iterator[str]:
    """Every file under ``root`` some provider claims, in a stable order.

    A file path is also accepted, so ``relore refs`` and ``relore map`` work on one file.
    """
    from relore.code.registry import claimed

    if os.path.isfile(root):
        if claimed(root):
            yield root
        return
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(names):
            path = os.path.join(directory, name)
            if claimed(path):
                yield path


def all_files(root: str) -> Iterator[str]:
    """Every file under ``root`` worth reading, claimed by a provider or not.

    :func:`source_files` is the *parsing* set, so it depends on which grammars are
    installed. A text search must not: "which files have this shape" is asked of YAML and
    docs too, and an answer that silently omits them is worse than a slower one.
    """
    if os.path.isfile(root):
        yield root
        return
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(names):
            if not name.startswith("."):
                yield os.path.join(directory, name)


def needle(name: str) -> bytes | None:
    """The bytes a file must contain to be worth parsing for ``name``, or ``None``.

    This is the prefilter that closed issue #45 and it is the whole of issue #83's lever 1.
    A provider reads a name out of the source, so a file whose bytes do not contain the
    identifier can neither define nor reference it. The result is still a *superset* -- a
    file that only mentions the name in a comment is parsed and rejected by the caller --
    which is the property that makes it safe: it can cost a wasted parse, never an answer.

    Reading is not the cost. On ``huggingface/transformers`` reading all 4,883 claimed files
    is 0.59 s and parsing them is 16.44 s, so a filter that still reads everything and skips
    the parse keeps ~97% of the win.

    Two things it deliberately does **not** do:

    * It matches the trailing identifier run, not the whole string. ``Foo.bar`` is spelled
      across two lines and the dotted form is in no file -- and rule 1 makes the separator
      the *language's*, so splitting on ``.`` alone would over-filter a provider that writes
      ``Foo::bar`` or ``Foo#bar``. Anything that is not an identifier character ends the run.
    * It returns ``None`` for a non-ASCII name rather than guessing an encoding. A
      ``coding:`` declaration means a file's bytes need not be UTF-8, and a filter that
      guesses wrong silently drops real matches -- so the caller falls back to parsing
      everything, which is slow and right.
    * It returns ``None`` when there is no trailing run at all. Falling back to the whole
      string would make ``Vec<T>`` or ``Foo.<lambda>`` its own needle, which is in no file,
      so every file would be rejected and the verb would answer "nothing found" with
      complete confidence -- the one outcome a *superset* filter must never produce.

    Every ``None`` above means the same thing: filter nothing, parse everything. Slow is the
    only direction this function is allowed to be wrong in.
    """
    leaf = _TRAILING_IDENTIFIER.search(name)
    if leaf is None:
        return None
    text = leaf.group(0)
    return text.encode() if text.isascii() else None


def read(path: str) -> bytes | None:
    """A file's bytes, or ``None`` when it is too big or unreadable.

    Unreadable is not an error here: a repo map that dies on one broken symlink is a repo
    map nobody can run, and the missing file is one symbol rather than a wrong answer.
    """
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None
