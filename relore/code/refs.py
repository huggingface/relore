"""References to a name, across a local checkout.

``refs`` is optional throughout (section 9's rule 2): a language whose provider does not
offer it says so and exits 0, rather than returning an empty list a caller would read as
"nothing calls this".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from relore.code import registry
from relore.code.api import REFS, MissingParser
from relore.code.registry import provider_for
from relore.code.walk import needle, read, source_files


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    name: str
    #: What the occurrence *is* -- ``call``, ``definition``, ``attribute``, ``name``. The
    #: provider names its own kinds (rule 1); core groups by them and reports the counts.
    kind: str = "reference"


@dataclass(frozen=True)
class RefResult:
    """References found, plus the files nobody could answer for.

    ``unsupported`` is the honest part: a checkout of mixed languages where only some have
    a ``refs`` tier produces a partial answer, and a caller that cannot see which files
    were skipped will read the gap as an absence.
    """

    hits: tuple[Hit, ...]
    searched: int
    unsupported: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unsupported

    @property
    def by_kind(self) -> dict[str, int]:
        """How many of each kind, because a bare total is the thing that misled.

        21 plausible-looking lines with no denominator and no statement of what was left
        out reads as the whole truth; ``21 call, 38 definition, 214 attribute`` is a
        number a caller can act on -- and can compare against ``grep``.
        """
        counts: dict[str, int] = {}
        for hit in self.hits:
            counts[hit.kind] = counts.get(hit.kind, 0) + 1
        return dict(sorted(counts.items()))


def references(root: str, symbol: str) -> RefResult:
    """Every occurrence of ``symbol`` in the tree, by kind.

    Only the files whose bytes contain the name are parsed (issue #83, lever 1). This is
    :func:`~relore.code.walk.needle`, the same prefilter ``copies`` and ``symbol`` have had
    since issue #45, and the argument carries over unchanged because ``refs`` also takes a
    single name: a file that does not spell it cannot reference it. Measured on
    ``huggingface/transformers`` @ ``d9890f6``, ``refs use_kernels`` goes 10.7 s -> 0.6 s,
    9 files parsed instead of 4,883, and **the answer is byte-identical** -- 47 references,
    same lines, same kinds, same summary.

    The order of the three tests below is load-bearing:

    * the provider is resolved and the ``refs``-tier check runs **first**, on every claimed
      file, so ``unsupported`` still names every language that cannot answer. Prefiltering
      ahead of it would let a Rust file that happens not to spell the name disappear before
      anyone noticed Rust has no reference tier, and :attr:`RefResult.complete` would then
      claim a partial answer was whole.
    * a file skipped by the prefilter still counts as ``searched``. It was searched -- read
      in full and answered for, definitively, without a parse -- and issue #37 is the
      standing reminder that a file count which drifts from what it says is a wrong answer
      in the direction that looks like diligence.

    Where it stops helping is the honest part: the filter is only as selective as the name
    is rare. ``use_kernels`` is in 9 of 4,883 files, ``forward`` in 1,648 and ``config`` in
    3,182, so a name written nearly everywhere filters to nearly everything (issue #83's
    table). It is never *slower* than the full walk -- the read already happened -- but on a
    ubiquitous name it is close to no filter at all.

    **The cache does not belong here, and the reason is now structural.** Issue #83 proposed
    that ``refs`` consult :mod:`relore.code.cache` for an exact name set instead of a byte
    superset. It was built and measured, and it lost -- 1.97 s against 1.37 s on
    ``compute_default_rope_parameters``, 10.98 s against 9.66 s on ``config`` -- because
    opening a 4.2 MB cache costs ~0.4 s and bought back only a read the prefilter had already
    made cheap.

    Then the cache stopped skipping reads altogether (its rule 1: git's idea of a clean file
    is a stat comparison, so freshness has to be the sha of the bytes, which means reading
    them). What remains of it is the ability to skip a *parse* -- which is exactly and only
    what the prefilter above already does for this verb. There is nothing left for a cache to
    contribute here, at any price.
    """
    registry.require_any()
    wanted = needle(symbol)
    hits: list[Hit] = []
    unsupported: set[str] = set()
    searched = 0
    for path in source_files(root):
        try:
            provider = provider_for(path)
        except MissingParser:
            continue
        if REFS not in provider.capabilities:
            unsupported.add(provider.name)
            continue
        source = read(path)
        if source is None:
            continue
        searched += 1
        if wanted is not None and wanted not in source:
            continue
        hits.extend(_matches(provider, path, source, symbol))
    return RefResult(tuple(hits), searched, tuple(sorted(unsupported)))


def _matches(provider, path: str, source: bytes, symbol: str) -> Iterator[Hit]:
    """Compare, never parse.

    Rule 1 makes ``qualname`` opaque above the provider interface, so ``symbol`` is matched
    whole against either the name written at the call site or a qualname the provider
    resolved -- and **not** by splitting it on a dot. Splitting would be core deciding what
    the separator is, which is the one thing rule 1 forbids: it is ``::`` in one language
    and ``#`` in another, and the split would quietly do the wrong thing in both.

    The consequence is a real and declared limit: ``relore refs Gemma3Model.forward``
    matches only where a provider resolves callees, and Python's does not (see
    :class:`~relore.code.providers.python.PythonProvider`). ``relore refs forward`` is the
    query that works today.
    """
    for reference in provider.refs(path, source):
        if symbol in (reference.name, reference.qualname):
            yield Hit(path=path, line=reference.line, name=reference.name, kind=reference.kind)
