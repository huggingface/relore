"""The ranked repo map: what a stranger should read first.

Ranked by **how often a definition's name is written elsewhere, divided by how many
definitions share that name.** Both halves are load-bearing, and the division is there
because the undivided count produced a map with no information in it.

Measured on ``huggingface/transformers``: 35 of the top 40 were ``__init__`` at exactly
9,198 and the other 5 were ``.to`` at exactly 12,637 -- ``Rotation.to`` in an openfold
utility credited with every ``.to(`` in the repository. That list is "the most common
method names in Python", which is the same list for every project, and it spends the top of
a caller's context budget on it. Dividing by the number of definitions sharing the name is
the cheapest defensible correction: a name defined once and written 400 times outranks one
defined 400 times and written 9,000. Dunders are dropped outright -- they are the same name
everywhere by construction, so the division cannot rescue them and they are never the
answer to "what matters here".

It remains a count of a *written name*, not a resolved call graph. Nothing here infers a
type, so ``map`` is an orientation aid and not evidence about a particular definition.

Note what this does *not* do: it never splits a ``qualname``. A provider reports both the
whole qualified name and the bare ``name`` a call site would have written, so the call
counter is keyed on the latter and the former is passed through untouched -- section 9's
rule 1, which exists because the separator is the language.

Providers without a ``refs`` tier still contribute their definitions; they just cannot
contribute edges, so their symbols rank by the tie-break rather than by callers. That is
the per-tier degradation of rule 2, and :attr:`RepoMap.ranked_by` says which happened so a
flat ranking is not mistaken for a flat codebase.

**Same-named methods share a count**, and both numbers are printed so it cannot be
mistaken for anything else: three classes each defining ``info`` report the same
``matches`` and a ``shared_by`` of 3. That is the honest ceiling of a graph keyed on the
written name, and lifting it is section 1's symbol-disambiguation tier -- resolving a call
site to ``Gemma3Model.forward`` needs a whole tree in scope, which a pure function of one
file does not have. Read ``matches`` as "this name is busy", not "this definition is".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from relore.code import cache, registry
from relore.code.api import REFS, MissingParser
from relore.code.registry import provider_for
from relore.code.walk import source_files


@dataclass(frozen=True)
class Entry:
    qualname: str
    kind: str
    path: str
    line: int
    #: Occurrences of this definition's bare *name* elsewhere in the tree -- calls,
    #: attribute reads and plain mentions, never the definitions themselves.
    matches: int
    #: How many definitions in the tree carry that name. ``matches`` is credited to every
    #: one of them, so this is the factor by which the number overstates this row.
    shared_by: int = 1


@dataclass(frozen=True)
class RepoMap:
    entries: tuple[Entry, ...]
    files: int
    definitions: int
    #: ``"name matches per definition"`` when at least one provider supplied a reference
    #: graph, else ``"declaration order"``.
    ranked_by: str
    #: Dunder definitions left out of the ranking. Reported rather than silently dropped:
    #: 35 of a previous top 40 were ``__init__``, so their absence is a fact about the
    #: ranking a reader should be told.
    excluded_dunders: int = 0


def repo_map(root: str, *, limit: int = 40) -> RepoMap:
    """The ranked map of ``root``, and the only writer of :mod:`relore.code.cache`.

    ``map`` has no query symbol, so it cannot be prefiltered the way ``refs`` is -- visiting
    everything *is* the verb -- and it already computes both halves of a cache entry, so it
    is the one place population is free. 18.5 s -> 1.0 s on ``huggingface/transformers``.
    ``RELORE_NO_CACHE`` takes the other path and must print the same bytes.

    A definition is not a use of itself; counting it would give every copy of a duplicated
    function a floor of one. That rule lives in :meth:`~relore.code.cache.Session.store`, so
    the two paths cannot drift apart on it.
    """
    registry.require_any()
    found: list[tuple[str, str, str, int, str]] = []  # qualname, name, kind, line, path
    calls: Counter[str] = Counter()
    files = 0
    saw_refs = False
    defined: Counter[str] = Counter()

    with cache.open_session(root) as session:
        for path in source_files(root):
            try:
                provider = provider_for(path)
            except MissingParser:
                continue
            entry, source = session.lookup(path, provider.name)
            if entry is None:
                if source is None:
                    continue
                entry = session.store(path, provider, source)
            files += 1
            for qualname, name, kind, line in entry.definitions:
                found.append((qualname, name, kind, line, path))
                defined[name] += 1
            if REFS in provider.capabilities:
                # Asked of the live provider, never of the entry: "this language has no
                # reference tier" is what `ranked_by` reports, and it is a fact about the
                # install rather than about the file.
                saw_refs = True
                for name, count in entry.calls:
                    calls[name] += count

    entries = [
        Entry(qualname, kind, path, line, calls.get(name, 0), max(defined[name], 1))
        for qualname, name, kind, line, path in found
        if not _is_dunder(name)
    ]
    # Most attributable matches first, then a stable order, so two runs over an unchanged
    # tree agree -- which is what lets a caller read a diff of two maps as a code change.
    entries.sort(key=lambda e: (-e.matches / e.shared_by, e.path, e.line))
    return RepoMap(
        entries=tuple(entries[:limit]),
        files=files,
        definitions=len(found),
        ranked_by="name matches per definition" if saw_refs else "declaration order",
        excluded_dunders=len(found) - len(entries),
    )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")
