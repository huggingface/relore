"""The frozen evaluation set of section 10.

One file, JSONL, first line a header. It holds three slices -- ``rationale``,
``failure``, ``precedent`` -- because they have different ranking regimes and a weight
set that helps one can hurt another; the runner scores each separately and never pools
them.

Two things travel with the set rather than with a run, because without them a number is
not comparable to anything:

* **the corpus window.** Section 5.5's sample indexes a bounded window, so `relore`
  cannot find what it never indexed. Every baseline has to be restricted to the same
  window, and the header carries it so the runner can do that rather than trust whoever
  reads the table.
* **who judged.** A model-judged example and a human-judged one are both usable and are
  not the same evidence. ``judged_by`` is per example and the runner reports the split.

``frozen_at`` is the point of the file: section 10 says freeze the set before tuning, and
a set that can still grow while the weights move is not a benchmark. :func:`save` refuses
to write a frozen set that has changed.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from relore.search.queries import QUERY_KINDS

#: Section 10's three slices. Deliberately the same names as the query kinds -- an
#: example is a query of that kind, and giving the slice its own vocabulary would invite
#: labelling one slice and scoring it under another kind's trust floor.
SLICES = QUERY_KINDS

JUDGES = ("human", "model")

VERSION = 1


@dataclass(frozen=True)
class Relevant:
    """One thread that answers the query, and why it does.

    ``why`` is not decoration: an example whose ground truth nobody can re-check is a
    number without an argument, and section 10's caveat about file-overlap recall is
    exactly the kind of thing a later reader needs the ``why`` to notice.
    """

    repo: str
    number: int
    why: str = ""


@dataclass(frozen=True)
class Example:
    """One query and the threads that answer it.

    The query is expressed the way a caller actually sends one: a short text term set
    (``search`` ANDs every content term, so a pasted sentence matches nothing) plus the
    section 5.3 signal filters. Both halves are scored -- a baseline gets the same terms.
    """

    id: str
    kind: str
    query: str = ""
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    relevant: tuple[Relevant, ...] = ()
    #: Section 13's cutoff, for a set whose examples come from different points in history.
    #: A run-level ``--before`` puts one instant on every example, which is right for a
    #: smoke test and wrong for a set where each example's window is its own: twenty
    #: examples drawn from twenty issues have twenty cutoffs, and a flag cannot hold them.
    #: ISO 8601 and timezone-aware, or refused. ``None`` leaves the window to the run.
    before: str | None = None
    #: Threads this example withholds by number, on top of the run's own list. The cutoff
    #: cannot express it: the thread an example was drawn from usually predates the
    #: example's own cutoff and is still the answer.
    exclude: tuple[int, ...] = ()
    #: ``None`` means mined but unjudged -- a candidate, not yet ground truth.
    judged_by: str | None = None
    note: str = ""
    #: How it was mined: the document it came from, the cue that surfaced it. Kept so a
    #: labeller can open the evidence rather than take the miner's word for it.
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SLICES:
            raise ValueError(f"unknown slice {self.kind!r}; one of {list(SLICES)}")
        if self.judged_by is not None and self.judged_by not in JUDGES:
            raise ValueError(f"judged_by must be one of {list(JUDGES)} or null")
        if self.judged_by is not None and not self.relevant:
            raise ValueError(f"{self.id}: a judged example needs at least one relevant thread")
        if not (self.query or self.files or self.symbols or self.errors or self.tests):
            raise ValueError(f"{self.id}: an example with no query and no filter asks nothing")
        if self.before is not None and self.window() is None:
            raise ValueError(f"{self.id}: before={self.before!r} is not a timezone-aware instant")

    def window(self) -> dt.datetime | None:
        """``before`` as an instant, or ``None`` if it is unset or unusable.

        Naive is unusable rather than merely awkward: a cutoff with no zone is a different
        instant depending on who runs the set, which is the one thing a frozen set exists to
        prevent. ``SearchQuery`` refuses one for the same reason.
        """
        if self.before is None:
            return None
        try:
            parsed = dt.datetime.fromisoformat(self.before.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None

    @property
    def judged(self) -> bool:
        return self.judged_by is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Example:
        row = dict(row)
        rel = tuple(Relevant(**r) for r in row.pop("relevant", ()))
        for key in ("files", "symbols", "errors", "tests", "exclude"):
            row[key] = tuple(row.get(key, ()))
        return cls(relevant=rel, **row)


@dataclass(frozen=True)
class Corpus:
    """One indexed window, copied out of ``repo_sample`` at mine time.

    A full backfill has no window; it is recorded with ``since=None``, which the runner
    reads as "do not restrict the baselines".
    """

    repo: str
    thread_type: str
    since: str | None
    selector: str = ""
    threads: int = 0
    documents: int = 0


@dataclass(frozen=True)
class Dataset:
    examples: tuple[Example, ...] = ()
    corpus: tuple[Corpus, ...] = ()
    frozen_at: str | None = None
    version: int = VERSION

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for ex in self.examples:
            if ex.id in seen:
                raise ValueError(f"duplicate example id {ex.id!r}")
            seen.add(ex.id)

    @property
    def frozen(self) -> bool:
        return self.frozen_at is not None

    def slice(self, kind: str) -> tuple[Example, ...]:
        return tuple(ex for ex in self.examples if ex.kind == kind)

    def judged(self) -> Dataset:
        """The scoreable subset. Unjudged candidates are carried in the same file so
        mining and labelling are one loop, and dropped here so they cannot inflate a
        denominator."""
        return replace(self, examples=tuple(ex for ex in self.examples if ex.judged))

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for kind in SLICES:
            rows = self.slice(kind)
            out[kind] = {
                "examples": len(rows),
                "judged": sum(1 for r in rows if r.judged),
                **{j: sum(1 for r in rows if r.judged_by == j) for j in JUDGES},
            }
        return out

    def since(self, repo: str) -> str | None:
        """The floor for this repository: the *oldest* window over its thread types, so a
        baseline restricted by it cannot be narrower than what the index holds."""
        floors = [c.since for c in self.corpus if c.repo == repo]
        if not floors or any(f is None for f in floors):
            return None
        return min(f for f in floors if f is not None)

    def add(self, examples: Iterable[Example]) -> Dataset:
        if self.frozen:
            raise ValueError("this set is frozen; unfreeze it deliberately before adding")
        known = {ex.id for ex in self.examples}
        new = tuple(ex for ex in examples if ex.id not in known)
        return replace(self, examples=self.examples + new)

    def freeze(self) -> Dataset:
        return replace(self, frozen_at=dt.datetime.now(dt.timezone.utc).isoformat())


def load(path: Path) -> Dataset:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return Dataset()
    head = json.loads(lines[0])
    if "dataset" not in head:
        raise ValueError(f"{path}: first line must be the dataset header")
    meta = head["dataset"]
    if meta.get("version") != VERSION:
        raise ValueError(f"{path}: dataset version {meta.get('version')!r}, expected {VERSION}")
    return Dataset(
        examples=tuple(Example.from_dict(json.loads(ln)) for ln in lines[1:]),
        corpus=tuple(Corpus(**c) for c in meta.get("corpus", ())),
        frozen_at=meta.get("frozen_at"),
    )


def save(dataset: Dataset, path: Path) -> None:
    """Write, refusing to change a frozen set.

    The check is against what is already on disk, not against an in-memory flag: the way
    a frozen benchmark actually rots is a later run appending to it, and by then whoever
    set the flag is out of context.
    """
    if path.exists():
        prior = load(path)
        if prior.frozen and prior.examples != dataset.examples:
            raise ValueError(
                f"{path} was frozen at {prior.frozen_at}; section 10 scores against the "
                "frozen set. Write a new file rather than editing this one."
            )
    header = {
        "dataset": {
            "version": dataset.version,
            "frozen_at": dataset.frozen_at,
            "corpus": [asdict(c) for c in dataset.corpus],
        }
    }
    body = "\n".join([json.dumps(header)] + [json.dumps(ex.as_dict()) for ex in dataset.examples])
    path.write_text(body + "\n")
