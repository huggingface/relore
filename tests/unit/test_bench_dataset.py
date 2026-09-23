"""The evaluation set's guarantees: it freezes, ids are unique, and a judgement is
inspectable (the build plan section 10)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from relore.bench import dataset as ds


def _example(**kwargs) -> ds.Example:
    kwargs.setdefault("id", "rationale:owner/name#1:c1")
    kwargs.setdefault("kind", "rationale")
    kwargs.setdefault("query", "load_adapter")
    return ds.Example(**kwargs)


def test_an_example_with_no_query_and_no_filter_asks_nothing() -> None:
    with pytest.raises(ValueError, match="asks nothing"):
        _example(query="")


def test_a_judged_example_needs_a_relevant_thread() -> None:
    """The failure this prevents is a silent zero: an example judged 'nothing answers
    this' scores 0 recall for every system, and averages that into the slice."""
    with pytest.raises(ValueError, match="at least one relevant"):
        _example(judged_by="human")


def test_an_unknown_slice_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown slice"):
        _example(kind="ranking")


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate example id"):
        ds.Dataset(examples=(_example(), _example()))


def test_unjudged_candidates_do_not_reach_the_score(tmp_path: Path) -> None:
    judged = _example(id="a", judged_by="model", relevant=(ds.Relevant("owner/name", 1),))
    data = ds.Dataset(examples=(judged, _example(id="b")))

    assert [ex.id for ex in data.judged().examples] == ["a"]
    assert data.counts()["rationale"] == {
        "examples": 2,
        "judged": 1,
        "human": 0,
        "model": 1,
    }


def test_a_round_trip_keeps_the_window_and_the_judge(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    data = ds.Dataset(
        examples=(_example(judged_by="model", relevant=(ds.Relevant("owner/name", 7, "why"),)),),
        corpus=(ds.Corpus("owner/name", "pr", "2026-03-01", "merged", 10, 100),),
    )
    ds.save(data, path)

    back = ds.load(path)
    assert back == data
    assert back.since("owner/name") == "2026-03-01"


def test_a_full_history_declares_no_floor() -> None:
    """Section 10 restricts every baseline by the corpus window. A backfill has none, and
    saying so is not the same as forgetting to record one."""
    data = ds.Dataset(corpus=(ds.Corpus("owner/name", "all", None, "full history"),))

    assert data.since("owner/name") is None
    assert data.since("owner/absent") is None


def test_the_oldest_window_is_the_floor() -> None:
    data = ds.Dataset(
        corpus=(
            ds.Corpus("owner/name", "issue", "2026-03-01", ""),
            ds.Corpus("owner/name", "pr", "2026-06-01", ""),
        )
    )

    assert data.since("owner/name") == "2026-03-01"


def test_a_frozen_set_refuses_to_grow_on_disk(tmp_path: Path) -> None:
    """The check is against the file, not an in-memory flag: the way a frozen benchmark
    actually rots is a later run appending to it."""
    path = tmp_path / "eval.jsonl"
    ds.save(ds.Dataset(examples=(_example(),)).freeze(), path)

    frozen = ds.load(path)
    assert frozen.frozen
    with pytest.raises(ValueError, match="frozen"):
        ds.save(ds.Dataset(examples=(_example(), _example(id="b"))), path)

    # Rewriting the same examples is not a change, so re-saving is allowed.
    ds.save(frozen, path)


def test_add_refuses_on_a_frozen_set() -> None:
    with pytest.raises(ValueError, match="frozen"):
        ds.Dataset(examples=(_example(),)).freeze().add([_example(id="b")])


def test_add_is_idempotent() -> None:
    data = ds.Dataset().add([_example()]).add([_example()])

    assert len(data.examples) == 1


def test_an_example_may_carry_its_own_cutoff() -> None:
    """A set whose examples come from different points in history needs one window per
    example; a run-level flag holds exactly one."""
    ex = _example(before="2026-03-01T00:00:00+00:00", exclude=(41310,))
    assert ex.window() == dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    assert ex.exclude == (41310,)


def test_a_trailing_z_is_a_cutoff_like_any_other() -> None:
    assert _example(before="2026-03-01T00:00:00Z").window() == dt.datetime(
        2026, 3, 1, tzinfo=dt.timezone.utc
    )


def test_a_naive_cutoff_is_refused() -> None:
    """A cutoff with no zone is a different instant depending on who runs the set, which is
    the one thing a frozen set exists to prevent. `SearchQuery` refuses one too."""
    with pytest.raises(ValueError, match="timezone-aware instant"):
        _example(before="2026-03-01T00:00:00")


def test_an_unparseable_cutoff_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware instant"):
        _example(before="march the first")


def test_no_cutoff_leaves_the_window_to_the_run() -> None:
    assert _example().window() is None


def test_the_window_survives_the_file(tmp_path: Path) -> None:
    ex = _example(before="2026-03-01T00:00:00+00:00", exclude=(7, 9))
    path = tmp_path / "set.jsonl"
    ds.save(ds.Dataset(examples=(ex,)), path)
    assert ds.load(path).examples == (ex,)
