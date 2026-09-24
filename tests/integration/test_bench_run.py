"""Scoring section 10's set: the metrics, and the two rules the runner enforces so the
reader does not have to (the build plan section 10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_github import FakeGitHub
from sqlalchemy import Engine

from relore.bench import run as br
from relore.bench.dataset import Corpus, Dataset, Example, Relevant
from relore.ingest.index_thread import index_thread

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


def _result(ranked, relevant, **kwargs) -> br.Result:
    return br.Result(
        example_id="e",
        kind="failure",
        ranked=tuple((REPO, n) for n in ranked),
        relevant=tuple((REPO, n) for n in relevant),
        **kwargs,
    )


# -- the metrics -----------------------------------------------------------


def test_recall_counts_the_relevant_found_not_the_results_returned() -> None:
    result = _result(ranked=[9, 8, 1, 7, 6, 5, 4, 3, 2, 10], relevant=[1, 10])

    assert result.recall_at(5) == 0.5
    assert result.recall_at(10) == 1.0
    assert result.reciprocal_rank() == pytest.approx(1 / 3)


def test_a_miss_is_zero_not_undefined() -> None:
    result = _result(ranked=[2, 3], relevant=[1])

    assert result.recall_at(10) == 0.0
    assert result.reciprocal_rank() == 0.0


def test_an_empty_result_is_scored_not_skipped() -> None:
    """Section 10: for the strongest use cases the grep baseline comes back empty, and
    that is the whole point. Dropping the example would hide exactly that."""
    score = br.Score.over([_result(ranked=[], relevant=[1]), _result(ranked=[1], relevant=[1])])

    assert score.n == 2
    assert score.recall[10] == 0.5
    assert score.empty == 1


# -- the two rules ---------------------------------------------------------


def test_a_report_refuses_a_second_backend() -> None:
    """``ts_rank_cd`` and ``bm25`` are different engines (section 4.1)."""
    report = br.Report(backend={"name": "postgresql", "ranking": "ts_rank_cd"})
    run = br.Run(system="index", backend={"name": "sqlite", "ranking": "bm25"})

    with pytest.raises(br.BackendMismatch, match="refuses to merge"):
        report.add(run)


def test_the_github_baseline_carries_the_window() -> None:
    system = br.GitHubSearchSystem(client=None, repos=(REPO,), since="2026-03-01")
    example = Example(id="e", kind="failure", query="backend mapping")

    query = system.query_string(example)

    assert "repo:owner/name" in query
    assert "updated:>=2026-03-01" in query


def test_an_error_reaches_the_github_baseline_as_a_literal() -> None:
    """Unquoted, GitHub tokenizes the message and the two systems are not asked the same
    question."""
    system = br.GitHubSearchSystem(client=None, repos=(REPO,), since=None)
    example = Example(id="e", kind="failure", errors=("ValueError: backend should be defined",))

    assert '"ValueError: backend should be defined"' in system.query_string(example)


def test_a_placeholder_never_reaches_a_baseline() -> None:
    """Section 5.3's normalized form is what makes the term match across runs, and what
    makes it match nothing outside this index. A baseline handed `<n>` is being asked an
    impossible question, not the same one badly."""
    assert br.error_literal("the size of tensor a (<n>) must match tensor b (<n>) here") == (
        "the size of tensor a"
    )
    assert br.error_literal("expected <n> got <n>") == ""


def test_the_github_query_is_cut_to_fit_and_keeps_its_scope() -> None:
    """``/search/issues`` 422s over 256 characters, and a normalized error is routinely
    longer than that on its own. Dropping a term loses specificity; dropping the scope
    would silently widen the search past the corpus window."""
    system = br.GitHubSearchSystem(client=None, repos=(REPO,), since="2026-03-01")
    example = Example(
        id="e",
        kind="failure",
        query="short",
        errors=("ValueError: " + "verbose " * 60,),
    )

    query = system.query_string(example)

    assert len(query) <= br.MAX_GITHUB_QUERY_CHARS
    assert "repo:owner/name" in query and "updated:>=2026-03-01" in query
    assert "short" in query


def test_a_path_is_a_term_not_a_code_search_qualifier() -> None:
    """`/search/issues` accepts `path:`, matches nothing with it, and reports no error --
    a handicapped baseline that looks like a real zero."""
    system = br.GitHubSearchSystem(client=None, repos=(REPO,), since=None)

    query = system.query_string(Example(id="e", kind="rationale", files=("src/mod.py",)))

    assert "path:" not in query
    assert '"src/mod.py"' in query


def test_a_symbol_reaches_the_github_baseline_too() -> None:
    """A filter the baseline is never handed is a handicap, not a comparison."""
    system = br.GitHubSearchSystem(client=None, repos=(REPO,), since=None)

    assert "build_mask" in system.query_string(
        Example(id="e", kind="rationale", symbols=("build_mask",))
    )


def test_a_bare_scope_is_never_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``repo:owner/name`` alone matches the whole repository, and its top ten would score
    as if the baseline had answered."""

    class Recorder:
        def get_json(self, url, params=None):  # pragma: no cover - must not be called
            raise AssertionError("a scope-only query was sent")

    system = br.GitHubSearchSystem(client=Recorder(), repos=(REPO,), since="2026-03-01")
    monkeypatch.setattr(system, "query_string", lambda example: "repo:owner/name")

    assert system.search(Example(id="e", kind="rationale", query="x")) == (
        (),
        "no expressible query",
    )


# -- the rationale leak split ----------------------------------------------


def test_the_rationale_slice_reports_the_leak_free_subset_apart() -> None:
    """A human finding is itself a retrievable document in the answer's thread, so a
    number pooled over both subsets describes neither (mine.py)."""
    report = br.Report(backend={"name": "sqlite", "ranking": "bm25"})
    run = br.Run(system="index")
    run.results = [
        br.Result("a", "rationale", ((REPO, 1),), ((REPO, 1),), leaks=True),
        br.Result("b", "rationale", (), ((REPO, 2),), leaks=False),
    ]
    report.add(run)

    (row,) = report.table()

    assert row["recall@10"] == 0.5
    assert row["leak_free"] == {"n": 1, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "empty": 1}


# -- the systems, against a real index -------------------------------------


def test_the_index_system_answers_through_the_served_backend(
    engine: Engine, fake: FakeGitHub
) -> None:
    fake.add_issue(1, body="ValueError: backend should be defined in the mapping")
    fake.add_issue(2, body="something else entirely")
    with fake.client() as client:
        index_thread(engine, client, REPO, 1)
        index_thread(engine, client, REPO, 2)

    system = br.IndexSystem(engine, (REPO,))
    ranked, _ = system.search(Example(id="e", kind="failure", query="backend mapping"))

    assert ranked == ((REPO, 1),)


def test_one_thread_is_one_result_however_many_documents_matched(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The API returns up to three documents per thread (section 6); counting a chatty
    thread twice would make one answer look like two."""
    issue = fake.add_issue(1, body="the mapping backend is wrong")
    fake.add_comment(issue, 10, "the mapping backend again")
    fake.add_comment(issue, 11, "still the mapping backend")
    with fake.client() as client:
        index_thread(engine, client, REPO, 1)

    ranked, _ = br.IndexSystem(engine, (REPO,)).search(
        Example(id="e", kind="failure", query="mapping backend")
    )

    assert ranked == ((REPO, 1),)


def test_the_grep_baseline_maps_files_onto_threads(
    engine: Engine, fake: FakeGitHub, tmp_path: Path
) -> None:
    clone = tmp_path / "clone"
    (clone / "src").mkdir(parents=True)
    (clone / "src" / "mod.py").write_text("def build_mask():\n    return None\n")
    _git(clone)

    system = br.GrepSystem({REPO: clone}, {REPO: {"src/mod.py": [(REPO, 7)]}})
    ranked, note = system.search(Example(id="e", kind="failure", query="build_mask"))

    assert ranked == ((REPO, 7),)
    assert "1 files matched" in note


def test_the_grep_baseline_greps_every_clone_it_was_given(tmp_path: Path) -> None:
    """A corpus can span repositories, and an agent in front of two checkouts greps both.

    Scoping the grep to whichever repository holds the answer would be reading the ground
    truth, so both are searched and the matches pooled.
    """
    other = "huggingface/other"
    clones = {}
    for repo in (REPO, other):
        clone = tmp_path / repo.replace("/", "-")
        (clone / "src").mkdir(parents=True)
        (clone / "src" / "mod.py").write_text("def build_mask():\n    return None\n")
        _git(clone)
        clones[repo] = clone

    system = br.GrepSystem(
        clones,
        {
            REPO: {"src/mod.py": [(REPO, 7)]},
            other: {"src/mod.py": [(other, 9)]},
        },
    )
    ranked, note = system.search(Example(id="e", kind="failure", query="build_mask"))

    assert set(ranked) == {(REPO, 7), (other, 9)}
    # Two repositories matched one path each: the path is keyed by its repository, so
    # they are two matches and not one overwriting the other.
    assert "2 files matched" in note


def test_the_grep_baseline_greps_the_literal_before_the_prose(tmp_path: Path) -> None:
    """Section 10.2: a baseline that cannot express the query is a bug, not a result.

    The query's own words are prose -- ``name``, ``object``, ``module`` -- and taking them
    first pushed the error literal past the term cap, so the failure slice was scored on a
    question grep had never been asked.
    """
    clone = tmp_path / "clone"
    clone.mkdir()
    _git(clone)

    terms = br.GrepSystem({REPO: clone}, {}).terms(
        Example(
            id="e",
            kind="failure",
            query="NameError name nn is not defined",
            errors=("NameError: name 'nn' is not defined",),
        )
    )

    assert terms[0] == "NameError: name 'nn' is not defined"
    assert len(terms) == br.GrepSystem.MAX_TERMS


def test_grep_coming_back_empty_is_the_finding(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "readme.md").write_text("nothing relevant\n")
    _git(clone)

    ranked, note = br.GrepSystem({REPO: clone}, {}).search(
        Example(id="e", kind="rationale", query="build_mask")
    )

    assert ranked == ()
    assert note == "grep found nothing"


def _git(path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)


# -- running the set -------------------------------------------------------


def test_only_judged_examples_reach_the_score(engine: Engine, fake: FakeGitHub) -> None:
    fake.add_issue(1, body="ValueError: backend should be defined in the mapping")
    with fake.client() as client:
        index_thread(engine, client, REPO, 1)

    data = Dataset(
        examples=(
            Example(
                id="judged",
                kind="failure",
                query="backend mapping",
                relevant=(Relevant(REPO, 1),),
                judged_by="human",
            ),
            Example(id="candidate", kind="failure", query="backend mapping"),
        ),
        corpus=(Corpus(REPO, "issue", "2026-03-01", ""),),
    )

    run = br.run_system(br.IndexSystem(engine, (REPO,)), data)

    assert [r.example_id for r in run.results] == ["judged"]
    assert run.backend is not None


# -- the caveat that is counted rather than asserted ------------------------


def test_a_report_under_a_cutoff_carries_no_caveat_it_cannot_support() -> None:
    """The note this replaces was printed on every cutoff run and said ordering was
    inflated. It is not, now that the signal tables carry dates and ``_overlap`` reads
    them -- and a caveat printed whether or not it holds is one a reader learns to skip."""
    report = br.Report(backend={"name": "x", "ranking": "y"}, cutoff={"before": "T", "exclude": []})

    assert report.as_dict()["caveats"] == []


def test_an_index_with_undated_signals_says_so_and_says_how_many() -> None:
    """What can still be true: migrated, not yet re-derived. A null date fails closed, so
    the run is strict rather than inflated and its recall is a floor -- which is a fact
    about one database at one moment, so it is counted."""
    caveat = br.undated_caveat({"thread_files": 12, "thread_symbols": 3})

    assert "thread_files 12" in caveat and "thread_symbols 3" in caveat
    assert "floor" in caveat
    assert "relored derive" in caveat


def test_undated_signals_counts_only_what_has_no_date(engine: Engine, fake: FakeGitHub) -> None:
    from sqlalchemy import update

    from relore.bench.corpus import undated_signals
    from relore.store import schema as s

    fake.add_pr(1, body="an opening naming src/mod.py")
    with fake.client() as client:
        index_thread(engine, client, REPO, 1)

    assert undated_signals(engine) == {}, "a freshly derived index has a date on every row"

    with engine.begin() as conn:
        conn.execute(update(s.thread_files).values(first_seen_at=None))

    assert undated_signals(engine) == {"thread_files": 1}
