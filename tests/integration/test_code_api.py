"""The code lens over HTTP (section 1, issue #7).

Scoped like every other endpoint -- a token cannot read a tree it cannot read threads from
-- and degrading rather than failing when a repository has no clone, which is build plan
14.4(b): the conversation index must never depend on a checkout.
"""

from __future__ import annotations

import subprocess

import pytest
from fake_github import FakeGitHub
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select

from relore import __version__
from relore.api.server import build_app
from relore.api.tokens import Authenticator, Token
from relore.code.blame import blame_line
from relore.code.clone import CLONE_ARGS
from relore.ingest.index_thread import index_thread
from relore.render import render_why
from relore.store import schema as s
from relore.wire import CLIENT_HEADER

REPO = "owner/name"

BODY = """\
def compute_default_rope_parameters(config):
    dim = int(config.head_dim * config.partial_rotary_factor)
    return dim
"""

DIVERGED = BODY.replace(" * config.partial_rotary_factor", "")


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO)


@pytest.fixture
def clone_root(tmp_path):
    """A real git repository, because the endpoints report HEAD."""
    root = tmp_path / "clones"
    tree = root / REPO.replace("/", "__")
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "modeling_llama.py").write_text(BODY)
    (tree / "src" / "modeling_gpt_neox_japanese.py").write_text(DIVERGED)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "initial"],
    ):
        subprocess.run(argv, cwd=tree, check=True, capture_output=True)
    return str(root)


@pytest.fixture
def client(engine: Engine, fake: FakeGitHub, clone_root) -> TestClient:
    with fake.client() as github:
        index_thread(engine, github, REPO, 1) if fake.threads else None
    return TestClient(
        build_app(engine, clone_root=clone_root), headers={CLIENT_HEADER: __version__}
    )


@pytest.fixture(autouse=True)
def indexed(engine: Engine, fake: FakeGitHub):
    """`_one_repo` resolves against what is indexed, so the repo has to exist as history."""
    fake.add_pr(1, title="the refactor")
    with fake.client() as github:
        index_thread(engine, github, REPO, 1)


def test_copies_groups_the_definitions_so_the_outlier_is_the_answer(client) -> None:
    response = client.get(
        "/api/v1/code/copies", params={"symbol": "compute_default_rope_parameters"}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 2
    assert [group["count"] for group in payload["groups"]] == [1, 1]
    assert payload["head"]


def test_copies_can_be_asked_for_the_exact_text_instead(client) -> None:
    """Grouping by what the body does is the default (issue #35); grouping by its text is
    still a real question, so it is a flag rather than something you cannot ask for."""
    params = {"symbol": "compute_default_rope_parameters", "exact": "true"}
    payload = client.get("/api/v1/code/copies", params=params).json()

    assert payload["exact"] is True
    assert all(group["bodies"] == 1 for group in payload["groups"])
    assert all(copy["shape_hash"] == copy["body_hash"] for copy in payload["groups"][0]["copies"])


def test_grep_answers_the_pattern_question(client) -> None:
    response = client.get(
        "/api/v1/code/grep",
        params={"pattern": "partial_rotary_factor", "path": "src/*.py"},
    )

    payload = response.json()
    assert [hit["path"] for hit in payload["hits"]] == ["src/modeling_llama.py"]
    # Both denominators, as separate fields (issue #37): one file matched, two were read,
    # and a caller given only the second reads it as the first.
    assert payload["files_with_hits"] == 1
    assert payload["files_searched"] == 2


def test_symbol_returns_a_body_and_says_how_many_there_are(client) -> None:
    """Two copies here, 38 in `transformers`: serving one as *the* body without the count
    is a wrong answer the caller cannot see."""
    response = client.get(
        "/api/v1/code/symbol", params={"qualname": "compute_default_rope_parameters"}
    )

    payload = response.json()
    assert payload["definitions_total"] == 2
    assert payload["path"].startswith("src/")
    assert "def compute_default_rope_parameters" in payload["body"]


def test_symbol_with_a_qualifier_no_class_defines_is_a_404(client) -> None:
    """Not another class's method of the same name (#59)."""
    tree = client.app.state.deps.clones.path(REPO)
    (tree / "src" / "model.py").write_text(
        "class Model:\n    def forward(self):\n        return 1\n"
    )

    response = client.get("/api/v1/code/symbol", params={"qualname": "Missing.forward"})

    assert response.status_code == 404
    assert "Missing.forward is not defined in" in response.json()["detail"]


def test_defs_and_refs_answer_from_the_clone(client) -> None:
    defs = client.get("/api/v1/code/defs", params={"path": "src/modeling_llama.py"}).json()
    assert [d["qualname"] for d in defs["definitions"]] == ["compute_default_rope_parameters"]

    refs = client.get(
        "/api/v1/code/refs", params={"symbol": "compute_default_rope_parameters"}
    ).json()
    assert len(refs["hits"]) == 2


# -- why PATH:LINE (issue #9) ----------------------------------------------


def test_why_names_the_pull_request_that_last_changed_the_line(client, engine, fake) -> None:
    """`git blame` gives the commit; the index gives the argument. The link is
    `thread_commits`, and the squash-merge subject is the fallback."""
    response = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["blame"]["sha"]
    assert "partial_rotary_factor" in payload["blame"]["text"]
    # Nothing in this index carries that commit, and saying so is the answer -- not an
    # error, and not a silently empty page.
    assert payload["number"] is None


def test_why_resolves_the_pull_request_from_a_staged_commit(client, engine, fake) -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=f"{client.app.state.deps.clones.path(REPO)}",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with engine.begin() as conn:
        conn.execute(
            s.thread_commits.insert().values(
                thread_id=conn.execute(select(s.threads.c.id)).scalar_one(),
                sha=sha,
                message="the refactor",
            )
        )

    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2}).json()

    assert payload["number"] == 1
    assert payload["resolved_by"] == "commit"
    assert payload["thread"]["title"] == "the refactor"


def test_a_query_never_fetches_from_the_remote(monkeypatch, clone_root) -> None:
    """Blobless was the first answer and `why` paid for it: blame walks back through a
    file's history, so every query fetched from the remote mid-answer, and on `transformers`
    the walk reached objects it would not serve at all. Both halves are the property -- a
    clone that carries its blobs, and a blame that refuses to fetch if one somehow does not.
    """
    assert not [arg for arg in CLONE_ARGS if arg.startswith("--filter")]

    seen: dict[str, str] = {}
    real = subprocess.run

    def record(argv, **kwargs):
        seen.update(kwargs.get("env") or {})
        return real(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    blame_line(f"{clone_root}/{REPO.replace('/', '__')}", "src/modeling_llama.py", 2)

    assert seen["GIT_NO_LAZY_FETCH"] == "1"


def test_why_carries_the_origin_chain_and_the_word_it_pickaxed(client) -> None:
    """The composed answer (huggingface/relore#57). `git log -L` follows a range and loses a
    rewritten block; this follows a word. Both lists are on the page, separately, and the
    word is named so the choice can be argued with rather than taken."""
    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2}).json()

    assert "origin" in payload
    assert payload["origin_term"] == "" or payload["origin_term"]
    # Whatever it found or did not, it says what it tried: a silent absence and "there was
    # nothing to find" are the two answers this verb keeps having to tell apart.
    assert [c["term"] for c in payload["origin_considered"]]


def test_the_origin_chain_renders_as_a_different_question_from_the_revisions(client) -> None:
    """A reader who cannot tell them apart cannot tell "this line was reformatted eight
    times" from "this behaviour was argued here"."""
    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2}).json()
    payload["origin_term"] = "partial_rotary_factor"
    payload["origin"] = [{"sha": "abc123abc123", "date": "2024-03-06", "summary": "origin (#7)"}]

    out = render_why(payload)

    assert "changed `partial_rotary_factor` in this file" in out
    assert "follows the word, not the line" in out


def test_a_review_comment_carries_a_date_and_not_only_an_age(client, engine, fake) -> None:
    """huggingface/relore#66. `why` is the verb most often asked for "the comment, with its
    author and its date", and its own revision rows have carried a date since #57 -- so
    without this the page was inconsistent with itself, and an agent bounded the date from
    the merge commit and said so."""
    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2}).json()

    for group in ("anchored", "on_file", "reviews"):
        for entry in payload.get(group) or []:
            assert "age" in entry, "the age is what the page prints"
            assert "date" in entry, "and this is what a citation needs"


def test_why_on_a_line_that_is_not_there_says_blame_has_nothing_to_read(client) -> None:
    response = client.get("/api/v1/why", params={"path": "src/nope.py", "line": 4})

    assert response.status_code == 404
    assert "blame has nothing to read" in response.json()["detail"]


def test_a_repository_with_no_clone_says_so_rather_than_failing(engine: Engine, tmp_path) -> None:
    """14.4(b): the lens degrades, the index does not depend on it."""
    client = TestClient(
        build_app(engine, clone_root=str(tmp_path / "empty")), headers={CLIENT_HEADER: __version__}
    )

    response = client.get("/api/v1/code/grep", params={"pattern": "x"})

    assert response.status_code == 503
    assert "no working clone" in response.json()["detail"]
    assert "relored clone" in response.json()["detail"]


def test_a_token_cannot_read_a_tree_it_cannot_read_threads_from(engine: Engine, clone_root) -> None:
    auth = Authenticator(tokens=(Token(name="agent", secret="tok", repos=("other/repo",)),))
    client = TestClient(
        build_app(engine, auth=auth, clone_root=clone_root),
        headers={CLIENT_HEADER: __version__, "authorization": "Bearer tok"},
    )

    response = client.get("/api/v1/code/grep", params={"pattern": "x", "repo": REPO})

    assert response.status_code == 404


# -- the widening, and the line's history (huggingface/relore#57, #63, #64) -------


@pytest.fixture
def argued(engine: Engine, fake: FakeGitHub, clone_root):
    """A line whose argument is one level up, which is where arguments usually are.

    The line carries no comment of its own; #1 holds a review comment forty lines away and
    an approving review whose body states the condition. Both are invisible to a window
    around the line, and the second is invisible to any window at all.
    """
    thread = fake.threads[1]
    fake.add_review_comment(
        thread, 900, "this branch is the compile path", path="src/modeling_llama.py", line=400
    )
    fake.add_review(thread, 901, "approving, but check the fullgraph case", state="APPROVED")
    with fake.client() as github:
        index_thread(engine, github, REPO, 1)
    with engine.begin() as conn:
        conn.execute(
            s.thread_commits.insert().values(
                thread_id=conn.execute(select(s.threads.c.id)).scalar_one(),
                sha=subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=f"{clone_root}/{REPO.replace('/', '__')}",
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
                message="the refactor",
            )
        )
    return thread


def test_why_widens_past_the_line_and_says_which_level_answered(client, argued) -> None:
    """`0 comments on this line` is a fact about a window and reads as a fact about the
    code. The levels past it are returned, and named (#64)."""
    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 2}).json()

    assert payload["anchored"] == []
    assert [c["line"] for c in payload["on_file"]] == [400]
    assert payload["level"] == "file"
    # A review body carries no line anchor, so no widening of a *window* reaches it (#63).
    assert [r["text"] for r in payload["reviews"]] == ["approving, but check the fullgraph case"]
    assert payload["reviews"][0]["state"] == "APPROVED"


def test_why_returns_the_lines_revisions_not_only_its_last_touch(client, clone_root) -> None:
    """Blame names the last commit; on a reformatted line that is a refactor standing in
    front of the pull request that decided anything (#57)."""
    tree = f"{clone_root}/{REPO.replace('/', '__')}"
    edited = BODY.replace("    return dim", "    return dim  # keep the partial factor")
    (__import__("pathlib").Path(tree) / "src" / "modeling_llama.py").write_text(edited)
    for argv in (["git", "add", "-A"], ["git", "commit", "-qm", "tidy the return (#77)"]):
        subprocess.run(argv, cwd=tree, check=True, capture_output=True)

    payload = client.get("/api/v1/why", params={"path": "src/modeling_llama.py", "line": 3}).json()

    summaries = [commit["summary"] for commit in payload["history"]]
    assert summaries == ["tidy the return (#77)", "initial"]
    # The squash-merge convention is read here too, so a commit no per-PR pass staged still
    # names its pull request rather than reporting nothing.
    assert payload["history"][0]["number"] == 77
