"""``brief`` -- one page before a person reads a new thread. On both dialects (section 12).

The page is a composition of queries that already have their own tests, so these are about
what the composition can get wrong: the thread itself leaking into its own history, a
competing pull request missed because it was found through an issue rather than directly,
an empty section vanishing instead of saying so, and machine authors counted as people.
"""

from __future__ import annotations

import pytest
from fake_github import FakeGitHub
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from relore import __version__
from relore.api.server import AsOf, build_app
from relore.ingest.index_thread import index_thread
from relore.render import render_brief
from relore.wire import CLIENT_HEADER

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    fake = FakeGitHub(REPO)
    fake.add_issue(1, title="rope partial factor ignored", author="reporter", assoc="NONE")
    earlier = fake.add_pr(
        5,
        title="refactor rotary init",
        author="maint",
        assoc="OWNER",
        merged_at="2025-06-01T00:00:00Z",
    )
    fake.add_review_comment(
        earlier, 500, "partial factor has to come from the config", path="src/rope.py"
    )
    fake.add_issue(3, title="docs typo", author="alice")

    pr = fake.add_pr(10, title="fix rope partial factor", body="Fixes #1", author="alice")
    fake.add_review_comment(
        pr,
        1000,
        "read it from the config instead",
        path="src/rope.py",
        author="maint",
        assoc="OWNER",
    )
    fake.add_comment(pr, 1001, "thanks!", author="passerby")
    fake.add_comment(pr, 1002, "CI report", author="ci-bot", bot=True)

    fake.add_pr(11, title="another rope fix", body="Closes #1", author="someone")
    return fake


def _index(engine: Engine, fake: FakeGitHub) -> None:
    with fake.client() as client:
        for number in sorted(fake.threads):
            index_thread(engine, client, fake.repo, number)


def _brief(engine: Engine, number: int, *, as_of: AsOf | None = None, **params) -> dict:
    client = TestClient(build_app(engine, as_of=as_of), headers={CLIENT_HEADER: __version__})
    response = client.get(f"/api/v1/brief/{number}", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_pull_requests_brief_names_its_issue_its_rival_and_the_earlier_pr(
    engine: Engine, fake: FakeGitHub
) -> None:
    _index(engine, fake)

    body = _brief(engine, 10)

    assert body["thread"]["number"] == 10
    assert [entry["number"] for entry in body["closes"]] == [1]
    assert body["closes"][0]["thread"]["title"] == "rope partial factor ignored"
    # Found through #1, not directly: #11 never mentions #10.
    assert [claim["number"] for claim in body["competing"]["claims"]] == [11]
    history = [t["number"] for t in body["file_history"]["threads"]]
    assert history == [5]
    assert body["file_history"]["threads"][0]["merged"] is True


def test_the_thread_never_appears_in_its_own_history(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)

    body = _brief(engine, 10)

    listed = [
        *[c["number"] for c in body["competing"]["claims"]],
        *[t["number"] for t in body["file_history"]["threads"]],
        *[t["number"] for t in body["similar"]["threads"]],
    ]
    assert 10 not in listed
    assert len(listed) == len(set(listed)), "a thread is named in one section only"


def test_people_are_counted_by_tier_and_machines_are_not_people(
    engine: Engine, fake: FakeGitHub
) -> None:
    _index(engine, fake)

    people = _brief(engine, 10)["people"]

    assert {p["author"] for p in people["listed"]} == {"maint", "passerby"}
    assert people["listed"][0]["author"] == "maint", "write access first"
    assert people["authoritative"] == 1
    assert people["machine_comments"] == 1


def test_the_opener_record_excludes_the_thread_itself(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)

    opener = _brief(engine, 10)["opener"]

    assert (opener["author"], opener["issues"], opener["prs"]) == ("alice", 1, 0)


def test_an_issues_brief_lists_what_claims_to_close_it(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)

    body = _brief(engine, 1)

    assert {claim["number"] for claim in body["competing"]["claims"]} == {10, 11}
    assert body["closes"] == []


def test_empty_sections_say_so(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)

    text = render_brief(_brief(engine, 3))

    assert "IN FLIGHT" in text and "none." in text
    assert "replies: nobody but the author" in text


def test_the_rendered_page_quotes_retrieved_text(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)

    text = _brief(engine, 10, render="true")["rendered"]

    assert "> fix rope partial factor" in text
    assert "> partial factor has to come from the config" in text
    assert "@maint [authoritative]" in text


def test_a_pinned_daemon_withholds_threads_from_every_section(
    engine: Engine, fake: FakeGitHub
) -> None:
    _index(engine, fake)

    body = _brief(engine, 10, as_of=AsOf(exclude=(11, 5)))

    assert body["competing"]["claims"] == []
    assert [t["number"] for t in body["file_history"]["threads"]] == []


def test_an_unindexed_thread_is_a_404(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)
    client = TestClient(build_app(engine), headers={CLIENT_HEADER: __version__})

    assert client.get("/api/v1/brief/4242").status_code == 404
