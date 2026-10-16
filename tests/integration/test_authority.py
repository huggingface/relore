"""Section 6.2: `MEMBER` is not authority, and cannot be treated as either yes or no.

The failure this suite exists for was measured, not imagined. On `huggingface/serge` the
`authoritative` tier held **zero** of 415 documents, because both maintainers report as
`MEMBER` -- their write access comes through a team rather than a direct collaborator
invite. A rationale query with an `authoritative` floor would have returned nothing at all.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_github import FakeGitHub, FakeGraphQL
from sqlalchemy import Engine, select

from relore.ingest.authority import resolve_authority
from relore.ingest.backfill import backfill
from relore.store import repository as repo_layer
from relore.store import schema as s

REPO = "owner/name"


@pytest.fixture
def fake() -> FakeGitHub:
    """One maintainer who reports as MEMBER, one who does not, and a passer-by.

    `zoe` is the case the whole section exists for: an org member with real write access
    whom `author_association` alone cannot distinguish from the other thousands.
    """
    fake = FakeGitHub(REPO)
    issue = fake.add_issue(1, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(issue, 100, "that is intentional", author="zoe", assoc="MEMBER")
    fake.add_comment(issue, 101, "me too", author="passerby", assoc="CONTRIBUTOR")
    fake.add_comment(issue, 102, "nightly run failed", author="ci", assoc="CONTRIBUTOR", bot=True)

    pr = fake.add_pr(
        2, title="fix it", author="zoe", assoc="MEMBER", merged_at="2026-02-02T00:00:00Z"
    )
    pr.files = ["src/mod.py"]
    pr.merged_by = "zoe"  # the fallback's whole signal: merging is the write act
    fake.add_review_comment(pr, 300, "belongs in the parent", author="owner", assoc="OWNER")
    return fake


def _index(engine: Engine, fake: FakeGitHub, *, graphql: bool = True) -> None:
    """Everything but the authority pass, so each test drives that itself."""
    gql = FakeGraphQL(fake).client() if graphql else None
    with fake.client() as client:
        try:
            backfill(engine, client, REPO, gql=gql, authority=False)
        finally:
            if gql:
                gql.close()


def _trust_of(engine: Engine, author: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row.trust
            for row in conn.execute(
                select(s.documents.c.trust).where(s.documents.c.author == author)
            )
        }


def _resolve(engine: Engine, fake: FakeGitHub, **kwargs) -> object:
    with fake.client() as client:
        return resolve_authority(engine, REPO, client=client, **kwargs)


# -- the tiers -------------------------------------------------------------


def test_an_unresolved_member_is_reported(engine: Engine, fake: FakeGitHub) -> None:
    """Fail closed: until write access is confirmed, an org member is making a claim."""
    _index(engine, fake)
    assert _trust_of(engine, "zoe") == {"reported"}


def test_a_member_with_confirmed_write_access_is_authoritative(
    engine: Engine, fake: FakeGitHub
) -> None:
    _index(engine, fake)
    fake.permissions["zoe"] = "write"

    result = _resolve(engine, fake)

    assert result.authoritative == 1
    assert result.promoted == {"zoe"}
    assert _trust_of(engine, "zoe") == {"authoritative"}


def test_an_owner_is_authoritative_without_costing_a_request(
    engine: Engine, fake: FakeGitHub
) -> None:
    """OWNER and COLLABORATOR are unambiguous, so resolution never asks about them."""
    _index(engine, fake)
    fake.requests.clear()

    _resolve(engine, fake)

    assert _trust_of(engine, "owner") == {"authoritative"}
    assert not [p for p in fake.requests if "/collaborators/owner/" in p]


def test_a_human_contributor_stays_reported(engine: Engine, fake: FakeGitHub) -> None:
    _index(engine, fake)
    fake.permissions["passerby"] = "write"  # never asked: not a MEMBER

    _resolve(engine, fake)

    assert _trust_of(engine, "passerby") == {"reported"}
    assert not [p for p in fake.requests if "/collaborators/passerby/" in p]


def test_a_bot_is_machine_whatever_its_association(engine: Engine, fake: FakeGitHub) -> None:
    """`user.type` decides, so a bot carrying CONTRIBUTOR is still excluded."""
    _index(engine, fake)
    _resolve(engine, fake)
    assert _trust_of(engine, "ci") == {"machine"}


def test_triage_and_read_are_not_write_access(engine: Engine, fake: FakeGitHub) -> None:
    """The tier is the *write* act. Being able to label an issue is not deciding one."""
    _index(engine, fake)
    fake.permissions["zoe"] = "triage"

    result = _resolve(engine, fake)

    assert result.authoritative == 0
    assert _trust_of(engine, "zoe") == {"reported"}


# -- the reconcile trap ----------------------------------------------------


def test_a_promotion_rewrites_the_tier_though_no_byte_of_text_changed(
    engine: Engine, fake: FakeGitHub
) -> None:
    """The failure this whole change turns on.

    `trust` is derived from the payload *plus* `repo_authority`, so resolving an author
    changes the tier while `content_hash` stays identical. A reconcile comparing hashes
    alone reports "unchanged" and leaves the stale tier in place for ever -- and section
    6.2 requires a rebuild to reproduce the tier.
    """
    _index(engine, fake)
    with engine.connect() as conn:
        before = dict(
            conn.execute(
                select(s.documents.c.id, s.documents.c.content_hash).where(
                    s.documents.c.author == "zoe"
                )
            ).all()
        )
    fake.permissions["zoe"] = "admin"

    result = _resolve(engine, fake)

    assert result.documents_written > 0, "the promotion must reach the documents"
    assert _trust_of(engine, "zoe") == {"authoritative"}
    with engine.connect() as conn:
        after = dict(
            conn.execute(
                select(s.documents.c.id, s.documents.c.content_hash).where(
                    s.documents.c.author == "zoe"
                )
            ).all()
        )
    assert after == before, "the text did not change, so neither may the hash"


def test_resolving_twice_writes_no_documents(engine: Engine, fake: FakeGitHub) -> None:
    """Idempotency survives the new comparison: a settled tier is a no-op."""
    _index(engine, fake)
    fake.permissions["zoe"] = "write"
    _resolve(engine, fake)

    again = _resolve(engine, fake, refresh_after=dt.timedelta(0))

    assert again.documents_written == 0
    assert again.promoted == set()


def test_a_rebuild_reproduces_the_tier(engine: Engine, fake: FakeGitHub) -> None:
    """Section 6.2: derive at derive time, so `derive` alone puts the tier back."""
    _index(engine, fake)
    fake.permissions["zoe"] = "write"
    _resolve(engine, fake)

    from relore.ingest.index_thread import derive_thread

    with engine.begin() as conn:
        conn.execute(s.documents.update().values(trust="reported"))
        for number in (1, 2):  # zoe wrote on both
            derive_thread(conn, REPO, number)

    assert _trust_of(engine, "zoe") == {"authoritative"}


# -- the fallback ----------------------------------------------------------


def test_a_token_without_push_access_falls_back_to_merged_by(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Merging is the write act itself, so a positive answer here is definitive."""
    _index(engine, fake)
    fake.permission_endpoint_forbidden = True

    result = _resolve(engine, fake)

    assert result.endpoint_available is False
    assert result.sources["merged_by"] == 1
    assert _trust_of(engine, "zoe") == {"authoritative"}


def test_a_forbidden_endpoint_is_asked_about_once_not_once_per_author(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A token without push access fails identically for every login, so spending one
    doomed request per author buys nothing."""
    fake.add_comment(fake.threads[1], 103, "and another", author="yves", assoc="MEMBER")
    fake.add_comment(fake.threads[1], 104, "and one more", author="xena", assoc="MEMBER")
    _index(engine, fake)
    fake.permission_endpoint_forbidden = True
    fake.requests.clear()

    result = _resolve(engine, fake)

    asked = [p for p in fake.requests if "/collaborators/" in p]
    assert len(asked) == 1
    assert result.candidates >= 3


def test_offline_resolution_needs_no_client_at_all(engine: Engine, fake: FakeGitHub) -> None:
    """The merged_by half is local, so an operator who would rather not spend the
    requests still gets the maintainers who have merged something."""
    _index(engine, fake)

    result = resolve_authority(engine, REPO, client=None)

    assert result.sources["merged_by"] == 1
    assert _trust_of(engine, "zoe") == {"authoritative"}


def test_a_login_that_is_not_a_collaborator_is_an_answer_not_a_failure(
    engine: Engine, fake: FakeGitHub
) -> None:
    """404 means "not a collaborator", and must not trip the fallback for everyone else."""
    _index(engine, fake)
    fake.permissions.clear()  # every lookup 404s

    result = _resolve(engine, fake)

    assert result.endpoint_available is True
    assert result.sources["permission_api"] >= 1
    assert _trust_of(engine, "zoe") == {"reported"}


# -- the refresh window ----------------------------------------------------


def test_a_freshly_checked_author_is_not_asked_about_again(
    engine: Engine, fake: FakeGitHub
) -> None:
    """People gain and lose access rarely; re-resolving belongs on a slow schedule."""
    _index(engine, fake)
    fake.permissions["zoe"] = "write"
    _resolve(engine, fake)
    fake.requests.clear()

    result = _resolve(engine, fake)

    assert result.skipped_fresh >= 1
    assert not [p for p in fake.requests if "/collaborators/" in p]


def test_a_pass_that_skipped_everyone_still_reports_who_has_access(
    engine: Engine, fake: FakeGitHub
) -> None:
    """A healthy pass must not read like a failure.

    Counting only what *this* pass resolved makes a fully-resolved repository report
    "0 of 3 have write access" on every subsequent run -- the silent-success class
    inverted, and the operator would go looking for a bug that is not there.
    """
    _index(engine, fake)
    fake.permissions["zoe"] = "write"
    _resolve(engine, fake)

    again = _resolve(engine, fake)

    assert again.resolved == 0 and again.skipped_fresh == 1
    assert again.authoritative == 1


def test_re_resolution_can_move_a_document_back_down_a_tier(
    engine: Engine, fake: FakeGitHub
) -> None:
    """Standing is mutable (section 6.2's caveat). A maintainer who loses access is
    correctly demoted -- that is the table answering "today", not drift."""
    _index(engine, fake)
    fake.permissions["zoe"] = "write"
    _resolve(engine, fake)
    assert _trust_of(engine, "zoe") == {"authoritative"}

    fake.permissions["zoe"] = "read"
    _resolve(engine, fake, refresh_after=dt.timedelta(0))

    assert _trust_of(engine, "zoe") == {"reported"}


def test_only_member_authors_are_ever_candidates(engine: Engine, fake: FakeGitHub) -> None:
    """What bounds the pass to a few hundred requests on a repository with 47k threads."""
    _index(engine, fake)
    with engine.connect() as conn:
        assert repo_layer.member_authors(conn, REPO) == ["zoe"]


# -- CI directives ---------------------------------------------------------


def _trust_of_body(engine: Engine, needle: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row.trust
            for row in conn.execute(
                select(s.documents.c.trust).where(s.documents.c.body_text.like(f"%{needle}%"))
            )
        }


def test_a_ci_directive_is_machine_however_senior_its_author(
    engine: Engine, fake: FakeGitHub
) -> None:
    """`run-slow: whisper` fires the GPU suite. It is perfectly believable and says nothing,
    and it is maintainers who send it -- so without this it arrives at the top tier and
    outranks real discussion."""
    issue = fake.add_issue(9, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(issue, 900, "run-slow: whisper, gemma4", author="owner", assoc="OWNER")
    _index(engine, fake)
    _resolve(engine, fake)

    assert _trust_of_body(engine, "run-slow") == {"machine"}


def test_the_author_of_a_directive_is_not_marked_a_bot(engine: Engine, fake: FakeGitHub) -> None:
    """The document is demoted, not the person: their other comments are still evidence."""
    issue = fake.add_issue(9, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(issue, 900, "run-slow: whisper", author="owner", assoc="OWNER")
    fake.add_comment(issue, 901, "deliberate, see the adapter", author="owner", assoc="OWNER")
    _index(engine, fake)
    _resolve(engine, fake)

    with engine.connect() as conn:
        flags = (
            conn.execute(select(s.documents.c.author_is_bot).where(s.documents.c.author == "owner"))
            .scalars()
            .all()
        )
    assert flags and not any(flags)
    assert _trust_of_body(engine, "see the adapter") == {"authoritative"}


def test_a_directive_followed_by_prose_stays_human(engine: Engine, fake: FakeGitHub) -> None:
    """The tail has to be an identifier list. Someone who triggers CI *and* explains why is
    contributing the explanation."""
    issue = fake.add_issue(9, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(
        issue,
        900,
        "run-slow: whisper\n\nThe cache reorder is skipped here (see #41).",
        author="owner",
        assoc="OWNER",
    )
    _index(engine, fake)
    _resolve(engine, fake)

    assert _trust_of_body(engine, "cache reorder") == {"authoritative"}


def test_a_bot_command_is_machine(engine: Engine, fake: FakeGitHub) -> None:
    issue = fake.add_issue(9, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(issue, 900, "@bot /style", author="zoe", assoc="MEMBER")
    _index(engine, fake)
    _resolve(engine, fake)

    assert _trust_of_body(engine, "/style") == {"machine"}


def test_prose_that_merely_mentions_a_directive_is_untouched(
    engine: Engine, fake: FakeGitHub
) -> None:
    issue = fake.add_issue(9, title="a bug", body="it crashes", author="stranger", assoc="NONE")
    fake.add_comment(
        issue, 900, "We should run-slow this before merging", author="owner", assoc="OWNER"
    )
    _index(engine, fake)
    _resolve(engine, fake)

    assert _trust_of_body(engine, "before merging") == {"authoritative"}
