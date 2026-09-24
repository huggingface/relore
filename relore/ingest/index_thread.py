"""``index_thread`` -- the only mutation primitive (section 5.1).

Fetch the thread's canonical current state, stage it verbatim, build the desired document
set, reconcile against what is stored, **in one transaction**. Idempotent by
construction: a label change rewrites one ``threads`` row and touches no document.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, Engine

from relore.github.client import GitHubClient
from relore.github.fetch_thread import AUTHORITATIVE_TYPES, ThreadPayload, fetch_thread
from relore.ingest.authority import WRITE_PERMISSIONS, bot_accounts
from relore.ingest.chunk import chunk, content_hash
from relore.ingest.extract import extract_signals
from relore.ingest.normalize import normalize, redact
from relore.ingest.relationships import extract_links
from relore.ingest.timestamps import parse_timestamp, text_existed_at
from relore.ingest.versions import CHUNKING_VERSION, EXTRACTOR_VERSION
from relore.store import repository as repo_layer
from relore.store.dialect import utcnow

log = logging.getLogger(__name__)

#: The staged object types that *are* a thread. Everything else staged under a thread
#: number is a child of one, and a number with only children cannot be derived at all --
#: which is a real state, not a corruption: 20 of the 47,928 numbers staged for
#: `huggingface/transformers` are comments whose issue was deleted or transferred upstream
#: after the comment walk saw it. `relored derive` reads this to pick what it can derive
#: and reports the rest; `relored sweep` is what prunes them.
THREAD_HEAD_TYPES = ("issue", "pr")


@dataclass
class IndexResult:
    repo: str
    number: int
    thread_id: int
    stats: repo_layer.ReconcileStats
    updated_at: dt.datetime | None
    raw_pruned: int = 0
    redactions: Counter[str] = field(default_factory=Counter)
    signals: repo_layer.SignalStats = field(default_factory=repo_layer.SignalStats)

    @property
    def wrote(self) -> int:
        """Document writes. Signal writes are reported separately (``signals.wrote``):
        the section 5.1 idempotency claim is about documents, and folding a second
        number into it would make a green assertion mean less than it does."""
        return self.stats.wrote


def index_thread(engine: Engine, client: GitHubClient, repo: str, number: int) -> IndexResult:
    """Fetch (network) then stage-and-derive (one transaction)."""
    payload = fetch_thread(client, repo, number)
    with engine.begin() as conn:
        return index_payload(conn, payload)


def index_payload(conn: Connection, payload: ThreadPayload) -> IndexResult:
    """The transactional half, separated so tests can drive it without a network."""
    now = utcnow()
    objects = payload.raw_objects()
    repo_layer.stage_raw(
        conn,
        payload.repo,
        [
            {
                "repo": payload.repo,
                "object_type": object_type,
                "object_id": object_id,
                "thread_number": payload.number,
                "payload": raw,
                "fetched_at": now,
                "github_updated_at": parse_timestamp(raw.get("updated_at")),
            }
            for object_type, object_id, raw in objects
        ],
    )
    # The fetch is the authority on what this thread currently contains, so anything
    # staged and not returned has been deleted on GitHub. Prune before deriving, or the
    # derive faithfully rebuilds a document for a comment that no longer exists.
    pruned = repo_layer.prune_raw(
        conn,
        payload.repo,
        payload.number,
        keep={(object_type, object_id) for object_type, object_id, _raw in objects},
        object_types=AUTHORITATIVE_TYPES,
    )
    result = derive_thread(conn, payload.repo, payload.number)
    result.raw_pruned = pruned
    return result


def derive_thread(
    conn: Connection, repo: str, number: int, *, authority: Mapping[str, str] | None = None
) -> IndexResult:
    """Rebuild one thread's rows from ``raw_objects``. No network, re-runnable at will.

    ``authority`` is section 6.2's resolved write access, ``login -> permission``. Pass it
    when deriving many threads in a row -- it is the same few hundred rows every time --
    and leave it None to have this read it per thread.
    """
    raw = repo_layer.load_raw(conn, repo, number)
    thread_raw = _sole(raw.get("pr") or raw.get("issue"))
    if thread_raw is None:
        raise LookupError(f"{repo}#{number} is not staged; fetch it first")
    thread_type = "pr" if raw.get("pr") else "issue"

    if authority is None:
        authority = repo_layer.get_authority(conn, repo)

    detail = _sole(raw.get("pr_details"))
    thread_row, labels = _thread_row(
        repo, number, thread_type, thread_raw, detail, reviews=raw.get("review") or []
    )
    thread_id = repo_layer.upsert_thread(conn, thread_row)
    repo_layer.set_labels(conn, thread_id, labels)

    documents, redactions, sources = build_documents(number, thread_raw, raw, authority=authority)
    stats = repo_layer.reconcile_documents(conn, thread_id, documents)
    # Section 5.3, in the same transaction as the documents it was extracted from: a
    # thread whose signals belong to a body it no longer has is worse than one with no
    # signals, because the `--file`/`--error` filters would answer from it.
    rows = extract_signals(sources, detail=detail, merged_at=thread_row["merged_at"]).rows()
    # Section 13.3: the claim comes from this thread alone, the resolution from the index.
    # A target nobody has indexed yet is stored unresolved rather than dropped -- an open
    # pull request closing an unindexed issue is the *normal* case on a sampled index, and
    # dropping the edge would answer "is anyone fixing this?" with silence.
    rows["links"] = _resolve_links(
        conn, repo, extract_links(thread_type, number, documents, detail, repo=repo)
    )
    signals = repo_layer.reconcile_signals(conn, thread_id, rows)
    return IndexResult(
        repo=repo,
        number=number,
        thread_id=thread_id,
        stats=stats,
        updated_at=thread_row["updated_at"],
        redactions=redactions,
        signals=signals,
    )


def _resolve_links(
    conn: Connection, repo: str, links: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fill in ``target_thread_id`` for the targets that are in the index."""
    if not links:
        return links
    ids = repo_layer.thread_ids_by_number(conn, repo, [link["target_number"] for link in links])
    return [{**link, "target_thread_id": ids.get(link["target_number"])} for link in links]


# -- rows ------------------------------------------------------------------


def _thread_row(
    repo: str,
    number: int,
    thread_type: str,
    raw: dict[str, Any],
    detail: dict[str, Any] | None = None,
    reviews: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build the ``threads`` row.

    ``detail`` is the GraphQL node from the per-PR pass, when it has run: the bulk
    ``/issues`` walk carries no diff stats and no ``merged_by``. It is a separate raw
    object rather than merged into the thread payload, because section 4 stores every
    object verbatim as fetched and two endpoints' output in one row is neither.

    ``reviews`` are the staged review objects, read only for the decision they add up to.
    """
    reviews = reviews or []
    labels = [
        label["name"] if isinstance(label, dict) else str(label)
        for label in raw.get("labels") or []
    ]
    row = {
        "repo": repo,
        "github_number": number,
        "thread_type": thread_type,
        "work_kind": None,  # section 9; inferred from changed-file topology, milestone 4
        "title": raw.get("title") or "",
        "author": (raw.get("user") or {}).get("login"),
        "state": raw.get("state"),
        "url": raw.get("html_url"),
        "created_at": parse_timestamp(raw.get("created_at")),
        "updated_at": parse_timestamp(raw.get("updated_at")),
        "closed_at": parse_timestamp(raw.get("closed_at")),
        # Three shapes, one per pass: GET /pulls/{n} puts merged_at at the top level, the
        # bulk /issues walk nests it under `pull_request`, GraphQL calls it mergedAt. A
        # backfill only ever sees the middle one, so missing it would leave every
        # backfilled PR unmerged -- and section 6.2 treats a merged PR as precedent
        # whoever wrote it.
        "merged_at": parse_timestamp(
            raw.get("merged_at")
            or (raw.get("pull_request") or {}).get("merged_at")
            or (detail or {}).get("mergedAt")
        ),
        "github_node_id": raw.get("node_id"),
        "indexed_at": utcnow(),
        "metadata": {
            key: raw[key]
            for key in (
                "author_association",
                "draft",
                "merged",
                "comments",
                "review_comments",
                "commits",
                "additions",
                "deletions",
                "changed_files",
                "locked",
            )
            if key in raw
        },
    }
    if detail:
        row["metadata"].update(
            {
                key: detail[graphql_key]
                for key, graphql_key in (
                    ("additions", "additions"),
                    ("deletions", "deletions"),
                    ("changed_files", "changedFiles"),
                )
                if detail.get(graphql_key) is not None
            }
        )
        merged_by = (detail.get("mergedBy") or {}).get("login")
        if merged_by:
            # Section 6.2's authority fallback: merging is the write act itself, so this
            # is definitive where `author_association` is only a proxy -- and it is
            # available nowhere cheaper than this pass.
            row["metadata"]["merged_by"] = merged_by
        if detail.get("truncated_files") or detail.get("truncated_commits"):
            row["metadata"]["graphql_truncated"] = True
    row["metadata"].update(_events(raw, detail, reviews))
    return row, labels


# -- events (issue #22) -----------------------------------------------------

# COMMENTED and PENDING decide nothing -- GitHub's own rule.
_DECISIVE_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})


def _events(
    raw: dict[str, Any], detail: dict[str, Any] | None, reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    """A closure is an event, not a comment, so no ranking finds it (issue #22).

    Every field comes from a payload :func:`relore.github.fetch_thread.fetch_thread`
    already stages, so a thread the poll has touched carries them and one known only from
    a backfill's bulk ``/issues`` walk does not. Keys are omitted rather than nulled.
    """
    out: dict[str, Any] = {}
    if raw.get("state_reason"):
        out["state_reason"] = raw["state_reason"]
    closed_by = (raw.get("closed_by") or {}).get("login")
    if closed_by:
        out["closed_by"] = closed_by
    requested = [
        login
        for reviewer in raw.get("requested_reviewers") or []
        if (login := (reviewer or {}).get("login"))
    ]
    if requested:
        out["requested_reviewers"] = requested

    decision, deciders = _review_decision(reviews, detail)
    if decision:
        out["review_decision"] = decision
        out["review_decision_by"] = deciders
    return out


def _review_decision(
    reviews: list[dict[str, Any]], detail: dict[str, Any] | None
) -> tuple[str | None, list[str]]:
    """Latest decisive review per reviewer; one ``changes_requested`` outranks any approvals.

    Computed rather than read from GraphQL's ``reviewDecision``, which the per-PR pass only
    fetches for merged pull requests -- absent on exactly the threads this is asked about.
    """
    latest: dict[str, tuple[Any, str]] = {}
    for review in [*reviews, *_graphql_reviews(detail)]:
        state = str(review.get("state") or "").upper()
        if state not in _DECISIVE_REVIEW_STATES:
            continue
        login = (review.get("user") or {}).get("login")
        if not login:
            continue
        submitted = parse_timestamp(review.get("submitted_at"))
        previous = latest.get(login)
        # No timestamp sorts oldest.
        if previous is None or (
            submitted is not None and (previous[0] is None or submitted >= previous[0])
        ):
            latest[login] = (submitted, state)

    blocking = sorted(
        login for login, (_at, state) in latest.items() if state == "CHANGES_REQUESTED"
    )
    if blocking:
        return "changes_requested", blocking
    approving = sorted(login for login, (_at, state) in latest.items() if state == "APPROVED")
    if approving:
        return "approved", approving
    return None, []


def _graphql_reviews(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The per-PR pass's reviews, in the REST shape the rule above reads."""
    if not detail:
        return []
    return [_review_from_graphql(node) for node in (detail.get("reviews") or {}).get("nodes") or []]


def build_documents(
    number: int,
    thread_raw: dict[str, Any],
    raw: dict[str, list[dict[str, Any]]],
    *,
    authority: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    """One document per GitHub-native unit (section 5.4).

    Returns the documents, the redaction counts, and the **unchunked** redacted text of
    each source object. The third is what section 5.3's extraction reads. A pasted
    traceback is one paragraph with no blank line in it, so past ``MAX_CHUNK_CHARS`` the
    chunker has no natural boundary and cuts on length -- through the middle of a line. An
    error line, a node id or a path can therefore land half in one chunk and half in the
    next and be extracted from neither, while the person pasting that traceback into
    ``--error`` has all of it. The documents are the chunks; the signals are the thread's.
    """
    redactions: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    def emit(
        *,
        source_type: str,
        source_id: str,
        body: str | None,
        raw_obj: dict[str, Any],
        split: bool,
        metadata: dict[str, Any] | None = None,
        commit_sha: str | None = None,
    ) -> None:
        if not (body or "").strip():
            return
        clean, found = redact(body or "")
        redactions.update(found)
        # Before `chunk`, on purpose: see this function's docstring.
        #
        # `existed_at` is what dates the signal rows read out of this text. It is computed
        # here rather than in the extractor because the rule needs `source_type`, which is
        # a fact about the GitHub object and stops being one by the time the extractor has
        # a string.
        sources.append(
            {
                "body_markdown": clean,
                "metadata": metadata or {},
                "existed_at": text_existed_at(
                    source_type,
                    parse_timestamp(raw_obj.get("created_at")),
                    parse_timestamp(raw_obj.get("updated_at")),
                ),
            }
        )
        author = raw_obj.get("user") or {}
        is_bot, trust = _trust(author, raw_obj.get("author_association"), authority)
        for piece in chunk(clean, split=split):
            text = normalize(piece.text)
            out.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "chunk_index": piece.index,
                    "author": author.get("login"),
                    "author_assoc": raw_obj.get("author_association"),
                    "author_is_bot": is_bot,
                    "trust": trust,
                    "url": raw_obj.get("html_url"),
                    # body_markdown is verbatim apart from redaction, which is not
                    # optional: the markdown copy is what gets shown (section 11.3).
                    "body_markdown": piece.text,
                    "body_text": text,
                    "content_hash": content_hash(text, metadata or {}),
                    "chunking_version": CHUNKING_VERSION,
                    "extractor_version": EXTRACTOR_VERSION,
                    "embedding_model": None,
                    "github_created_at": parse_timestamp(raw_obj.get("created_at")),
                    "github_updated_at": parse_timestamp(raw_obj.get("updated_at")),
                    "commit_sha": commit_sha,
                    "enclosing_symbol": None,  # the code lens fills this, milestone 4
                    "metadata": metadata or {},
                }
            )

    # A title is one document and is never split.
    emit(
        source_type="title",
        source_id=str(number),
        body=thread_raw.get("title"),
        raw_obj=thread_raw,
        split=False,
    )
    emit(
        source_type="body",
        source_id=str(number),
        body=thread_raw.get("body"),
        raw_obj=thread_raw,
        split=True,
    )
    for comment in raw.get("issue_comment", []):
        emit(
            source_type="issue_comment",
            source_id=comment["_object_id"],
            body=comment.get("body"),
            raw_obj=comment,
            split=True,
        )
    for review in raw.get("review", []):
        # A review submission with no body is a bare approval: real, but not a document.
        emit(
            source_type="review",
            source_id=review["_object_id"],
            body=review.get("body"),
            raw_obj=review,
            split=True,
            metadata={"state": review.get("state")},
            commit_sha=review.get("commit_id"),
        )
    # Reviews the per-PR GraphQL pass saw and the REST pass did not. A backfill has no
    # per-thread REST reviews at all (there is no repo-wide reviews endpoint), so on that
    # path this is the only source of review bodies.
    rest_reviews = {str(review["_object_id"]) for review in raw.get("review", [])}
    for detail in raw.get("pr_details", []):
        for node in (detail.get("reviews") or {}).get("nodes") or []:
            source_id = str(node.get("databaseId") or "")
            if not source_id or source_id in rest_reviews:
                continue
            adapted = _review_from_graphql(node)
            emit(
                source_type="review",
                source_id=source_id,
                body=adapted.get("body"),
                raw_obj=adapted,
                split=True,
                metadata={"state": adapted.get("state")},
                commit_sha=adapted.get("commit_id"),
            )

    for comment in raw.get("review_comment", []):
        # Never split: an inline review comment is one complete thought tied to one file
        # and line, and splitting it destroys what made it worth retrieving.
        emit(
            source_type="review_comment",
            source_id=comment["_object_id"],
            body=comment.get("body"),
            raw_obj=comment,
            split=False,
            metadata={
                key: comment.get(key)
                for key in (
                    "path",
                    "line",
                    "original_line",
                    "start_line",
                    "side",
                    "diff_hunk",
                    "in_reply_to_id",
                    "pull_request_review_id",
                    "subject_type",
                )
                if comment.get(key) is not None
            },
            commit_sha=comment.get("commit_id") or comment.get("original_commit_id"),
        )
    return out, redactions, sources


def _review_from_graphql(node: dict[str, Any]) -> dict[str, Any]:
    """Adapt a GraphQL review node to the shape ``emit`` reads.

    Read-time adaptation, deliberately: the raw object stays verbatim GraphQL (section 4),
    so a later extractor can still reach a field this version happens to ignore.
    ``__typename`` is ``"Bot"`` for a bot, which is exactly what REST's ``user.type`` says.
    """
    author = node.get("author") or {}
    submitted = node.get("submittedAt")
    return {
        "id": node.get("databaseId"),
        "body": node.get("body"),
        "state": node.get("state"),
        "user": {"login": author.get("login"), "type": author.get("__typename")},
        "author_association": node.get("authorAssociation"),
        "created_at": submitted,
        "updated_at": submitted,
        "html_url": node.get("url"),
        "commit_id": (node.get("commit") or {}).get("oid"),
    }


def _trust(
    user: dict[str, Any], assoc: str | None, authority: Mapping[str, str] | None = None
) -> tuple[bool, str]:
    """Section 6.2's tier, derived from the payload plus ``repo_authority``.

    ``OWNER`` and ``COLLABORATOR`` are unambiguous write access. ``MEMBER`` is *not* --
    on a large org it is thousands of people with no standing over this code -- so it is
    admitted only where :mod:`relore.ingest.authority` has resolved that login to write
    access on this repository. An unresolved ``MEMBER`` stays ``reported``, which is
    section 6.2's own last resort: a maintainer demoted to a claim costs one lost result,
    a stranger promoted to authority costs a wrong patch.

    Deriving it here rather than storing it at fetch time is what makes a rebuild
    reproduce the tier, and what lets a re-resolution move documents between tiers.
    """
    login = ((user or {}).get("login") or "").lower()
    # Section 6.2: `user.type` catches the accounts GitHub knows are apps; the
    # deployment's own list catches the ones it does not -- `HuggingFaceDocBuilderDev`
    # presents as a `User` and a `MEMBER`. Both are the machine tier, excluded from every
    # default result set (section 11).
    if (user or {}).get("type") == "Bot" or login in bot_accounts():
        return True, "machine"
    if assoc in ("OWNER", "COLLABORATOR"):
        return False, "authoritative"
    if (
        assoc == "MEMBER"
        and (authority or {}).get((user or {}).get("login") or "") in WRITE_PERMISSIONS
    ):
        return False, "authoritative"
    return False, "reported"


def _sole(items: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    return items[0] if items else None
