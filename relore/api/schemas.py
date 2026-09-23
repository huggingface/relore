"""Request and response shapes, versioned from the first commit (section 7).

The moment a second consumer exists, renaming a field breaks a stranger -- so the fields
of section 7 are all present now, including the ones whose *effect* is a later milestone.
``kind`` is the clearest case: it selects section 6's decay regime and section 6.2's trust
floor, neither of which exists yet, and it is accepted and echoed anyway because adding it
later would be a breaking change to a shipped API for no reason.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from relore.freshness import THREAD_STAMP_NOTE
from relore.search.queries import (
    HUMAN_TRUST,
    MACHINE_TRUST,
    MAX_HITS,
    QUERY_KINDS,
    SORTS,
    Hit,
    InflightView,
    ThreadView,
    WhyView,
    render_stamp,
)


class SearchRequest(BaseModel):
    """``POST /api/v1/search``.

    ``limit`` is clamped, not validated: a client asking for 500 gets 10 (section 6). The
    signal filters read the tables section 5.3's extraction pass fills.
    """

    query: str = ""
    kind: str | None = Field(default=None, description=f"one of {list(QUERY_KINDS)}")
    trust: str | None = Field(
        default=None,
        description=(
            f"raise the floor to one of {list(HUMAN_TRUST)}, or {MACHINE_TRUST!r} to ask the "
            "separate question of what the deployment's own bots claimed. The server decides "
            "the default and a request can never lower it"
        ),
    )
    repos: list[str] = Field(
        default=[],
        description=(
            "narrow to these repositories. It can only ever *subtract* from the token's "
            "scope (section 11): a name the token cannot see is dropped, not granted, so "
            "asking for one returns nothing rather than an error. Empty means every "
            "repository already in scope"
        ),
    )
    files: list[str] = []
    symbols: list[str] = []
    errors: list[str] = []
    tests: list[str] = []
    labels: list[str] = []
    since: dt.datetime | None = None
    before: dt.datetime | None = Field(
        default=None,
        description=(
            "only documents written before this instant. The complement of `since`, and a "
            "different question: `since` asks what is recent, this reconstructs the index "
            "as it stood at a past moment. A document whose creation date is unknown is "
            "excluded rather than admitted"
        ),
    )
    exclude: list[int] = Field(
        default=[],
        description=(
            "thread numbers this call may not reach. A cutoff cannot express these: a "
            "temporal benchmark's own task issue and the pull request that fixed it are "
            "older than the cutoff and are still the answer"
        ),
    )
    limit: int = MAX_HITS
    sort: str = Field(
        default="relevance",
        description=(
            f"one of {list(SORTS)}. Ordering only -- it reorders the same hits relevance "
            "would have chosen, and never changes which ones they are"
        ),
    )
    compact: bool = Field(
        default=False, description="shorter snippets, no score internals, for a tight budget"
    )
    render: bool = Field(
        default=False,
        description=(
            "also return the exact text the CLI would print, envelope included. What "
            "section 8's 'view as the model sees it' displays; off by default because an "
            "agent reading the JSON would be paying for the same content twice"
        ),
    )
    presentation: bool = Field(
        default=False,
        description=(
            "render for a person at a TTY: the same facts plus the backend tag and the "
            "flags worth trying next. The CLI sends it when stdout is a terminal (#13)"
        ),
    )
    expand: bool = Field(
        default=True,
        description=(
            "section 6's query expansion: fan the call out into error, test, symbol, file "
            "and free-text legs and merge them. On by default because the plain search "
            "ANDs every content term, which is what makes a pasted traceback match "
            "nothing. Turn it off to ask exactly one question"
        ),
    )


class LabelRequest(BaseModel):
    """``POST /api/v1/label`` -- section 8's relevance judgement, section 10's ground truth."""

    query: str
    repo: str
    number: int
    source_type: str
    verdict: str = Field(description="relevant | not_relevant | decisive")
    kind: str | None = None
    #: The signal filters the judged search actually carried. A label is only
    #: interpretable next to the whole query, and two examples can share the text and
    #: differ only in their `--file` -- folding the labels back into section 10's
    #: evaluation set has no way to tell them apart without this.
    filters: dict[str, list[str]] = {}
    note: str = ""


def hit_json(hit: Hit, *, compact: bool = False) -> dict[str, Any]:
    """One hit, as served.

    ``trust`` and ``age`` are always present and always rendered: age changes what a model
    concludes, authority changes it more (section 6, section 6.2). ``compact`` drops the
    score and its breakdown, which only a person tuning weights has any use for.
    """
    out: dict[str, Any] = {
        "repo": hit.repo,
        "number": hit.number,
        "type": hit.thread_type,
        "title": hit.title,
        "source_type": hit.source_type,
        "url": hit.url,
        "author": hit.author,
        "trust": hit.trust,
        "age": hit.age,
        # The age is for reading, this is for citing. JSON only, and never instead of the
        # age: a bare timestamp makes a model do arithmetic it will skip
        # (`render_stamp`, huggingface/relore#66).
        "date": render_stamp(hit.created_at),
        "snippet": hit.snippet,
        # Which comment, and which piece of it (huggingface/relore#18). `url` is not an
        # identity: a chunk of a comment carries the comment's URL, so two hits from one
        # 9,827-character comment were indistinguishable from two people agreeing.
        # `document_id` is stable across rebuilds -- it is GitHub's id plus the chunk --
        # where the row's primary key is not.
        "document_id": f"{hit.source_type}:{hit.source_id}:{hit.chunk_index}",
        "source_id": hit.source_id,
        "chunk_index": hit.chunk_index,
    }
    # Absent rather than 1 where nobody counted: corpus-wide search does not pay for the
    # aggregate, and "1 of 1" would be a claim it cannot make.
    if hit.passages:
        out["passages"] = hit.passages
    # Only where it is true, and only where somebody measured it: `search` does not, and a
    # `false` it did not check would be a claim. The page uses it to decide whether it owes
    # the caller the address of the rest (huggingface/relore#71).
    if hit.truncated:
        out["truncated"] = True
    if not compact:
        out["score"] = round(hit.score, 6)
        out["breakdown"] = {k: round(v, 6) for k, v in hit.breakdown.items()}
    return out


def inflight_json(view: InflightView) -> dict[str, Any]:
    """``GET /api/v1/inflight/{n}`` -- what claims to close this thread.

    ``links_indexed`` is here for the reason every count in this module is: an index with
    no relationship rows answers every question with an empty list, and an agent cannot
    tell that from "nobody is working on this" -- which are opposite instructions.
    """
    return {
        "repo": view.repo,
        "number": view.number,
        "claims": [
            {
                "repo": claim.repo,
                "number": claim.number,
                "type": claim.thread_type,
                "title": claim.title,
                "url": claim.url,
                "author": claim.author,
                "state": claim.state,
                "draft": claim.draft,
                "merged": claim.merged,
                "state_reason": claim.state_reason,
                "closed_by": claim.closed_by,
                "review_decision": claim.review_decision,
                "age": claim.age,
                "date": render_stamp(claim.created_at),
                "relationship": claim.relationship,
            }
            for claim in view.claims
        ],
        "claims_returned": len(view.claims),
        "claims_total": view.total,
        "links_indexed": view.links_indexed,
    }


def _outline_json(hit: Hit) -> dict[str, Any]:
    """One outline row. Deliberately not :func:`hit_json`: an outline line carries what is
    needed to *choose* a comment and nothing that would let it be quoted -- no score, no
    breakdown, no repeated repo and number, and a snippet cut at
    :data:`~relore.search.queries.OUTLINE_CHARS`. The whole value of the view is that it
    is cheap enough to serve complete."""
    return {
        "id": hit.source_id,
        "source_type": hit.source_type,
        "author": hit.author,
        "trust": hit.trust,
        "age": hit.age,
        "date": render_stamp(hit.created_at),
        "url": hit.url,
        "passages": hit.passages,
        "snippet": hit.snippet,
    }


def why_json(view: WhyView, *, blame: Any) -> dict[str, Any]:
    """``GET /api/v1/why`` -- the commit, the pull request, and what was said on the line.

    ``number: null`` and an empty ``anchored`` are different answers: no pull request could
    be resolved for the commit, versus one was and nobody reviewed this line.
    """
    return {
        "repo": view.repo,
        "path": view.path,
        "line": view.line,
        "blame": {
            "sha": blame.sha,
            "author": blame.author,
            "summary": blame.summary,
            "text": blame.text,
        },
        "number": view.number,
        "resolved_by": view.resolved_by,
        "thread": thread_json(view.thread) if view.thread else None,
        "anchored": [dict(comment) for comment in view.anchored],
        # The widening, and what it reached (relore#63, #64). Separate keys rather than one
        # merged list: a comment forty lines away and an approval on the pull request are
        # both evidence and are not the same claim, and a caller merging them cannot say
        # which it read.
        "on_file": [dict(comment) for comment in view.on_file],
        "reviews": [dict(review) for review in view.reviews],
        "level": view.level,
        "history": [dict(commit) for commit in view.history],
        # Separate from `history` and never merged into it: one follows the line, the other
        # follows a word, and a caller that cannot tell them apart cannot tell "this line
        # was reformatted eight times" from "this behaviour was argued here".
        "origin": [dict(commit) for commit in view.origin],
        "origin_term": view.origin_term,
        "origin_considered": [
            {"term": term, "commits": count} for term, count in view.origin_considered
        ],
        # Why the chain is empty, when it is. `comment` | `no-candidate` | `unreadable`,
        # and they are three different answers (huggingface/relore#71).
        "origin_declined": view.origin_declined,
    }


def _stamp(moment: dt.datetime | None) -> str | None:
    """A freshness mark a reader can compare with a GitHub timestamp, to the second.

    One function for every timestamp this API serves, so a caller reading ``indexed_at``
    next to a comment's ``date`` never has to parse two formats
    (:func:`~relore.search.queries.render_stamp`).
    """
    return render_stamp(moment)


def thread_json(view: ThreadView, *, compact: bool = False) -> dict[str, Any]:
    return {
        "repo": view.repo,
        "number": view.number,
        "type": view.thread_type,
        "title": view.title,
        "url": view.url,
        "author": view.author,
        "state": view.state,
        "age": view.age,
        "date": render_stamp(view.created_at),
        "labels": list(view.labels),
        # Events, not prose (issue #22).
        "state_reason": view.state_reason,
        "closed_by": view.closed_by,
        "merged": view.merged,
        "review_decision": view.review_decision,
        "review_decision_by": list(view.review_decision_by),
        "requested_reviewers": list(view.requested_reviewers),
        "body": view.body,
        # A cap a caller cannot see is a cap a caller reads as the whole document. `full`
        # serves the rest; these two say whether there is a rest.
        "body_chars": view.body_chars,
        "body_truncated": view.body_truncated,
        # Two keys, never one. Presence in `files_changed` says the thread changed the
        # path; presence in `files_mentioned` says somebody typed it, which is not the
        # same claim and was indistinguishable while both shared an array
        # (huggingface/relore#17).
        "files_changed": list(view.files_changed),
        # Also the diff, and also not part of the collected page: an inline review comment
        # can only hang on a changed file, so this recovers paths past the 100-row cap.
        "files_anchored": list(view.files_anchored),
        "files_mentioned": list(view.files_mentioned),
        # The denominator of the changed-file list, and `len(files_changed)` is its
        # numerator -- they are the same quantity, which the old bare count was not. A
        # truncated list must never be able to answer a membership question:
        # `files_collected < files_total` means a path that is absent may still have been
        # touched.
        "files_total": view.files_total,
        "files_collected": view.files_collected,
        "links": [dict(link) for link in view.links],
        # Named so the cap is visible in the response rather than inferred from a short
        # list: a caller that cannot tell truncation from a quiet thread will read ten
        # comments as the whole argument.
        "comments": [hit_json(hit, compact=compact) for hit in view.comments],
        "comments_returned": len(view.comments),
        # Comments, not documents: a long comment is several documents and counting rows
        # called one comment two (huggingface/relore#18). Every comment the thread has,
        # including the ones the trust floor withheld -- `comments_total` minus
        # `comments_machine_suppressed` is what the cap and the sampling compose with, and
        # a total computed post-filter said `0 of 0` about a thread we hold a comment for
        # (huggingface/relore#28).
        "comments_total": view.total_documents + view.machine_suppressed,
        # The gap, named rather than left to be inferred from a short list. Machine
        # authors are excluded and not down-weighted (section 6.2), so this is a filter a
        # reader has to know fired before concluding a thread is quiet.
        "comments_machine_suppressed": view.machine_suppressed,
        # When this response's thread was last rebuilt from GitHub. Per document, because
        # `status`'s per-source high-water rows cannot be composed into an answer about
        # one thread and two field runs composed them wrongly in opposite directions
        # (huggingface/relore#32). A string, not a datetime: `render` takes plain JSON
        # shapes and must read the same on both sides of the wire.
        "indexed_at": _stamp(view.indexed_at),
        "indexed_at_note": THREAD_STAMP_NOTE,
        # HOW those comments were chosen. A positional sample and a ranked top ten look
        # identical on the page, and the first five and last five of a 97-comment thread
        # were read as its ten best (huggingface/relore#16).
        "selection": view.selection,
        # A focus orders the comments and never selects them, so the count that matters is
        # how many carried every term: zero next to ten returned comments says "your
        # question matched nothing, this is the thread in order" rather than "nothing here".
        "focus": view.focus,
        "focus_matched": view.focus_matched,
        # One line per comment for the whole thread (relore#70). Present and empty unless
        # `--outline` asked, so a caller can tell "not requested" from "nothing to show".
        "outline": [_outline_json(c) for c in view.outline],
        "outline_returned": len(view.outline),
        # An outline narrowed by a focus is complete *of the comments carrying a word*,
        # which is a different completeness from the unnarrowed one and has to be
        # distinguishable without reading the rows (huggingface/relore#71). `widened` is
        # the case where the focus carried nothing and the whole outline came back
        # instead, so a caller never has to infer that from a count that did not shrink.
        "outline_matched": view.outline_matched,
        "outline_widened": view.outline_widened,
        # What `--comment` asked for and what became of it. `absent` and `suppressed` are
        # both "no comment came back" and are opposite next actions, so the page says which
        # rather than leaving it to be inferred from an empty list (huggingface/relore#71).
        "comment": view.comment,
        "comment_status": view.comment_status,
        # Where a sweep resumed. A page that silently begins in the middle is worse than
        # one that stops, so the cursor travels with the page that honoured it.
        "after": view.after,
    }
