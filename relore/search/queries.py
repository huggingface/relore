"""The query vocabulary, the result caps, and the two things every hit must carry.

Nothing in here touches a database: it is what the backends (section 4.1) agree on and
what the API and the CLI both render. The caps are the interesting part.

It imports one thing from the ingest side -- section 5.3's error normalizer -- because the
error filter is a two-sided normal form and both sides have to use the same function. That
module is stdlib-only for the same reason.

**Result caps are the contract, not a default.** The number behind them is measured: in
the motivating deployment every task session resends its whole conversation each turn at
~40k input tokens, and all seven sessions that ran LLM turns were stopped by a budget
guard rather than by finishing (use-cases section 7d). A retrieval tool that returns
generously does not add context, it subtracts turns from the patch. A client may ask for
less. Never more.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from relore.ingest.extract import error_query_form

#: Section 6. Requests are clamped to these, never rejected for exceeding them -- a
#: client asking for 50 gets 10, because failing the call helps nobody.
MAX_HITS = 10
MAX_SNIPPET_CHARS = 400
#: ``--compact`` for a client on a tight context budget (section 6). A client may ask
#: for less; never more.
COMPACT_SNIPPET_CHARS = 160
#: Section 6's dedup keeps the best chunk per ``(thread_id, source_type)``; this is the
#: other half of "one chatty thread cannot occupy the whole result page", since a single
#: thread has five source types to spend slots on.
MAX_HITS_PER_THREAD = 3
#: ``thread`` is never returnable in full: a 200-comment thread is the case this exists
#: for. The body gets a longer window than a comment because it is the one document whose
#: opening is reliably worth reading.
MAX_THREAD_COMMENTS = 10
MAX_BODY_CHARS = 800
#: How far past :data:`MAX_BODY_CHARS` a body is served whole rather than cut. The notice
#: that a body was truncated costs ~40 characters, so withholding fewer than a sentence's
#: worth spends more of the budget than it saves -- and an advisory that fires when
#: nothing meaningful was withheld is one a reader learns to skip, and then misses the one
#: that matters (huggingface/relore#31).
BODY_SLACK_CHARS = 80
#: What an unfocused ``thread`` does with those ten slots, and the name it reports.
#: Two at each end and the rest spread across the middle, because a *contested* thread
#: resolves in the middle: on ``huggingface/transformers#39847`` the first five and last
#: five were four emoji, a ``cc @``, a ``run-slow:`` line and CI chatter, while the review
#: that decided the question sat at position 51 of 97 and was structurally unreachable
#: (huggingface/relore#16).
ENDS_AND_MIDDLE = "ends+middle"
#: What a page asked with ``--after`` does instead: the next comments in order. Named,
#: because it is a third selection and a reader has to know which one produced the page --
#: a sweep's page is neither a sample nor a ranking.
AFTER_CURSOR = "after"
#: What ``--outline`` reports: not a page at all, which is the fourth thing a reader has to
#: be able to tell apart. Chronological and complete to its cap, so neither "sampled" nor
#: "ranked" describes it.
OUTLINE_VIEW = "outline"
#: What ``--comment <id>`` reports. Not a page either: one addressed document, served
#: whole, which is the same bargain ``--full`` makes for the opening post -- a caller who
#: named one thing gets that thing, because the caps exist to stop a *sample* being
#: mistaken for a corpus, and a document asked for by its id is the opposite of a sample.
COMMENT_VIEW = "comment"
#: How many of the ten each end gets. Small on purpose: the opening states the problem and
#: the last word is usually the resolution, but everything else that matters is between
#: them.
THREAD_ENDS = 2
#: How much more than a page the focused fetch asks for, so that collapsing a comment's
#: chunks into one hit still leaves ten comments to show.
PASSAGE_OVERFETCH = 3
#: The outline's own cap. Ten times the page, because the outline exists to be *complete*
#: where the page cannot be: measured on one agent run against ``transformers#37866``, a
#: 70-comment thread served ten at a time, the agent came back **six times** with six
#: different ``--focus`` strings and spent 3,558 tokens seeing overlapping samples of it.
#: **Measured, and the first guess was wrong.** At 200 rows and 90 characters each, an
#: outline of ``transformers#46419`` (644 comments) came to 6,626 tokens -- five times the
#: page it replaces, and still not complete. Nothing is cheap enough to serve a
#: 644-comment thread whole, so the cap is set where the *worst* page is about one
#: ``--full``: a hundred rows at ~25 tokens each. That is complete for the threads this
#: was built for -- 70 on ``#37866``, where the agent spent 3,558 tokens on six
#: overlapping samples -- and loudly capped past it, with ``--after`` to sweep the rest.
MAX_OUTLINE_COMMENTS = 100
#: How much of a comment one outline line carries. Enough to recognize the comment you are
#: looking for and never enough to answer with: the outline's job is to say *which* comment
#: to ask for, and a line long enough to quote would make it a second, worse page. Sixty
#: characters is most of a first sentence, and it is two thirds of what the snippet cost at
#: ninety -- which is the whole difference between an outline that pays for itself and one
#: that does not.
OUTLINE_CHARS = 60
#: ``inflight`` answers "is somebody already fixing this?", and the useful answer is one
#: pull request. Ten is the same cap as a result page for the same reason.
MAX_CLAIMS = 10

#: The query kinds of section 6's decay table and section 6.2's trust table. The per-kind
#: **trust floor** is applied (see :func:`trust_policy`); kind-aware **decay** is not, and
#: waits for the weighted ranking in milestone 3.
QUERY_KINDS = ("failure", "precedent", "rationale")

#: Human tiers, weakest first. ``machine`` is deliberately not on this ladder: it is not a
#: lower rung, it is excluded (section 6.2, section 11).
HUMAN_TRUST = ("reported", "authoritative")
MACHINE_TRUST = "machine"

#: How a page is *ordered*. It does not change which documents are on it: section 10.6 is
#: the record of what happens when recency decides selection -- each thread contributes its
#: newest document, which on a merged pull request is the approving review, and a real query
#: came back nine-tenths ``LGTM``. So ``newest`` reorders the same best-per-thread hits that
#: ``relevance`` would have returned; it never picks different ones.
#:
#: Sorting is not decay. Decay (section 6) folds age into the score and can demote a correct
#: old answer; this leaves the score alone and answers a different question -- "what has
#: moved lately" rather than "what answers this".
SORTS = ("relevance", "newest")

#: How a query's free text decides *admission*. ``all`` is the conjunction every search has
#: always been: ``plainto_tsquery`` on Postgres, one quoted term per word on FTS5, and it
#: is what stops a pasted sentence matching everything. ``any`` is the same terms disjoined,
#: ranked so that a row carrying more of them sorts first.
#:
#: **``any`` is a fallback, never a mode a caller picks.** It is asked for exactly once, by
#: :func:`relore.search.expansion.search_best`, and only after ``all`` came back with
#: nothing -- so no query that had an answer can be widened out of it. The reason it exists
#: is measured: over three head-to-head runs (``relore/docs/gh-vs-relore-2026-09-14.md``)
#: every zero-result page came from a prose query of six to nine terms, and every one cost
#: a full turn to reformulate. The term count is the signature, the same one
#: :meth:`relore.search.backends.base.SearchBackend._focused` documents inside one thread:
#: a monotonic decrease ending in an empty page that reads as "the corpus has nothing",
#: which is the one thing it did not mean.
MATCH_ALL = "all"
MATCH_ANY = "any"
MATCHES = (MATCH_ALL, MATCH_ANY)


class QueryError(ValueError):
    """A request that cannot be served as asked. Distinct from an empty result, which is
    not an error: no results is a successful call with nothing in it, always."""


@dataclass(frozen=True)
class SearchQuery:
    """One ``/search`` request, after validation and clamping.

    ``repos`` is **required and concrete**. Section 11 puts per-token repo scoping in the
    query layer rather than in a handler so that a new endpoint cannot forget it, and an
    empty tuple therefore means *no repository is in scope* -- an empty result, never
    everything. Resolving "this token may see all of them" into a list of names happens
    where the token is read, so nothing below here has a wildcard to mishandle.
    """

    repos: tuple[str, ...]
    text: str = ""
    kind: str | None = None
    trust: str | None = None
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    since: dt.datetime | None = None
    #: An upper bound on document age: only what was written *before* this instant is
    #: eligible. ``since`` narrows a search to recent discussion; this reconstructs the
    #: index as it stood at a past moment, which is a different question and the one a
    #: temporal benchmark asks -- an evaluation that can retrieve the answer written after
    #: the task measures nothing. Applied in the query layer beside the repo scope and the
    #: trust floor (``backends/base._filters``) so a new endpoint cannot forget it, and
    #: carried by every expansion leg because a leg is a ``replace()`` of this object.
    before: dt.datetime | None = None
    #: Thread numbers this call may not reach, whatever else matches. A temporal benchmark
    #: needs it for the one thing a cutoff cannot express: the task's own issue and the
    #: pull request that fixed it are *older* than the cutoff and are still the answer, so
    #: the corpus has to be able to withhold a named thread rather than an era.
    exclude: tuple[int, ...] = ()
    limit: int = MAX_HITS
    compact: bool = False
    sort: str = "relevance"
    #: :data:`MATCH_ALL` or :data:`MATCH_ANY`. Set by the widening fallback, not by a
    #: caller -- there is no flag and no request field for it.
    match: str = MATCH_ALL

    def __post_init__(self) -> None:
        # Section 6's error term matches the *normalized* form (section 5.3), so a caller
        # who pastes what their terminal printed has to be put in that form here -- or
        # every `--error` query silently matches nothing, which looks exactly like an
        # index that has nothing to say. `error_query_form` pulls the raised error out of
        # a whole traceback, so pasting the traceback works too.
        if self.errors:
            object.__setattr__(
                self,
                "errors",
                tuple(dict.fromkeys(filter(None, map(error_query_form, self.errors)))),
            )
        if self.kind is not None and self.kind not in QUERY_KINDS:
            raise QueryError(f"unknown query kind {self.kind!r}; one of {list(QUERY_KINDS)}")
        if self.trust is not None and self.trust not in (*HUMAN_TRUST, MACHINE_TRUST):
            raise QueryError(f"unknown trust tier {self.trust!r}")
        if self.since is not None and self.since.tzinfo is None:
            raise QueryError("since must be timezone-aware")
        if self.before is not None and self.before.tzinfo is None:
            raise QueryError("before must be timezone-aware")
        # An empty window is a caller error, not an empty result: `since > before` asks for
        # documents written after one instant and before an earlier one, which no corpus
        # can satisfy. Answering it with silence is indistinguishable from a quiet index,
        # and a benchmark that misconfigures its cutoff would read that silence as a
        # baseline scoring zero.
        if self.since is not None and self.before is not None and self.since >= self.before:
            raise QueryError(
                f"empty window: since {self.since.isoformat()} is not before "
                f"before {self.before.isoformat()}"
            )
        if self.sort not in SORTS:
            raise QueryError(f"unknown sort {self.sort!r}; one of {list(SORTS)}")
        if self.match not in MATCHES:
            raise QueryError(f"unknown match {self.match!r}; one of {list(MATCHES)}")
        object.__setattr__(self, "limit", max(1, min(int(self.limit), MAX_HITS)))

    @property
    def terms(self) -> tuple[str, ...]:
        return tokenize(self.text)

    @property
    def snippet_chars(self) -> int:
        return COMPACT_SNIPPET_CHARS if self.compact else MAX_SNIPPET_CHARS

    @property
    def has_signal_filter(self) -> bool:
        return bool(self.files or self.symbols or self.errors or self.tests)


@dataclass(frozen=True)
class Hit:
    """One result. ``trust`` and ``age`` are rendered, not implied (section 6, section 6.2).

    Age changes what a model concludes; authority changes it more. A four-year-old comment
    about a file that still exists may be exactly right about intent and badly wrong about
    the current code, and the cheapest way to get an agent to treat it that way is to tell
    it how old it is. ``[MEMBER]`` versus ``[contributor claim]`` in front of a snippet
    costs four tokens and is the difference between a fact and someone's opinion.
    """

    repo: str
    number: int
    thread_type: str
    title: str
    source_type: str
    url: str | None
    author: str | None
    trust: str
    age: str
    snippet: str
    score: float
    created_at: dt.datetime | None = None
    #: Which comment this is a piece of, and which piece. A long comment is chunked into
    #: several documents, and without these two a ranked page can carry the same comment
    #: twice -- same URL, same author, same tier -- reading as two people agreeing
    #: (huggingface/relore#18). ``source_id`` is the comment's own GitHub id, so it is the
    #: identity ``url`` is not: a chunk has no URL of its own.
    source_id: str = ""
    chunk_index: int = 0
    #: How many chunks that comment has, where the caller knows. ``0`` means "not
    #: counted here" rather than "one": the corpus-wide search path does not pay for the
    #: aggregate, and the renderer says nothing rather than something wrong.
    passages: int = 0
    #: Whether ``snippet`` is a window onto something longer. The window discloses itself
    #: with an ellipsis, which says *that* it was cut and not how to undo it -- and until
    #: ``--comment`` there was no way to undo it at all: ``--full`` is the opening post,
    #: ``--focus`` re-ranks and snippets again, ``--after`` serves the *next* comments. So
    #: the page that cut a comment is the page that has to name the address, and this is
    #: what tells it one was cut (huggingface/relore#71).
    truncated: bool = False
    #: Every term of the score, for the human tuning the weights (section 8). An agent has
    #: no use for *why* something ranked; the person has nothing else. Dropped by
    #: ``--compact``.
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreadView:
    """One thread, capped. Section 6: a 200-comment thread must never come back in full."""

    repo: str
    number: int
    thread_type: str
    title: str
    url: str | None
    author: str | None
    state: str | None
    age: str
    labels: tuple[str, ...]
    body: str
    comments: tuple[Hit, ...]
    #: The opening post's full length, and whether ``body`` is a window onto it. A body
    #: cut at :data:`MAX_BODY_CHARS` with no marker is the gap that sent a diagnosis
    #: session back to ``git clone``: an issue template spends its first ~700 characters
    #: on environment boilerplate, so ``### Reproduction`` began at almost exactly the cut.
    body_chars: int = 0
    body_truncated: bool = False
    #: The pull request's diff, as far as it was collected. **Only this list is about
    #: what the thread changed.**
    files_changed: tuple[str, ...] = ()
    #: Paths an inline review comment is anchored to. Also the diff -- GitHub will not
    #: anchor a comment anywhere else -- but *not* part of the page the per-PR pass
    #: collected, so it is counted separately and is the one source that recovers a path
    #: the 100-row cap dropped.
    files_anchored: tuple[str, ...] = ()
    #: Paths read out of prose. Kept, because a traceback's path is often the most useful
    #: thing in a thread -- and kept *apart*, because it is an extraction rather than a
    #: fact: a bare basename, a file the thread never touched, or the same file the diff
    #: already names in full all land here (huggingface/relore#17).
    files_mentioned: tuple[str, ...] = ()
    #: The changed-file list's denominator. The per-PR pass takes the diff 100 at a time,
    #: so ``files_changed`` is routinely a subset. A *missing* entry in a list is read as
    #: a negative fact, and on ``huggingface/transformers#39847`` (323 changed files, 100
    #: collected) that absence would have exonerated the pull request that caused the bug
    #: being diagnosed.
    files_total: int | None = None
    files_collected: int = 0
    links: tuple[dict[str, Any], ...] = ()
    #: Events, not prose (issue #22): ``state: closed`` cannot separate *withdrawn by its
    #: author* from *closed as a duplicate*, and ``focus`` never finds a closure because
    #: there is no comment to rank. A review decision with no comments at all is the other
    #: half -- ``0 of 0 comments`` cannot say whether nobody looked or nobody typed.
    state_reason: str | None = None
    closed_by: str | None = None
    merged: bool = False
    review_decision: str | None = None
    review_decision_by: tuple[str, ...] = ()
    requested_reviewers: tuple[str, ...] = ()
    #: Comments, not documents. A long comment is several documents, so counting rows
    #: overstated the thread and made ``10 of 89 comments`` a count of two different
    #: things (huggingface/relore#18).
    total_documents: int = 0
    #: Comments this thread has that the trust floor did not admit. Machine-authored
    #: documents are excluded rather than down-weighted (section 6.2) -- correct, they are
    #: not evidence -- but a thread whose only comment is one of them rendered as ``0 of
    #: 0``, which is the sentence for a thread nobody has touched. Counted so the filter
    #: can be announced instead of inferred (huggingface/relore#28).
    machine_suppressed: int = 0
    #: When this thread was last rebuilt from GitHub. Per response and per document, which
    #: is exactly what ``status``'s per-source high-water rows are not: nobody can compose
    #: ``[issue_comments]`` and ``[threads]`` into "are the comments on *this* thread
    #: current?", and two field runs read that strip wrongly in opposite directions --
    #: once discounting comments that were complete (huggingface/relore#32).
    indexed_at: dt.datetime | None = None
    #: When the thread was opened -- the moment ``age`` was rendered from, for a caller that
    #: has to cite it rather than read it (:func:`render_stamp`, huggingface/relore#66).
    created_at: dt.datetime | None = None
    #: How the returned comments were chosen -- ``focus`` when a query ordered them,
    #: :data:`ENDS_AND_MIDDLE` otherwise. A positional sample and a ranked top-ten are
    #: indistinguishable on the page unless the page says which it is, and the first ten
    #: of a 97-comment thread were read as its ten best (huggingface/relore#16).
    selection: str = ""
    #: What ``focus`` asked, and how many comments carry *every* one of its terms. A focus
    #: **orders** a thread's comments and never selects them (see
    #: :meth:`relore.search.backends.base.SearchBackend._focused`), so this is the
    #: denominator that says whether the ranking had anything to work with -- and a zero
    #: here next to ten returned comments is the honest shape of "your question matched
    #: nothing, so this is the thread in order".
    focus: str = ""
    focus_matched: int | None = None
    #: One line per comment, the whole thread, instead of a page of ten (relore#70).
    #: Empty unless ``--outline`` asked for it. This is the ``defs`` move applied to a
    #: discussion: ask for the shape, then read the parts -- because the page cap is the
    #: contract (section 6) and the answer to "ten of seventy" cannot be a bigger page.
    outline: tuple[Hit, ...] = ()
    outline_total: int = 0
    #: How many of the thread's comments carry *any* of ``focus``'s terms, when an outline
    #: was narrowed by one (huggingface/relore#71) -- ``None`` when no focus was passed.
    #: Any rather than every: a conjunction is what empties a thread page as soon as a
    #: caller passes a sentence, and an empty outline would read as "no comment here
    #: mentions that".
    outline_matched: int | None = None
    #: Whether that narrowing found nothing and the view fell back to the whole outline.
    #: A zero-row page whose emptiness is a property of the question is the failure
    #: huggingface/relore#47 was opened for; this is the same answer, one verb over.
    outline_widened: bool = False
    #: The comment ``--comment`` asked for, echoed, and what happened to it. A caller who
    #: gets an empty answer to an id cannot tell "this thread has no such comment" from
    #: "that comment is our own bot's and the trust floor dropped it" (section 6.2), and
    #: those are opposite next actions -- so the status is named rather than inferred from
    #: an absence. ``served`` | ``absent`` | ``suppressed``.
    comment: str = ""
    comment_status: str = ""
    #: The comment this page starts after, if the caller was sweeping (``--after``). A page
    #: that silently begins in the middle is the one thing worse than a page that stops.
    after: str = ""


@dataclass(frozen=True)
class Claim:
    """One thread that claims to close another (section 13.3).

    ``state``, ``draft`` and ``merged`` are all here because they imply opposite next
    actions: an approved pull request means stop, a stale draft means supersede it, and a
    merged one means the fix is in and the issue may simply be un-closed.
    """

    repo: str
    number: int
    thread_type: str
    title: str
    url: str | None
    author: str | None
    state: str | None
    draft: bool
    merged: bool
    age: str
    relationship: str
    #: The moment ``age`` was rendered from. JSON only -- see :func:`render_stamp`.
    created_at: dt.datetime | None = None
    #: How it stopped, not only that it did (issue #22).
    state_reason: str | None = None
    closed_by: str | None = None
    review_decision: str | None = None


@dataclass(frozen=True)
class WhyView:
    """Why one line is the way it is (issue #9).

    ``git blame`` names the commit; everything else here is the argument that produced it.
    ``number is None`` is a real answer -- the line's commit reached the tree through no
    pull request this index holds -- and it has to be distinguishable from "nothing was
    said", which is an empty ``anchored`` on a thread that exists.
    """

    repo: str
    path: str
    line: int
    sha: str
    summary: str = ""
    number: int | None = None
    #: ``commit`` when ``thread_commits`` held the sha, ``summary`` when only the
    #: squash-merge convention did. The second is a guess from a string and says so.
    resolved_by: str = ""
    thread: ThreadView | None = None
    #: The part `git blame` cannot give: what reviewers said *on this line* while it was
    #: being written.
    anchored: tuple[dict[str, Any], ...] = ()
    #: Review comments on the same file, outside the line window. One level out, and the
    #: difference between "nobody discussed this line" and "nobody discussed this code".
    on_file: tuple[dict[str, Any], ...] = ()
    #: The pull request's review *bodies* (relore#63). No line anchor exists on these, so
    #: no widening of a window reaches them -- and an approval's conditions live here.
    reviews: tuple[dict[str, Any], ...] = ()
    #: Which of the three carried the answer: ``line``, ``file``, ``review`` or ``none``.
    #: A caller that cannot tell reads a widened answer as an exact one (relore#64).
    level: str = "none"
    #: The line's revision chain, newest first, each resolved to its pull request where the
    #: index holds one (relore#57). ``blame`` is this chain's first entry, never its whole.
    history: tuple[dict[str, Any], ...] = ()
    #: The line's *string* chain: ``git log -S`` on one word from it, oldest last. A
    #: revision chain follows a range and so loses a block that was **rewritten** rather
    #: than moved -- which is the common case, and is why three separate runs left this
    #: verb and ran the pickaxe by hand. Empty when no candidate word found anything the
    #: revision chain did not already have, which is a real answer and not a gap.
    origin: tuple[dict[str, Any], ...] = ()
    #: Which word was pickaxed, and what every candidate reached. The choice is a
    #: judgement, and a page that does not show its working cannot be argued with.
    origin_term: str = ""
    origin_considered: tuple[tuple[str, int], ...] = ()
    #: Why the origin chain is empty, when it is. Three causes a reader cannot separate
    #: from the absence -- the line carries no code, every candidate was rejected, the file
    #: could not be read -- and only the middle one means "the revision chain is the whole
    #: story". The page printed *nothing* for an empty origin, which is that answer given
    #: by omission (huggingface/relore#71).
    origin_declined: str = ""


@dataclass(frozen=True)
class InflightView:
    """What claims to close one thread, and whether the question could be answered.

    ``links_indexed`` is the difference between "nothing claims this" and "this index has
    no relationship rows at all" -- an index derived before section 13.3 existed answers
    every ``inflight`` with silence, and silence here is the answer an agent acts on.
    """

    repo: str
    number: int
    claims: tuple[Claim, ...]
    total: int
    links_indexed: int


@dataclass(frozen=True)
class BackendInfo:
    """What engine answered, and what it can do (section 4.1).

    ``status`` prints this so a surprising result set is diagnosable rather than
    mysterious, and section 10's benchmark records it so a recall number measured on
    ``bm25`` is never compared with one measured on ``ts_rank_cd``.

    ``capabilities`` lists what this backend can answer **today**, not which indexes
    happen to exist: a declared capability nothing implements is a lie a reader would
    reasonably act on.
    """

    name: str
    ranking: str
    capabilities: frozenset[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ranking": self.ranking,
            "capabilities": sorted(self.capabilities),
        }


#: Section 6.2's per-query-kind floors. The same query kinds that need different recency
#: need different authority, and the reason is that the corpus holds two different things
#: that look identical in the schema: *reports* ("I hit this error"), which a stranger can
#: make as well as anyone, and *judgements* ("that's intentional"), which are only worth
#: retrieving from someone entitled to make them.
_KIND_FLOOR: dict[str, str] = {
    # The whole value is that someone entitled to decide has decided. An unauthorised
    # answer here is a guess the agent will act on, and it is worse than an empty result.
    "rationale": "authoritative",
    # The opinion needs authority; the artifact does not -- see `merged_pr_is_precedent`.
    "precedent": "authoritative",
    # A stranger's traceback carrying the same exception is real evidence. They are
    # reporting, not adjudicating.
    "failure": "reported",
}


@dataclass(frozen=True)
class TrustPolicy:
    """Which documents a query may see, as data rather than as SQL.

    This module stays free of any database import (see the header), so the policy is
    described here and rendered into a predicate by the backend.
    """

    tiers: tuple[str, ...]
    #: Section 6.2's one disjunction: on a *precedent* query a merged pull request is
    #: precedent whoever wrote it, because merging is the maintainer's act and authority
    #: attaches to the merge rather than to the author. A first-time contributor's merged
    #: change is precedent; their unmerged proposal is not.
    merged_pr_is_precedent: bool = False

    def __contains__(self, trust: str) -> bool:
        return trust in self.tiers


def trust_policy(requested: str | None = None, kind: str | None = None) -> TrustPolicy:
    """Section 6.2's filter: authority is a floor, and the server decides the default.

    Machine-authored documents are **excluded, not down-weighted** (section 6.2,
    section 11). The corpus contains the deployment's own agent's comments; returning one
    as "prior discussion" makes the agent's unreviewed output its own evidence -- a
    poisoning loop with a database in the middle, and one that looks like retrieval
    working. So ``trust=machine`` is not a lowered floor, it is a separate, explicit
    question ("what did the bot claim here?") and the only way to see them; a query kind
    does not change that either way.

    Otherwise the floor is the **higher** of the query kind's default and whatever the
    caller asked for. A caller may raise it and can never lower it, which is why this
    takes the maximum rather than preferring the request.
    """
    if requested == MACHINE_TRUST:
        return TrustPolicy((MACHINE_TRUST,))

    floors = [_KIND_FLOOR.get(kind or "", HUMAN_TRUST[0])]
    if requested is not None:
        floors.append(requested)
    effective = max(HUMAN_TRUST.index(floor) for floor in floors)

    return TrustPolicy(
        tiers=HUMAN_TRUST[effective:],
        # Dropped when the caller raises the floor explicitly: asking for `authoritative`
        # is asking for authoritative documents, not for an exemption from the request.
        merged_pr_is_precedent=(kind == "precedent" and requested is None),
    )


def admissible_trust(requested: str | None = None, kind: str | None = None) -> tuple[str, ...]:
    """The tiers alone, for callers that render the floor rather than filter on it."""
    return trust_policy(requested, kind).tiers


_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Split free text into the terms both backends will match on.

    Identifiers keep their underscores and lose everything else, because
    ``to_tsvector('english', ...)`` and FTS5's ``unicode61`` both shred ``snake_case`` and
    ``camelCase`` the same way (section 1). Nothing here is a substitute for the engine's
    own parser -- it exists so a snippet can be centred on a match, and so the SQLite
    backend can build a ``MATCH`` expression out of user text without inheriting FTS5's
    query syntax.
    """
    return tuple(m.group(0) for m in _WORD.finditer(text))


def render_age(then: dt.datetime | None, *, now: dt.datetime | None = None) -> str:
    """``3d``, ``7mo``, ``4y`` -- rendered, because a bare timestamp makes a model do
    arithmetic it will skip (section 6)."""
    if then is None:
        return "?"
    now = now or dt.datetime.now(dt.timezone.utc)
    seconds = (now - then).total_seconds()
    if seconds < 0:
        return "0h"
    hours = seconds / 3600
    if hours < 1:
        return "<1h"
    if hours < 48:
        return f"{int(hours)}h"
    days = hours / 24
    if days < 60:
        return f"{int(days)}d"
    months = days / 30.44
    if months < 24:
        return f"{int(months)}mo"
    return f"{days / 365.25:.0f}y"


def render_stamp(then: dt.datetime | None) -> str | None:
    """The same moment :func:`render_age` renders, as a date a caller can *cite*.

    Age is the right unit for reading and it is what the rendered page keeps: a bare
    timestamp makes a model do arithmetic it will skip. It is not enough to quote one, and
    quoting is what an agent is asked for -- "the decisive comment, with its author and its
    date". With only an age, a careful agent bounds the date from the merge commit and says
    it did; a less careful one states the inferred month as a fact (huggingface/relore#66).

    So both, on different surfaces: the age in the text, this in the JSON. ``Z`` rather than
    an offset and no microseconds, so it compares character-for-character with what the
    GitHub API hands over.
    """
    if then is None:
        return None
    return then.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snippet(text: str, terms: tuple[str, ...] = (), *, limit: int = MAX_SNIPPET_CHARS) -> str:
    """A window of ``text``, centred on the first matching term.

    Centring rather than taking the head is what makes section 8's question -- "are these
    results any good?" -- answerable by a person reading ten results: a 6000-character
    chunk's opening paragraph usually does not contain the term that matched it.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    start = _first_match(text, terms)
    start = max(0, min(start - limit // 3, len(text) - limit))
    window = text[start : start + limit]
    if start:  # do not begin mid-word
        window = window[window.find(" ") + 1 :] if " " in window else window
    return ("…" if start else "") + window.rstrip() + "…"


def _first_match(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    positions = [p for term in terms if (p := lowered.find(term.lower())) >= 0]
    return min(positions) if positions else 0
