"""Turning an API response into the text a model reads. One implementation, two callers.

The CLI prints this, and section 8's "view as the model sees it" shows the *same* string --
including the untrusted envelope and its delimiter scrubbing -- so that a formatting or
injection bug is caught by a person reading it rather than by an agent meeting it mid-task.
Two renderers would drift, and the one that drifted would be the one nobody was looking at.

**With no MCP server, stdout is an API** (#13), so the piped form is a contract:
``presentation=False`` -- what a caller gets when stdout is not a TTY -- prints facts only,
in a documented line grammar (``docs/cli.md``). ``presentation=True`` adds advice and
diagnostics for the person at the terminal: the backend tag, `--full`, `--focus`,
`relored derive`. **Facts are never presentation**, so every count, cap and caveat is in
both forms; only the suggestions move.

**Everything here takes plain dictionaries -- the JSON shapes of section 7 -- and imports
nothing but the standard library.** That is what lets it live on both sides of the
boundary: :mod:`relore.cli` may not reach a database driver or a web server (AGENTS.md
invariant 1), and importing :mod:`relore.search` would pull SQLAlchemy in through the
package's ``__init__``. Dicts in, text out.
"""

from __future__ import annotations

from typing import Any

from relore.freshness import COLLECTOR_NOTE, THREAD_STAMP_NOTE, pass_note
from relore.security.untrusted import envelope, quote

#: What each tier is called in front of a snippet. Four tokens, and the difference between
#: a fact and someone's opinion (section 6.2).
TRUST_LABEL = {
    "authoritative": "authoritative",
    "reported": "contributor claim",
    "machine": "MACHINE — our own bot, not evidence",
}


#: What ``--compact`` shortens one line of quoted prose to. The same budget the server
#: applies to a search snippet (``relore.search.queries.COMPACT_SNIPPET_CHARS``), which is
#: where a caller's expectation for the flag comes from -- spelled again rather than
#: imported, because this module may not reach :mod:`relore.search` (AGENTS.md invariant 1)
#: and ``test_render`` pins the two together so they cannot drift.
COMPACT_CHARS = 160

#: How many comments one ``--outline`` reaches
#: (``relore.search.queries.MAX_OUTLINE_COMMENTS``). Needed here only to tell a capped
#: page which sentence is true -- "lists all 68" or "a hundred at a time, and ``--after``
#: sweeps the rest" -- and spelled again rather than imported, because this module may not
#: reach :mod:`relore.search` (AGENTS.md invariant 1). ``test_render`` pins the two
#: together so they cannot drift.
OUTLINE_CAP = 100

#: The flag each signal filter was spelled as on the command line. A zero page echoes
#: them, because the caller's own `--error` is a *correct* fact about their bug that can
#: still zero a query answering at rank 1 without it (huggingface/relore#47) -- and with
#: only the query text on the page, that reads as an absence in the corpus rather than as
#: a property of the question.
FILTER_FLAG = {
    "files": "--file",
    "symbols": "--symbol",
    "errors": "--error",
    "tests": "--test",
    "labels": "--label",
}


def trim(text: str, *, limit: int = COMPACT_CHARS) -> tuple[str, bool]:
    """One line of prose, shortened for ``--compact``, and whether it was.

    The flag returns a *second* value rather than quietly appending an ellipsis because a
    shortened line has to be disclosed by whoever prints it: an agent that cannot tell a
    trimmed line from a whole one reads the tail it never got as absent from the source,
    which is the failure this project keeps re-finding (section 13.3).
    """
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "…", True


def render_search(
    payload: dict[str, Any], *, compact: bool = False, presentation: bool = False
) -> str:
    """The result page, wrapped in the envelope.

    Empty is not an error: no hits renders as a sentence saying so, still enveloped, still
    exit 0.
    """
    query = payload.get("query") or {}
    hits = payload.get("hits") or []
    backend = payload.get("backend") or {}
    header = [
        f"{len(hits)} hit{'' if len(hits) == 1 else 's'}"
        + (f" for {query.get('text')!r}" if query.get("text") else "")
        + (f"   [{backend.get('name', '?')}/{backend.get('ranking', '?')}]" if presentation else "")
    ]
    floor = query.get("trust_floor") or []
    if floor and floor != ["reported", "authoritative"]:
        # A raised floor and a quiet corpus produce the same empty page, so say which.
        header.append(f"trust floor: {', '.join(floor)}")
    widened = str(query.get("widened") or "")
    if widened and hits:
        # Never silent: the page in front of the caller is not the page they asked for,
        # and a hit that carries two terms of seven has to be read as one (section 6).
        header.append(
            "widened: nothing carried every term, so these carry some of them — "
            "most of the query first."
        )
    if not hits:
        applied = _filters_applied(query)
        if applied:
            header.append(f"filters: {', '.join(applied)}   (they AND)")
        header.append(
            # Widening already asked the cheaper question. Saying so is what stops the
            # next turn being another reformulation of the same words.
            "nothing matched, with every term or with any of them."
            if widened
            else "nothing matched."
        )
    untested = int(payload.get("files_untested") or 0)
    if untested:
        # A `--file` page is read as "these are the threads that touched it". Threads with
        # no collected changed-file list are absent from it without ever being tested, and
        # they skew new -- so the page has to say so or it overstates what it knows.
        header.append(
            f"{untested} more thread{'' if untested == 1 else 's'} matched but "
            f"{'has' if untested == 1 else 'have'} no collected changed-file list, "
            f"so --file could not test {'it' if untested == 1 else 'them'}."
        )

    body = [*header, ""]
    for index, hit in enumerate(hits, start=1):
        body += _hit_lines(index, hit, note=_passage_note(hit, hits))
    return envelope("\n".join(body).rstrip(), compact=compact)


def _filters_applied(query: dict[str, Any]) -> list[str]:
    """Every filter but the text, spelled the way the caller passed it. See :data:`FILTER_FLAG`."""
    filters = query.get("filters") or {}
    applied = [f"--kind {query['kind']}"] if query.get("kind") else []
    for name, flag in FILTER_FLAG.items():
        # `--error` is echoed in section 5.3's normal form, which is what it actually
        # filtered on and is rarely what was typed -- `RuntimeError: nope` filters as
        # `nope`. Echoing the input instead would hide the one transformation on this
        # page that can produce a surprising zero all by itself.
        note = " [normalized]" if name == "errors" else ""
        applied += [f"{flag} {value}{note}" for value in filters.get(name) or []]
    if query.get("since"):
        applied.append(f"--since {query['since']}")
    repos = query.get("repos")
    # An empty scope is the one that means *nothing is searchable*, and it is section 11
    # failing closed rather than a quiet corpus (server.py says the same of the field).
    if repos is not None and not repos:
        applied.append("no repository in scope")
    return applied


def _hit_lines(index: int, hit: dict[str, Any], note: str = "") -> list[str]:
    """One hit. ``compact`` does not reach here: it shortens the snippet server-side and
    drops the score breakdown from the JSON, and the score is no longer rendered at all.

    ``note`` is for what the head line cannot imply -- today, that this hit is one passage
    of a longer comment rather than a comment of its own.
    """
    tier = TRUST_LABEL.get(str(hit.get("trust")), str(hit.get("trust")))
    head = (
        f"{index}. {hit.get('repo')}#{hit.get('number')} {hit.get('type')}  "
        f"[{tier}]  {hit.get('age')}  {hit.get('source_type')}"
    )
    if hit.get("author"):
        head += f"  @{hit['author']}"
    if note:
        head += f"  ({note})"
    # The two fields that are somebody else's words get marked as such; the head line
    # above -- tier, age, author, source type -- is ours (huggingface/relore#12).
    lines = [head]
    # Skipped when empty rather than printed blank: a comment carries its thread's title,
    # and a blank line still costs the caller a token.
    lines += [f"   {quote(text)}" for text in (hit.get("title"), hit.get("snippet")) if text]
    if hit.get("url"):
        lines.append(f"   {hit['url']}")
    # No score. The ranking is already expressed by the order, and `score 0.03009` above
    # `score 0.03125` supports no decision a caller can act on -- it is four tokens per
    # hit spent on a number whose scale is a property of the backend. It stays in `--json`
    # with its breakdown, for section 8's page and whoever is tuning the weights.
    lines.append("")
    return lines


def _comment_lines(index: int, hit: dict[str, Any], note: str = "") -> list[str]:
    """One comment *inside a thread page*, which is not the same line as one search hit.

    A search hit has to say which thread it came from; a comment on a thread page does not,
    and said it anyway. Measured on five real pages, ``owner/repo#N pr``, the thread's
    title re-quoted under every comment, and a 76-character URL were **24-33% of the whole
    page** -- three constants the head line already carries, reprinted ten times
    (huggingface/relore#71).

    What replaces the URL is the id, which was already in the payload and is what
    ``--after`` and ``--comment`` take. A URL opens a browser and the caller reading this
    page is not a browser; the id is the address, and :func:`render_thread` spends one line
    at the foot of the page saying so instead of ten repeating it. Same three fields a
    reader actually weighs a comment by -- tier, age, author -- in the same order as an
    outline row, so the two views read as one grammar.
    """
    tier = TRUST_LABEL.get(str(hit.get("trust")), str(hit.get("trust")))
    head = (
        f"{index}. {hit.get('source_id') or ''}  [{tier}]  "
        f"{hit.get('age')}  {hit.get('source_type')}"
    )
    if hit.get("author"):
        head += f"  @{hit['author']}"
    if note:
        head += f"  ({note})"
    lines = [head]
    if hit.get("snippet"):
        lines.append(f"   {quote(hit['snippet'])}")
    lines.append("")
    return lines


def _one_comment_lines(thread: dict[str, Any], *, presentation: bool = False) -> list[str]:
    """``--comment <id>``: the comment that id names, whole (huggingface/relore#71).

    The three outcomes are three different sentences, because an empty answer cannot
    distinguish them and they are opposite next actions: this thread has no such comment
    (stop), or it has one and it is our own bot's, which section 6.2 excludes from evidence
    (ask ``search --trust machine`` if the bot's claim is what you wanted).
    """
    asked = thread.get("comment") or ""
    status = thread.get("comment_status") or ""
    served = (thread.get("comments") or [None])[0] if status == "served" else None
    if served is None:
        if status == "suppressed":
            return [
                f"-- comment {asked}: MACHINE — our own bot, not evidence, so this view "
                "excludes it (section 6.2) --"
                + (
                    "\n`relore search --trust machine` asks what the bots claimed."
                    if presentation
                    else ""
                )
            ]
        return [
            f"-- comment {asked}: this thread holds no comment with that id. It is a "
            "comment's own GitHub id, off an outline line or a page's head line -- not a "
            "position, and not the thread number --"
        ]
    tier = TRUST_LABEL.get(str(served.get("trust")), str(served.get("trust")))
    text = served.get("snippet") or ""
    clauses = [
        f"-- comment {asked}, {served.get('source_type')}",
        f"[{tier}]",
        f"{served.get('age')}",
    ]
    if served.get("author"):
        clauses.append(f"@{served['author']}")
    passages = int(served.get("passages") or 1)
    if passages > 1:
        # It was indexed in pieces and is served rejoined, which is worth one clause: a
        # reader who met this comment on a page as "passage 4 of 7" needs to know this is
        # not a fourth seventh.
        clauses.append(f"{passages} passages, rejoined")
    clauses.append(f"{len(text)} characters, whole")
    lines = [", ".join(clauses) + " --", quote(text)]
    if served.get("url"):
        # Once, on the one page a person is likely to want to cite or open.
        lines.append(served["url"])
    return lines


def _outline_lines(
    thread: dict[str, Any], outline: list[dict[str, Any]], *, presentation: bool = False
) -> list[str]:
    """One line per comment, for the whole thread (relore#70).

    The ``defs`` move applied to a discussion. The page serves ten of seventy and the cap
    is the contract (section 6), so the answer to "I need the other sixty" cannot be a
    bigger page -- it is a cheaper *complete* view, read to decide which comments to ask
    for. Measured on the run this came from: the agent spent six ``--focus`` calls and
    3,558 tokens taking overlapping samples of one 70-comment thread.

    Chronological and unranked, because ranking it is what ``--focus`` already does and
    the reason to read a whole thread is usually that ranking has not worked. Each line
    carries the id to ask for it by, so the next call is ``--after`` or a search, never a
    re-roll of the same lottery.

    A ``focus`` **narrows** it rather than ranking it (huggingface/relore#71), so the head
    line has two denominators to keep apart: how many rows are here, and how many comments
    they were drawn from. ``12 of 12 comments, carrying any of 'rope scaling' — 12 of 68
    do`` is complete; ``12 of 68`` alone would read as a cap.
    """
    total = int(thread.get("comments_total") or 0)
    suppressed = int(thread.get("comments_machine_suppressed") or 0)
    shown = len(outline)
    visible = max(total - suppressed, shown)
    focus = thread.get("focus") or ""
    matched = thread.get("outline_matched")
    widened = bool(thread.get("outline_widened"))
    # What the rows are *of*. A narrowed outline is still complete -- of the comments
    # carrying a word -- and the denominator has to say which of the two completenesses
    # this is, or a caller reads "12 of 68" as a cap (huggingface/relore#71).
    reachable = matched if (focus and matched and not widened) else visible
    clauses = [f"-- outline: {shown} of {reachable} comments, oldest first"]
    if focus and not widened:
        clauses.append(f"carrying any of {focus!r} — {matched} of {visible} do")
    elif focus:
        # Never an empty outline. The rows are the whole thread again, and the sentence
        # for that is the caller's question finding nothing -- not the thread being quiet.
        clauses.append(f"no comment carries any of {focus!r}, so this is the unnarrowed outline")
    if suppressed:
        clauses.append(f"{suppressed} machine-tier suppressed")
    if shown < reachable:
        # The outline exists to be complete, so the one case where it is not has to be
        # louder here than a cap normally would be.
        clauses.append(f"CAPPED at {shown}: {reachable - shown} more this view did not reach")
    if thread.get("indexed_at"):
        clauses.append(f"indexed at {thread['indexed_at']} ({THREAD_STAMP_NOTE})")
    lines = [", ".join(clauses) + " --"]
    for index, entry in enumerate(outline, start=1):
        tier = TRUST_LABEL.get(str(entry.get("trust")), str(entry.get("trust")))
        head = (
            f"{index:3}. {entry.get('id')}  [{tier}]  {entry.get('age')}  "
            f"{entry.get('source_type')}"
        )
        if entry.get("author"):
            head += f"  @{entry['author']}"
        if int(entry.get("passages") or 1) > 1:
            head += f"  ({entry['passages']} passages)"
        lines.append(head)
        lines.append(quote(str(entry.get("snippet", ""))))
    if presentation:
        lines += [
            "",
            'read one: `relore thread <n> --focus "<words from its line above>"`; '
            "sweep from here: `relore thread <n> --after <id>`"
            + ("" if focus else '; narrow this list: `--outline --focus "<words>"`'),
        ]
    return lines


def render_thread(
    payload: dict[str, Any],
    *,
    compact: bool = False,
    presentation: bool = False,
    files: bool = False,
) -> str:
    """One thread, with the cap stated rather than implied.

    A caller that cannot tell truncation from a quiet thread will read ten comments as the
    whole argument, so the count is on the page -- and so is every other reason a comment
    is not on it. The tier filter used to be the silent one: a thread whose only comment
    is machine-authored rendered as ``0 of 0``, which is the sentence for an untouched
    thread (huggingface/relore#28).

    The head line also carries when this thread was last rebuilt from GitHub
    (huggingface/relore#32). ``status`` has that number per ingestion *source*, which no
    reader can compose into an answer about one document -- and two runs have now drawn
    opposite wrong conclusions trying to.
    """
    thread = payload.get("thread") or {}
    lines = [
        f"{thread.get('repo')}#{thread.get('number')} {thread.get('type')}  "
        f"{thread.get('state')}  {thread.get('age')}",
        quote(thread.get("title", "")),
    ]
    if thread.get("comment"):
        # **A page's worth of preamble is not what "read this one comment" asked for.**
        # The events, the diff's counts, the links and the opening post are what a reader
        # weighs a *thread* by; a caller who named one comment has already been on that
        # page -- that is where the id came from. Two lines of identity so the quote can be
        # cited, then the comment. Measured on `transformers#46419`: 1,941 bytes with the
        # full header, 480 without (huggingface/relore#71).
        lines += ["", *_one_comment_lines(thread, presentation=presentation)]
        return envelope("\n".join(lines).rstrip(), source=thread.get("url"), compact=compact)
    if thread.get("author"):
        lines.append(f"opened by @{thread['author']}")
    lines += _event_lines(thread)
    if thread.get("labels"):
        lines.append(f"labels: {', '.join(thread['labels'])}")
    lines += _file_lines(thread, compact=compact, presentation=presentation, files=files)
    if thread.get("links"):
        lines.append("links: " + ", ".join(_link(link) for link in thread["links"]))
    lines += ["", quote(thread.get("body", ""))]
    if thread.get("body_truncated"):
        lines.append(
            f"(body truncated: {len(thread.get('body') or '')} of "
            f"{thread.get('body_chars')} characters."
            + (
                " `--full` serves the rest, which on an issue template is where the "
                "reproduction starts."
                if presentation
                else ""
            )
            + ")"
        )
    lines.append("")

    outline = thread.get("outline") or []
    if outline:
        lines += _outline_lines(thread, outline, presentation=presentation)
        return envelope("\n".join(lines).rstrip(), compact=compact)

    comments = thread.get("comments") or []
    returned = thread.get("comments_returned", len(comments))
    # `comments_total` is every comment on the thread. `visible` is how many of them this
    # trust floor admits, and it -- never the total -- is what the cap and the sampling
    # compose with: the machine tier is a separate question, not a page we ran out of room
    # for (huggingface/relore#28).
    total = thread.get("comments_total", 0)
    suppressed = int(thread.get("comments_machine_suppressed") or 0)
    visible = max(total - suppressed, returned)
    focus, matched = thread.get("focus") or "", thread.get("focus_matched")
    clauses = []
    if suppressed:
        # Filtered is not empty. `0 of 0` on a thread we hold one bot comment for asserts
        # the thread is untouched, which is the one thing it did not mean -- and it is the
        # failure signature #11 was opened to eliminate (huggingface/relore#28).
        clauses.append(
            f"{suppressed} machine-tier suppressed"
            + (
                " (`relore search --trust machine` asks what the bots claimed)"
                if presentation
                else ""
            )
        )
    if focus:
        clause = f"best first for {focus!r}"
        if matched is not None:
            clause += f" ({matched} of {visible} carry every term)"
        clauses.append(clause)
    if thread.get("after"):
        # Where this page starts. A page that silently begins in the middle is worse than
        # one that stops, and the count beside it is still "of the whole thread".
        clauses.append(f"sweeping in order from after {thread['after']}")
    elif not focus and thread.get("selection") == "ends+middle" and visible > returned:
        # The selection, named. Ten comments under a bare count read as the ten best, and
        # an agent that believes it has read the best ten stops (huggingface/relore#16).
        clauses.append("SAMPLED not ranked: the first and last few and a spread of the middle")
    # This is an indexing visit, not proof of completeness (huggingface/relore#46).
    if thread.get("indexed_at"):
        clauses.append(f"indexed at {thread['indexed_at']} ({THREAD_STAMP_NOTE})")
    head = ", ".join([f"-- {returned} of {total} comments", *clauses])
    lines.append(head + " --")
    if not returned and thread.get("after") and visible:
        # An empty sweep page is the one result here that can be read as "that was the end
        # of it" while being nothing of the kind: `--after` taken from a *sampled* page
        # starts at the thread's last comment, because that is where the sample ends. The
        # page has to say which of the two happened, or the caller stops having seen ten of
        # seventy and believing it read the thread (huggingface/relore#11).
        lines.append(
            f"nothing follows that comment. It is at or after the last of this thread's "
            f"{visible} — which is also what you get by sweeping from a SAMPLED page, "
            "whose last row is the thread's last comment. `--outline` lists every "
            "comment, in order, with the id to resume from."
        )
    for index, comment in enumerate(comments, start=1):
        lines += _comment_lines(index, comment, note=_passage_note(comment, comments))
    # **The anchor, once, and only when it is owed.** A window that ends in an ellipsis
    # says it was cut and nothing about undoing it, and until `--comment` nothing could:
    # `--full` is the opening post, `--focus` re-ranks and snippets again, `--after` serves
    # the *next* comments. The id in each head line is the address; this is the sentence
    # that makes it one, and it costs ~20 tokens where the ten URLs it replaced cost ~250
    # (huggingface/relore#71). Not printed on a page that withheld nothing.
    if any(c.get("truncated") for c in comments):
        cut = sum(1 for c in comments if c.get("truncated"))
        lines.append(
            f"({cut} of these {'is' if cut == 1 else 'are'} cut to a window: "
            "`--comment <id>` serves one whole, by the id in its line above.)"
        )
    if visible > returned:
        # **The way to the rest is a fact, in both forms** (huggingface/relore#71). This
        # line is the one place a caller learns what a capped page means, and it used to
        # end at "never returnable in full" -- true of *this* page and not of the verb,
        # since `--outline` reaches every comment it is hiding. Piped there was no hint at
        # all, so the flag was discoverable only by reading `--help`, which a caller reads
        # once already stuck: measured on one run, the agent found `--outline` at call 36
        # of 46 and had spent five `--focus` rolls on this exact thread first. The
        # `--focus` suggestion stays, and stays presentation, because it is advice about
        # what to want; naming the view that does not withhold is a property of the page.
        reach = (
            f" `--outline` lists all {visible} in one line each, with the id to read any "
            "of them by."
            if visible <= OUTLINE_CAP
            else f" `--outline` lists them {OUTLINE_CAP} at a time, oldest first, with the "
            "id to read any of them by; `--after <id>` sweeps the rest."
        )
        lines.append(
            f"({visible - returned} not shown: a page is never the whole thread."
            + reach
            + (
                ' `--focus "<what you care about>"` ranks all of them.'
                if presentation and not focus
                else ""
            )
            + ")"
        )
    return envelope("\n".join(lines).rstrip(), source=thread.get("url"), compact=compact)


def _passage_note(hit: dict[str, Any], page: list[dict[str, Any]]) -> str:
    """Whether this hit is a piece of a longer comment, and how to read the page if so.

    A 9,827-character comment is several indexed documents, and two of them came back as
    hits 1 and 2 with the same URL, author, age and tier -- which reads as two sources
    agreeing rather than one comment quoted twice (huggingface/relore#18). Both halves of
    that are worth saying: that the snippet is an excerpt of something longer, and that
    another slot on this page came from the same comment.
    """
    passages = int(hit.get("passages") or 0)
    index = int(hit.get("chunk_index") or 0)
    identity = (hit.get("url"), hit.get("source_id"))
    if not hit.get("source_id"):
        return ""
    same = sum(1 for other in page if (other.get("url"), other.get("source_id")) == identity)
    parts = []
    if passages > 1:
        parts.append(f"passage {index + 1} of {passages} in this comment")
    elif index:
        parts.append(f"passage {index + 1} in this comment")
    if same > 1:
        parts.append(f"{same} of its passages are on this page")
    return "; ".join(parts)


def _claim_state(claim: dict[str, Any]) -> list[str]:
    """``closed (duplicate) by @maintainer`` rather than ``closed`` (issue #22).

    Two claimants both reading ``closed`` is the state this verb is worst at: withdrawn by
    its author means review the survivor, ruled a duplicate means read the triage.
    """
    if claim.get("merged"):
        return [claim.get("state") or "", "merged"]
    parts = [claim.get("state") or "", "draft" if claim.get("draft") else ""]
    if claim.get("state") == "closed":
        reason, closer = claim.get("state_reason"), claim.get("closed_by")
        if reason and reason != "completed":
            parts.append(f"({reason.replace('_', ' ')})")
        if closer:
            parts.append("by its author" if closer == claim.get("author") else f"by @{closer}")
    elif claim.get("review_decision"):
        parts.append(f"({claim['review_decision'].replace('_', ' ')})")
    return parts


def _event_lines(thread: dict[str, Any]) -> list[str]:
    """How the thread got to its state, and what the reviewers settled (issue #22).

    `closed` is the same word for *withdrawn by its author* and *ruled a duplicate*, which
    imply opposite next actions; and a review decision is the only thing that tells
    ``0 of 0 comments`` apart from *approved without typing*.
    """
    lines = []
    if thread.get("merged"):
        lines.append("merged")
    elif thread.get("state") == "closed":
        reason, closer = thread.get("state_reason"), thread.get("closed_by")
        parts = ["closed"]
        if reason and reason != "completed":
            parts.append(f"as {reason.replace('_', ' ')}")
        if closer:
            parts.append("by its author" if closer == thread.get("author") else f"by @{closer}")
        if len(parts) > 1:
            lines.append(" ".join(parts))

    decision, by = thread.get("review_decision"), thread.get("review_decision_by") or []
    requested = thread.get("requested_reviewers") or []
    if decision:
        who = ", ".join(f"@{login}" for login in by)
        lines.append(f"review: {decision.replace('_', ' ')}" + (f" by {who}" if who else ""))
    elif requested:
        who = ", ".join(f"@{login}" for login in requested)
        lines.append(f"review: requested from {who}, no verdict yet")
    elif thread.get("type") == "pr" and thread.get("state") == "open":
        lines.append("review: nobody has approved or blocked it")
    return lines


def _link(link: dict[str, Any]) -> str:
    """One relationship edge. ``indexed`` is on the page because an unresolved target is
    the normal case on a sampled index, and "we have not indexed #12" is a different fact
    from "#12 does not exist"."""
    suffix = "" if link.get("indexed", True) else " (not indexed)"
    return f"{link['relationship']} #{link['target']}{suffix}"


def _file_lines(
    thread: dict[str, Any],
    *,
    compact: bool = False,
    presentation: bool = False,
    files: bool = False,
) -> list[str]:
    """The changed files and the mentioned ones, on separate lines, each with its meaning.

    **The counts and the caveats are always here; the diff's paths are behind ``--files``**
    (relore#70). Measured on one agent run asking why a line is written the way it is:
    ``thread 43121 --full`` spent **1,350 tokens on 98 paths** -- 51% of that page and 7%
    of every tool result in the whole run -- and not one of them was referred to again.
    What made this page load-bearing was never the paths: it is the *count* and the
    truncation notice, because a short list reads as a weak positive and a missing entry
    reads as a negative fact. Those cost a line and stay. ``--files`` is for the question
    the paths answer, which is a different question from the one this verb is usually
    asked. Only the diff's own page is gated: the anchored and mentioned lists are a
    handful of paths by nature and are each other's cross-check.

    **The withholding itself is a fact, and 0.3.15 shipped it as advice**
    (huggingface/relore#71). The pointer to the flag was ``presentation`` only, so the
    piped form -- which is the only form an agent ever reads -- said ``98 of 98 —
    complete`` and stopped: a page that had served no paths at all, announcing itself
    complete, with no sign that ninety-eight of them were one flag away. That is this
    project's recurring failure shape (section 13.3) reintroduced by a flag meant to cut
    tokens. *That* the paths are withheld and reachable is a caveat and is in both forms;
    only what to do about it would have been advice, and here the two are one clause.

    A short list of *hits* is read as a weak positive; a *missing entry* is read as a
    negative fact. ``huggingface/transformers#39847`` is the case this exists for: 323
    changed files, 100 collected because the per-PR pass takes one page, and no
    ``gpt_neox`` entry among them. The pull request does touch ``gpt_neox_japanese``, so
    the absence would have exonerated the change that caused the bug being diagnosed.

    The inverse error is the reason for the split (huggingface/relore#17). Merged into one
    array, five prose-derived filenames sat among a hundred real paths: ``config.json``,
    which that pull request does not touch, answered "did it change this?" with yes. A
    truncated list must not answer a membership question silently in *either* direction,
    so the two provenances are two lines and each says what it is.

    **Which 100** is its own fact (huggingface/relore#56). The page is
    ``files(first: 100)``, and GitHub serves a diff in path order -- checked against
    ``#39847``, whose first page is exactly the 100 byte-order-first of its 323 paths, in
    both the REST and the GraphQL form this pass uses. So the cut is a prefix, not a
    sample: everything after ``lfm2_moe`` was missing from that thread's list, including
    the one model directory the reader was there for. ``100 of 323`` alone reads as a
    spread, and a reader who believes it is a spread reads the gap as thin coverage rather
    than as a wall.
    """
    changed = thread.get("files_changed") or []
    anchored = thread.get("files_anchored") or []
    mentioned = thread.get("files_mentioned") or []
    total = thread.get("files_total")
    collected = thread.get("files_collected") or len(changed)
    if not changed and not anchored and not mentioned and total is None:
        return []

    lines = []
    if total is None:
        if changed:  # no `changed_files` count, yet a diff: an index predating migration 7
            lines.append(f"changed files: {len(changed)} (no total recorded for this thread)")
            lines.append("  " + ", ".join(changed))
    elif collected >= total:
        lines.append(f"changed files: {collected} of {total} — complete")
        # Not shaped under `--compact`: a complete list is the one that *can* answer a
        # membership question, and answering it is the whole of what the paths are for.
        if files:
            lines.append("  " + ", ".join(changed))
        else:
            # Both forms. *That* the paths were withheld is a caveat, not advice, and
            # 0.3.15 shipped it as advice: piped -- which is every agent -- the line read
            # `98 of 98 — complete` and nothing else, so a page that had served no paths
            # at all announced itself as complete (huggingface/relore#71).
            lines.append("  (paths withheld; `--files` serves them)")
    elif collected:
        lines.append(
            f"changed files: {collected} of {total} collected. TRUNCATED: a path that is "
            "absent here may still have been touched"
        )
        lines.append(
            f"  the collected page is the first {collected} in path order, cut after "
            f"{max(changed)} — a path sorting after that one is absent whether or not the "
            "thread touched it"
        )
        if files:
            lines += _path_block(changed, compact=compact, presentation=presentation)
        else:
            lines.append("  (paths withheld; `--files` serves the collected page)")
    else:
        lines.append(
            f"changed files: none collected of {total}, so this thread cannot answer "
            "whether it touched a path"
        )
    if anchored and lines:
        lines.append(
            f"  + {len(anchored)} more the diff must contain, from the files inline "
            "review comments are anchored to (not part of the collected page)"
        )
        lines.append("  " + ", ".join(anchored))
    elif anchored:
        # No changed-file line to continue from -- an issue, or a pull request whose
        # per-PR pass has not run. The anchors are still the diff, and saying so as a
        # fragment under nothing would read as a footnote to a list that is not there.
        lines.append(
            f"changed files: not collected for this thread, but {len(anchored)} are known "
            "from the files inline review comments are anchored to (a comment can only "
            "hang on a changed file). Absence is still not evidence"
        )
        lines.append("  " + ", ".join(anchored))
    if mentioned:
        lines.append(
            f"mentioned in the discussion: {len(mentioned)} — named by somebody, NOT the "
            "diff. A bare filename here is not evidence the thread changed it"
        )
        lines.append("  " + ", ".join(mentioned))
    return lines


def _path_block(paths: list[str], *, compact: bool, presentation: bool) -> list[str]:
    """The truncated changed-file list: the paths, or under ``--compact`` their shape.

    ``--compact`` is a context budget, and this list was spending it on the one thing
    huggingface/relore#11 forbids concluding from. On ``huggingface/transformers#39847``
    it is 5,320 of the 10,651 bytes a compact ``thread`` renders -- half the page for 100
    paths the line above them has just said prove nothing by their absence, and the 223
    the reader may not even ask about are not there at all (huggingface/relore#56). What
    that reader took from the list was *"a wide mechanical refactor across model
    directories"*, and that is a sentence. The paths stay in the default render and in
    ``--json``, which is where a caller that wants to grep them is already looking.

    Only this branch is shaped. A *complete* list answers membership and keeps its paths;
    the anchored and mentioned lists are short by construction and their value is the
    names themselves -- a bare ``config.json`` somebody typed does not have a shape.
    """
    if not compact:
        return ["  " + ", ".join(paths)]
    return [
        "  " + _path_shape(paths),
        "  (paths omitted under --compact"
        + ("; `--json` serves them, and so does the default render" if presentation else "")
        + ")",
    ]


#: How many prefix groups :func:`_path_shape` counts out before it totals the rest. Three
#: fits the shapes that occur -- source, tests, docs -- and the line stays one line, which
#: is the point of it.
SHAPE_GROUPS = 3


def _path_shape(paths: list[str]) -> str:
    """Where a list of paths sits and how far it spreads, in one line.

    Grouped by the first two segments, which is the level a repository's layout is legible
    at (``src/transformers/``, ``tests/models/``), and the directory count is the part
    that separates one edited package from a sweep across thirty-three of them. It is the
    two facts the full list was being read for, at ~2% of its bytes.
    """
    groups: dict[str, list[str]] = {}
    for path in paths:
        segments = path.split("/")
        prefix = "/".join(segments[: min(2, len(segments) - 1)])
        groups.setdefault(prefix + "/" if prefix else "", []).append(path)
    ranked = sorted(groups.items(), key=lambda group: (-len(group[1]), group[0]))
    parts = []
    for prefix, members in ranked[:SHAPE_GROUPS]:
        part = f"{len(members)} under {prefix}" if prefix else f"{len(members)} at the root"
        # Named only when it is a spread: "across 1 directories" is noise, and the count
        # is here to tell one package apart from ninety.
        directories = {member.rsplit("/", 1)[0] for member in members if "/" in member}
        if len(directories) > 1:
            part += f" across {len(directories)} directories"
        parts.append(part)
    if len(ranked) > SHAPE_GROUPS:
        rest = ranked[SHAPE_GROUPS:]
        count = sum(len(members) for _, members in rest)
        parts.append(f"{count} under {len(rest)} other prefixes")
    return ", ".join(parts)


def _chains(payload: dict[str, Any], *, presentation: bool = False) -> list[str]:
    """The two histories behind a line, as two lists that never merge.

    They answer different questions and a reader who cannot tell them apart cannot tell
    "this line was reformatted eight times" from "this behaviour was argued here". The
    revision chain follows the *range* and loses a block that was rewritten; the origin
    chain follows a *word* and loses one that was renamed. Neither is provenance, both are
    evidence, and the word is printed so the choice can be argued with rather than taken.
    """
    lines: list[str] = []
    history = payload.get("history") or []
    if len(history) > 1:
        # Blame is this list's first row, not the answer (relore#57). The rest is what a
        # reader has to see to know the top one is a refactor rather than a decision.
        lines += ["", f"-- this line has {len(history)} revisions, newest first --"]
        for commit in history:
            mark = f"#{commit['number']}" if commit.get("number") else "no pull request indexed"
            lines.append(f"   {str(commit.get('sha', ''))[:12]}  {commit.get('date')}  {mark}")
            lines.append(quote(str(commit.get("summary", ""))))

    origin = payload.get("origin") or []
    if origin:
        term = payload.get("origin_term") or "?"
        lines += [
            "",
            f"-- {len(origin)} commit(s) changed `{term}` in this file, newest first. "
            "This follows the word, not the line, so it reaches a block that was rewritten "
            "rather than moved --",
        ]
        for commit in origin:
            mark = f"#{commit['number']}" if commit.get("number") else "no pull request indexed"
            lines.append(f"   {str(commit.get('sha', ''))[:12]}  {commit.get('date')}  {mark}")
            lines.append(quote(str(commit.get("summary", ""))))
        if presentation:
            considered = payload.get("origin_considered") or []
            if considered:
                lines.append(
                    "   (tried: "
                    + ", ".join(f"{c.get('term')}={c.get('commits')}" for c in considered)
                    + f"; `{term}` reached furthest back without filling the page)"
                )
        return lines

    # **An empty origin used to render as nothing at all**, which is the answer
    # "the revision chain is the whole story" given by omission -- and it is not even
    # always that answer (huggingface/relore#71). Three causes, three sentences, because a
    # reader cannot separate them from a silence and they do not mean the same thing.
    declined = payload.get("origin_declined") or ""
    if declined == "comment":
        lines.append(
            "(no origin chain: this line carries no code, so there is no behaviour on it "
            "to follow back. The revisions above are its whole history.)"
        )
    elif declined == "no-candidate":
        tried = payload.get("origin_considered") or []
        lines.append(
            "(no origin chain: no word on this line reaches further back than the "
            f"revisions above, so they are the whole story — not a gap. {len(tried)} "
            "candidate(s) tried.)"
        )
    elif declined == "unreadable":
        lines.append(
            "(no origin chain: this file could not be read from the working clone, so the "
            "word history was never asked for. Absence here is not evidence.)"
        )
    return lines


def render_why(
    payload: dict[str, Any], *, compact: bool = False, presentation: bool = False
) -> str:
    """Why one line is the way it is (issue #9).

    Blame's commit first, then the pull request that carried it, then the part blame
    cannot give: what reviewers said *on this line* while it was being written.

    The review comments are the only unbounded prose on this page, so they are what
    ``--compact`` trims -- and every trimmed one is counted at the bottom, because a
    reader who cannot tell a shortened comment from a whole one reads the tail it never
    got as something nobody said (huggingface/relore#55).
    """
    blame = payload.get("blame") or {}
    thread = payload.get("thread") or {}
    lines = [
        f"{payload.get('repo')} {payload.get('path')}:{payload.get('line')}",
        quote(blame.get("text", "")),
        "",
        f"last changed in {blame.get('sha', '')[:12]} by {blame.get('author', 'someone')}",
        quote(blame.get("summary", "")),
    ]
    if payload.get("number") is None:
        lines.append(
            "\nno pull request in this index carries that commit, so the argument behind "
            "it is not retrievable here. It may predate the index, or have reached the "
            "branch outside a pull request."
        )
        # The chains, still -- and especially here. This branch used to return on that
        # sentence, which threw away both of them on the one page where they are the whole
        # remaining recourse: blame's own commit is unresolvable, and a revision or a
        # pickaxe hit that *is* indexed is the only way left in.
        lines += _chains(payload, presentation=presentation)
        return envelope("\n".join(lines).rstrip(), compact=compact)

    guessed = (
        " (from the commit subject, not a staged commit row)"
        if (payload.get("resolved_by") == "summary")
        else ""
    )
    lines += [
        "",
        f"{thread.get('repo')}#{payload['number']} {thread.get('state', '')}{guessed}",
        quote(thread.get("title", "")),
    ]
    if thread.get("url"):
        lines.append(thread["url"])
    if thread.get("links"):
        lines.append("links: " + ", ".join(_link(link) for link in thread["links"]))

    lines += _chains(payload, presentation=presentation)

    anchored = payload.get("anchored") or []
    lines += ["", f"-- {len(anchored)} review comment(s) on this line while it was written --"]
    trimmed = 0
    for index, comment in enumerate(anchored, start=1):
        tier = TRUST_LABEL.get(str(comment.get("trust")), str(comment.get("trust")))
        head = f"{index}. [{tier}]  {comment.get('age')}"
        if comment.get("author"):
            head += f"  @{comment['author']}"
        if comment.get("line"):
            head += f"  (line {comment['line']})"
        text = str(comment.get("text", ""))
        if compact:
            text, was_trimmed = trim(text)
            trimmed += was_trimmed
        lines += [head, quote(text)]
        if comment.get("url"):
            lines.append(f"   {comment['url']}")
        lines.append("")
    if trimmed:
        lines.append(
            f"({trimmed} comment{'' if trimmed == 1 else 's'} shortened to {COMPACT_CHARS} "
            "characters by --compact"
            + ("; `--json` serves them whole" if presentation else "")
            + ")"
        )
    lines += _widened(payload, compact=compact)
    if presentation and payload.get("level") != "line":
        # A next command, never a full stop (relore#64): the verb is holding the thread it
        # just fetched, and an agent told only that a window was empty goes looking for
        # what it was handed -- eleven calls of it, measured.
        lines.append(
            f"(nothing was said on this line itself. `relore thread {payload['number']} "
            "--full` reads the discussion this came from.)"
        )
    return envelope("\n".join(lines).rstrip(), source=thread.get("url"), compact=compact)


def _widened(payload: dict[str, Any], *, compact: bool) -> list[str]:
    """The levels past the line window, each labelled with what it is.

    Printed always, not only when the line window came back empty: a comment on this file
    and an approving review are evidence about the same code either way, and the reader
    who most needs them is the one who thinks two anchored comments were the whole story.
    """
    out: list[str] = []
    for key, heading in (
        ("on_file", "elsewhere in this file, in the same pull request"),
        ("reviews", "reviews on this pull request"),
    ):
        group = payload.get(key) or []
        if not group:
            continue
        out += ["", f"-- {len(group)} {heading} --"]
        for index, item in enumerate(group, start=1):
            tier = TRUST_LABEL.get(str(item.get("trust")), str(item.get("trust")))
            head = f"{index}. [{tier}]  {item.get('age')}"
            if item.get("author"):
                head += f"  @{item['author']}"
            if item.get("state"):
                head += f"  {str(item['state']).lower()}"
            if item.get("line"):
                head += f"  (line {item['line']})"
            text = str(item.get("text", ""))
            if compact:
                text, _ = trim(text)
            out += [head, quote(text)]
            if item.get("url"):
                out.append(f"   {item['url']}")
    return out


def render_inflight(
    payload: dict[str, Any], *, compact: bool = False, presentation: bool = False
) -> str:
    """What already claims to close a thread.

    Empty is the answer this verb exists to give, so it has to be a sentence rather than a
    blank page -- and it has to be distinguishable from *unanswerable*: an index with no
    relationship rows at all says so, because "nobody is working on this" and "this index
    cannot tell you" imply opposite next actions.
    """
    claims = payload.get("claims") or []
    subject = f"{payload.get('repo')}#{payload.get('number')}"
    total = payload.get("claims_total", len(claims))
    if not claims:
        lines = [f"nothing in the index claims to close {subject}."]
        if not payload.get("links_indexed"):
            lines.append(
                "(and this repository has no relationship rows at all, so that is not an "
                "answer yet"
                + (": re-derive it with `relored derive` to fill them" if presentation else "")
                + ".)"
            )
        return envelope("\n".join(lines), compact=compact)

    shortened = 0
    claim_word = "thread claims" if total == 1 else "threads claim"
    lines = [f"{total} {claim_word} to close {subject}", ""]
    for index, claim in enumerate(claims, start=1):
        state = " ".join(part for part in _claim_state(claim) if part)
        head = (
            f"{index}. {claim.get('repo')}#{claim.get('number')} {claim.get('type')}  "
            f"{state}  {claim.get('age')}  {claim.get('relationship')}"
        )
        if claim.get("author"):
            head += f"  @{claim['author']}"
        lines.append(head)
        if claim.get("title"):
            # The one unbounded string on this page. It used to be served whole under
            # `--compact` because the flag's only effect here was dropping the envelope
            # sentence -- which is to say the page's entire "compact saving" was its
            # untrusted-content warning. That sentence stays now, so the trimming has to
            # be real.
            title = str(claim["title"])
            if compact:
                title, was_trimmed = trim(title)
                shortened += was_trimmed
            lines.append(f"   {quote(title)}")
        if claim.get("url"):
            lines.append(f"   {claim['url']}")
        lines.append("")
    if total > len(claims):
        lines.append(f"({total - len(claims)} more, not shown.)")
    if shortened:
        lines.append(
            f"({shortened} title{'' if shortened == 1 else 's'} shortened to {COMPACT_CHARS} "
            "characters by --compact)"
        )
    return envelope("\n".join(lines).rstrip(), compact=compact)


def render_status(payload: dict[str, Any]) -> str:
    """No envelope: this is our own numbers, not retrieved text."""
    backend = payload.get("backend") or {}
    schema = payload.get("schema") or {}
    lines = [
        f"version   {payload.get('version')}",
        f"backend   {backend.get('name')} / {backend.get('ranking')}  "
        f"capabilities: {', '.join(backend.get('capabilities') or []) or 'none'}",
        f"schema    applied {schema.get('applied')}, pending {schema.get('pending')}",
        f"index     {payload.get('threads')} threads, {payload.get('documents')} documents, "
        f"{payload.get('raw_objects')} raw objects",
    ]
    lines.append(payload.get("passes_note") or COLLECTOR_NOTE)
    for row in payload.get("passes") or []:
        lines.append(
            f"  {row['repo']} [{row['pass']}] high-water {row['high_water']} "
            f"last-ok {row['last_ok_at']}"
            + (f" ({note})" if (note := row.get("note") or pass_note(row["pass"])) else "")
        )
    usage = payload.get("usage")
    if usage:
        lines.append(
            f"quota     {usage['per_minute']['used']}/{usage['per_minute']['limit']} per minute, "
            f"{usage['daily']['used']}/{usage['daily']['limit']} today"
        )
    return "\n".join(lines)
