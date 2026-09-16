"""What the text output *says*, given a response.

The renderer takes the JSON shapes of section 7 as plain dictionaries, so these are unit
tests with no database and no server: what is asserted here is the part an agent reads,
and specifically the part that says how much of the answer it is not being shown.

With no MCP server (``api/ui.py``), stdout is the agent-facing contract. A count or a
caveat that exists only in ``--json`` is a caveat the CLI's callers do not have.
"""

from __future__ import annotations

from relore.render import (
    COMPACT_CHARS,
    OUTLINE_CAP,
    render_inflight,
    render_search,
    render_thread,
    render_why,
)
from relore.security.untrusted import BEGIN, END


def _thread(**fields):
    base = {
        "repo": "owner/name",
        "number": 1,
        "type": "pr",
        "title": "a refactor",
        "state": "closed",
        "age": "3d",
        "body": "the opening",
        "comments": [],
        "comments_returned": 0,
        "comments_total": 0,
    }
    return {"thread": {**base, **fields}}


# -- the piped grammar (issue #13) -----------------------------------------


def test_the_piped_form_carries_no_advice_and_no_backend_tag() -> None:
    """With no MCP server stdout is an API, so the piped form is a contract: facts only,
    and nothing that moves when a flag is renamed."""
    payload = {
        "query": {"text": "rope"},
        "backend": {"name": "postgresql", "ranking": "ts_rank_cd"},
        "hits": [],
    }

    assert "[postgresql/ts_rank_cd]" not in render_search(payload)
    assert "[postgresql/ts_rank_cd]" in render_search(payload, presentation=True)


def test_a_hit_keeps_its_field_order_in_both_forms() -> None:
    hit = {
        "repo": "owner/name",
        "number": 48630,
        "type": "pr",
        "trust": "authoritative",
        "age": "3d",
        "source_type": "review_comment",
        "author": "ydshieh",
        "snippet": "this cast is load-bearing",
        "url": "https://github.com/owner/name/pull/48630",
    }
    payload = {"query": {"text": "cast"}, "backend": {}, "hits": [hit]}

    for out in (render_search(payload), render_search(payload, presentation=True)):
        head = [line for line in out.splitlines() if line.startswith("1. ")][0]
        assert head.startswith("1. owner/name#48630 pr  [authoritative]  3d  review_comment")
        assert head.endswith("@ydshieh")


def test_an_unanswerable_inflight_states_the_fact_in_both_forms() -> None:
    page = {"repo": "owner/name", "number": 1, "claims": [], "links_indexed": 0}

    piped, terminal = render_inflight(page), render_inflight(page, presentation=True)
    assert "no relationship rows at all" in piped and "no relationship rows at all" in terminal
    assert "relored derive" not in piped
    assert "relored derive" in terminal


# -- events (issue #22) ----------------------------------------------------


def test_a_closure_says_who_and_why() -> None:
    out = render_thread(
        _thread(author="truongsontung", state_reason="duplicate", closed_by="ydshieh")
    )
    assert "closed as duplicate by @ydshieh" in out


def test_a_thread_closed_by_its_author_says_so() -> None:
    out = render_thread(_thread(author="truongsontung", closed_by="truongsontung"))
    assert "closed by its author" in out


def test_a_review_decision_is_shown_with_no_comments_at_all() -> None:
    """`-- 0 of 0 comments --` is true and cannot say whether nobody looked or nobody
    typed. The decision is the only thing that separates them."""
    out = render_thread(_thread(review_decision="approved", review_decision_by=["ydshieh"]))
    assert "0 of 0 comments" in out
    assert "review: approved by @ydshieh" in out


def test_an_open_pull_request_nobody_has_reviewed_says_that() -> None:
    out = render_thread(_thread(state="open", requested_reviewers=["vasqu"]))
    assert "review: requested from @vasqu, no verdict yet" in out
    assert "review: nobody has approved or blocked it" in render_thread(_thread(state="open"))


# -- the changed-file list -------------------------------------------------


def test_a_truncated_file_list_says_it_is_truncated() -> None:
    """`huggingface/transformers#39847`: 323 changed files, 100 collected, and the absence
    of a `gpt_neox` path read as evidence the pull request did not touch it."""
    out = render_thread(
        _thread(
            files_changed=[f"f{i}.py" for i in range(100)], files_total=323, files_collected=100
        )
    )

    assert "100" in out and "323" in out
    assert "TRUNCATED" in out


def test_a_truncated_file_list_names_the_path_it_was_cut_after() -> None:
    """`100 of 323` reads as a spread. It is a prefix: the per-PR pass takes GitHub's first
    page and GitHub serves a diff in path order, so everything sorting after the last
    collected path is missing whether or not it was touched. On the thread this came from
    the cut was at `lfm2_moe` and the one interesting model was `mistral4`
    (huggingface/relore#56)."""
    payload = _thread(
        files_changed=["src/a/one.py", "src/a/two.py"], files_total=323, files_collected=2
    )

    for out in (render_thread(payload), render_thread(payload, compact=True)):
        assert "cut after src/a/two.py" in out
        assert "absent whether or not the thread touched it" in out


def test_compact_serves_a_truncated_file_list_as_its_shape() -> None:
    """Half of a compact `thread 39847` was 100 paths the line above them says prove
    nothing by their absence -- a budget flag spending its budget on the one list nothing
    may be concluded from (huggingface/relore#11, huggingface/relore#56)."""
    paths = [f"src/transformers/models/m{i}/modeling_m{i}.py" for i in range(100)]
    paths += ["tests/models/test_rope.py", "docs/source/en/rope.md"]
    payload = _thread(files_changed=paths, files_total=323, files_collected=102)

    # `--files`, because the paths are behind it now (relore#70) — what is under test
    # here is what `--compact` does to the list once it has been asked for.
    full, out = (
        render_thread(payload, files=True),
        render_thread(payload, compact=True, files=True),
    )

    assert "100 under src/transformers/ across 100 directories" in out
    # The cut point stays -- it is a caveat, not a path in a list.
    assert "modeling_m50.py" not in out, "the shape replaces the paths"
    assert "paths omitted under --compact" in out
    assert len(out) < len(full) / 3
    # The count, the denominator and the caveat are facts, so `--compact` keeps them.
    assert "102 of 323 collected" in out and "TRUNCATED" in out
    assert "src/transformers/models/m0/modeling_m0.py" in full


def test_compact_keeps_a_complete_file_list() -> None:
    """The shape is what a list that cannot answer a membership question is worth. A
    complete one can, and answering it is the whole of what the paths are for."""
    payload = _thread(files_changed=["src/a.py", "src/b.py"], files_total=2, files_collected=2)

    out = render_thread(payload, compact=True, files=True)

    assert "src/a.py, src/b.py" in out
    assert "paths omitted" not in out


def test_the_shape_counts_directories_only_when_they_spread() -> None:
    """One edited package and a sweep across thirty-three are the two readings the line
    exists to separate; `across 1 directories` separates nothing and costs tokens."""
    payload = _thread(
        files_changed=["pkg/mod/a.py", "pkg/mod/b.py", "README.md"],
        files_total=9,
        files_collected=3,
    )

    out = render_thread(payload, compact=True, files=True)

    assert "2 under pkg/mod/, 1 at the root" in out
    assert "across" not in out.split("cut after")[1].split("(paths omitted")[0]


def test_a_complete_file_list_does_not_cry_truncation() -> None:
    out = render_thread(_thread(files_changed=["a.py", "b.py"], files_total=2, files_collected=2))

    assert "complete" in out
    assert "TRUNCATED" not in out


def test_the_diff_and_the_discussion_are_two_lists() -> None:
    """A filename somebody typed is not a changed file, and merged into one array it
    answered "did this pull request touch it?" with yes. `config.json` is the case: named
    in `huggingface/transformers#39847`'s discussion, absent from its 323-file diff."""
    out = render_thread(
        _thread(
            files_changed=["src/transformers/modeling_rope_utils.py"],
            files_mentioned=["config.json", "modeling_rope_utils.py"],
            files_total=323,
            files_collected=1,
        )
    )

    changed, mentioned = out.index("changed files:"), out.index("mentioned in the discussion:")
    assert changed < mentioned
    assert "NOT the diff" in out
    # The bare basename and the real path are both present and on different lines: the
    # thing that made them indistinguishable was sharing one array.
    assert out.index("src/transformers/modeling_rope_utils.py") < mentioned
    assert "config.json" in out[mentioned:]


def test_an_anchor_with_no_collected_diff_is_a_sentence_not_a_dangling_fragment() -> None:
    """An issue, or a pull request whose per-PR pass has not run, has anchors and no
    changed-file line for them to continue -- and `  + 1 more…` under nothing reads as a
    footnote to a list that is not on the page. Found rendering a real serge thread."""
    out = render_thread(_thread(files_anchored=["reviewbot/llm_client.py"], files_total=None))

    assert "not collected for this thread" in out
    assert "can only hang on a changed file" in out
    assert "  + 1 more" not in out


def test_a_thread_with_only_mentioned_paths_claims_no_diff() -> None:
    """An issue has no changed-file list, and a pull request whose per-PR pass has not run
    yet has an empty one -- neither is "this thread touched a file"."""
    out = render_thread(_thread(files_mentioned=["a.py"], files_total=None, files_collected=0))
    assert "mentioned in the discussion" in out
    assert "changed files:" not in out

    unvisited = render_thread(_thread(files_mentioned=["a.py"], files_total=12, files_collected=0))
    assert "none collected of 12" in unvisited
    assert "cannot answer whether it touched a path" in unvisited


# -- the body --------------------------------------------------------------


def test_a_truncated_body_says_how_much_is_missing_and_how_to_get_it() -> None:
    payload = _thread(body="x" * 800, body_chars=5214, body_truncated=True)

    piped = render_thread(payload)
    assert "5214" in piped, "the cap is a fact, so it is in both forms"
    assert "--full" not in piped

    assert "--full" in render_thread(payload, presentation=True)


def test_an_untruncated_body_is_quiet() -> None:
    out = render_thread(_thread(body="short", body_chars=5, body_truncated=False))

    assert "truncated" not in out


# -- the focus denominator -------------------------------------------------


def test_a_focus_that_matched_nothing_says_so_next_to_the_comments() -> None:
    out = render_thread(
        _thread(
            focus="what is the resolution",
            focus_matched=0,
            comments_returned=3,
            comments_total=30,
            comments=[{"repo": "owner/name", "number": 1, "snippet": "a comment"}] * 3,
        )
    )

    assert "0 of 30 carry every term" in out


def test_an_unfocused_thread_still_suggests_a_focus() -> None:
    payload = _thread(comments_returned=10, comments_total=30)

    assert "--focus" in render_thread(payload, presentation=True)
    assert "20 not shown" in render_thread(payload), "the cap is a fact; the flag is advice"


# -- the tier filter, announced (huggingface/relore#28) -------------------


def test_a_capped_page_names_the_view_that_does_not_withhold() -> None:
    """The truncated page is the one place a caller learns what to do next, and it named
    `--focus` -- which returns ten of seventy too, so the caller reformulates and gets
    another ten. Measured: five `--focus` rolls on one thread before `--outline` was
    reached at call 36 of 46 (huggingface/relore#71). A fact, so it is in both forms."""
    payload = _thread(comments_returned=10, comments_total=68)

    for out in (render_thread(payload), render_thread(payload, presentation=True)):
        assert "`--outline` lists all 68 in one line each" in out
    assert "a thread is never returnable in full" not in render_thread(payload), (
        "true of this page, not of the verb: --outline reaches every comment it hides"
    )


def test_a_page_too_long_to_outline_whole_is_told_how_to_sweep_it() -> None:
    payload = _thread(comments_returned=10, comments_total=644)

    out = render_thread(payload)

    assert f"`--outline` lists them {OUTLINE_CAP} at a time" in out
    assert "`--after <id>` sweeps the rest" in out


def test_a_thread_whose_only_comment_is_a_bots_does_not_claim_to_be_empty() -> None:
    """`0 of 0` is the sentence for a thread nobody has touched, and it was printed for a
    thread the index holds one machine-tier comment for. An agent reading it cannot tell
    whether CI has spoken or whether the thread is genuinely untouched -- which is the
    failure signature #11 was opened to eliminate."""
    payload = _thread(comments_returned=0, comments_total=1, comments_machine_suppressed=1)

    piped = render_thread(payload)
    assert "0 of 1 comments" in piped
    assert "1 machine-tier suppressed" in piped, "the filter is a fact, so it is in both forms"
    assert "--trust machine" not in piped, "how to see them is advice"

    assert "--trust machine" in render_thread(payload, presentation=True)


def test_a_suppressed_comment_is_not_counted_as_one_the_cap_dropped() -> None:
    """`not shown` means the page ran out of room, which is a different fact and a
    different next action from a tier this view never admits."""
    out = render_thread(
        _thread(comments_returned=0, comments_total=1, comments_machine_suppressed=1),
        presentation=True,
    )

    assert "not shown" not in out
    assert "--focus" not in out, "ranking cannot surface a comment the floor excluded"


def test_both_axes_are_stated_when_both_apply() -> None:
    payload = _thread(
        selection="ends+middle",
        comments_returned=10,
        comments_total=90,
        comments_machine_suppressed=1,
    )

    out = render_thread(payload)
    assert "10 of 90 comments" in out
    assert "1 machine-tier suppressed" in out
    assert "SAMPLED not ranked" in out
    assert "79 not shown" in out, "the cap dropped 79 of the 89 the floor admits, not 80"


def test_a_thread_with_nothing_suppressed_says_nothing_about_tiers() -> None:
    out = render_thread(_thread(comments_returned=2, comments_total=2))

    assert "machine-tier" not in out


# -- freshness travels with the answer (huggingface/relore#32) ------------


def test_the_comment_page_says_what_it_is_current_to() -> None:
    """`status` reports freshness per ingestion source, and no reader can compose those
    rows into "are the comments on *this* thread current?". Two field runs tried and drew
    opposite wrong conclusions -- one trusting stale comments, one discounting complete
    ones. Per-response, the question has an answer."""
    out = render_thread(
        _thread(comments_returned=2, comments_total=2, indexed_at="2026-09-09T09:02:49Z")
    )

    assert (
        "-- 2 of 2 comments, indexed at 2026-09-09T09:02:49Z "
        "(body and comments last indexed; later GitHub changes unknown) --"
    ) in out


def test_an_index_with_no_stamp_says_nothing_rather_than_none() -> None:
    assert "indexed at" not in render_thread(_thread(indexed_at=None))


# -- what the ten comments actually are (huggingface/relore#16, #18) -------


def test_an_unfocused_page_says_it_is_a_sample_and_not_a_ranking() -> None:
    """`-- 10 of 89 comments --` is indistinguishable from a ranked top ten, and was read
    as one: the agent concluded the thread held nothing better and stopped, while the
    review that answered its question sat at position 51 of 97."""
    payload = _thread(selection="ends+middle", comments_returned=10, comments_total=89)

    assert "SAMPLED not ranked" in render_thread(payload), "which ten these are is a fact"
    assert "--focus" in render_thread(payload, presentation=True)


def test_a_focused_page_is_not_called_a_sample() -> None:
    out = render_thread(
        _thread(selection="focus", focus="rope", focus_matched=2, comments_total=89)
    )

    assert "SAMPLED" not in out
    assert "best first" in out


def test_a_chunked_comment_says_which_passage_it_is() -> None:
    """Two hits with one URL, one author and one tier read as two people agreeing. They
    were two pieces of a 9,827-character comment, one of them prose from inside a
    collapsed `<details>` block."""
    passage = {
        "repo": "owner/name",
        "number": 1,
        "url": "https://github.com/owner/name/pull/1#issuecomment-3223571427",
        "source_type": "issue_comment",
        "source_id": "3223571427",
        "snippet": "a passage",
        "passages": 7,
    }
    out = render_search({"hits": [{**passage, "chunk_index": 3}, {**passage, "chunk_index": 4}]})

    assert "passage 4 of 7 in this comment" in out
    assert "passage 5 of 7 in this comment" in out
    assert "2 of its passages are on this page" in out


def test_an_unchunked_comment_says_nothing_about_passages() -> None:
    out = render_search(
        {
            "hits": [
                {
                    "repo": "owner/name",
                    "number": 1,
                    "source_type": "issue_comment",
                    "source_id": "42",
                    "chunk_index": 0,
                    "passages": 1,
                    "snippet": "the whole comment",
                }
            ]
        }
    )

    assert "passage" not in out


# -- the diff's paths are behind a flag (huggingface/relore#70) -------------


def test_the_file_count_and_its_caveat_are_never_behind_a_flag() -> None:
    """What made this page load-bearing was never the paths: it is the count and the
    truncation notice, because a short list reads as a weak positive and a missing entry
    reads as a negative fact. Those cost a line and stay."""
    payload = _thread(
        files_changed=["src/aaa.py", "src/zzz.py"], files_total=323, files_collected=2
    )

    out = render_thread(payload)

    assert "2 of 323 collected" in out and "TRUNCATED" in out
    # `src/zzz.py` survives as the *cut point*, which is a caveat rather than a path in a
    # list. The list itself is gone.
    assert "cut after src/zzz.py" in out
    assert "src/aaa.py" not in out


def test_the_paths_come_back_with_files() -> None:
    payload = _thread(files_changed=["src/a.py", "src/b.py"], files_total=2, files_collected=2)

    assert "src/a.py" not in render_thread(payload)
    assert "src/a.py, src/b.py" in render_thread(payload, files=True)


def test_a_complete_list_says_where_the_paths_went_in_both_forms() -> None:
    """*That* the paths were withheld is a caveat, not advice (huggingface/relore#71).

    0.3.15 put the pointer behind `presentation`, so the piped form -- the only form an
    agent reads -- said `1 of 1 — complete` and nothing else: a page that had served no
    paths at all, announcing itself complete.
    """
    payload = _thread(files_changed=["src/a.py"], files_total=1, files_collected=1)

    for out in (render_thread(payload), render_thread(payload, presentation=True)):
        assert "paths withheld" in out and "`--files`" in out


def test_a_truncated_list_says_the_paths_were_withheld_too() -> None:
    payload = _thread(files_changed=["src/a.py", "src/z.py"], files_total=300, files_collected=2)

    assert "paths withheld" in render_thread(payload)
    assert "src/a.py" not in render_thread(payload)


def test_the_anchored_and_mentioned_paths_are_not_gated() -> None:
    """Only the diff's own page is gated. These two are a handful of paths by nature and
    are each other's cross-check (huggingface/relore#17)."""
    payload = _thread(
        files_changed=["src/aaa.py", "src/zzz.py"],
        files_total=9,
        files_collected=2,
        files_anchored=["src/anchored.py"],
        files_mentioned=["src/mentioned.py"],
    )

    out = render_thread(payload)

    assert "src/anchored.py" in out and "src/mentioned.py" in out
    assert "src/aaa.py" not in out, "the diff's page is gated; these two are not"


# -- the comment line, and the address on it (huggingface/relore#71) -------


def test_a_comment_on_a_thread_page_does_not_reprint_the_page() -> None:
    """A search hit has to say which thread it came from; a comment on a thread page does
    not, and said it anyway. Measured on five real pages, `owner/repo#N pr`, the title
    re-quoted under every comment and a 76-character URL were 24-33% of the whole page."""
    payload = _thread(
        comments_returned=1,
        comments_total=1,
        comments=[
            {
                "repo": "owner/name",
                "number": 1,
                "type": "pr",
                "title": "a refactor",
                "source_id": "2086096507",
                "trust": "authoritative",
                "age": "16mo",
                "source_type": "review_comment",
                "author": "maintainer",
                "url": "https://github.com/owner/name/pull/1#discussion_r2086096507",
                "snippet": "we can improve that description",
            }
        ],
    )

    out = render_thread(payload)
    body = out.split("-- 1 of 1 comments")[1]

    assert "2086096507  [authoritative]  16mo  review_comment  @maintainer" in body
    assert "discussion_r2086096507" not in body, "the id is the address; the URL is not"
    assert "a refactor" not in body, "the title is on the page's own head line"
    assert "owner/name#1" not in body, "so is the thread"


def test_a_search_hit_still_carries_the_thread_it_came_from() -> None:
    """The other half: `search` spans threads, so every one of those three is load-bearing
    there. One grammar per question, not one grammar."""
    out = render_search(
        {
            "query": {"text": "rope"},
            "hits": [
                {
                    "repo": "owner/name",
                    "number": 7,
                    "type": "pr",
                    "title": "a refactor",
                    "trust": "authoritative",
                    "age": "3d",
                    "source_type": "issue_comment",
                    "url": "https://github.com/owner/name/pull/7#issuecomment-1",
                    "snippet": "the answer",
                }
            ],
        }
    )

    assert "owner/name#7" in out and "a refactor" in out and "issuecomment-1" in out


def test_a_page_that_cut_a_comment_says_how_to_read_it_whole() -> None:
    """A window ends in an ellipsis, which says *that* it was cut and nothing about undoing
    it -- and until `--comment` nothing could. One line, and a fact, so both forms."""
    payload = _thread(
        comments_returned=2,
        comments_total=2,
        comments=[
            {"source_id": "11", "snippet": "short", "trust": "reported", "age": "3d"},
            {
                "source_id": "12",
                "snippet": "a long one…",
                "trust": "reported",
                "age": "3d",
                "truncated": True,
            },
        ],
    )

    for out in (render_thread(payload), render_thread(payload, presentation=True)):
        assert "1 of these is cut to a window" in out
        assert "`--comment <id>` serves one whole" in out


def test_a_page_that_withheld_nothing_does_not_advertise_the_address() -> None:
    """Most insight for the fewest tokens: the line is owed only where something was cut."""
    payload = _thread(
        comments_returned=1,
        comments_total=1,
        comments=[{"source_id": "11", "snippet": "short", "trust": "reported", "age": "3d"}],
    )

    assert "--comment" not in render_thread(payload)


def test_a_comment_served_whole_is_not_wrapped_in_a_page() -> None:
    """A caller who named one comment has already been on the page -- that is where the id
    came from. Two lines of identity so the quote can be cited, then the comment."""
    payload = _thread(
        comment="12",
        comment_status="served",
        body="the whole opening post, which is not what was asked for",
        files_changed=["src/a.py"],
        files_total=1,
        files_collected=1,
        labels=["bug"],
        comments=[
            {
                "source_id": "12",
                "trust": "authoritative",
                "age": "4mo",
                "source_type": "issue_comment",
                "author": "maintainer",
                "snippet": "the whole comment, all of it",
                "passages": 3,
            }
        ],
    )

    out = render_thread(payload)

    assert "-- comment 12, issue_comment, [authoritative], 4mo, @maintainer" in out
    assert "3 passages, rejoined" in out and "28 characters, whole" in out
    assert "the whole comment, all of it" in out
    assert "the whole opening post" not in out
    assert "changed files" not in out and "labels" not in out


def test_an_id_this_thread_does_not_hold_says_which_kind_of_id_it_wanted() -> None:
    payload = _thread(comment="999", comment_status="absent", comments=[])

    out = render_thread(payload)

    assert "this thread holds no comment with that id" in out
    assert "not a position, and not the thread number" in out


def test_a_suppressed_comment_is_not_reported_as_a_missing_one() -> None:
    """Opposite next actions: stop looking, or ask what the bot claimed. An empty answer
    cannot tell them apart (section 6.2, huggingface/relore#28)."""
    payload = _thread(comment="13", comment_status="suppressed", comments=[])

    piped = render_thread(payload)

    assert "MACHINE" in piped and "excludes it" in piped
    assert "--trust machine" not in piped, "the remedy is advice"
    assert "--trust machine" in render_thread(payload, presentation=True)


# -- the outline (huggingface/relore#70) -----------------------------------


def test_the_outline_replaces_the_page_rather_than_joining_it() -> None:
    """Two views of the same comments on one page would be the cost of both and the use of
    neither."""
    payload = _thread(
        comments_total=70,
        outline=[
            {
                "id": "1",
                "author": "bob",
                "trust": "reported",
                "age": "3d",
                "source_type": "issue_comment",
                "snippet": "the first one",
            },
        ],
        comments=[{"repo": "o/n", "number": 1, "snippet": "a served comment"}],
    )

    out = render_thread(payload)

    assert "the first one" in out
    assert "a served comment" not in out


def test_an_outline_that_did_not_reach_the_end_shouts_about_it() -> None:
    """The outline's whole promise is completeness, so the case where it cannot deliver has
    to be louder than a cap normally is."""
    payload = _thread(
        comments_total=644,
        outline=[
            {
                "id": str(n),
                "author": "bob",
                "trust": "reported",
                "age": "3d",
                "source_type": "issue_comment",
                "snippet": f"comment {n}",
            }
            for n in range(100)
        ],
    )

    out = render_thread(payload)

    assert "outline: 100 of 644 comments" in out
    assert "CAPPED at 100: 544 more this view did not reach" in out


def _outline_rows(n, start=0):
    return [
        {
            "id": str(i),
            "author": "bob",
            "trust": "reported",
            "age": "3d",
            "source_type": "issue_comment",
            "snippet": f"comment {i}",
        }
        for i in range(start, start + n)
    ]


def test_a_narrowed_outline_says_what_it_is_complete_of() -> None:
    """`12 of 68` reads as a cap. A narrowed outline is complete -- of the comments
    carrying a word -- and the page has two denominators to keep apart
    (huggingface/relore#71)."""
    payload = _thread(
        comments_total=68,
        focus="rope scaling",
        outline_matched=12,
        outline=_outline_rows(12),
    )

    out = render_thread(payload)

    assert "outline: 12 of 12 comments" in out
    assert "carrying any of 'rope scaling' — 12 of 68 do" in out
    assert "CAPPED" not in out


def test_a_narrowed_outline_that_is_still_capped_counts_against_the_matches() -> None:
    payload = _thread(
        comments_total=644,
        focus="cache",
        outline_matched=130,
        outline=_outline_rows(100),
    )

    out = render_thread(payload)

    assert "outline: 100 of 130 comments" in out
    assert "CAPPED at 100: 30 more this view did not reach" in out


def test_a_focus_that_carried_nothing_widens_instead_of_emptying() -> None:
    """An empty outline would read as "no comment in this thread mentions that", which is
    the one thing it did not mean (huggingface/relore#47, one verb over)."""
    payload = _thread(
        comments_total=8,
        focus="quantization",
        outline_matched=0,
        outline_widened=True,
        outline=_outline_rows(8),
    )

    out = render_thread(payload)

    assert "no comment carries any of 'quantization', so this is the unnarrowed outline" in out
    assert "outline: 8 of 8 comments" in out
    assert "CAPPED" not in out


def test_a_complete_outline_does_not_cry_truncation() -> None:
    payload = _thread(
        comments_total=2,
        outline=[
            {
                "id": "1",
                "author": "bob",
                "trust": "reported",
                "age": "3d",
                "source_type": "issue_comment",
                "snippet": "one",
            },
            {
                "id": "2",
                "author": "bob",
                "trust": "reported",
                "age": "3d",
                "source_type": "issue_comment",
                "snippet": "two",
            },
        ],
    )

    assert "CAPPED" not in render_thread(payload)


# -- nothing matched -------------------------------------------------------


def test_no_hits_is_a_sentence_not_an_empty_page() -> None:
    out = render_search({"query": {"text": "nothing"}, "hits": []})

    assert "nothing matched" in out


def test_a_widened_page_says_it_was_widened() -> None:
    """The page in front of the caller is not the page they asked for: a hit carrying two
    terms of seven has to be read as one, and silence would let it read as a whole match."""
    out = render_search(
        {"query": {"text": "a b c", "widened": "any-term"}, "hits": [{"repo": "o/n", "number": 1}]}
    )

    assert "widened" in out
    assert "nothing carried every term" in out


def test_a_page_that_answered_says_nothing_about_widening() -> None:
    """It is a disclosure, not a banner. A query that worked was never widened, so the
    sentence would be false as well as noise."""
    out = render_search({"query": {"text": "a b c"}, "hits": [{"repo": "o/n", "number": 1}]})

    assert "widened" not in out


def test_an_empty_widened_page_says_the_cheaper_question_was_asked_too() -> None:
    """The answer that saves a turn: the caller's words are not what zeroed this, so the
    next turn should not be another reformulation of them."""
    out = render_search({"query": {"text": "a b c", "widened": "any-term"}, "hits": []})

    assert "nothing matched, with every term or with any of them." in out


def test_an_empty_page_echoes_the_filters_that_produced_it() -> None:
    """huggingface/relore#47: a *correct* `--error` can zero a query that answers at rank 1
    without it. With only the query text on the page that reads as an absence in the corpus
    rather than as a property of the question."""
    out = render_search(
        {
            "query": {
                "text": "size of tensor must match",
                "kind": "failure",
                "filters": {"symbols": ["GPTNeoXJapaneseAttention"], "errors": ["RuntimeError"]},
                "since": "2026-03-01T00:00:00+00:00",
            },
            "hits": [],
        }
    )

    assert "--kind failure" in out
    assert "--symbol GPTNeoXJapaneseAttention" in out
    assert "--error RuntimeError [normalized]" in out, "section 5.3's form, not the input"
    assert "--since 2026-03-01T00:00:00+00:00" in out
    assert "(they AND)" in out


def test_a_page_with_hits_does_not_echo_the_filters() -> None:
    """They are the diagnosis of a zero, and every line costs the caller tokens forever
    (section 6's caps). A page that answered needs no diagnosis."""
    out = render_search(
        {
            "query": {"text": "rope", "filters": {"errors": ["RuntimeError"]}},
            "hits": [{"repo": "o/n", "number": 1}],
        }
    )

    assert "--error" not in out


def test_an_out_of_scope_page_says_the_scope_is_empty() -> None:
    """Section 11 fails closed, and a closed scope and a quiet corpus render identically
    unless the page says which happened."""
    out = render_search({"query": {"text": "rope", "repos": []}, "hits": []})

    assert "no repository in scope" in out


def test_an_unresolvable_commit_still_gets_both_chains() -> None:
    """The page where they matter most. Blame's own commit is in no indexed pull request,
    so the revisions and the pickaxe are the only way in -- and this branch used to return
    on the apology and throw both away."""
    out = render_why(
        {
            "blame": {"sha": "0" * 40, "author": "someone", "summary": "s", "text": "t"},
            "number": None,
            "history": [
                {"sha": "a" * 12, "date": "2026-01-15", "number": 43121, "summary": "refactor"},
                {"sha": "b" * 12, "date": "2026-01-07", "number": 43126, "summary": "moes"},
            ],
            "origin": [{"sha": "c" * 12, "date": "2025-05-22", "number": 37866, "summary": "why"}],
            "origin_term": "fullgraph",
        }
    )

    assert "no pull request in this index carries that commit" in out
    assert "this line has 2 revisions" in out
    assert "changed `fullgraph` in this file" in out
    assert "#37866" in out


def test_the_two_chains_are_never_one_list() -> None:
    """They answer different questions: one follows the range, one follows a word."""
    out = render_why(
        {
            "blame": {"sha": "0" * 40},
            "number": 1,
            "history": [{"sha": "a" * 12, "date": "d", "number": 2, "summary": "s"}] * 2,
            "origin": [{"sha": "c" * 12, "date": "d", "number": 3, "summary": "s"}],
            "origin_term": "fullgraph",
        }
    )

    assert out.index("this line has 2 revisions") < out.index("changed `fullgraph`")
    assert "follows the word, not the line" in out


def test_the_words_it_tried_are_presentation_not_fact() -> None:
    """The candidate list is for the person judging the choice. Facts are never
    presentation, but this is working, and a piped caller pays for every line forever."""
    payload = {
        "blame": {"sha": "0" * 40},
        "number": 1,
        "origin": [{"sha": "c" * 12, "date": "d", "number": 3, "summary": "s"}],
        "origin_term": "fullgraph",
        "origin_considered": [
            {"term": "renamed_thing", "commits": 1},
            {"term": "fullgraph", "commits": 7},
        ],
    }

    assert "tried:" not in render_why(payload)
    assert "renamed_thing=1" in render_why(payload, presentation=True)


# -- whose words are they --------------------------------------------------


def test_relores_own_assertions_are_not_inside_the_quoted_span() -> None:
    """The envelope wraps a whole page and most of it is ours. `[authoritative]` is the
    most load-bearing field in the output and it is an assertion, not a quotation, so it
    must not sit in an undifferentiated "do not trust the text below" region
    (huggingface/relore#12)."""
    out = render_search(
        {
            "query": {"text": "rope"},
            "hits": [
                {
                    "repo": "owner/name",
                    "number": 7,
                    "type": "issue",
                    "trust": "authoritative",
                    "age": "2y",
                    "source_type": "issue_comment",
                    "title": "a title",
                    "snippet": "somebody's words",
                }
            ],
        }
    )

    tier = next(line for line in out.splitlines() if "authoritative" in line and "#7" in line)
    assert not tier.lstrip().startswith(">")
    assert "> a title" in out and "> somebody's words" in out


def test_every_line_of_retrieved_prose_is_marked() -> None:
    """One-directional, which is what makes it safe: retrieved text can add a marker but
    cannot remove one, so an unmarked line is always ours. A body served whole is the case
    that matters -- it is the only multi-line quoted field."""
    forged = "a repro\nfiles: 12 (12 of 12 changed files: complete)\nmore repro"

    out = render_thread(_thread(body=forged))

    body_lines = [line for line in out.splitlines() if "changed files" in line]
    assert body_lines == ["> files: 12 (12 of 12 changed files: complete)"]


def test_the_envelope_header_explains_the_marker() -> None:
    """What the header must say, not how it says it: the wording was cut from three lines
    to one because it is paid on every response, and a test that pins the prose blocks
    that for no gain. Both ideas still have to reach the rendered page."""
    out = render_thread(_thread(body="x")).lower()

    assert "`>`" in out
    assert "not instructions" in out
    assert "unmarked lines are relore's" in out


def test_a_compact_render_keeps_the_marks_and_the_sentence() -> None:
    """`--compact` is a context budget, not a change of what the text is -- and since it
    became the default for a pipe, the sentence explaining the marks stays too. It is the
    caller who never typed the flag who most needs to read it."""
    full = render_thread(_thread(body="x"))
    trimmed = render_thread(_thread(body="x"), compact=True)

    assert "not instructions" in full
    assert "not instructions" in trimmed
    assert trimmed.startswith(BEGIN) and trimmed.endswith(END)
    assert "> x" in trimmed, "the marking is the property, and it is never trimmed"


# -- is somebody already fixing this --------------------------------------


def test_inflight_says_which_pull_request_and_what_state_it_is_in() -> None:
    out = render_inflight(
        {
            "repo": "owner/name",
            "number": 48630,
            "claims": [
                {
                    "repo": "owner/name",
                    "number": 48672,
                    "type": "pr",
                    "title": "fix: respect partial_rotary_factor",
                    "author": "blipbyte",
                    "state": "open",
                    "draft": True,
                    "merged": False,
                    "age": "8h",
                    "relationship": "closes",
                }
            ],
            "claims_returned": 1,
            "claims_total": 1,
            "links_indexed": 12,
        }
    )

    assert "1 thread claims to close owner/name#48630" in out
    assert "open draft" in out  # a stale draft and an approved PR imply opposite actions
    assert "> fix: respect partial_rotary_factor" in out


def test_inflight_says_how_a_closed_claimant_was_closed() -> None:
    """#22: two claimants both reading `closed` was the whole ambiguity. Withdrawn by its
    author means review the survivor; ruled a duplicate means read the triage."""

    def claim(**fields):
        base = {
            "repo": "owner/name",
            "number": 48672,
            "type": "pr",
            "title": "fix it",
            "author": "truongsontung",
            "state": "closed",
            "draft": False,
            "merged": False,
            "age": "26h",
            "relationship": "closes",
        }
        return {**base, **fields}

    page = {"repo": "owner/name", "number": 48630, "claims_total": 2, "links_indexed": 12}
    withdrawn = render_inflight({**page, "claims": [claim(closed_by="truongsontung")]})
    duplicate = render_inflight(
        {**page, "claims": [claim(state_reason="duplicate", closed_by="ydshieh")]}
    )

    assert "closed by its author" in withdrawn
    assert "closed (duplicate) by @ydshieh" in duplicate


def test_inflight_distinguishes_a_clean_answer_from_an_unanswerable_one() -> None:
    """ "Nobody is working on this" and "this index cannot tell you" are opposite
    instructions, and both come back as an empty list."""
    clean = render_inflight({"repo": "owner/name", "number": 1, "claims": [], "links_indexed": 12})
    unanswerable = render_inflight(
        {"repo": "owner/name", "number": 1, "claims": [], "links_indexed": 0}
    )

    assert "nothing in the index claims to close" in clean
    assert "no relationship rows" not in clean
    assert "no relationship rows" in unanswerable


# -- --compact, on every page that has one (huggingface/relore#55) ---------


def test_the_compact_budget_is_the_one_the_server_applies_to_a_snippet() -> None:
    """Two constants with one meaning: `render` may not import `relore.search` (the module
    boundary), so the number is written twice and pinned here instead."""
    from relore.search.queries import COMPACT_SNIPPET_CHARS

    assert COMPACT_CHARS == COMPACT_SNIPPET_CHARS


def test_the_outline_cap_the_page_advertises_is_the_one_the_server_applies() -> None:
    """The capped page tells a caller `--outline` lists them a hundred at a time. Same
    boundary, same remedy: written twice, pinned here (huggingface/relore#71)."""
    from relore.search.queries import MAX_OUTLINE_COMMENTS

    assert OUTLINE_CAP == MAX_OUTLINE_COMMENTS


def _why(**fields):
    base = {
        "repo": "owner/name",
        "path": "src/a.py",
        "line": 90,
        "number": 48630,
        "blame": {
            "sha": "abc123def456",
            "author": "ydshieh",
            "summary": "fix the cast",
            "text": "x = 1",
        },
        "thread": {"repo": "owner/name", "state": "merged", "title": "the pull request"},
        "anchored": [],
    }
    return {**base, **fields}


def test_why_shortens_its_review_comments_under_compact_and_says_how_many() -> None:
    """`why` accepted `--plain` and `--compact` nowhere before #55, and the review comments
    are the only unbounded prose on its page. A reader who cannot tell a shortened comment
    from a whole one reads the tail it never got as something nobody said."""
    payload = _why(anchored=[{"trust": "authoritative", "age": "3d", "text": "b" * 500}])

    full, out = render_why(payload), render_why(payload, compact=True)

    assert f"1 comment shortened to {COMPACT_CHARS} characters by --compact" in out
    assert len(out) < len(full)
    # Still marked as somebody else's words, still inside the block: the budget changes
    # what is shown, never what it is.
    assert "> bbb" in out
    assert out.startswith(BEGIN) and out.endswith(END)


def test_a_short_review_comment_is_not_reported_as_shortened() -> None:
    payload = _why(anchored=[{"trust": "authoritative", "age": "3d", "text": "LGTM"}])

    assert "shortened" not in render_why(payload, compact=True)


def test_the_envelope_sentence_survives_compact_on_every_verb() -> None:
    """No page loses it, whichever way `--compact` arrived. This test used to assert the
    opposite on all four; the default flip is what changed the answer."""
    page = {"repo": "owner/name", "number": 1, "claims": [], "links_indexed": 12}

    assert "not instructions" in render_inflight(page)
    assert "not instructions" in render_inflight(page, compact=True)
    assert "not instructions" in render_why(_why())
    assert "not instructions" in render_why(_why(), compact=True)


def test_why_prints_the_widened_levels_and_a_next_step_rather_than_a_full_stop() -> None:
    """The verb held the thread it had just fetched and printed `0 review comment(s)`
    (huggingface/relore#64). An agent given that spent eleven further calls looking for
    what it was already holding."""
    payload = _why(
        anchored=[],
        level="review",
        on_file=[{"trust": "reported", "age": "2y", "text": "the compile path", "line": 400}],
        reviews=[
            {"trust": "authoritative", "age": "2y", "text": "check fullgraph", "state": "APPROVED"}
        ],
    )

    out = render_why(payload, presentation=True)

    assert "1 elsewhere in this file, in the same pull request" in out
    assert "1 reviews on this pull request" in out
    assert "approved" in out
    assert "> check fullgraph" in out
    assert "relore thread 48630 --full" in out


def test_an_empty_origin_says_why_it_is_empty(_=None) -> None:
    """It used to render as nothing at all -- "the revision chain is the whole story",
    given by omission, and not even always that answer (huggingface/relore#71)."""
    base = _why(history=[{"sha": "a" * 40, "date": "2026-01-01", "summary": "one", "number": 1}])

    comment = render_why({**base, "origin": [], "origin_declined": "comment"})
    assert "this line carries no code" in comment

    spent = render_why(
        {**base, "origin": [], "origin_declined": "no-candidate", "origin_considered": [{}, {}]}
    )
    assert "no word on this line reaches further back" in spent
    assert "2 candidate(s) tried" in spent

    gone = render_why({**base, "origin": [], "origin_declined": "unreadable"})
    assert "could not be read from the working clone" in gone
    assert "Absence here is not evidence" in gone


def test_an_origin_that_answered_says_nothing_about_declining() -> None:
    out = render_why(
        _why(
            origin=[{"sha": "b" * 40, "date": "2024-03-06", "summary": "the one", "number": 9}],
            origin_term="fullgraph",
            origin_declined="",
        )
    )

    assert "no origin chain" not in out


def test_why_lists_the_lines_revisions_when_there_is_more_than_one() -> None:
    """Blame is the first row of this list, never the whole of it (#57)."""
    payload = _why(
        history=[
            {"sha": "a" * 40, "date": "2026-01-15", "summary": "tidy (#43121)", "number": 43121},
            {"sha": "b" * 40, "date": "2025-05-22", "summary": "rewrite (#37866)", "number": 37866},
        ]
    )

    out = render_why(payload)

    assert "this line has 2 revisions, newest first" in out
    assert "#37866" in out and "2025-05-22" in out


def test_one_revision_is_not_announced_as_a_history() -> None:
    """A line touched once has no chain to show, and a heading over a single row is noise
    on every `why` that was already right."""
    out = render_why(_why(history=[{"sha": "a" * 12, "date": "2026-01-15", "summary": "x"}]))

    assert "revisions" not in out


def test_compact_shortens_a_long_claim_title_and_counts_it() -> None:
    """`--compact` on `inflight` used to do exactly one thing: drop the envelope sentence.
    So the page's whole compact saving was its untrusted-content warning, and with that
    sentence kept the flag would have become a no-op here."""
    page = {
        "repo": "owner/name",
        "number": 1,
        "links_indexed": 12,
        "claims": [
            {"repo": "owner/name", "number": 2, "title": "t" * 500, "relationship": "closes"}
        ],
    }

    full, out = render_inflight(page), render_inflight(page, compact=True)

    assert len(out) < len(full)
    assert f"1 title shortened to {COMPACT_CHARS} characters by --compact" in out
    assert "not instructions" in out, "the warning is not what a budget trims"


def test_collector_times_cannot_be_mistaken_for_thread_freshness() -> None:
    from relore.render import render_status

    rows = [
        dict(repo="o/n", **{"pass": name}, high_water="2026-09-09", last_ok_at="2026-09-13")
        for name in ("threads", "issue_comments", "pr_comments")
    ]
    out = render_status({"passes": rows})
    assert "collector telemetry (not response freshness" in out
    assert "high-water is a source cursor; last-ok is collector completion" in out
    assert out.count("per-thread refreshes run via [threads] and do not update this row") == 2
    assert "[threads] high-water 2026-09-09 last-ok 2026-09-13\n" in out


def test_outline_also_labels_the_indexing_visit() -> None:
    out = render_thread(_thread(outline=[], indexed_at="2026-09-13T13:37:34Z", comments_total=0))
    assert "indexed at 2026-09-13T13:37:34Z" in out
    assert "body and comments last indexed; later GitHub changes unknown" in out
    assert "current to" not in out
