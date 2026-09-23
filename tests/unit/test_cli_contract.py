"""Contract bits that hold from commit one, so they cannot quietly regress later."""

from __future__ import annotations

import sys

import pytest

from relore import cli, daemon
from relore.code import registry


@pytest.mark.parametrize("mod", [cli, daemon])
def test_version_flag_exits_clean(mod, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        mod.main(["--version"])
    assert exc.value.code == 0


@pytest.mark.parametrize("verb", ["map", "defs", "refs"])
def test_code_verbs_without_a_parser_give_an_install_hint(monkeypatch, tmp_path, verb) -> None:
    """Never an ImportError traceback -- AGENTS.md, and the build plan section 2.

    Both halves are removed, because either alone can answer: no grammar (so the shipped
    tree-sitter provider fails to construct, and is skipped with a warning) *and* no ctags
    fallback. That is the only state in which there is genuinely nothing to say but "install
    something", and asserting it with a grammar still reachable would assert nothing.

    ``map`` is in here for a specific near-miss: it walks a tree, nothing claims any file,
    so before ``registry.require_any`` it returned an empty map and exit 0 -- a missing
    install wearing the answer to a quiet repository.
    """
    monkeypatch.setitem(sys.modules, "tree_sitter", None)
    monkeypatch.setenv(registry.DISABLE_CTAGS_ENV, "1")
    registry.reset()
    # A file that exists, so `defs` fails for the reason under test rather than on the read.
    source = tmp_path / "m.py"
    source.write_text("def f():\n    pass\n")
    argv = {"map": [str(tmp_path)], "defs": [str(source)], "refs": ["f"]}[verb]

    with pytest.raises(SystemExit) as exc:
        cli.main([verb, *argv])

    assert "pip install" in str(exc.value)
    registry.reset()


@pytest.mark.parametrize(
    "verb",
    ["search", "thread", "inflight", "precedent", "why", "status", "map", "defs", "refs"],
)
def test_every_documented_verb_is_registered(verb: str) -> None:
    actions = [a for a in cli.build_parser()._actions if hasattr(a, "choices") and a.choices]
    assert any(verb in a.choices for a in actions), f"{verb} is documented but not in the parser"


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "refs", "helper"],
        ["refs", "helper", "--json"],
        ["refs", "--json", "helper"],
    ],
)
def test_a_global_flag_is_accepted_on_either_side_of_the_verb(argv: list[str]) -> None:
    """`relore search ... --json` is what anyone writes by analogy with `git` and `gh`, and
    argparse called the flag *unrecognized* rather than misplaced -- which sends the reader
    looking for a typo (huggingface/relore#6)."""
    assert cli.build_parser().parse_args(argv).json is True


#: One minimal invocation per verb, so the matrix below can be asserted rather than
#: sampled. A verb added without a row here fails `test_every_verb_is_exercised_below`.
MINIMAL = {
    "search": ["search", "rope"],
    "thread": ["thread", "1"],
    "inflight": ["inflight", "1"],
    "precedent": ["precedent"],
    "why": ["why", "a.py:1"],
    "status": ["status"],
    "map": ["map"],
    "defs": ["defs", "a.py"],
    "refs": ["refs", "helper"],
    "symbol": ["symbol", "a.f"],
    "grep": ["grep", "rope"],
    "copies": ["copies", "f"],
}


def _verbs() -> list[str]:
    subparsers = next(a for a in cli.build_parser()._actions if hasattr(a, "choices") and a.choices)
    return list(subparsers.choices)


def test_every_verb_is_exercised_below() -> None:
    assert set(_verbs()) == set(MINIMAL), "a verb was added without a row in MINIMAL"


def test_thread_takes_the_two_ways_past_the_page_cap_and_the_file_flag() -> None:
    """huggingface/relore#70. The cap stays (section 6); these are how a caller gets the
    rest of a thread without it being raised, and `--files` is what the diff's path list
    costs when nobody asked for it."""
    args = cli.build_parser().parse_args(
        ["thread", "1", "--outline", "--after", "123456", "--files"]
    )

    assert args.outline is True
    assert args.after == "123456"
    assert args.files is True


def test_thread_narrows_an_outline_with_the_focus_it_already_has() -> None:
    """One flag, two behaviours, argued at `_outline`: a focus orders a *page* and narrows
    an *outline*. It is not a second flag because the caller's question is the same one
    (huggingface/relore#71)."""
    args = cli.build_parser().parse_args(["thread", "1", "--outline", "--focus", "rope"])

    assert (args.outline, args.focus) == (True, "rope")


def test_thread_addresses_one_comment_by_its_id() -> None:
    """The id in a page's head line is the address, and an address is only one if something
    takes it (huggingface/relore#71)."""
    args = cli.build_parser().parse_args(["thread", "1", "--comment", "2086096507"])

    assert args.comment == "2086096507"


def test_thread_defaults_to_the_page_it_always_served() -> None:
    """All four are additive: the default page is what it was before any of this."""
    args = cli.build_parser().parse_args(["thread", "1"])

    assert (args.outline, args.after, args.files, args.comment) == (False, "", False, "")


def _dest(flag: str) -> str:
    """argparse's own transformation, which a flag with an internal dash needs:
    `--no-compact` parses into `no_compact`."""
    return flag.lstrip("-").replace("-", "_")


@pytest.mark.parametrize("verb", sorted(MINIMAL))
@pytest.mark.parametrize("flag", [flag for flag, _ in cli.GLOBAL_FLAGS])
def test_every_global_flag_is_accepted_after_every_verb(verb: str, flag: str) -> None:
    """The matrix, asserted rather than sampled (huggingface/relore#55).

    The globals landed one `add_parser` at a time and ended up per-verb: `--json` on all
    twelve, `--compact` on the three that existed when #6 was closed -- one of them a stub
    -- and `--plain` on none. A flag that works after `thread` and fails after `grep` has
    no rule a caller can learn, and argparse calls it *unrecognized* rather than misplaced,
    so the repair an agent reaches for is to delete it. On `grep --compact` that is the
    opposite of what it wanted.
    """
    argv = [*MINIMAL[verb], flag, *(["http://127.0.0.1:9"] if flag == "--api" else [])]

    args = cli.build_parser().parse_args(argv)

    assert getattr(args, _dest(flag)) not in (False, None)


@pytest.mark.parametrize("flag", [flag for flag, _ in cli.GLOBAL_FLAGS])
def test_the_subparser_alias_does_not_overwrite_a_flag_given_before_the_verb(flag: str) -> None:
    """The trap in accepting it twice: a subparser default would silently turn
    `relore --json search` back off."""
    before = [flag, *(["http://127.0.0.1:9"] if flag == "--api" else []), "search", "anything"]

    args = cli.build_parser().parse_args(before)

    assert getattr(args, _dest(flag)) not in (False, None)
    # And nothing else was turned on by being adjacent to it.
    on = [f for f, _ in cli.GLOBAL_FLAGS if getattr(args, _dest(f)) not in (False, None)]
    assert on == [flag]


@pytest.mark.parametrize("verb", ["precedent"])
def test_an_unimplemented_verb_says_so_in_its_help(verb: str) -> None:
    """An agent builds its plan from `--help`, so a stub has to say it is one. `why` was
    on this list until #9 built it."""
    parser = cli.build_parser()
    subparsers = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    entry = next(c for c in subparsers._choices_actions if c.dest == verb)

    assert "NOT IMPLEMENTED" in (entry.help or "")


def test_why_is_implemented_and_takes_a_path_and_a_line() -> None:
    args = cli.build_parser().parse_args(["why", "src/model.py:90"])

    assert args.location == "src/model.py:90"
    assert "NOT IMPLEMENTED" not in (
        next(
            c
            for c in next(
                a for a in cli.build_parser()._actions if hasattr(a, "choices") and a.choices
            )._choices_actions
            if c.dest == "why"
        ).help
        or ""
    )


def test_precedent_takes_a_limit_like_every_other_listing_verb() -> None:
    assert cli.build_parser().parse_args(["precedent", "--limit", "3"]).limit == 3


def test_the_default_api_is_the_deployment_and_is_overridable() -> None:
    """The default moved to `ghlore.huggingface.tech` on 2026-09-11, when the name began
    to resolve. It is pinned here because it was reverted once for a good reason — the name
    existed and did not resolve — so a future change to it should be deliberate rather than
    incidental. The override is the part that has to keep working: a caller with their own
    index must never be silently pointed at production.
    """
    assert cli.DEFAULT_API == "https://ghlore.huggingface.tech"
    assert cli.build_parser().parse_args(["--api", "http://127.0.0.1:9999", "status"]).api == (
        "http://127.0.0.1:9999"
    )


# -- RELORE_REPO (issue #36) -----------------------------------------------


@pytest.mark.parametrize("verb,argv", [("thread", ["42"]), ("why", ["a.py:1"]), ("grep", ["x"])])
def test_the_environment_supplies_the_repository_when_the_flag_does_not(
    monkeypatch, verb: str, argv: list[str]
) -> None:
    """The per-call tax this removes: a daemon serving three repositories makes `--repo`
    mandatory on every bare number, and one field run passed it on ~20 consecutive calls."""
    monkeypatch.setenv(cli.REPO_ENV, "huggingface/transformers")
    args = cli.build_parser().parse_args([verb, *argv])

    assert cli._repo(args) == "huggingface/transformers"


def test_the_flag_beats_the_environment(monkeypatch) -> None:
    """So a session with a default set can still ask about another repository without
    unsetting anything."""
    monkeypatch.setenv(cli.REPO_ENV, "huggingface/transformers")
    args = cli.build_parser().parse_args(["thread", "42", "--repo", "huggingface/trl"])

    assert cli._repo(args) == "huggingface/trl"


def test_an_empty_variable_reads_as_unset_rather_than_as_a_repository(monkeypatch) -> None:
    """`export RELORE_REPO=` must hand the question back to the daemon, not ask it about a
    repository named "", which is a 404 with a confusing sentence in it."""
    monkeypatch.setenv(cli.REPO_ENV, "")

    assert cli._repo(cli.build_parser().parse_args(["thread", "42"])) is None
    assert cli._repos(cli.build_parser().parse_args(["search", "x"])) == []


def test_search_takes_the_default_too_but_the_flag_replaces_it(monkeypatch) -> None:
    """`search` spans the whole scope by design, so this is the one verb the variable
    *narrows*. That is what naming the session's repository asks for."""
    monkeypatch.setenv(cli.REPO_ENV, "huggingface/transformers")

    assert cli._repos(cli.build_parser().parse_args(["search", "x"])) == [
        "huggingface/transformers"
    ]
    assert cli._repos(
        cli.build_parser().parse_args(["search", "x", "--repo", "huggingface/trl"])
    ) == ["huggingface/trl"]


def test_the_environment_never_moves_defs_and_refs_off_the_working_tree(monkeypatch) -> None:
    """For these two `--repo` does not name a repository, it *switches the verb* to the
    daemon's clone of HEAD. The tree you are standing in, uncommitted edits included, is the
    only thing they offer that nothing else does, so only the flag may give it up."""
    monkeypatch.setenv(cli.REPO_ENV, "huggingface/transformers")

    assert cli.build_parser().parse_args(["defs", "a.py"]).repo is None
    assert cli.build_parser().parse_args(["refs", "helper"]).repo is None


# -- what the code verbs print ---------------------------------------------


def test_grep_names_both_denominators() -> None:
    """Issue #37: the summary read `3 in 4 files`, where 4 counted the files *searched*.
    The natural reading is the other one, and an agent sizing blast radius acts on it."""
    text = cli._grep_text(
        {
            "hits": [
                {"path": "a.py", "line": 1, "text": "x"},
                {"path": "a.py", "line": 2, "text": "x"},
                {"path": "b.py", "line": 9, "text": "x"},
            ],
            "files_searched": 4,
        }
    )

    assert text.splitlines()[-1] == "-- 3 hits in 2 of 4 files searched"


def test_grep_says_one_hit_rather_than_1_hits() -> None:
    text = cli._grep_text({"hits": [{"path": "a.py", "line": 1, "text": "x"}], "files_searched": 1})

    assert text.splitlines()[-1] == "-- 1 hit in 1 of 1 file searched"


def test_grep_still_discloses_its_cap_with_the_remedy_in_it() -> None:
    """The one thing this command already got right, and worth keeping while rewording
    around it: the cap says so, inline, with what to do about it.

    The remedy moved behind `presentation` when `--plain` became accepted on this verb
    (huggingface/relore#55) -- the cap is a fact and is printed either way, `--path` is
    advice. A caveat is never dropped for brevity; a suggestion is not a caveat.
    """
    payload = {"hits": [], "files_searched": 3, "truncated": True}

    assert "(capped: narrow it with --path)" in cli._grep_text(payload, presentation=True)
    assert "(capped)" in cli._grep_text(payload)


def test_grep_shortens_its_matched_lines_under_compact_and_counts_them() -> None:
    """`--compact` was rejected by `grep` and accepted by `thread`, so an agent that had
    seen the second write the first and was told the flag was *unrecognized*
    (huggingface/relore#55). It is accepted now, and the matched line -- 300 characters of
    a minified bundle, at worst -- is what there is to trim here."""
    payload = {"hits": [{"path": "a.py", "line": 1, "text": "x" * 400}], "files_searched": 1}

    text = cli._grep_text(payload, compact=True)

    assert "…" in text
    assert f"1 line shortened to {cli.COMPACT_CHARS} characters by --compact" in text
    assert len(text) < len(cli._grep_text(payload))
    # The row and both denominators survive: the flag shortens prose, it never drops a hit.
    assert "a.py:1" in text and "-- 1 hit in 1 of 1 file searched" in text


def test_a_short_grep_line_is_not_marked_as_shortened() -> None:
    payload = {"hits": [{"path": "a.py", "line": 1, "text": "rope"}], "files_searched": 1}

    assert "shortened" not in cli._grep_text(payload, compact=True)


def test_symbol_keeps_its_body_whole_under_compact() -> None:
    """The body is the answer this verb exists to give. A shortened one is a wrong answer,
    not a cheaper one, so `--compact` is accepted here and deliberately does nothing."""
    payload = {
        "path": "a.py",
        "start_line": 1,
        "kind": "function",
        "qualname": "f",
        "body": "def f():\n    " + "return 1  # " + "x" * 400,
    }

    assert payload["body"] in cli._symbol_text(payload)


def test_symbol_names_the_count_in_both_forms_and_the_remedy_in_one() -> None:
    payload = {
        "path": "a.py",
        "start_line": 1,
        "kind": "function",
        "qualname": "f",
        "body": "...",
        "definitions_total": 7,
    }

    assert "7 definitions of this name" in cli._symbol_text(payload)
    assert "`relore copies` groups them" not in cli._symbol_text(payload)
    assert "`relore copies` groups them" in cli._symbol_text(payload, presentation=True)


def test_copies_calls_out_the_shape_of_one() -> None:
    """On a symbol with 186 definitions the singleton is the row the question was about,
    and it sorts last, under everything else (issue #35)."""
    text = cli._copies_text(
        {
            "symbol": "compute_default_rope_parameters",
            "total": 3,
            "groups": [
                {"count": 2, "copies": [{"path": "a.py", "start_line": 1}] * 2},
                {"count": 1, "copies": [{"path": "odd.py", "start_line": 90}]},
            ],
        }
    )

    assert "-- shape 1: 2 copies (the majority shape)" in text
    assert "-- shape 2: 1 copy  <- the only one of its shape" in text


def test_copies_says_what_it_normalized_away() -> None:
    """The answer changed shape -- a group is now "these do the same thing" rather than
    "these are the same text" -- and a reader not told that reads a divergence as a typo."""
    grouped = cli._copies_text({"symbol": "f", "total": 1, "groups": [{"count": 1, "copies": []}]})
    exact = cli._copies_text(
        {"symbol": "f", "total": 1, "exact": True, "groups": [{"count": 1, "copies": []}]}
    )

    assert "normalized away" in grouped
    assert "exact text" in exact and "normalized away" not in exact
    # What was normalized away is a caveat and is in both forms; the flag that turns it
    # off is advice, so it is in the terminal form only (#13, huggingface/relore#55).
    assert "--exact" not in grouped
    assert "--exact" in cli._copies_text(
        {"symbol": "f", "total": 1, "groups": [{"count": 1, "copies": []}]}, presentation=True
    )


def test_compact_is_the_default_for_a_pipe_and_not_for_a_terminal(monkeypatch) -> None:
    """The saving that depends on the caller remembering a flag does not happen.

    A tool result is re-sent with every later turn that reads it, so its cost is its size
    times the turns remaining: measured on one agent run, relore's output was 32,349
    tokens and 892,197 once re-billing was counted — 47% of that run. The agent used
    `--compact` twice in twenty-three calls. So the pipe gets it by default, a terminal
    does not (a person pays once), and `--no-compact` is the way back.
    """
    args = cli.build_parser().parse_args(["search", "anything"])

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    assert cli._compact(args) is True
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli._compact(args) is False


def test_no_compact_undoes_the_pipe_default_and_compact_overrides_a_terminal(monkeypatch) -> None:
    piped = cli.build_parser().parse_args(["--no-compact", "search", "anything"])
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    assert cli._compact(piped) is False

    asked = cli.build_parser().parse_args(["--compact", "search", "anything"])
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli._compact(asked) is True


def test_json_is_not_trimmed_by_the_pipe_default(monkeypatch) -> None:
    """`--json` is the machine surface and asked for the structure: trimming it by default
    would drop fields from a program's input to save a model's context, which is the wrong
    caller paying. Asking for both still trims, because then somebody asked."""
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)

    assert cli._compact(cli.build_parser().parse_args(["--json", "search", "x"])) is False
    assert (
        cli._compact(cli.build_parser().parse_args(["--json", "--compact", "search", "x"])) is True
    )


# -- the temporal verbs (section 13) ---------------------------------------


@pytest.mark.parametrize("verb", ["export-corpus", "bench", "mine", "judge", "serve"])
def test_every_daemon_verb_is_registered(verb: str) -> None:
    actions = [a for a in daemon.build_parser()._actions if hasattr(a, "choices") and a.choices]
    assert any(verb in a.choices for a in actions), f"{verb} is documented but not in the parser"


@pytest.mark.parametrize(
    "argv",
    [
        ["export-corpus", "--repo", "owner/name", "--before", "2026-03-01", "--out", "c.jsonl"],
        ["bench", "--repo", "owner/name", "--set", "s.jsonl", "--before", "2026-03-01"],
        ["serve", "--as-of", "2026-03-01"],
    ],
)
def test_a_naive_cutoff_is_refused_before_anything_runs(argv, monkeypatch, tmp_path) -> None:
    """A naive cutoff parses, filters, and is wrong by the reader's offset from UTC --
    which on a leakage boundary is a document either side of it. Refused at the flag."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(daemon.DB_URL_ENV, "sqlite://")

    with pytest.raises(SystemExit) as exc:
        daemon.main(argv)

    assert "no timezone" in str(exc.value)


def test_a_cutoff_run_refuses_the_baselines_that_cannot_honour_it(capsys, tmp_path) -> None:
    """Their numbers beside a cutoff-restricted `index` would be a bounded system against
    two unbounded ones, with the gap read as a result."""
    code = daemon.main(
        [
            "bench",
            "--repo",
            "owner/name",
            "--set",
            str(tmp_path / "never-read.jsonl"),
            "--before",
            "2026-03-01T00:00:00Z",
            "--system",
            "github",
        ]
    )

    assert code == 2
    assert "cannot be honoured" in capsys.readouterr().err
