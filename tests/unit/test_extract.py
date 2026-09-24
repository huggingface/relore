"""Section 5.3's extraction, by failure class.

The rule under test throughout is **precision over exhaustiveness**: a wrong symbol costs
ranking quality on every future query, a missed one costs one query. So roughly half of
these name something the extractor must *not* claim.
"""

from __future__ import annotations

import datetime as dt

import pytest

from relore.ingest.extract import (
    error_query_form,
    extract_signals,
    normalize_error_message,
)


def _signals(*bodies: str, detail: dict | None = None, hunk: str | None = None):
    documents = [
        {"body_markdown": body, "metadata": {"diff_hunk": hunk} if hunk else {}} for body in bodies
    ]
    return extract_signals(documents, detail=detail)


def _dated(*documents: tuple[str, str | None], **kwargs):
    """``(body, iso date)`` pairs, which is what the date rules below are about."""
    rows = [
        {"body_markdown": body, "metadata": {}, "existed_at": _at(when)} for body, when in documents
    ]
    return extract_signals(rows, **kwargs)


def _at(when: str | None) -> dt.datetime | None:
    return None if when is None else dt.datetime.fromisoformat(when).replace(tzinfo=dt.timezone.utc)


def _paths(*bodies: str) -> set[str]:
    return {row["path"] for row in _signals(*bodies).files}


def _symbols(*bodies: str, hunk: str | None = None) -> set[str]:
    return {row["symbol"] for row in _signals(*bodies, hunk=hunk).symbols}


# -- files -----------------------------------------------------------------


def test_a_repository_shaped_path_in_prose_is_a_file() -> None:
    assert _paths("this belongs in src/models/mod.py instead") == {"src/models/mod.py"}


def test_a_bare_filename_needs_a_source_extension() -> None:
    """With no directory, a path is only distinguishable from an abbreviation by its
    extension -- ``e.g.`` is otherwise a file named ``e``."""
    assert _paths("see README.md") == {"README.md"}
    assert _paths("e.g. this one, cf. that one") == set()
    assert _paths("bumped to 1.2.3 in the release") == set()


def test_an_absolute_path_is_skipped_rather_than_guessed_at() -> None:
    """Mapping one to a repository-relative path needs milestone 4's working clone. The
    ``site-packages`` case is the exception: everything after it *is* the path inside the
    installed distribution."""
    assert _paths("/home/me/checkout/src/mod.py exploded") == set()
    assert _paths("/usr/lib/python3.10/site-packages/torch/nn/mod.py") == {"torch/nn/mod.py"}


def test_a_url_is_not_a_path() -> None:
    assert _paths("see https://github.com/owner/repo/blob/main/src/mod.py") == set()


def test_the_changed_file_list_wins_over_prose() -> None:
    """The per-PR pass knows the change type; prose does not. One row per path either
    way -- section 6's overlap terms count rows."""
    detail = {"files": {"nodes": [{"path": "src/mod.py", "changeType": "ADDED"}]}}
    rows = _signals("touches src/mod.py", detail=detail).files

    assert rows == [
        {
            "path": "src/mod.py",
            "change_type": "ADDED",
            "source": "changed",
            "first_seen_at": None,
        }
    ]


def test_an_inline_comments_own_path_is_definitive() -> None:
    documents = [{"body_markdown": "wrong here", "metadata": {"path": "src/deep/mod.py"}}]

    assert [row["path"] for row in extract_signals(documents).files] == ["src/deep/mod.py"]


def test_a_definitive_path_is_not_filtered_through_the_prose_heuristic() -> None:
    """The heuristic decides whether an ambiguous *token* is a path; it answers no for
    ``Dockerfile`` and ``.gitignore``. GitHub already told us these are files, so running
    a guess over them would drop data that was never in question."""
    detail = {
        "files": {
            "nodes": [
                {"path": "docker/Dockerfile", "changeType": "ADDED"},
                {"path": ".gitignore", "changeType": "MODIFIED"},
            ]
        }
    }

    assert {row["path"] for row in _signals("no paths here", detail=detail).files} == {
        "docker/Dockerfile",
        ".gitignore",
    }
    assert _paths("rebuild the Dockerfile first") == set(), "still not a path in prose"


# -- symbols ---------------------------------------------------------------


def test_a_stack_frame_yields_its_symbol_and_its_path() -> None:
    signals = _signals('  File "src/mod.py", line 412, in forward\n    raise ValueError("x")')

    assert [(r["symbol"], r["path"], r["symbol_type"]) for r in signals.symbols] == [
        ("forward", "src/mod.py", "frame")
    ]
    assert [r["path"] for r in signals.files] == ["src/mod.py"]


def test_a_pytest_frame_is_the_same_signal_in_another_rendering() -> None:
    assert _symbols("tests/test_mod.py:88: in test_shapes\n    assert False") >= {"test_shapes"}


def test_a_definition_in_a_diff_is_a_symbol() -> None:
    hunk = "@@ -1,3 +1,5 @@\n class Gemma3Model:\n+    def forward(self, x):\n+        return x"
    signals = _signals("looks wrong", hunk=hunk)

    assert {(r["symbol"], r["symbol_type"]) for r in signals.symbols} == {
        ("Gemma3Model", "class"),
        ("forward", "function"),
    }


def test_a_code_span_needs_a_marker_of_code_to_count() -> None:
    """The shape test alone admits ``true`` and ``main``. Requiring CamelCase, an
    underscore or a dot is what keeps prose out of ``thread_symbols``."""
    assert _symbols("set `MyConfig.use_cache` on the `config_object`") == {
        "MyConfig.use_cache",
        "config_object",
    }
    assert _symbols("pass `true` and read `docs`") == set()


def test_a_filename_in_a_code_span_is_a_file_and_not_a_symbol() -> None:
    """``webapp.py`` passes every identifier test -- it has a dot. Section 6 weights the
    file and symbol overlaps separately, so counting a path in both double-counts the
    same evidence."""
    signals = _signals("look at `webapp.py` and `MyConfig.use_cache`")

    assert [row["symbol"] for row in signals.symbols] == ["MyConfig.use_cache"]
    assert [row["path"] for row in signals.files] == ["webapp.py"]


def test_a_call_span_records_that_it_was_a_call() -> None:
    rows = {(r["symbol"], r["symbol_type"]) for r in _signals("call `build_model()`").symbols}

    assert rows == {("build_model", "function")} or rows == {("build_model", "call")}


def test_a_fenced_block_is_read_for_definitions_not_as_one_span() -> None:
    body = "```python\nclass Thing:\n    pass\n```"

    assert _symbols(body) == {"Thing"}


# -- errors ----------------------------------------------------------------


def test_an_error_line_yields_its_type_and_a_normalized_message() -> None:
    rows = _signals("ValueError: expected 768 features, got 1024").errors

    assert rows == [
        {
            "exception_type": "ValueError",
            "message_norm": "expected <n> features, got <n>",
            "first_seen_at": None,
        }
    ]


def test_a_dotted_exception_is_recorded_under_its_class_name() -> None:
    """A query is written as ``OutOfMemoryError``, not as the import path it was raised
    through."""
    rows = _signals("torch.cuda.OutOfMemoryError: CUDA out of memory").errors

    assert rows[0]["exception_type"] == "OutOfMemoryError"


def test_a_runner_gutter_does_not_hide_the_error() -> None:
    assert _signals("E       RuntimeError: shapes do not match").errors[0]["exception_type"] == (
        "RuntimeError"
    )
    assert (
        _signals("FAILED tests/t.py::test_a - KeyError: 'mask'").errors[0]["exception_type"]
        == "KeyError"
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        ("at 0x7f3a2b1c object", "at 0x? object"),
        ("failed in /home/me/checkout/src/mod.py", "failed in mod.py"),
        ("tensor([0.11, 0.22, 0.33, 0.44]) mismatch", "tensor([<values>]) mismatch"),
        ("expected  4\n  got 5", "expected <n> got <n>"),
    ],
)
def test_normalization_removes_what_differs_between_two_of_the_same_failure(
    message: str, expected: str
) -> None:
    assert normalize_error_message(message) == expected


def test_markdown_a_terminal_never_printed_is_unwrapped() -> None:
    """GitHub autolinks a bare host inside a pasted error, and autolinking an already
    linked one nests it. The caller's paste has none of that, and the term matches by
    containment -- so the stored form must not carry it."""
    message = (
        "HTTPSConnectionPool(host='[[hf-mirror.com](http://hf-mirror.com/)]"
        "(http://hf-mirror.com/)', port=443): max retries"
    )

    assert normalize_error_message(message) == (
        "httpsconnectionpool(host='hf-mirror.com', port=<n>): max retries"
    )


def test_the_message_is_capped_so_a_tensor_dump_cannot_become_the_row() -> None:
    assert len(normalize_error_message("boom " + "x" * 5000)) <= 300


def test_a_pasted_traceback_is_put_into_the_indexed_form() -> None:
    """Section 6's error term matches by containment against the *normalized* form, so a
    query that skipped this would match nothing -- which looks exactly like an index with
    nothing to say."""
    pasted = (
        "Traceback (most recent call last):\n"
        '  File "/home/me/src/mod.py", line 412, in forward\n'
        "    raise ValueError(msg)\n"
        "ValueError: expected 768 features, got 1024\n"
    )

    assert error_query_form(pasted) == "expected <n> features, got <n>"


def test_a_paste_with_no_error_line_falls_back_to_its_last_line() -> None:
    """A terminal cuts a traceback where it likes. Normalizing the whole block would
    produce a string longer than any stored message, so it could not match by containment
    even in principle -- the last line at least can."""
    cut = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 9, in <module>\n'
        "    model = load()\n"
        '  File "/pkg/loader.py", line 88, in load'
    )

    assert error_query_form(cut) == 'file "loader.py", line <n>, in load'


def test_a_partial_phrase_is_normalized_rather_than_discarded() -> None:
    assert error_query_form("expected 768 features") == "expected <n> features"


# -- tests -----------------------------------------------------------------


def test_a_node_id_is_stored_written_and_parameter_stripped() -> None:
    """Section 5.3 wants both readings: the exact id, and the form under which
    ``test_x[a]`` and ``test_x[b]`` unify."""
    rows = _signals("tests/test_mod.py::TestMask::test_shapes[cuda] fails").tests

    assert {row["test_id"] for row in rows} == {
        "tests/test_mod.py::TestMask::test_shapes",
        "tests/test_mod.py::TestMask::test_shapes[cuda]",
    }
    assert {row["test_function"] for row in rows} == {"test_shapes"}


def test_a_node_id_naming_a_class_records_no_function() -> None:
    """``tests/t.py::SomeTests`` is a valid node id whose last component is a class.
    Recording it as ``test_function`` would put a class name in the column a query
    filters functions by."""
    assert _signals("tests/t.py::ProviderTests is red").tests == [
        {"test_id": "tests/t.py::ProviderTests", "test_function": None, "first_seen_at": None}
    ]


def test_a_node_id_does_not_swallow_the_next_word() -> None:
    assert {row["test_id"] for row in _signals("tests/t.py::test_a fails often").tests} == {
        "tests/t.py::test_a"
    }


def test_a_dotted_unittest_id_counts() -> None:
    rows = _signals("tests.models.test_gemma.GemmaTest.test_generate is red").tests

    assert rows == [
        {
            "test_id": "tests.models.test_gemma.GemmaTest.test_generate",
            "test_function": "test_generate",
            "first_seen_at": None,
        }
    ]


def test_a_bare_function_name_is_not_a_test_id() -> None:
    """It is still retrievable -- as a symbol. ``thread_tests`` holds runner ids, and an
    exact filter on one is only worth having if the rows in it are ids."""
    assert _signals("`test_generate` is red").tests == []
    assert _symbols("`test_generate` is red") == {"test_generate"}


# -- commits ---------------------------------------------------------------


def test_commits_come_from_the_pr_pass_and_never_from_prose() -> None:
    detail = {"commits": {"nodes": [{"commit": {"oid": "abc1234", "messageHeadline": "fix it"}}]}}

    assert _signals("reverted in deadbeef", detail=detail).commits == [
        {"sha": "abc1234", "message": "fix it", "first_seen_at": None}
    ]
    assert _signals("reverted in deadbeef").commits == []


# -- redaction -------------------------------------------------------------


def test_a_secret_in_a_diff_hunk_does_not_reach_a_signal_table() -> None:
    """``diff_hunk`` is stored verbatim because it is metadata rather than body, so this
    module redacts it on the way in. A signal table is not a way around section 11.3."""
    hunk = '+ TOKEN = "ghp_' + "a" * 36 + '"\n+ def load_token():\n+     pass'
    signals = _signals("hardcoded", hunk=hunk)

    assert "load_token" in {row["symbol"] for row in signals.symbols}
    assert "ghp_" not in repr(signals)


# -- the empty case --------------------------------------------------------


def test_a_thread_with_nothing_extractable_yields_nothing() -> None:
    signals = _signals("Thanks, merged!", "LGTM")

    assert signals.total == 0


# -- first_seen_at (temporal-cutoff-map.md 8) ------------------------------
#
# One row stands for every document that produced it, which is what keeps `_overlap`'s
# counts and production ranking unchanged. So the row's date is a `min` over those
# documents, and the cases below are the ways that `min` can be got wrong.


def test_a_signal_is_dated_by_the_earliest_document_that_carried_it() -> None:
    signals = _dated(
        ("the fix is in src/mod.py", "2026-06-01"),
        ("as I said, src/mod.py", "2026-01-01"),
    )

    assert [row["first_seen_at"] for row in signals.files] == [_at("2026-01-01")]


def test_a_signal_only_a_later_document_carries_is_dated_by_that_one() -> None:
    signals = _dated(
        ("nothing quotable here", "2026-01-01"),
        ("the fix is in src/mod.py", "2026-06-01"),
    )

    assert [row["first_seen_at"] for row in signals.files] == [_at("2026-06-01")]


def test_an_undated_contributor_does_not_unsettle_a_dated_one() -> None:
    """The claim ``first_seen_at < T`` makes is *the thread certainly carried this by* ``T``,
    and one dated document settles that alone. An undated one can only mean the true first
    appearance is earlier, so passing over it cannot admit a value the thread did not have.

    Every ``review`` document in the corpus is undated -- GitHub dates a review with
    ``submitted_at`` -- so the stricter reading would take 3% of ``thread_files`` out of
    every cutoff for no soundness at all.
    """
    signals = _dated(("src/mod.py is wrong", None), ("still src/mod.py", "2026-01-01"))

    assert [row["first_seen_at"] for row in signals.files] == [_at("2026-01-01")]


def test_a_row_no_dated_document_produced_is_undated() -> None:
    """And there the fail-closed direction does apply: nothing is known, so the ``<``
    comparison drops it."""
    signals = _dated(("src/mod.py is wrong", None))

    assert [row["first_seen_at"] for row in signals.files] == [None]


def test_every_kind_of_signal_carries_a_date() -> None:
    signals = _dated(
        (
            'File "src/mod.py", line 9, in forward\n'
            "ValueError: bad 3 shapes\n"
            "tests/test_mod.py::test_shapes fails",
            "2026-01-01",
        )
    )
    dates = {
        row["first_seen_at"]
        for rows in (signals.files, signals.symbols, signals.errors, signals.tests)
        for row in rows
    }

    assert signals.total > 3
    assert dates == {_at("2026-01-01")}


def test_the_changed_file_list_is_dated_by_the_merge() -> None:
    """Nothing in the per-PR node carries a date, and fetching one would make a re-derive
    a network operation. The merge is the one instant the list is known to have been
    final."""
    detail = {"files": {"nodes": [{"path": "src/mod.py", "changeType": "ADDED"}]}}

    rows = _dated(
        ("no paths in this body", "2026-01-01"), detail=detail, merged_at=_at("2026-06-01")
    ).files

    assert [(row["source"], row["first_seen_at"]) for row in rows] == [
        ("changed", _at("2026-06-01"))
    ]


def test_an_unmerged_pull_request_leaves_its_diff_undated() -> None:
    """It had some prefix of this list at any earlier instant and nothing can reconstruct
    which, so the row fails closed rather than claiming the whole diff."""
    detail = {"files": {"nodes": [{"path": "src/mod.py", "changeType": "ADDED"}]}}

    rows = _dated(("no paths in this body", "2026-01-01"), detail=detail).files

    assert [row["first_seen_at"] for row in rows] == [None]


def test_a_path_both_written_and_diffed_is_dated_by_the_writing() -> None:
    """The row is one row -- the changed-file list overwrites its ``change_type`` and its
    ``source`` -- but the thread did carry that path from the moment somebody typed it, so
    the date is the earlier of the two and not the merge."""
    detail = {"files": {"nodes": [{"path": "src/mod.py", "changeType": "ADDED"}]}}

    rows = _dated(
        ("touches src/mod.py", "2026-01-01"), detail=detail, merged_at=_at("2026-06-01")
    ).files

    assert [(row["source"], row["first_seen_at"]) for row in rows] == [
        ("changed", _at("2026-01-01"))
    ]


def test_commits_are_dated_by_the_merge_too() -> None:
    detail = {"commits": {"nodes": [{"commit": {"oid": "abc1234", "messageHeadline": "fix it"}}]}}

    rows = _dated(("an opening", "2026-01-01"), detail=detail, merged_at=_at("2026-06-01")).commits

    assert [row["first_seen_at"] for row in rows] == [_at("2026-06-01")]
