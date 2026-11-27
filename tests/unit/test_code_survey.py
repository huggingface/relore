"""`grep` and `copies` over a tree (issue #7).

`copies` is the verb the report ranked first: a repository that duplicates model code on
purpose asks "which of the 38 copies diverge", and that is a grouping rather than a list.
"""

from __future__ import annotations

import os

import pytest

from relore.code import survey
from relore.code.survey import copies, grep, symbol_body

CANONICAL = """\
def compute_default_rope_parameters(config):
    base = config.rope_theta
    dim = int(config.head_dim * config.partial_rotary_factor)
    return base, dim
"""

DIVERGED = """\
def compute_default_rope_parameters(config):
    base = config.rope_theta
    dim = int(config.head_dim)
    return base, dim
"""


@pytest.fixture
def tree(tmp_path):
    models = tmp_path / "src" / "models"
    models.mkdir(parents=True)
    for name in ("llama", "gemma", "mistral"):
        (models / f"modeling_{name}.py").write_text(CANONICAL)
    (models / "modeling_gpt_neox_japanese.py").write_text(DIVERGED)
    (tmp_path / "README.md").write_text("partial_rotary_factor is documented here")
    return tmp_path


def test_copies_groups_by_body_so_the_outlier_is_visible(tree) -> None:
    """`huggingface/transformers#48630` in miniature: one model out of four diverging, and
    the useful answer is which one -- not where the 38 definitions are."""
    result = copies(str(tree), "compute_default_rope_parameters")

    assert len(result.copies) == 4
    groups = list(result.groups.values())
    assert [len(group) for group in groups] == [3, 1]
    assert groups[1][0].path.endswith("modeling_gpt_neox_japanese.py")


def test_copies_ignores_indentation_so_a_lifted_function_is_not_a_divergence(tmp_path) -> None:
    (tmp_path / "a.py").write_text(CANONICAL)
    (tmp_path / "b.py").write_text(
        "class Model:\n" + "".join(f"    {line}\n" for line in CANONICAL.splitlines())
    )

    result = copies(str(tmp_path), "compute_default_rope_parameters")

    assert len(result.copies) == 2
    assert len(result.groups) == 1


# The shape of the corpus this verb is pointed at: `transformers` generates every
# `modeling_*.py` from a `modular_*.py`, so two models' copies of one function are the same
# code with a different config type annotated on the parameter and a different model named
# in the docstring.
GENERATED = '''\
def compute_default_rope_parameters(config: {config}, device=None, **kwargs):
    """RoPE parameters for {model}.

    Args:
        config ([`~transformers.{config}`]): the model configuration.
    """
    base: float = config.rope_theta  # theta, per the paper
    partial = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(config.head_dim * partial)
    return base, dim
'''

DROPS_THE_FACTOR = """\
def compute_default_rope_parameters(config: GPTNeoXJapaneseConfig, device=None, **kwargs):
    base = config.rope_theta
    dim = int(config.head_dim)
    return base, dim
"""


@pytest.fixture
def generated(tmp_path):
    """Five models' worth, each generated from its own modular file, and one that diverged."""
    models = tmp_path / "src" / "models"
    models.mkdir(parents=True)
    for model in ("llama", "gemma", "mistral", "phi", "persimmon"):
        config = f"{model.title()}Config"
        for prefix in ("modeling", "modular"):
            (models / f"{prefix}_{model}.py").write_text(
                GENERATED.format(config=config, model=model)
            )
    (models / "modeling_gpt_neox_japanese.py").write_text(DROPS_THE_FACTOR)
    return tmp_path


def test_an_annotation_is_not_a_divergence(generated) -> None:
    """Issue #35, the failure verbatim: grouping on the body's text made every model its
    own shape -- 186 definitions in 172 shapes on the real corpus, with a "majority shape"
    of two -- because `config: LlamaConfig` and `config: GPTNeoXConfig` are different text.
    The one model that had actually diverged was invisible among 171 that had not."""
    result = copies(str(generated), "compute_default_rope_parameters")

    assert len(result.copies) == 11
    assert [len(group) for group in result.groups.values()] == [10, 1]
    outlier = list(result.groups.values())[1][0]
    assert outlier.path.endswith("modeling_gpt_neox_japanese.py")


def test_exact_keeps_the_old_grouping_for_whoever_wants_the_text(generated) -> None:
    """It is a real question -- "are these byte-identical" -- just never the question this
    verb is reached for. Five models, five shapes of two, plus the outlier."""
    result = copies(str(generated), "compute_default_rope_parameters", exact=True)

    assert [len(group) for group in result.groups.values()] == [2, 2, 2, 2, 2, 1]


def test_a_docstring_and_a_comment_are_not_a_divergence(tmp_path) -> None:
    (tmp_path / "a.py").write_text('def f(config):\n    """One."""\n    return config.x  # x\n')
    (tmp_path / "b.py").write_text("def f(config):\n    return config.x\n")

    assert len(copies(str(tmp_path), "f").groups) == 1
    assert len(copies(str(tmp_path), "f", exact=True).groups) == 2


def test_a_body_that_is_only_a_docstring_still_groups(tmp_path) -> None:
    """`ast.unparse` cannot render an empty suite, and removing the docstring empties one.
    A function whose body was only a docstring does nothing, and `pass` is how that is
    written."""
    (tmp_path / "a.py").write_text('def f():\n    """Nothing to do."""\n')
    (tmp_path / "b.py").write_text("def f():\n    pass\n")

    result = copies(str(tmp_path), "f")

    assert len(result.groups) == 1
    assert all(copy.normalized for copy in result.copies)


def test_what_the_body_does_still_separates_shapes(tmp_path) -> None:
    """The rule normalizes what is not computation. It must not normalize computation."""
    (tmp_path / "a.py").write_text("def f(c):\n    return c.x + 1\n")
    (tmp_path / "b.py").write_text("def f(c):\n    return c.x + 2\n")

    assert len(copies(str(tmp_path), "f").groups) == 2


@pytest.mark.parametrize(
    "path,body",
    [
        # A provider that declares no `extents` capability gives no end line, so the body
        # is the signature alone -- which is not a parseable statement.
        ("m.py", ["def f(config: Config):"]),
        # Another language. There is no normalizer for it, and inventing one from Python's
        # rules would be worse than admitting there is none.
        ("m.rs", ["fn f(c: &Config) -> u32 {", "    c.x", "}"]),
    ],
)
def test_a_body_with_no_normalizer_falls_back_to_its_text(path: str, body: list[str]) -> None:
    """Falling back is right; doing it silently is not. Two shapes grouped under different
    rules are not comparable, so :attr:`Copy.normalized` records which rule was used."""
    assert survey._shape(path, body) is None


def test_the_fallback_is_marked_on_the_copy(tmp_path, monkeypatch) -> None:
    """What a caller sees when the fallback fires: the shape key *is* the text hash, and
    the copy says it was never normalized."""
    monkeypatch.setattr(survey, "_shape", lambda path, body: None)
    (tmp_path / "a.py").write_text("def f(c):\n    return c.x\n")

    result = copies(str(tmp_path), "f")

    assert result.copies
    assert not any(copy.normalized for copy in result.copies)
    assert all(copy.shape_hash == copy.body_hash for copy in result.copies)


def test_grep_answers_the_pattern_question_a_symbol_lookup_cannot(tree) -> None:
    """ "which files have this shape of code" is not "where is this symbol defined"."""
    result = grep(str(tree), "partial_rotary_factor", path_glob="src/**/modeling_*.py")

    assert [hit.path for hit in result.hits] == [
        "src/models/modeling_gemma.py",
        "src/models/modeling_llama.py",
        "src/models/modeling_mistral.py",
    ]
    assert all(hit.line == 3 for hit in result.hits)


def test_grep_path_glob_excludes_what_it_should(tree) -> None:
    everywhere = grep(str(tree), "partial_rotary_factor")
    assert any(hit.path == "README.md" for hit in everywhere.hits)


def test_symbol_returns_a_body_and_how_many_there_are(tree) -> None:
    """One of 38 served as *the* body is a wrong answer the caller cannot see, so the
    count travels with it."""
    found = symbol_body(str(tree), "compute_default_rope_parameters")

    assert found is not None
    assert found.definition.qualname == "compute_default_rope_parameters"
    assert found.total == 4
    assert found.path.startswith("src/models/")
    assert "def compute_default_rope_parameters" in found.body


def test_a_symbol_that_is_not_there_is_none_rather_than_an_error(tree) -> None:
    assert symbol_body(str(tree), "not_a_function") is None


def test_a_qualifier_that_matches_no_class_is_none_not_another_classs_method(tmp_path) -> None:
    """`Axk2Model.forward` returned an example's `TestAttention.forward` (#59): the qualifier
    was dropped and the first same-named method won. A wrong answer the caller cannot see."""
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "a.py").write_text(
        "class TestAttention:\n    def forward(self):\n        return 1\n"
    )
    (tmp_path / "m.py").write_text(
        "class AXK2Model:\n    def forward(self, x):\n        return x\n"
    )

    assert symbol_body(str(tmp_path), "Axk2Model.forward") is None
    assert symbol_body(str(tmp_path), "ZzzNotARealClass.forward") is None
    found = symbol_body(str(tmp_path), "AXK2Model.forward")
    assert found is not None and found.path == "m.py"
    # A bare name has no qualifier to contradict, so it still picks among all of them.
    bare = symbol_body(str(tmp_path), "forward")
    assert bare is not None and bare.total == 2


def test_a_partial_qualifier_still_resolves_a_nested_definition(tmp_path) -> None:
    (tmp_path / "m.py").write_text(
        "class Outer:\n    class Inner:\n        def run(self):\n            return 1\n"
    )

    found = symbol_body(str(tmp_path), "Inner.run")

    assert found is not None
    assert found.definition.qualname == "Outer.Inner.run"


def test_two_definitions_of_a_name_in_one_file(tmp_path) -> None:
    """A 500 in production: selection sorted `(path, Definition, body)` tuples, so two
    definitions in the same file tied on the path and fell through to comparing
    `Definition`, which a frozen dataclass does not order. Every fixture here had one
    definition per file, which is why it passed. `transformers` does not."""
    (tmp_path / "modeling_two.py").write_text(
        "class A:\n"
        "    def compute_default_rope_parameters(self, config):\n"
        "        return 1\n"
        "\n"
        "\n"
        "class B:\n"
        "    def compute_default_rope_parameters(self, config):\n"
        "        return 2\n"
    )

    found = symbol_body(str(tmp_path), "compute_default_rope_parameters")

    assert found is not None
    assert found.total == 2
    # The earlier of the two, because selection has to be deterministic to be reportable.
    assert found.definition.qualname == "A.compute_default_rope_parameters"
    assert "return 1" in found.body


# -- the latency of both verbs is which files get parsed (issue #45) -------------------


def _parse_counter(monkeypatch) -> list[str]:
    """Records the files handed to the parser, which is what the fix is about."""
    parsed: list[str] = []
    real = survey.definitions

    def counted(path: str, source: bytes):
        parsed.append(os.path.basename(path))
        return real(path, source)

    monkeypatch.setattr(survey, "definitions", counted)
    return parsed


def test_only_files_that_can_hold_the_name_are_parsed(tmp_path, monkeypatch) -> None:
    """Issue #45: 20 s warm on `transformers` against a 30 s client timeout, because all
    4,882 claimed files were parsed to answer about one name."""
    (tmp_path / "has_it.py").write_text(CANONICAL)
    (tmp_path / "mentions_it.py").write_text("# compute_default_rope_parameters is fine\n")
    for index in range(20):
        (tmp_path / f"unrelated_{index}.py").write_text("def something_else(c):\n    return c\n")
    parsed = _parse_counter(monkeypatch)

    found = symbol_body(str(tmp_path), "compute_default_rope_parameters")

    assert found is not None
    assert found.path == "has_it.py"
    # A file that merely mentions the name is still parsed and then rejected -- the filter
    # is a superset of the answer, never a second matching rule.
    assert sorted(parsed) == ["has_it.py", "mentions_it.py"]


def test_the_filter_matches_the_leaf_so_a_qualname_still_resolves(tmp_path, monkeypatch) -> None:
    """The dotted form is in no file, so filtering on it would 404 every qualname."""
    (tmp_path / "m.py").write_text("class Model:\n    def forward(self, x):\n        return x\n")
    parsed = _parse_counter(monkeypatch)

    found = symbol_body(str(tmp_path), "Model.forward")

    assert found is not None
    assert found.definition.qualname == "Model.forward"
    assert parsed == ["m.py"]


def test_a_non_ascii_name_drops_the_filter_rather_than_the_match(tmp_path, monkeypatch) -> None:
    """A `coding:` declaration means the bytes need not be UTF-8, so unsure parses
    everything rather than missing the match."""
    (tmp_path / "m.py").write_text("def café(c):\n    return c\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def plain(c):\n    return c\n")
    parsed = _parse_counter(monkeypatch)

    found = symbol_body(str(tmp_path), "café")

    assert found is not None
    assert sorted(parsed) == ["m.py", "other.py"]


def test_copies_is_filtered_too_since_it_shares_the_walk(tree, monkeypatch) -> None:
    """Same cost for the same reason (#43); one walk serves both verbs."""
    for index in range(20):
        (tree / "src" / "models" / f"other_{index}.py").write_text(
            "def elsewhere(c):\n    return c\n"
        )
    parsed = _parse_counter(monkeypatch)

    result = copies(str(tree), "compute_default_rope_parameters")

    assert len(result.copies) == 4
    assert len(parsed) == 4
