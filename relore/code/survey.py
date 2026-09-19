"""Questions about a whole tree rather than one file (issue #7): `grep` and `copies`.

`copies` is the one this corpus is shaped for. `transformers` duplicates model code on
purpose, so "the same function, copied into 38 files, which copies diverge" is the native
shape of a bug there -- and `huggingface/transformers#48630` was exactly that: one model
diverging from the other 37. The answer is a grouping, not a list, because the question is
never "where is it" but "which ones are different".

Grouping is by what the body *does*, not by its text (issue #35). Body identity was the
first implementation and it made the verb unusable on exactly the corpus it was built for:
``compute_default_rope_parameters`` came back as **186 definitions in 172 shapes**, every
group one model's generated file paired with its own ``modular_*.py``, and a "majority
shape" of two. The whole split was one token --

    def compute_default_rope_parameters(config: GPTNeoXConfig, device=None, **kwargs)
    def compute_default_rope_parameters(config: LlamaConfig,  device=None, **kwargs)

-- so every model was its own shape before any semantic difference was considered, and the
one model that had actually diverged was invisible among 171 that had not. Normalizing
annotations, docstrings and comments away collapses that to the handful of shapes that
differ in what they compute, which is the question. ``--exact`` keeps the old grouping for
whoever wants the text.
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from relore.code.api import Definition
from relore.code.defs import definitions
from relore.code.registry import claimed
from relore.code.walk import all_files, needle, read, source_files

MAX_MATCHES = 500
MAX_COPIES = 200


@dataclass(frozen=True)
class GrepHit:
    path: str
    line: int
    text: str


@dataclass
class GrepResult:
    pattern: str
    hits: list[GrepHit] = field(default_factory=list)
    files_searched: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class Copy:
    path: str
    qualname: str
    start_line: int
    end_line: int | None
    #: The exact text, indentation stripped. Reported, never grouped on unless asked:
    #: it is what tells a reader that two members of one shape are not byte-identical.
    body_hash: str
    #: What this copy *does*, and the key :attr:`CopiesResult.groups` uses. Equal to
    #: ``body_hash`` under ``--exact``, or wherever the body could not be parsed.
    shape_hash: str
    lines: int
    #: False when the shape hash fell back to the text -- an unparseable body, a language
    #: with no normalizer, or a provider that gave no end line so there is only one line to
    #: read. A caller comparing two shapes deserves to know one of them was not normalized.
    normalized: bool


@dataclass
class CopiesResult:
    symbol: str
    copies: list[Copy] = field(default_factory=list)
    truncated: bool = False
    #: Whether grouping was asked to use the exact text (``--exact``).
    exact: bool = False

    @property
    def groups(self) -> dict[str, list[Copy]]:
        """Copies by shape, largest group first: the majority shape, then the outliers.

        Largest-first is the ordering the answer is read in -- what most copies do, then
        what differs from it -- and a shape of one at the bottom is the row the question
        was usually about.
        """
        out: dict[str, list[Copy]] = {}
        for copy in self.copies:
            out.setdefault(copy.shape_hash, []).append(copy)
        return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def grep(root: str, pattern: str, *, path_glob: str | None = None) -> GrepResult:
    """A regular expression over every file in the tree, not only the parseable ones.

    The audit question -- "which files have this shape of code" -- reaches YAML, docs and
    templates, so the set here is the walker's, not the providers'.
    """
    regex = re.compile(pattern)
    result = GrepResult(pattern=pattern)
    for path in all_files(root):
        relative = _relative(root, path)
        if path_glob and not fnmatch(relative, path_glob):
            continue
        source = read(path)
        if source is None:
            continue
        result.files_searched += 1
        for number, line in enumerate(source.decode("utf-8", "replace").splitlines(), start=1):
            if regex.search(line):
                result.hits.append(GrepHit(path=relative, line=number, text=line.strip()[:300]))
                if len(result.hits) >= MAX_MATCHES:
                    result.truncated = True
                    return result
    return result


def copies(root: str, symbol: str, *, exact: bool = False) -> CopiesResult:
    """Every definition of ``symbol``, grouped by whether the bodies agree.

    Indentation is stripped either way, so a function lifted into a class groups with its
    original rather than reading as a divergence. Beyond that, ``exact=False`` -- the
    default -- groups on what the body computes: see :func:`_shape` for what is normalized
    away and why the alternative did not work on generated code.
    """
    result = CopiesResult(symbol=symbol, exact=exact)
    for path, definition, source in _definitions_named(root, symbol):
        body = _body(source, definition)
        body_hash = _hash(body)
        shape = None if exact else _shape(path, body)
        result.copies.append(
            Copy(
                path=_relative(root, path),
                qualname=definition.qualname,
                start_line=definition.start_line,
                end_line=definition.end_line,
                body_hash=body_hash,
                shape_hash=_hash(shape.splitlines()) if shape is not None else body_hash,
                lines=len(body),
                normalized=shape is not None,
            )
        )
        if len(result.copies) >= MAX_COPIES:
            result.truncated = True
            break
    result.copies.sort(key=lambda copy: copy.path)
    return result


@dataclass(frozen=True)
class SymbolBody:
    path: str
    definition: Definition
    body: str
    #: How many definitions of this name the tree holds. Never 1 by assumption: in a
    #: repository that duplicates model code, serving the first of 38 as *the* body is a
    #: wrong answer a caller cannot see, so the count travels with it and `copies` groups
    #: them.
    total: int


def symbol_body(root: str, qualname: str) -> SymbolBody | None:
    """The source of one definition, so a caller can read it without cloning (#7 item 2)."""
    name = qualname.rsplit(".", 1)[-1]
    found = [
        (_relative(root, path), definition, "\n".join(_body(source, definition)))
        for path, definition, source in _definitions_named(root, name)
    ]
    if not found:
        return None
    exact = [item for item in found if item[1].qualname == qualname]
    # By an explicit key, never by the tuple: two definitions of one name in the same file
    # tie on the path, and the fallback comparison is `Definition < Definition`, which a
    # frozen dataclass does not implement. That was a 500 on `symbol`, and only for names
    # duplicated *within* a file -- so it passed every test with one definition per path.
    path, definition, body = min(exact or found, key=lambda item: (item[0], item[1].start_line))
    return SymbolBody(path=path, definition=definition, body=body, total=len(found))


def _definitions_named(root: str, name: str) -> Iterator[tuple[str, Definition, list[str]]]:
    """Every definition of ``name`` in the tree, parsing only the files that can hold one.

    The filter is the latency of `symbol` and `copies` (issue #45): without it both parse
    every claimed file to answer about one name -- 4,882 files on `transformers`, 6.5 s of
    the 7 s, against 0.46 s to read them all. The parse is the cost, not the tree.

    The filter itself is :func:`~relore.code.walk.needle`, shared with ``refs`` since issue
    #83 applied the same argument there -- a superset keyed on the trailing identifier run,
    and ``None`` for a non-ASCII name rather than a guess at an encoding.
    """
    wanted = needle(name)
    for path in source_files(root):
        if not claimed(path):
            continue
        raw = read(path)
        if raw is None:
            continue
        if wanted is not None and wanted not in raw:
            continue
        source = raw.decode("utf-8", "replace").splitlines()
        for definition in definitions(path, raw):
            if definition.name == name or definition.qualname.endswith(f".{name}"):
                yield path, definition, source


def _body(source: list[str], definition: Definition) -> list[str]:
    end = definition.end_line or definition.start_line
    return source[definition.start_line - 1 : end]


def _shape(path: str, body: list[str]) -> str | None:
    """What this body computes, with the parts that are not computation removed.

    Three things are dropped, and each one split a real group in the field report:

    * **Type annotations**, parameter and return. ``config: GPTNeoXConfig`` against
      ``config: LlamaConfig`` is the single token that turned 186 copies of one function
      into 172 shapes -- the corpus is generated from ``modular_*.py``, so this is the
      common case rather than an edge one.
    * **Docstrings**, which name the model they were generated for.
    * **Comments and formatting**, which :func:`ast.unparse` removes by rebuilding the
      source from the tree -- so line breaks, redundant parentheses and quote style stop
      counting as divergence too.

    ``None`` when the body is not parseable Python: a provider that declares no extents
    hands us one line, a copy can be C or Rust, and a fixture can be a fragment. The caller
    falls back to the text hash and marks the copy un-normalized rather than guessing -- a
    silently different grouping rule for some rows would make the shapes incomparable,
    which is worse than a coarser answer.
    """
    if not path.endswith(".py"):
        return None
    try:
        tree = ast.parse(textwrap.dedent("\n".join(body)))
    except (SyntaxError, ValueError):
        return None
    return ast.unparse(_Shape().visit(tree))


class _Shape(ast.NodeTransformer):
    """Strip annotations and docstrings in place. Mutating is fine: the tree is ours."""

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        self.generic_visit(node)
        if node.value is None:
            # `x: int` declares a type and does nothing, so under this rule it is nothing.
            return None
        return ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._definition(node)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self._definition(node)

    def _definition(self, node: Any) -> ast.AST:
        self.generic_visit(node)
        if getattr(node, "returns", None) is not None:
            node.returns = None
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        # `ast.unparse` cannot render an empty suite, and a body that was only a docstring
        # now is one. `pass` is what that function actually does.
        if not node.body and not isinstance(node, ast.Module):
            node.body = [ast.Pass()]
        return node


def _hash(body: list[str]) -> str:
    stripped = "\n".join(line.strip() for line in body if line.strip())
    return hashlib.sha256(stripped.encode()).hexdigest()[:12]


def _relative(root: str, path: str) -> str:
    prefix = root.rstrip("/") + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path
