"""The split between ``relore`` and ``relored`` is a security boundary, not packaging
taste. Assert it statically, so "the agent's binary cannot write to the index" is a
property of the build rather than a promise in a document.

the build plan section 2 and section 12; AGENTS.md "The four invariants".
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys

import pytest

PKG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "relore"

# A database driver or a web server reachable from the client means the sandbox install
# just grew one, whether or not the code path is ever taken.
DB_AND_SERVER = ("psycopg", "psycopg2", "asyncpg", "sqlalchemy", "fastapi", "uvicorn", "starlette")
# The ML stack the client exists to avoid carrying.
ML = (
    "torch",
    "sentence_transformers",
    "transformers",
    "lancedb",
    "numpy",
    "fastembed",
    "onnxruntime",
)


def _module_file(mod: str) -> pathlib.Path | None:
    if mod != "relore" and not mod.startswith("relore."):
        return None
    base = PKG_ROOT.joinpath(*mod.split(".")[1:])
    if base.is_dir():
        init = base / "__init__.py"
        return init if init.exists() else None
    leaf = base.with_suffix(".py")
    return leaf if leaf.exists() else None


def _imports(path: pathlib.Path, mod: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative -- resolve against the containing package
                parts = mod.split(".")
                anchor = parts if path.name == "__init__.py" else parts[:-1]
                base = ".".join(anchor[: len(anchor) - node.level + 1])
                out.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                out.add(node.module)
    return out


def reachable(root: str) -> set[str]:
    """Every module name reachable from ``root`` by following intra-package imports."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = _module_file(mod)
        if path is None:
            continue
        for imp in _imports(path, mod):
            seen.add(imp)
            if imp.startswith("relore"):
                stack.append(imp)
    return seen


def _offenders(mods: set[str], forbidden: tuple[str, ...]) -> set[str]:
    return {m for m in mods for f in forbidden if m == f or m.startswith(f + ".")}


def test_client_reaches_no_database_or_server() -> None:
    assert _module_file("relore.cli") is not None, (
        "relore.cli must exist for this test to mean anything"
    )
    assert _offenders(reachable("relore.cli"), DB_AND_SERVER) == set()


def test_client_reaches_no_ml_stack() -> None:
    assert _offenders(reachable("relore.cli"), ML) == set()


@pytest.mark.parametrize("forbidden", ["relore.store", "relore.github", "relore.api"])
def test_client_reaches_no_server_side_package(forbidden: str) -> None:
    assert _offenders(reachable("relore.cli"), (forbidden,)) == set()


@pytest.mark.parametrize("forbidden", ["relore.store", "relore.github", "relore.api"])
def test_code_lens_is_a_pure_function_of_a_tree(forbidden: str) -> None:
    """``relore.code`` runs in both binaries -- against a working tree in the client and a
    historical blob in the daemon. That only works if it knows about neither."""
    assert _module_file("relore.code") is not None
    assert _offenders(reachable("relore.code"), (forbidden,)) == set()


def test_the_client_actually_runs_with_nothing_server_side_importable() -> None:
    """The same invariant at runtime, which the AST walk above cannot reach.

    The static test proves the *import graph*; this proves the *install*. It catches what
    the walk structurally cannot: a parent package's ``__init__`` pulled in as a side
    effect of importing a leaf module. ``from relore.search.queries import Hit`` looks
    stdlib-only to an AST reader and executes ``relore/search/__init__.py`` -- which
    imports SQLAlchemy.
    """
    blocked = (*DB_AND_SERVER, *ML, "psycopg", "pydantic")
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"the client reached {name!r}; AGENTS.md invariant 1 is broken")
        return real_import(name, *args, **kwargs)

    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m.split(".")[0] in blocked}
    monkey = builtins.__import__
    builtins.__import__ = guard
    try:
        importlib.reload(importlib.import_module("relore.cli"))
        importlib.reload(importlib.import_module("relore.render"))
        importlib.reload(importlib.import_module("relore.code.repomap"))
    finally:
        builtins.__import__ = monkey
        sys.modules.update(saved)
