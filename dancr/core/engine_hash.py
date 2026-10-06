"""The code fingerprint: what invalidates every cached result when DANCR's own code changes.

A step's output is cached under a hash that folds in the version of the code that produces it, so upgrading
DANCR (or a library that computes or reads) recomputes everything. The modules a run executes are discovered
from their import statements, so nothing is listed by hand and nothing can be forgotten. Split out of
``executor.py`` so the fingerprint is independently testable and does not sit in the middle of the runner.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import polars as pl

IMPL_VERSION = "4"     # bump to invalidate every cache

LIBRARIES = ("polars", "numpy", "fastexcel", "scipy", "matplotlib", "xlsxwriter")


def _version_of(package: str) -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(package)
    except PackageNotFoundError:
        return "none"


def engine_versions() -> dict[str, str]:
    """The version of DANCR's compute and read libraries, as a run manifest records them (a build has them
    pinned, but a manifest should say so explicitly)."""
    return {m: (pl.__version__ if m == "polars" else _version_of(m)) for m in LIBRARIES}


def engine_files() -> list[Path]:
    """The DANCR modules a run executes: this module, every step module, and every DANCR module they import,
    read from their import statements (imports inside functions too). Nothing is listed by hand, so a module
    can be neither forgotten nor included for no reason (the window, the answer engine)."""
    import ast
    package = Path(__file__).resolve().parent.parent          # dancr/
    top = package.parent

    def file_of(module: str) -> Path | None:
        base = top.joinpath(*module.split("."))
        # A package wins over a same-named module, exactly as Python's importer resolves them; otherwise a
        # leftover ``foo.py`` beside a live ``foo/`` package would be hashed instead of the code that runs.
        return next((f for f in (base / "__init__.py", base.with_suffix(".py")) if f.is_file()), None)

    todo = [Path(__file__).resolve()]
    # executor.py is not imported by the step modules, but its materialization and sampling shape every result,
    # so it must stay in the fingerprint even though this module is now the seed
    ex = Path(__file__).resolve().with_name("executor.py")
    if ex.is_file():
        todo.append(ex)
    todo += sorted((Path(__file__).resolve().parent / "nodes").glob("*.py"))
    seen: set[Path] = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        parts = list(f.relative_to(top).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        else:
            todo += [p / "__init__.py" for p in f.parents if top in p.parents]      # its packages run first
        here = parts if f.name == "__init__.py" else parts[:-1]         # the package relative imports start from
        try:
            tree = ast.parse(f.read_bytes())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(here[:len(here) - node.level + 1]) if node.level else ""
                mod = ".".join(x for x in (base, node.module or "") if x)
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]     # "from . import nodes" names a module
            for name in names:
                if name.split(".")[0] == package.name and (g := file_of(name)) is not None:
                    todo += [g] + [top.joinpath(*name.split(".")[:i]) / "__init__.py" for i in range(1, name.count(".") + 1)]
    return sorted(f for f in seen if f.is_file())


BAKED_FINGERPRINT = Path(__file__).resolve().parent.parent / "fingerprint.txt"     # written by packaging/dancr.spec


def _code_fingerprint() -> str:
    """Hash of the code that can shape a step's output, so cached outputs are invalidated when it changes:
    the modules a run executes (see ``engine_files``) and the versions of the libraries that compute and read.
    A frozen app has no source to read, so the build computes this from the source and bakes it in."""
    if getattr(sys, "frozen", False):
        try:
            return BAKED_FINGERPRINT.read_text().strip()
        except OSError as e:
            raise RuntimeError(f"this build of DANCR is incomplete: {BAKED_FINGERPRINT} is missing") from e
    import numpy
    top = Path(__file__).resolve().parent.parent.parent
    h = hashlib.sha1(IMPL_VERSION.encode())
    for lib in (pl.__version__, numpy.__version__, *(_version_of(m) for m in ("fastexcel", "scipy", "matplotlib", "xlsxwriter"))):
        h.update(lib.encode())
    for f in engine_files():
        try:
            h.update(f.relative_to(top).as_posix().encode())
            h.update(f.read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


CODE_FINGERPRINT = _code_fingerprint()
