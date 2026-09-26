"""What the command line and the MCP server share: finding steps, adding them, reporting them, charting and
reading results. Both front ends call these, so a step is reported the same way wherever it is asked for."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from .core import Pipeline, PipelineError, registry
from .core.dtypes import json_safe
from .core.executor import Executor, NodeState
from .core.model import Node


class StepFailed(ValueError):
    """A step asked for could not be computed (the CLI exits 1, as for a failed run, not 2 for a usage error)."""


def require_node(p: Pipeline, node_id: str) -> str:
    """Every command or tool that names a step checks it here, so the message is always the same."""
    if node_id not in p.nodes:
        raise ValueError(f"No step called {node_id!r}. Steps: {list(p.nodes)}")
    return node_id


def resolve_type(key_or_label: str) -> str:
    """A step type by key (`keep_rows`) or by the label people see (`Filter rows`)."""
    if registry.has(key_or_label):
        return key_or_label
    match = [t for t in registry.all() if t.label.lower() == key_or_label.lower()]
    if not match:
        raise ValueError(f"Unknown step type {key_or_label!r}. Known: {sorted(t.key for t in registry.all())}")
    return match[0].key


def place(p: Pipeline, after: str | None) -> tuple[float, float]:
    """Where a new step goes on the map: right of the step it follows, else below the last one; never on top of another."""
    x, y = 0.0, 0.0
    if after and after in p.nodes:
        x, y = p.nodes[after].x + 280, p.nodes[after].y
    elif p.nodes:
        last = list(p.nodes.values())[-1]
        x, y = last.x, last.y + 120
    while any(abs(n.x - x) < 200 and abs(n.y - y) < 80 for n in p.nodes.values()):
        y += 120
    return x, y


def add_step(p: Pipeline, type_key: str, params: dict[str, Any] | None = None, title: str | None = None,
             node_id: str | None = None, after: str | None = None, port: str | None = None,
             also_after: list[str] | None = None) -> Node:
    """Add a step, connected after `after` (into `port`) and from each of `also_after`. Nothing is changed if any
    step it names does not exist."""
    type_key = resolve_type(type_key)
    for nid in [after, *(also_after or [])]:
        if nid:
            require_node(p, nid)
    x, y = place(p, after)
    node = p.add_node(type_key, title=title, params=params or {}, x=x, y=y, id=node_id)
    try:
        if after:
            p.connect(after, node.id, port)
        for extra in also_after or []:
            p.connect(extra, node.id)
    except Exception:
        p.remove_node(node.id)
        raise
    return node


def node_record(p: Pipeline, st: NodeState) -> dict[str, Any]:
    """How a step is reported by both front ends: node_id, title, type, status, rows, columns ({name, dtype}),
    error, messages, report, elapsed, from_cache, and the rest of its state."""
    n = p.nodes[st.node_id]
    return json_safe({**st.to_dict(), "title": n.title, "type": n.type})


def run_record(p: Pipeline, ex: Executor, res: dict[str, NodeState], elapsed: float) -> dict[str, Any]:
    """How a run is reported by both front ends; `nodes` is keyed by step id. ``findings`` are what the steps
    said (a sentence each), most interesting first, and ``headline`` is the best of them."""
    from .core.findings import collect, record as finding_record, headline as finding_headline
    failed = [nid for nid, s in res.items() if s.status == "failed"]
    found = collect(finding_record(p.nodes[nid].type, p.nodes[nid].title, s.report, s.status)
                    for nid, s in res.items())
    return {"ok": not failed, "failed": failed, "nodes": {nid: node_record(p, s) for nid, s in res.items()},
            "problems": p.problems(), "elapsed": elapsed, "cache_dir": str(ex.cache_dir),
            "findings": found, "headline": finding_headline(found)}


def result_frame(p: Pipeline, ex: Executor, node_id: str, run: bool) -> pl.LazyFrame:
    """A step's output, running it first when `run` is set and it is not computed yet."""
    require_node(p, node_id)
    st = ex.state(node_id)
    if st.status != "done":
        if not run:
            raise ValueError(f"{node_id} hasn't been run yet (it's {st.status}). Run the project first, or ask for it to be run")
        res = ex.run(targets=[node_id])
        if res[node_id].status != "done":
            raise StepFailed(f"{node_id} failed: {res[node_id].error}")
    return ex.frame(node_id)


def select_columns(lf: pl.LazyFrame, columns: list[str] | None) -> pl.LazyFrame:
    """Only these columns, with a plain message for one that does not exist."""
    if not columns:
        return lf
    have = lf.collect_schema().names()
    missing = [c for c in columns if c not in have]
    if missing:
        raise ValueError(f"There is no column called {missing[0]!r}. Columns: {have[:20]}")
    return lf.select(columns)


def chart_params(node: Node, kind: str | None = None, x: str | None = None, y: list[str] | None = None,
                 column: str | None = None, title: str | None = None) -> dict[str, Any]:
    """A chart step's settings, or an ad-hoc chart of any step; each argument given overrides."""
    params = dict(node.params) if node.type == "chart" else {}
    if kind:
        params["kind"] = kind
    if x:
        params["x"] = x
    if y:
        params["series"] = [{"column": c} for c in y]
    if column:
        params["column"] = column
    if title:
        params["title"] = title
    params.setdefault("kind", "line")
    return params


def build_template(out: Path, template: str, data_file: str | Path | None) -> tuple[Pipeline, Path]:
    """A starter project at `out` on `data_file` (relative to the project's folder), or on a generated sample
    file when none is given. The template name and the data file are checked before anything is written."""
    from .core.samples import build_template as _build, write_sample, check_template
    check_template(template)
    out = out.expanduser().resolve()
    if data_file:
        data = Path(data_file).expanduser()
        data = (data if data.is_absolute() else out.parent / data).resolve()
        if not data.is_file():
            raise ValueError(f"No data file at {data}")
    else:
        data = write_sample(out.parent)
    pipe = Pipeline(out.stem)
    pipe.path = out
    _build(template, pipe, data)
    pipe.save(out)
    return pipe, data


def _path_settings(p: Pipeline, node_id: str) -> list[Path]:
    """The files named by a step's path settings, resolved (symlinks and `..` included)."""
    node = p.nodes[node_id]
    nt = registry.get(node.type)
    out = []
    for prm in nt.params:
        v = node.params.get(prm.name)
        if prm.kind == "path" and isinstance(v, str) and v.strip():
            target = Path(v).expanduser()
            out.append((target if target.is_absolute() else p.directory / target).resolve())
    if node.type == "report" and node.params.get("pdf", True):
        out += [t.with_suffix(".pdf") for t in out]           # the report's PDF sits next to it
    return out


def source_files(p: Pipeline) -> set[Path]:
    """Every file a source step of the project reads."""
    return {f for nid, n in p.nodes.items() if registry.get(n.type).kind == "source" for f in _path_settings(p, nid)}


def output_files(p: Pipeline, node_id: str) -> list[Path]:
    """The files a step saves (nothing for steps that do not save files)."""
    return _path_settings(p, node_id) if registry.get(p.nodes[node_id].type).kind == "sink" else []


def _same_file(a: Path, b: Path) -> bool:
    """One file under two spellings too: A.CSV and a.csv on a Mac or Windows disk, or a hard link."""
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:                        # either does not exist yet: then they are not the same file
        return False


def unsafe_write(p: Pipeline, target: Path, folder: Path) -> str | None:
    """Why writing `target` is refused (outside `folder`, or over a data file the project reads), else None."""
    target = target.resolve()
    if not target.is_relative_to(folder):
        return f"Can only save inside the project folder {folder}, not {target}"
    if any(_same_file(target, src) for src in source_files(p)):
        return f"Won't save over {target}, because the project reads its data from that file"
    return None


def unsafe_outputs(p: Pipeline, node_id: str, folder: Path) -> list[str]:
    """Why a step's files may not be written: each lands outside `folder` or over a file the project reads."""
    return [why for t in output_files(p, node_id) if (why := unsafe_write(p, t, folder))]


# ----------------------------------------------------------------- one writer at a time
LOCK_WAIT = 30.0          # seconds to wait for another program to finish changing a project file


class ProjectBusy(PipelineError):
    """Another program is changing the project file, or changed it while this was being done."""


def _try_lock(fd: int) -> bool:
    """True when the lock is ours, False when another holder has it. A folder that cannot lock at all (some
    network drives) is an error of its own, not a wait that ends in "being changed by another program"."""
    import errno
    busy = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK))
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in busy:
            return False
        raise PipelineError(f"The project's folder can't lock files ({e.strerror or e}), so two programs could save over each other "
                            "there. Move the project to a local folder.") from e
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def lock_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    return path.parent / ".dancr" / "locks" / f"{path.name}.lock"


_mine = threading.local()          # the project files whose lock this thread holds, so taking one again nests


@contextmanager
def project_lock(path: Path | str, wait: float | None = None) -> Iterator[None]:
    """Hold a project file's lock. The window, the command line and the MCP server each take it to read, change
    and write a project file, so none of them overwrites another's change. It is an operating-system lock on a
    file in the project's .dancr folder, so it goes away with the process however the process ends. A thread
    that already holds it (a save inside an edit) just carries on; other threads of the same process wait."""
    path = Path(path).expanduser().resolve()
    held = _mine.__dict__.setdefault("paths", set())
    if path in held:
        yield
        return
    held.add(path)
    try:
        with _os_lock(path, wait):
            yield
    finally:
        held.discard(path)


@contextmanager
def _os_lock(path: Path, wait: float | None) -> Iterator[None]:
    lock = lock_file(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + (LOCK_WAIT if wait is None else wait)
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise ProjectBusy(f"{path.name} is being changed by another program (the DANCR window, the command "
                                  "line or an agent). Try again in a moment.")
            time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def read_project(path: Path) -> tuple[Pipeline, str]:
    """A project file and the exact text it was read from."""
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"{path.name} is not valid JSON: {e}") from e
    return Pipeline.from_dict(data, path), text


@contextmanager
def editing(path: Path | str) -> Iterator[Pipeline]:
    """Load a project file, let the caller change it, and save it if anything changed, holding the file's lock
    throughout. A program that does not take the lock (a text editor) may still change the file meanwhile:
    then nothing is saved and the caller is told to try again."""
    path = Path(path).expanduser().resolve()
    with project_lock(path):
        p, text = read_project(path)
        before = p.dumps()                  # text, not the dict: the dict shares its parts with the project
        yield p
        if p.dumps() == before:
            return
        if path.read_text(encoding="utf-8") != text:
            raise ProjectBusy(f"{path.name} was changed by another program while this was running, so nothing was "
                              "saved. Try again.")
        p.save()


# ----------------------------------------------------------------- answers (shared by `dancr ask/suggest/answer` and MCP)
def add_files(p: Pipeline, files: list[str] | None) -> list[str]:
    """A load step for each file not loaded yet (a file already loaded keeps its step and its settings).
    Returns the load steps' ids, in the order given."""
    from .core.registry import resolve_path
    out = []
    for f in files or []:
        target = resolve_path(p.directory, str(f)).resolve()
        if not target.exists():
            raise ValueError(f"File not found: {target}")
        from .core.nodes.load import tables_in
        try:
            rel = str(target.relative_to(p.directory.resolve()))
        except ValueError:
            rel = str(target)
        for title, extra in tables_in(target):
            existing = next((nid for nid, n in p.nodes.items() if n.type == "load_file" and n.params.get("path")
                             and resolve_path(p.directory, str(n.params["path"])).resolve() == target
                             and str(n.params.get("sheet") or "") == str(extra.get("sheet") or "")), None)
            if existing is None:
                existing = add_step(p, "load_file", {"path": rel, **extra}, title=title).id
            out.append(existing)
    return out


def data_model(p: Pipeline, deep: bool = True):
    from .core.answers import model_for
    return model_for(p, Executor(p), deep=deep)


def suggestions(p: Pipeline, focus: str | None = None) -> list[dict[str, Any]]:
    from .core.recipes import suggest
    from .core.answers import model_for
    from .core.understand import default_tables
    if focus:
        require_node(p, focus)
    nodes = default_tables(p) + ([focus] if focus and focus not in default_tables(p) else [])
    m = model_for(p, Executor(p), nodes=nodes)
    return [{"index": i, **s.to_dict()} for i, s in enumerate(suggest(m, focus))]


def build_answer(p: Pipeline, spec: dict[str, Any], answer_id: str | None = None) -> dict[str, Any]:
    """Build (or rebuild) an answer from a spec; the caller saves the project."""
    from .core import answers
    m = answers.model_for(p, Executor(p), nodes=answers.tables_for(p, spec))
    a, plan = answers.build(p, m, spec, answer_id=answer_id)
    return {**answers.describe(p, a), "chips": plan.chips, "why": plan.why, "sentence": plan.sentence(),
            "set_aside": plan.set_aside, "note": answers.set_aside_note(plan.set_aside)}


def ask_question(p: Pipeline, text: str, build: bool = True) -> dict[str, Any]:
    """Read a question; with ``build`` add its answer to the project. ``ok`` is False (with a message and any
    "did you mean" hints) when the question could not be read."""
    from .core.ask import ask
    from .core.answers import model_for
    m = model_for(p, Executor(p))
    asked = ask(m, text)
    out: dict[str, Any] = {"question": asked.to_dict()}
    if asked.ok and build:
        out["answer"] = build_answer(p, asked.spec)
    return out


def change_answer(p: Pipeline, answer_id: str, key: str | None = None, value: Any = None,
                  assumption: int | None = None, choice: int = 0) -> dict[str, Any]:
    """Change an answer by one chip (``key`` = ``value``) or by choosing an alternative of one of its
    assumptions (``assumption`` index, ``choice`` index), and rebuild it in place."""
    from .core.recipes import apply_choice
    a = p.answer(answer_id)
    if a is None:
        raise ValueError(f"No answer called {answer_id!r}. Answers: {[x.id for x in p.answers]}")
    if assumption is not None:
        if not 0 <= assumption < len(a.assumptions):
            raise ValueError(f"The answer has {len(a.assumptions)} assumptions (numbered from 0)")
        choices = a.assumptions[assumption].get("choices") or []
        if not 0 <= choice < len(choices):
            raise ValueError(f"That assumption has {len(choices)} alternatives (numbered from 0)")
        spec = apply_choice(a.spec, "set", choices[choice]["set"])
    elif key:
        _check_choice(p, a, key, value)
        spec = apply_choice(a.spec, key, value)
    else:
        raise ValueError("Say what to change: a chip (key and value) or an assumption's alternative")
    return build_answer(p, spec, answer_id)


def _check_choice(p: Pipeline, a, key: str, value: Any) -> None:
    """A chip may only be set to one of its choices (what the window offers), so a typo never builds a broken answer."""
    from .core.answers import model_for, tables_for
    from .core.recipes import chips
    m = model_for(p, Executor(p), deep=False, nodes=tables_for(p, a.spec))
    cs = {c["key"]: c for c in chips(m, a.spec)}
    if key not in cs:
        raise ValueError(f"This answer has no choice called {key!r}. Choices: {', '.join(cs) or 'none'}")
    allowed = [ch["value"] for ch in cs[key]["choices"]]
    if value not in allowed:
        shown = ", ".join(json.dumps(v) for v in allowed[:12])
        raise ValueError(f"{key} cannot be {json.dumps(value)}. Choose one of: {shown}")
