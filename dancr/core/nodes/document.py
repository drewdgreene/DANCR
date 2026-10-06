"""Load documents: read PDF, Office, EPUB, HTML and other files as ordinary DANCR tables, through MinerU.

MinerU (Apache-2.0) turns a document into structured content (text, headings, tables, formulas, figures) with
stable page/block locators. This step exposes that as a table:

- ``what=blocks``: one row per content block — ``doc, doc_id, page, block, type, text, bbox, locator`` — so a
  document is searchable, filterable and citable.
- ``what=tables``: a catalog of the tables found (``doc, page, table_index, caption, n_rows, n_cols, csv,
  locator``), each table written to a CSV next to the project so it can be analysed like any other table.

Engines: ``output`` reads a MinerU result folder (no MinerU install needed), ``command`` runs the MinerU CLI,
``endpoint`` talks to a MinerU V1 server. Local by default; nothing is uploaded unless ``allow_remote`` is on.
Pure standard library + Polars; MinerU itself is an external tool, detected rather than bundled.
"""
from __future__ import annotations

import csv
import glob as _glob
import html.parser
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..secrets import expand_env
from .load import DOC_EXT, IMAGE_EXT, effective_ext

OUTPUT_EXTS = {".json", ".md", ".markdown", ".txt"}
BLOCK_COLUMNS = ["doc", "doc_id", "page", "block", "type", "text", "bbox", "locator"]
TABLE_COLUMNS = ["doc", "doc_id", "page", "table_index", "caption", "n_rows", "n_cols", "csv", "locator"]


# --------------------------------------------------------------------------- MinerU presence
_MEMO: dict[str, tuple[float, Any]] = {}
_MEMO_LOCK = threading.Lock()


def _memo(key: str, ttl: float, fn) -> Any:
    now = time.time()
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _MEMO_LOCK:
        if len(_MEMO) > 256:                 # bounded, as the connector's memo is: never grow without limit
            _MEMO.clear()
        _MEMO[key] = (now, value)
    return value


def _app_dirs() -> list[Path]:
    """Where a bundled MinerU env may live: an explicit home, then next to the app (frozen) / the repo (source)."""
    import sys
    dirs: list[Path] = []
    home = os.environ.get("DANCR_MINERU_HOME", "").strip()
    if home:
        dirs.append(Path(home).expanduser())
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent / ".dancr-mineru")
    dirs.append(Path(__file__).resolve().parents[3] / ".dancr-mineru")     # the repo root
    dirs.append(Path.home() / ".dancr-mineru")
    return dirs


def mineru_tool(params: dict[str, Any] | None = None) -> str | None:
    """The MinerU CLI to use: the step's setting, ``$DANCR_MINERU_CMD``, a bundled env, else PATH."""
    params = params or {}
    cmd = str(params.get("mineru_cmd") or "").strip() or os.environ.get("DANCR_MINERU_CMD", "").strip()
    if cmd:
        return cmd
    for d in _app_dirs():
        for sub in ("bin", "Scripts"):                    # POSIX venvs use bin/, Windows uses Scripts/ + .exe
            for name in ("mineru-kit", "mineru"):
                for cand in (d / sub / name, d / sub / (name + ".exe")):
                    if cand.exists():
                        return str(cand)
    for name in ("mineru-kit", "mineru"):
        found = shutil.which(name)
        if found:
            return found
    return None


def mineru_home() -> str | None:
    """A bundled MinerU home (holding its models), for ``$MINERU_HOME``, or None to use the default."""
    for d in _app_dirs():
        if (d / "models").exists():
            return str(d)
    return None


_SECRETISH_ENV = re.compile(r"(?i)(key|token|secret|password|passwd|pwd|credential|auth|session|cookie)")


def _subprocess_env() -> dict[str, str]:
    """The environment handed to the MinerU subprocess: this process's own, less anything that looks like a
    credential. MinerU needs paths and locale, not the person's API keys; an external tool must not inherit
    every secret DANCR was started with."""
    env = {k: v for k, v in os.environ.items() if not _SECRETISH_ENV.search(k)}
    if not env.get("MINERU_HOME"):
        home = mineru_home()
        if home:
            env["MINERU_HOME"] = home
    return env


def mineru_version(tool: str | None) -> str | None:
    if not tool:
        return None

    def probe() -> str | None:
        for args in (["version", "--json"], ["--version"]):
            try:
                r = subprocess.run([tool, *args], capture_output=True, text=True, timeout=30, env=_subprocess_env())
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                text = (r.stdout or r.stderr).strip()
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict) and (obj.get("version") or obj.get("mineru_version")):
                        return str(obj.get("version") or obj.get("mineru_version"))
                except ValueError:
                    pass
                m = re.search(r"\d+\.\d+[.\d]*", text)
                if m:
                    return m.group(0)
        return None

    return _memo(f"mineru-version:{tool}", 30.0, probe)


# --------------------------------------------------------------------------- files
def documents(directory: Path, params: dict[str, Any]) -> list[Path]:
    """The document (or MinerU-output) files a path/pattern matches, for the cache fingerprint and the step."""
    raw = str(params.get("path") or "").strip()
    if not raw:
        return []
    pattern = str(params.get("pattern") or "*").strip() or "*"
    recursive = bool(params.get("recursive", False))
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = directory / base
    try:
        if base.is_file():
            found = [base]
        elif base.is_dir():
            found = list(base.rglob(pattern) if recursive else base.glob(pattern))
        elif any(ch in raw for ch in "*?["):
            found = [Path(p) for p in _glob.glob(str(base), recursive=recursive)]
        else:
            return []
    except OSError:
        return []
    exts = set(DOC_EXT) | set(IMAGE_EXT) | OUTPUT_EXTS
    out = [f.resolve() for f in found
           if f.is_file() and not f.name.startswith(".") and effective_ext(f) in exts]
    return sorted(set(out), key=str)


def doc_node_for(path: str | Path) -> str | None:
    """The document step for this file (documents only; images need it explicitly), or None."""
    return "load_document" if effective_ext(Path(str(path))) in DOC_EXT else None


def _digest(directory: Path, params: dict[str, Any]) -> dict[str, Any]:
    """What changes when the document result would: the engine, tier, pages and the MinerU version in use."""
    tool = mineru_tool(params)
    return {"engine": str(params.get("engine") or "auto"), "tier": str(params.get("tier") or ""),
            "pages": str(params.get("pages") or ""), "what": str(params.get("what") or "blocks"),
            "cmd": tool or "", "version": mineru_version(tool) or ""}


# --------------------------------------------------------------------------- MinerU output -> normalized pages
class _HTMLTable(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = True
            self._buf = []
        elif tag == "br" and self._cell:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell and self._row is not None:
            self._row.append("".join(self._buf).strip())
            self._cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell:
            self._buf.append(data)


def _html_rows(text: str) -> list[list[str]]:
    p = _HTMLTable()
    try:
        p.feed(text)
    except Exception:  # noqa: BLE001 - malformed HTML simply yields what was parsed
        pass
    return [r for r in p.rows if r]


def _find_table_html(obj: Any) -> str | None:
    """The first HTML table string anywhere inside a block/entry."""
    if isinstance(obj, str):
        return obj if re.search(r"<\s*table", obj, re.I) else None
    if isinstance(obj, dict):
        for k in ("table_body", "table_html", "html", "content", "text"):
            v = obj.get(k)
            if isinstance(v, str) and re.search(r"<\s*table", v, re.I):
                return v
        for v in obj.values():
            found = _find_table_html(v)
            if found:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_table_html(v)
            if found:
                return found
    return None


def _block_text(block: dict[str, Any]) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for e in c:
            if isinstance(e, dict):
                v = e.get("content", e.get("text"))
                if isinstance(v, list):
                    v = " ".join(str(x) for x in v)
                if v:
                    parts.append(str(v))
            elif e:
                parts.append(str(e))
        return "\n".join(parts)
    return str(block.get("text") or "")


_TYPE_MAP = {"title": "heading", "doc_title": "heading", "interline_equation": "formula", "equation": "formula",
             "image": "figure", "figure": "figure", "table_body": "table", "table": "table", "list": "list"}


def _block(kind: Any, text: str, index: int, bbox: Any = None) -> dict[str, Any]:
    return {"type": _TYPE_MAP.get(str(kind or "text").lower(), str(kind or "text").lower()),
            "text": text, "index": index, "bbox": bbox}


def _pages_from_content_list(items: list, pages_out: list[dict[str, Any]]) -> None:
    """A flat content list (v1) or a list of pages (v2) -> normalized pages."""
    if items and isinstance(items[0], list):                       # v2: one list per page
        for i, page_items in enumerate(items):
            pages_out.append({"page": i + 1, "blocks": _blocks_from_items(page_items)})
        return
    by_page: dict[int, list] = {}
    order: list[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        p = it.get("page_idx", it.get("page", 0))
        p = int(p) if isinstance(p, (int, float)) else 0
        if p not in by_page:
            by_page[p] = []
            order.append(p)
        by_page[p].append(it)
    for p in order:
        pages_out.append({"page": p + 1, "blocks": _blocks_from_items(by_page[p])})


def _table_caption(obj: Any) -> str | None:
    """The caption of a table block, wherever MinerU nests a ``table_caption`` entry."""
    found: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if str(o.get("type")) == "table_caption":
                said = _block_text(o)
                if said:
                    found.append(said)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return "\n".join(found) if found else None


def _blocks_from_items(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items if isinstance(items, list) else []):
        if not isinstance(it, dict):
            continue
        kind = it.get("type")
        body = _find_table_html(it)
        caption = _table_caption(it)
        text = _block_text(it)
        if body:                                          # a table: keep the caption as text, not the raw HTML
            text = caption or "[table]"
        if not text and kind in ("table", "table_body") and it.get("table_body"):
            text = "[table]"
        blk = _block(kind, "" if text is None else str(text), i, it.get("bbox"))
        if body:
            blk["html"] = body
        if caption:
            blk["caption"] = caption
        out.append(blk)
    return out


def _normalize(obj: Any) -> list[dict[str, Any]]:
    """Any MinerU JSON shape -> [{'page': int|None, 'blocks': [{'type','text','index','bbox','html'?}]}]."""
    pages: list[dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("pages"), list):
        for pg in obj["pages"]:
            if not isinstance(pg, dict):
                continue
            raw_idx = pg.get("page_idx", pg.get("page", len(pages)))
            page = int(raw_idx) + 1 if isinstance(raw_idx, (int, float)) else None
            pages.append({"page": page, "blocks": _blocks_from_items(pg.get("blocks") or [])})
        return pages
    if isinstance(obj, list):
        _pages_from_content_list(obj, pages)
        return pages
    if isinstance(obj, dict) and "content" in obj:                 # doclib `mineru parse --json`: {parse, content}
        return _normalize(obj["content"])
    if isinstance(obj, dict) and isinstance(obj.get("blocks"), list):
        return [{"page": None, "blocks": _blocks_from_items(obj["blocks"])}]
    raise ValueError("This is not a MinerU result (no pages or blocks found).")


def _markdown_pages(text: str) -> list[dict[str, Any]]:
    """A last-resort reading of markdown: headings and paragraphs as blocks."""
    blocks: list[dict[str, Any]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(_block("text", "\n".join(para).strip(), len(blocks)))
            para.clear()

    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s", line):
            flush()
            blocks.append(_block("heading", re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip(), len(blocks)))
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return [{"page": None, "blocks": blocks}]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as e:
        raise ValueError(f"{path.name} is not valid JSON: {e}") from None


def _read_output(path: Path) -> list[dict[str, Any]]:
    """Read a MinerU result: a JSON file, a markdown file, or a folder holding them."""
    candidates: list[Path] = []
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        for name in ("middle_json.json", "structured_content.json", "content_list.json",
                     "content_list_v2.json", "markdown.md", "output.md"):
            hit = next(path.rglob(name), None)
            if hit is not None:
                candidates.append(hit)
        if not candidates:                                        # any json with pages/blocks, else any markdown
            jsons = sorted(path.rglob("*.json"))
            candidates = jsons or sorted(path.rglob("*.md"))
    for c in candidates:
        if c.suffix.lower() == ".json":
            return _normalize(_read_json(c))
        if c.suffix.lower() in (".md", ".markdown", ".txt"):
            return _markdown_pages(c.read_text(encoding="utf-8", errors="replace"))
    raise ValueError(f"No MinerU result found in {path.name}. Point at a folder holding middle_json.json, "
                     "structured_content.json or content_list.json (or markdown.md).")


# --------------------------------------------------------------------------- engines
def _run(cmd: list[str], timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env or _subprocess_env())
    except FileNotFoundError:
        raise ValueError(f"Could not find the MinerU command {cmd[0]!r}. Install it, or set the MinerU command.") from None
    except subprocess.TimeoutExpired:
        raise ValueError(f"MinerU took longer than {int(timeout)}s. Try a smaller page range or a faster tier.") from None


def _command_pages(file: Path, tool: str, tier: str, pages: str, timeout: float) -> list[dict[str, Any]]:
    if "kit" in Path(tool).name.lower():
        with tempfile.TemporaryDirectory(prefix="dancr-mineru-") as tmp:
            cmd = [tool, "parse", str(file), "--format", "middle_json", "-o", tmp]
            if tier and tier != "auto":
                cmd += ["--tier", tier]
            if pages:
                cmd += ["--pages", pages]
            r = _run(cmd, timeout)
            if r.returncode != 0 and "unrecognized arguments" in (r.stderr or "").lower():
                # fall back to the doclib reading interface
                return _command_pages_doclib(file, tool, tier, pages, timeout)
            if r.returncode != 0:
                raise ValueError(_last_line(r) or f"MinerU failed ({r.returncode})")
            for name in ("middle_json.json", "structured_content.json", "content_list.json"):
                hit = next(Path(tmp).rglob(name), None)
                if hit is not None:
                    return _normalize(_read_json(hit))
            hit = next(Path(tmp).rglob("*.json"), None)
            if hit is not None:
                return _normalize(_read_json(hit))
            raise ValueError("MinerU ran but produced no readable result. Check the MinerU command and version.")
    return _command_pages_doclib(file, tool, tier, pages, timeout)


def _command_pages_doclib(file: Path, tool: str, tier: str, pages: str, timeout: float) -> list[dict[str, Any]]:
    cmd = [tool, "parse", str(file), "--json"]
    if tier and tier != "auto":
        cmd += ["--tier", tier]
    if pages:
        cmd += ["--pages", pages]
    r = _run(cmd, timeout)
    if r.returncode != 0:
        raise ValueError(_last_line(r) or f"MinerU failed ({r.returncode})")
    try:
        return _normalize(json.loads(r.stdout))
    except ValueError:
        text = (r.stdout or "").strip()
        if text:
            return _markdown_pages(text)
        raise ValueError(_last_line(r) or "MinerU produced no readable result")


def _endpoint_pages(file: Path, params: dict[str, Any], tier: str, pages: str, timeout: float) -> list[dict[str, Any]]:
    import urllib.request
    endpoint = expand_env(params.get("endpoint") or os.environ.get("DANCR_MINERU_ENDPOINT", "")).rstrip("/")
    if not endpoint:
        raise ValueError("Set the MinerU endpoint URL for this step, or use the local MinerU command.")
    if not bool(params.get("allow_remote", False)):
        raise ValueError("Uploading a document to a MinerU endpoint needs 'Allow upload' turned on. "
                         "Leave it off to keep the document on this machine.")
    token = expand_env(params.get("token") or os.environ.get("DANCR_MINERU_API_KEY", ""))
    boundary = "----dancrMinerU" + os.urandom(8).hex()
    body = bytearray()

    def part(head: bytes, data: bytes) -> None:
        body.extend(b"--" + boundary.encode() + b"\r\n" + head + b"\r\n" + data + b"\r\n")

    part(b'Content-Disposition: form-data; name="tier"\r\n', tier.encode() if tier != "auto" else b"standard")
    if pages:
        part(b'Content-Disposition: form-data; name="pages"\r\n', pages.encode())
    part(b'Content-Disposition: form-data; name="file"; filename="' + file.name.encode()
         + b'"\r\nContent-Type: application/octet-stream\r\n', file.read_bytes())
    body.extend(b"--" + boundary.encode() + b"--\r\n")
    req = urllib.request.Request(endpoint + "/parse", data=bytes(body), method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - the user named this endpoint
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 - reported as a plain error below
        raise ValueError(f"Could not parse {file.name} at {endpoint}: {e}") from None
    try:
        return _normalize(data)
    except ValueError:
        raise ValueError("The MinerU endpoint returned a result this step does not understand (expected "
                         "pages/blocks or a content list).") from None


def _last_line(r: subprocess.CompletedProcess) -> str:
    text = (r.stderr or r.stdout or "").strip()
    return text.splitlines()[-1][:400] if text else ""


# --------------------------------------------------------------------------- reading one document
def _decide_engine(files: list[Path], params: dict[str, Any]) -> str:
    engine = str(params.get("engine") or "auto")
    if engine != "auto":
        return engine
    if all(effective_ext(f) in OUTPUT_EXTS for f in files):
        return "output"
    endpoint = str(params.get("endpoint") or os.environ.get("DANCR_MINERU_ENDPOINT", "")).strip()
    if mineru_tool(params):
        return "command"
    if endpoint:
        return "endpoint"
    return "command"                                          # no tool: the error names what to install


def _parse_one(file: Path, engine: str, params: dict[str, Any], tier: str, pages: str, timeout: float,
               preview: bool) -> list[dict[str, Any]]:
    if engine == "output":
        return _read_output(file)
    if engine == "command":
        tool = mineru_tool(params)
        if not tool:
            raise ValueError(
                "Reading PDF and Office documents needs MinerU (Apache-2.0). Install it — "
                "`uv tool install \"mineru>=4.0,<5\"` (or `pip install \"mineru>=4.0,<5\"`) — then set the MinerU "
                "command if it is not on PATH. Or point this step at a folder MinerU already produced "
                "(engine 'an existing MinerU result'), or set a MinerU endpoint.")
        return _command_pages(file, tool, tier, pages, timeout)
    if engine == "endpoint":
        return _endpoint_pages(file, params, tier, pages, timeout)
    raise ValueError(f"Unknown engine {engine!r}")


# --------------------------------------------------------------------------- node
def _load_document(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    files = documents(ctx.pipeline_dir, params)
    raw = str(params.get("path") or "").strip()
    if not files:
        raise ValueError(f"No documents matched {raw!r}. Give a PDF/Office file, a folder, or a glob such as "
                         "'reports/*.pdf' (or a MinerU result file)")
    what = str(params.get("what") or "blocks")
    engine = _decide_engine(files, params)
    tier = str(params.get("tier") or "standard")
    pages = str(params.get("pages") or "").strip()
    timeout = float(params.get("timeout") or 600)
    include = {str(t).lower() for t in (params.get("include") or [])}
    preview = bool(ctx.preview)
    blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    written: list[Path] = []
    docs: list[dict[str, Any]] = []
    out_dir = str(params.get("out_dir") or "extracted").strip() or "extracted"

    for f in files:
        doc_pages = pages
        if preview and not doc_pages and effective_ext(f) == ".pdf":
            doc_pages = "1"                                 # a preview reads the first page only
        found = _parse_one(f, engine, params, tier, doc_pages, timeout, preview)
        doc_id = None
        for pg in found:
            for blk in pg.get("blocks", []):
                doc_id = doc_id or blk.get("doc_id")
                locator = blk.get("locator") or _locator(f.name, pg.get("page"), blk.get("index"))
                if what == "blocks":
                    if include and blk["type"] not in include:
                        continue
                    blocks.append({"doc": f.name, "doc_id": doc_id, "page": pg.get("page"),
                                   "block": blk.get("index"), "type": blk["type"], "text": blk.get("text", ""),
                                   "bbox": _bbox_str(blk.get("bbox")), "locator": locator})
                elif blk["type"] == "table":
                    rows = _html_rows(blk.get("html") or "") or [[blk.get("text", "")]]
                    tables.append({"doc": f.name, "doc_id": doc_id, "page": pg.get("page"),
                                   "table_index": len(tables) + 1, "caption": blk.get("caption"),
                                   "n_rows": len(rows), "n_cols": max((len(r) for r in rows), default=0),
                                   "rows": rows, "locator": locator})
        docs.append({"doc": f.name, "engine": engine, "pages": len(found)})
        if preview and len(blocks) >= 200:
            break

    if what == "blocks":
        if preview:
            blocks = blocks[:200]
        cols = {c: [b.get(c) for b in blocks] for c in BLOCK_COLUMNS}
        frame = pl.DataFrame(cols).lazy()
        msg = (f"Read {len(blocks):,} block{'s' if len(blocks) != 1 else ''} from "
               f"{len(docs)} document{'s' if len(docs) != 1 else ''}"
               + (f" ({tier} tier)" if engine != "output" else "")
               + (" — preview, first page only" if preview else ""))
        report = {"kind": "document", "engine": engine, "tier": tier, "what": what, "documents": docs,
                  "blocks": len(blocks)}
        return NodeResult(frame, report=report, messages=[msg])

    # tables mode: write each table as a CSV next to the project, return the catalog
    if preview:
        tables = tables[:20]
    for i, t in enumerate(tables, start=1):
        rel = f"{out_dir}/{_safe(t['doc'])}/table_{i:03d}.csv"
        out = ctx.resolve_output(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out, t.pop("rows"))
        t["csv"] = rel
        written.append(out)
    cols = {c: [t.get(c) for t in tables] for c in TABLE_COLUMNS}
    frame = pl.DataFrame(cols).lazy()
    msg = (f"Found {len(tables):,} table{'s' if len(tables) != 1 else ''} in {len(docs)} document"
           f"{'s' if len(docs) != 1 else ''}"
           + (f" → {out_dir}/…" if tables else "")
           + (" — preview" if preview else ""))
    report = {"kind": "document", "engine": engine, "tier": tier, "what": what, "documents": docs,
              "tables": len(tables), "out_dir": out_dir}
    return NodeResult(frame, report=report, messages=[msg], files=written)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem) or "document"


def _locator(doc: str, page: Any, index: Any) -> str:
    bits = [f"{doc}"]
    if page is not None:
        bits.append(f"page {page}")
    if index is not None:
        bits.append(f"block {index}")
    return " · ".join(bits)


def _bbox_str(bbox: Any) -> str | None:
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)):
        return ", ".join(str(round(float(x), 1)) if isinstance(x, (int, float)) else str(x) for x in bbox)
    return str(bbox)


def _write_csv(out: Path, rows: list[list[str]]) -> None:
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(["" if c is None else c for c in r])


registry.register(NodeType(
    key="load_document", label="Load document (PDF/Office)", category="Get data", icon="document",
    kind="source", inputs=[],
    description="Read PDF, Word, PowerPoint, EPUB, HTML and other documents as a table through MinerU "
                "(Apache-2.0): one row per content block (with a page/block locator), or a catalog of the tables "
                "found, each written to a CSV. Local by default; MinerU is detected, not bundled.",
    apply=_load_document, summary=lambda p: Path(str(p.get("path") or "")).name or "no document chosen",
    source_files=documents, source_digest=_digest,
    params=[
        Param("path", "Document, folder or glob", "dir", required=True,
              help="A PDF/Office/EPUB/HTML file, a folder, or a glob such as 'reports/*.pdf'. "
                   "For the 'existing result' engine, a MinerU output folder."),
        Param("what", "Output", "choice", default="blocks",
              choices=[("blocks", "one row per content block"), ("tables", "a catalog of the tables found")]),
        Param("engine", "Engine", "choice", default="auto",
              choices=[("auto", "use MinerU if present, else an existing result"),
                       ("output", "read an existing MinerU result (no MinerU needed)"),
                       ("command", "run the MinerU command"),
                       ("endpoint", "a MinerU server")]),
        Param("tier", "Quality tier", "choice", default="standard",
              choices=[("flash", "flash (fastest, lower quality)"), ("basic", "basic"),
                       ("standard", "standard"), ("advanced", "advanced (slowest, best)")]),
        Param("pages", "Pages (PDF)", "text", default="", advanced=True,
              help="A range such as 1-10 or 'all'. Empty reads the document's default range"),
        Param("include", "Block types to keep", "text_list", default=[], advanced=True,
              help="Empty = every type. e.g. table, heading, text"),
        Param("out_dir", "Folder for extracted tables", "text", default="extracted", advanced=True,
              visible_when={"what": "tables"}),
        Param("pattern", "File name pattern", "text", default="*", advanced=True,
              help="Which files to read inside a folder, e.g. *.pdf"),
        Param("recursive", "Look inside sub-folders", "bool", default=False, advanced=True),
        Param("mineru_cmd", "MinerU command", "text", default="", advanced=True,
              help="The MinerU executable (default: $DANCR_MINERU_CMD, else mineru-kit/mineru on PATH)"),
        Param("endpoint", "MinerU endpoint", "text", default="", advanced=True,
              help="A MinerU V1 server URL (default: $DANCR_MINERU_ENDPOINT)"),
        Param("token", "Endpoint token", "text", default="", secret=True, advanced=True,
              help="An API key for the endpoint, or ${ENV_VAR} (default: $DANCR_MINERU_API_KEY)"),
        Param("allow_remote", "Allow uploading to the endpoint", "bool", default=False, advanced=True,
              help="Off keeps every document on this machine"),
        Param("timeout", "Seconds to wait", "int", default=600, min=1, advanced=True),
    ],
))
