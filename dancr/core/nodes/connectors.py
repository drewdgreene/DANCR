"""Read data that is not a local file: a table at a URL, or a database query.

- ``load_url`` reads CSV/TSV/text, Parquet or JSON over http(s) (or a ``file:`` URL). CSV/TSV are handed to the
  same reader as Load file (layout, dates, numbers), so a remote export is read like a local one. Only the
  standard library is needed; the bytes are cached under the project's results folder.
- ``load_sql`` runs a query (or reads a whole table) from SQLite (built in) or a database server (PostgreSQL and
  others, through SQLAlchemy and a driver, which ship with DANCR).

Connection strings and passwords should be written as ``${ENV_VAR}`` and kept in the environment; a connector's
settings are redacted wherever they are shown (see :mod:`dancr.core.secrets`). Reading here reaches the network,
so DANCR is no longer strictly offline once you add one of these steps.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from ..secrets import expand_env, redact
from ._memo import make_memo
from .load import scan_file
from .load_folder import SHARED

# ---------------------------------------------------------------- small caches
_memo = make_memo()


def _short(url: str) -> str:
    u = str(redact(url))
    return u if len(u) <= 80 else u[:77] + "…"


# =================================================================== load_url
MAX_DOWNLOAD_BYTES = 2 * 1024 ** 3      # 2 GiB: a limit, so an endless or huge URL cannot exhaust memory


def _download(ctx: Ctx, url: str, headers: dict[str, str], timeout: int, ext: str) -> tuple[Path, int]:
    """Stream a URL to a cached local file, hashing as it goes (no whole-body buffer), with a size cap.
    Returns (local path, byte count)."""
    import atexit
    import shutil
    import tempfile
    import urllib.error
    import urllib.request
    if ctx.cache_dir is not None:
        root = Path(ctx.cache_dir)
    else:                                        # no project cache: a private temp folder, removed at exit
        root = Path(tempfile.mkdtemp(prefix="dancr-url-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
    folder = root / ctx.node_id / ".downloads"
    folder.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=headers or {})
    h = hashlib.sha1()
    size = 0
    tmp = folder / f".{os.getpid()}.{uuid.uuid4().hex[:8]}.download.tmp"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the user named this URL
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"{_short(url)} is larger than {MAX_DOWNLOAD_BYTES // 2**30} GiB. "
                                         "Download it to a file and load that instead")
                    h.update(chunk)
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{_short(url)} answered {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Could not reach {_short(url)}: {e.reason}") from e
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    name = hashlib.sha1(url.encode()).hexdigest()[:16] + h.hexdigest()[:8] + ext
    target = folder / name
    if target.exists():
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, target)
    return target, size


def _head(url: str, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    import urllib.request
    _require_http(url)
    req = urllib.request.Request(url, headers=headers or {}, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                "length": r.headers.get("Content-Length")}


def _require_http(url: str) -> None:
    """Only http(s): urllib would otherwise read file://, ftp:// or data: URLs, letting a project read the local
    filesystem or an arbitrary scheme through what looks like a web request."""
    scheme = url.split(":", 1)[0].strip().lower() if ":" in url else ""
    if scheme not in ("http", "https"):
        raise ValueError(f"Only http and https URLs can be read, not {scheme or url!r}")


def _url_headers(params: dict[str, Any]) -> dict[str, str]:
    return {str(k): expand_env(v) for k, v in (params.get("headers") or {}).items()}


def _url_ext(url: str, fmt: str) -> str:
    if fmt in ("csv", "tsv", "txt", "parquet", "pq", "json", "ndjson"):
        return "." + ("parquet" if fmt == "pq" else fmt)
    path = url.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(path).suffix.lower()
    return suffix if suffix in (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".ndjson", ".dat") else ".csv"


def _url_digest(directory: Path, params: dict[str, Any]) -> Any:
    """What changes when the remote data does: the URL (redacted) plus, when it can be reached, the server's
    ETag/Last-Modified, remembered for a short time so a cache check does not hammer the server."""
    url = expand_env(params.get("url"))
    if not url:
        return None
    headers = _url_headers(params)
    base: dict[str, Any] = {"url": redact(url)}
    if params.get("check_remote", True):
        try:
            # hash the cache key: an Authorization header expanded from ${TOKEN} must not sit in a module global
            memo_key = "head:" + hashlib.sha1((url + repr(sorted(headers.items()))).encode()).hexdigest()
            base.update(_memo(memo_key, 30.0, lambda: _head(url, headers, 15)))
        except Exception:  # noqa: BLE001 - an unreachable URL simply contributes less to the hash
            pass
    return base


def _load_url(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    url = expand_env(params.get("url"))
    if not url:
        raise ValueError("Give a URL to read")
    _require_http(url)
    headers = _url_headers(params)
    fmt = str(params.get("format") or "auto").lower()
    ext = _url_ext(url, fmt)
    local, size = _download(ctx, url, headers, int(params.get("timeout") or 60), ext)
    note = f"Read {size:,} bytes from {_short(url)}"
    # the URL is redacted in the report too: it may carry a token or a password in its query string
    report_url = redact(url)
    if ext in (".parquet", ".pq"):
        return NodeResult(pl.read_parquet(local).lazy(), messages=[note], report={"url": report_url, "bytes": size})
    if ext in (".json", ".ndjson"):
        try:
            df = pl.read_json(local)
        except Exception:  # noqa: BLE001 - an ndjson file
            df = pl.read_ndjson(local)
        return NodeResult(df.lazy(), messages=[note], report={"url": report_url, "bytes": size})
    opts = {k: params.get(k) for k in SHARED if params.get(k) is not None}
    lf, msgs, rep = scan_file(ctx, {**opts, "path": str(local)})
    return NodeResult(lf, report={**rep, "url": report_url, "bytes": size}, messages=[note] + msgs)


registry.register(NodeType(
    key="load_url", label="Load from a URL", category="Get data", icon="file-csv", kind="source", inputs=[],
    description="Read a CSV, text, Parquet or JSON table from a URL over the network (http/https). CSV and text "
                "are read like a local file (layout, dates, numbers).",
    apply=_load_url, source_digest=_url_digest,
    summary=lambda p: _short(str(p.get("url") or "")) or "no URL",
    params=[
        Param("url", "URL", "text", required=True, placeholder="https://…/export.csv"),
        Param("format", "Format", "choice", default="auto",
              choices=[("auto", "detect from the address"), ("csv", "CSV / text"), ("tsv", "TSV"), ("parquet", "Parquet"),
                       ("json", "JSON / JSON Lines")]),
        Param("headers", "Request headers", "mapping", default={}, advanced=True, help="e.g. an Authorization header, with ${ENV_VAR} for a token"),
        Param("check_remote", "Ask the server whether it changed", "bool", default=True, advanced=True,
              help="A HEAD request while deciding whether to run again. Off means the step reruns only when its settings change"),
        Param("timeout", "Seconds to wait", "int", default=60, min=1, advanced=True),
        Param("has_header", "First row is column names", "bool", default=True, advanced=True),
        Param("parse_dates", "Detect dates", "bool", default=True, advanced=True),
        Param("parse_numbers", "Detect numbers written as text", "bool", default=True, advanced=True),
        Param("decimal_comma", "Numbers use decimal comma", "bool", default=False, advanced=True),
        Param("encoding", "Text encoding", "choice", default="utf8", advanced=True,
              choices=[("utf8", "UTF-8 (normal)"), ("latin1", "Latin-1 / Windows")]),
    ],
))


# =================================================================== load_sql
def _is_sqlite(conn: str) -> bool:
    c = conn.strip()
    if c in (":memory:", "") or c.lower().startswith("sqlite:"):
        return True
    first = c.split("://", 1)[0].lower()
    if "://" in c:
        return first in ("sqlite", "sqlite3")
    return Path(c).suffix.lower() in (".db", ".sqlite", ".sqlite3") or Path(c).exists()


def _sqlite_path(conn: str) -> str:
    c = conn.strip()
    for prefix in ("sqlite:///", "sqlite://", "sqlite:"):
        if c.lower().startswith(prefix):
            return c[len(prefix):] or ":memory:"
    return c


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_text(params: dict[str, Any]) -> str:
    query = str(params.get("query") or "").strip()
    if query:
        return query
    table = str(params.get("table") or "").strip()
    if not table:
        raise ValueError("Give a query, or the name of a table to read")
    schema = str(params.get("schema") or "").strip()
    return "SELECT * FROM " + ((_quote(schema) + ".") if schema else "") + _quote(table)


_READ_SQL = re.compile(r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/)|(?:\s))*\b(select|with)\b", re.IGNORECASE | re.DOTALL)


def _require_read_only_sql(sql: str) -> None:
    """A SQLite read must be a SELECT/WITH. A connection is read-only, but seeing the statement here also stops
    side-effecting ones (ATTACH DATABASE, PRAGMA) that Polars' read_database would otherwise execute."""
    if not _READ_SQL.match(sql or ""):
        raise ValueError("Only a SELECT (or WITH … SELECT) query can be read")


def _sqlite_connect(path: str):
    """A read-only SQLite connection (``mode=ro``), so reading a database cannot write to it or create files."""
    import sqlite3
    import pathlib
    uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _connect_read(conn: str, sql: str) -> pl.DataFrame:
    if _is_sqlite(conn):
        _require_read_only_sql(sql)
        con = _sqlite_connect(_sqlite_path(conn))
        try:
            return pl.read_database(sql, con)
        finally:
            con.close()
    try:
        import sqlalchemy  # noqa: F401  (SQLAlchemy is what turns a URI into a connection)
    except ImportError:
        raise ValueError("Reading a database server (PostgreSQL and others) needs SQLAlchemy, which is missing "
                         "from this build. Reinstall DANCR (or, in a source checkout, run 'uv sync')") from None
    return pl.read_database_uri(sql, conn)


def _max_value(conn: str, sql: str) -> Any:
    if _is_sqlite(conn):
        _require_read_only_sql(sql)
        con = _sqlite_connect(_sqlite_path(conn))
        try:
            row = con.execute(sql).fetchone()
            return None if row is None else row[0]
        finally:
            con.close()
    df = pl.read_database_uri(sql, conn)
    return df[0, 0] if df.height else None


def _sql_digest(directory: Path, params: dict[str, Any]) -> Any:
    """SQLite: the file's size/time. A server: the redacted connection and the query (or table), plus — when a
    ``version_column`` is named — the latest value in it, which changes when new rows arrive."""
    conn = expand_env(params.get("connection"))
    if not conn:
        return None
    if _is_sqlite(conn):
        p = Path(_sqlite_path(conn))
        try:
            st = p.stat()
            return {"sqlite": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            return {"sqlite": str(p)}
    base: dict[str, Any] = {"connection": redact(conn), "sql": params.get("query") or params.get("table")}
    vc = str(params.get("version_column") or "").strip()
    table = str(params.get("table") or "").strip()
    if vc and table:
        schema = str(params.get("schema") or "").strip()
        qualified = ((_quote(schema) + ".") if schema else "") + _quote(table)
        sql = f"SELECT max({_quote(vc)}) FROM {qualified}"
        try:
            base["max"] = str(_memo(f"sqlmax:{redact(conn)}:{sql}", 30.0, lambda: _max_value(conn, sql)))
        except Exception:  # noqa: BLE001
            pass
    return base


def _load_sql(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    conn = expand_env(params.get("connection"))
    if not conn:
        raise ValueError("Give a database connection (a SQLite file, or a server URL such as postgresql://…)")
    sql = _sql_text(params)
    df = _connect_read(conn, sql)
    where = "SQLite" if _is_sqlite(conn) else conn.split("://", 1)[0]
    return NodeResult(df.lazy(), messages=[f"Read {df.height:,} row{'s' if df.height != 1 else ''} from {where}"],
                      report={"rows": df.height, "columns": list(df.columns)})


registry.register(NodeType(
    key="load_sql", label="Load from a database", category="Get data", icon="table", kind="source", inputs=[],
    description="Read a query or a whole table from SQLite (built in) or a database server (PostgreSQL and "
                "others, through SQLAlchemy, included). Put credentials in ${ENV_VAR}.",
    apply=_load_sql, source_digest=_sql_digest,
    summary=lambda p: (str(p.get("table") or "query") + " from " + (str(p.get("connection") or "").split("://", 1)[0] or "?")),
    params=[
        Param("connection", "Connection", "text", required=True, secret=True,
              placeholder="sqlite:///data.db  or  ${PG_DSN}",
              help="A SQLite file, or a server URL. Put a password in ${ENV_VAR}, not here"),
        Param("query", "SQL query", "text", default="", placeholder="SELECT … FROM … WHERE …"),
        Param("table", "Or read a whole table", "text", default=""),
        Param("schema", "Schema", "text", default="", advanced=True),
        Param("version_column", "Changes indicator column", "text", default="", advanced=True,
              help="A column (an id or updated-at) whose latest value says whether new rows arrived (server databases)"),
    ],
))
