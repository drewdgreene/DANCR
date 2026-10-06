"""Connectors: reading a table from a URL and from a database, plus secret handling (env expansion, redaction)
and the cache digest that notices when remote or database data changes."""
import json
import sqlite3
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core import secrets
from dancr.core.nodes import connectors


def _run(p, node="src", **kw):
    ex = Executor(p)
    res = ex.run([node], **kw)
    return res[node]


def test_load_sql_sqlite(tmp_path):
    db = tmp_path / "data.db"
    con = sqlite3.connect(db)
    con.execute("create table t (a int, b text)")
    con.executemany("insert into t values (?, ?)", [(1, "x"), (2, "y")])
    con.commit(); con.close()
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_sql", params={"connection": str(db), "query": "select * from t order by a"}, id="src")
    p.save()
    st = _run(p)
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_load_sql_table_mode_and_digest(tmp_path):
    db = tmp_path / "data.db"
    con = sqlite3.connect(db)
    con.execute("create table t (a int)")
    con.execute("insert into t values (1)")
    con.commit(); con.close()
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_sql", params={"connection": str(db), "table": "t"}, id="src")
    ex = Executor(p)
    h1 = ex.plan_hash("src")
    con = sqlite3.connect(db); con.execute("insert into t values (2)"); con.commit(); con.close()
    assert ex.plan_hash("src") != h1            # the SQLite file changed -> rerun


def test_load_url_csv(tmp_path):
    served = tmp_path / "served"; served.mkdir()
    (served / "export.csv").write_text("a,b\n1,x\n2,y\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: SimpleHTTPRequestHandler(*a, directory=str(served), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/export.csv"
        p = Pipeline("p"); p.path = tmp_path / "p.json"
        p.add_node("load_url", params={"url": url}, id="src")
        p.save()
        st = _run(p)
        assert st.status == "done", st.error
        df = pl.read_parquet(st.output)
        assert df.to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    finally:
        srv.shutdown()


def test_load_url_digest_tracks_the_file(tmp_path):
    served = tmp_path / "served"; served.mkdir()
    f = served / "e.csv"; f.write_text("a\n1\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: SimpleHTTPRequestHandler(*a, directory=str(served), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/e.csv"
        p = Pipeline("p"); p.path = tmp_path / "p.json"
        p.add_node("load_url", params={"url": url}, id="src")
        ex = Executor(p)
        connectors._memo.clear()
        h1 = ex.plan_hash("src")
        f.write_text("a\n1\n2\n")
        connectors._memo.clear()                # the short HEAD memo is a cache, not a wrong answer
        assert ex.plan_hash("src") != h1
    finally:
        srv.shutdown()


def test_load_url_bad_host_is_a_plain_error(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_url", params={"url": "http://127.0.0.1:1/none.csv"}, id="src")
    st = _run(p)
    assert st.status == "failed" and "Could not reach" in st.error


# ---------------------------------------------------------------- secrets
def test_expand_env_and_redact(monkeypatch):
    monkeypatch.setenv("MY_DSN", "postgresql://user:secret@host/db")
    assert secrets.expand_env("${MY_DSN}") == "postgresql://user:secret@host/db"
    assert secrets.expand_env("$MISSING") == "$MISSING"
    assert secrets.redact("postgresql://user:secret@host/db") == "postgresql://user:****@host/db"
    assert "secret" not in secrets.redact("https://x/y?password=secret")


def test_connector_secrets_are_redacted_in_describe(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "p.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "load_sql", {"connection": "postgresql://u:pw@h/db", "query": "select 1"}, node_id="src")
    out = json.loads(srv.describe_pipeline(str(pj)))
    params = next(n for n in out["nodes"] if n["id"] == "src")["params"]
    assert params["connection"] == "****"


def test_params_schema_marks_secret(tmp_path):
    from dancr.core import registry
    nt = registry.get("load_sql")
    assert nt.param("connection").secret is True
    assert nt.to_json()["params"]
    assert any(p.get("secret") for p in nt.to_json()["params"])


# ---------------------------------------------------------------- science formats (optional)
def test_load_netcdf(tmp_path):
    xr = pytest.importorskip("xarray")
    import numpy as np
    da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"), name="temp")
    path = tmp_path / "d.nc"
    da.to_dataset().to_netcdf(path)
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_netcdf", params={"path": "d.nc", "variable": "temp"}, id="src")
    p.save()
    st = _run(p)
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.height == 6 and "temp" in df.columns


def test_load_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("values", data=np.arange(5.0))
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_hdf5", params={"path": "d.h5", "dataset": "values"}, id="src")
    p.save()
    st = _run(p)
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.height == 5 and df["value"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------- S8: SQLite reads are read-only
def test_sqlite_read_only_and_side_effects_refused(tmp_path):
    db = tmp_path / "data.db"
    con = sqlite3.connect(db); con.execute("create table t (a int)"); con.execute("insert into t values (1)")
    con.commit(); con.close()
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_sql", params={"connection": str(db), "query": "ATTACH DATABASE '/tmp/dancr-evil.db' AS evil"}, id="src")
    p.save()
    st = _run(p)
    assert st.status == "failed" and "Only a SELECT" in st.error


def test_load_url_refuses_a_non_http_scheme(tmp_path):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_url", params={"url": "file:///etc/passwd"}, id="src")
    p.save()
    st = _run(p)
    assert st.status == "failed" and "http" in st.error.lower()


def test_a_token_in_a_url_is_redacted_everywhere(tmp_path):
    assert secrets.redact("https://api.example.com/export.csv?token=SECRET123") == \
        "https://api.example.com/export.csv?token=****"
    assert connectors._short("https://api.example.com/export.csv?token=SECRET123&x=1") == \
        "https://api.example.com/export.csv?token=****&x=1"


def test_header_tokens_are_redacted_in_settings():
    from dancr.core import registry
    nt = registry.get("load_url")
    out = secrets.redact_params(nt, {"url": "https://x", "headers": {"Authorization": "Bearer SECRET",
                                                                    "X-Api-Key": "KEY"}})
    assert out["headers"] == {"Authorization": "****", "X-Api-Key": "****"}
