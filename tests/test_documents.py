"""Load document: reading PDF/Office/EPUB/HTML as tables through MinerU (and without it, from a result folder)."""
import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.nodes import document as doc


MIDDLE = {
    "schema": "docvortex.middle", "schema_version": "2.0",
    "pages": [{"page_idx": 0, "blocks": [
        {"type": "text", "index": 0, "bbox": [10, 10, 200, 30], "content": [{"type": "text", "content": "Hello world"}]},
        {"type": "title", "index": 1, "content": [{"type": "text", "content": "Introduction"}]},
        {"type": "table", "index": 2, "table_body":
            "<table><tr><th>gene</th><th>yield</th></tr><tr><td>A</td><td>65</td></tr></table>"},
    ]}],
}
CONTENT_LIST = [
    {"type": "text", "text": "Line one", "page_idx": 0},
    {"type": "table", "table_body": "<table><tr><td>x</td></tr><tr><td>y</td></tr></table>", "page_idx": 0},
    {"type": "text", "text": "Page two", "page_idx": 1},
]


def run(tmp_path: Path, params: dict) -> tuple:
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_document", params=params, id="n")
    ex = Executor(p)
    st = ex.run(targets=["n"])["n"]
    return st, ex


def rows(ex, node="n") -> list:
    return ex.frame(node).collect().to_dicts()


# ---------------------------------------------------------------- output engine (no MinerU)
def test_output_blocks_from_middle_json(tmp_path):
    (tmp_path / "middle_json.json").write_text(json.dumps(MIDDLE))
    st, ex = run(tmp_path, {"path": "middle_json.json", "what": "blocks", "engine": "output"})
    assert st.status == "done", st.error
    got = rows(ex)
    assert [r["type"] for r in got] == ["text", "heading", "table"]
    assert got[0]["text"] == "Hello world" and got[0]["page"] == 1
    assert got[0]["locator"] == "middle_json.json · page 1 · block 0"
    assert "10.0, 10.0" in got[0]["bbox"]


def test_output_blocks_from_content_list(tmp_path):
    (tmp_path / "content_list.json").write_text(json.dumps(CONTENT_LIST))
    st, ex = run(tmp_path, {"path": "content_list.json", "what": "blocks", "engine": "output"})
    got = rows(ex)
    assert [r["type"] for r in got] == ["text", "table", "text"]
    assert got[2]["page"] == 2 and got[2]["text"] == "Page two"


def test_output_blocks_from_markdown(tmp_path):
    (tmp_path / "markdown.md").write_text("# Title\n\nA paragraph.\n\nAnother.\n")
    st, ex = run(tmp_path, {"path": "markdown.md", "what": "blocks", "engine": "output"})
    got = rows(ex)
    assert [r["type"] for r in got] == ["heading", "text", "text"]
    assert got[0]["text"] == "Title"


def test_output_tables_writes_csv(tmp_path):
    (tmp_path / "middle_json.json").write_text(json.dumps(MIDDLE))
    st, ex = run(tmp_path, {"path": "middle_json.json", "what": "tables", "engine": "output"})
    assert st.status == "done", st.error
    got = rows(ex)
    assert got[0]["n_rows"] == 2 and got[0]["n_cols"] == 2
    csv_path = tmp_path / got[0]["csv"]
    assert csv_path.is_file()
    assert csv_path.read_text().splitlines()[0] == "gene,yield"
    assert any(Path(str(f)).name == "table_001.csv" for f in st.files)


def test_out_dir_cannot_escape_the_project_folder(tmp_path):
    (tmp_path / "middle_json.json").write_text(json.dumps(MIDDLE))
    st, _ex = run(tmp_path, {"path": "middle_json.json", "what": "tables", "engine": "output",
                             "out_dir": "../escape"})
    assert st.status == "failed" and "relative path" in (st.error or "")
    st2, _ex = run(tmp_path, {"path": "middle_json.json", "what": "tables", "engine": "output",
                              "out_dir": str(tmp_path / "abs")})
    assert st2.status == "failed" and "relative path" in (st2.error or "")


def test_output_include_filter(tmp_path):
    (tmp_path / "middle_json.json").write_text(json.dumps(MIDDLE))
    _, ex = run(tmp_path, {"path": "middle_json.json", "what": "blocks", "engine": "output",
                           "include": ["table", "heading"]})
    assert set(r["type"] for r in rows(ex)) == {"table", "heading"}


def test_output_bad_result_is_a_plain_error(tmp_path):
    (tmp_path / "middle_json.json").write_text('{"hello": "world"}')
    st, _ = run(tmp_path, {"path": "middle_json.json", "engine": "output"})
    assert st.status == "failed" and "MinerU result" in (st.error or "")


# ---------------------------------------------------------------- command engine (fake CLI)
FAKE = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
a = sys.argv[1:]
if a[:1] == ["version"]:
    print(json.dumps({"version": "9.9.9"})); sys.exit(0)
out = a[a.index("-o") + 1] if "-o" in a else "."
p = Path(out); p.mkdir(parents=True, exist_ok=True)
(p / "middle_json.json").write_text(json.dumps({
  "pages": [{"page_idx": 0, "blocks": [
    {"type": "text", "index": 0, "content": [{"type": "text", "content": "Scanned text"}]},
    {"type": "table", "index": 1, "table_body": "<table><tr><th>g</th></tr><tr><td>A</td></tr></table>"}]}]}))
'''


def fake_cli(tmp_path: Path) -> Path:
    exe = tmp_path / "mineru-kit"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_command_blocks(tmp_path):
    exe = fake_cli(tmp_path)
    (tmp_path / "report.pdf").write_bytes(b"%PDF x")
    st, ex = run(tmp_path, {"path": "report.pdf", "what": "blocks", "engine": "command", "mineru_cmd": str(exe)})
    assert st.status == "done", st.error
    assert rows(ex)[0]["text"] == "Scanned text"
    assert st.report["engine"] == "command"


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_command_tables(tmp_path):
    exe = fake_cli(tmp_path)
    (tmp_path / "report.pdf").write_bytes(b"%PDF x")
    st, ex = run(tmp_path, {"path": "report.pdf", "what": "tables", "engine": "command", "mineru_cmd": str(exe)})
    got = rows(ex)
    assert got and got[0]["n_rows"] == 2
    assert (tmp_path / got[0]["csv"]).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_digest_tracks_tier_and_version(tmp_path):
    exe = fake_cli(tmp_path)
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_document", params={"path": "r.pdf", "engine": "command", "mineru_cmd": str(exe),
                                        "tier": "flash"}, id="n")
    h1 = Executor(p).plan_hash("n")
    p.set_params("n", tier="standard")
    assert Executor(p).plan_hash("n") != h1


def test_missing_command_is_a_plain_error(tmp_path):
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    st, _ = run(tmp_path, {"path": "r.pdf", "engine": "command", "mineru_cmd": str(tmp_path / "nope")})
    assert st.status == "failed" and "MinerU command" in (st.error or "")


def test_no_tool_names_how_to_install(tmp_path, monkeypatch):
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(doc, "mineru_tool", lambda params=None: None)
    st, _ = run(tmp_path, {"path": "r.pdf", "engine": "command"})
    assert st.status == "failed"
    assert "MinerU" in (st.error or "") and "install" in (st.error or "").lower()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_env_var_supplies_the_command(tmp_path, monkeypatch):
    exe = fake_cli(tmp_path)
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("DANCR_MINERU_CMD", str(exe))
    st, ex = run(tmp_path, {"path": "r.pdf", "engine": "command"})
    assert st.status == "done" and rows(ex)[0]["text"] == "Scanned text"


# ---------------------------------------------------------------- endpoint engine (mock server)
def test_endpoint_needs_consent_and_reads(tmp_path):
    body = json.dumps({"pages": [{"page_idx": 0, "blocks": [
        {"type": "text", "index": 0, "content": [{"type": "text", "content": "From endpoint"}]}]}]}).encode()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        (tmp_path / "d.pdf").write_bytes(b"%PDF")
        endpoint = f"http://127.0.0.1:{srv.server_port}"
        st, _ = run(tmp_path, {"path": "d.pdf", "engine": "endpoint", "endpoint": endpoint})
        assert st.status == "failed" and "Allow upload" in (st.error or "")
        st, ex = run(tmp_path, {"path": "d.pdf", "engine": "endpoint", "endpoint": endpoint, "allow_remote": True})
        assert st.status == "done" and rows(ex)[0]["text"] == "From endpoint"
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- guard / routing / inspect / mcp
def test_guard_points_at_the_document_step(tmp_path):
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "x.pdf"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed" and "Load document" in (st.error or "")


def test_guard_for_an_image_mentions_ocr(tmp_path):
    (tmp_path / "x.png").write_bytes(b"\x89PNG")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "x.png"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed" and "Load document" in (st.error or "")


def test_add_files_routes_documents(tmp_path):
    import dancr.headless as hl
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.docx").write_bytes(b"PK")
    (tmp_path / "c.csv").write_text("a\n1\n")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    ids = hl.add_files(p, [str(tmp_path / f) for f in ("a.pdf", "b.docx", "c.csv")])
    assert [p.nodes[i].type for i in ids] == ["load_document", "load_document", "load_file"]


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_inspect_and_read_document(tmp_path, monkeypatch):
    exe = fake_cli(tmp_path)
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("DANCR_MINERU_CMD", str(exe))
    import dancr.headless as hl
    out = hl.inspect_file(tmp_path / "r.pdf", rows=2)
    assert "text" in [c["name"] for c in out["columns"]]
    rd = hl.read_document(tmp_path / "r.pdf", rows=5)
    assert rd["rows"] and rd["rows"][0]["text"] == "Scanned text"


@pytest.mark.skipif(sys.platform == "win32", reason="the fake MinerU CLI is a shebang script Windows cannot execute")
def test_mcp_read_document(tmp_path, monkeypatch):
    exe = fake_cli(tmp_path)
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("DANCR_MINERU_CMD", str(exe))
    import dancr.mcp_server as srv
    out = json.loads(srv.read_document(str(tmp_path / "r.pdf")))
    assert out["rows"][0]["text"] == "Scanned text"


def test_doc_node_for_and_documents(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.csv").write_text("a\n1\n")
    assert doc.doc_node_for("a.pdf") == "load_document"
    assert doc.doc_node_for("b.csv") is None
    found = doc.documents(tmp_path, {"path": str(tmp_path)})
    assert [f.name for f in found] == ["a.pdf"]


# ---------------------------------------------------------------- bundled MinerU detection
def test_bundled_mineru_is_detected(tmp_path, monkeypatch):
    home = tmp_path / "bundle"
    (home / "bin").mkdir(parents=True)
    exe = home / "bin" / "mineru-kit"
    exe.write_text("#!/bin/sh\necho x\n")
    exe.chmod(0o755)
    monkeypatch.setenv("DANCR_MINERU_HOME", str(home))
    monkeypatch.delenv("DANCR_MINERU_CMD", raising=False)
    monkeypatch.setattr(doc.shutil, "which", lambda name: None)
    assert doc.mineru_tool({}) == str(exe)


def test_mineru_home_detects_bundled_models(tmp_path, monkeypatch):
    home = tmp_path / "bundle"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DANCR_MINERU_HOME", str(home))
    assert doc.mineru_home() == str(home)


def test_bundled_mineru_windows_layout(tmp_path, monkeypatch):
    home = tmp_path / "bundle"
    (home / "Scripts").mkdir(parents=True)
    exe = home / "Scripts" / "mineru-kit.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("DANCR_MINERU_HOME", str(home))
    monkeypatch.delenv("DANCR_MINERU_CMD", raising=False)
    monkeypatch.setattr(doc.shutil, "which", lambda name: None)
    assert doc.mineru_tool({}) == str(exe)


def test_mineru_command_param_and_env_win(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_MINERU_CMD", "/env/mineru-kit")
    assert doc.mineru_tool({}) == "/env/mineru-kit"
    assert doc.mineru_tool({"mineru_cmd": "/p/mineru"}) == "/p/mineru"


def test_subprocess_env_sets_mineru_home(tmp_path, monkeypatch):
    home = tmp_path / "bundle"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DANCR_MINERU_HOME", str(home))
    monkeypatch.delenv("MINERU_HOME", raising=False)
    assert doc._subprocess_env().get("MINERU_HOME") == str(home)
