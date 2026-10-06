"""The repository-level store: layout, atomic writes, append-only logs and locks."""
import threading

import pytest

from dancr.core.repo import Repo, append_jsonl, read_json, read_jsonl, write_json_atomic, write_sqlite_atomic


def test_repo_layout(tmp_path):
    r = Repo(tmp_path)
    assert r.home == tmp_path / ".dancr"
    assert r.graph_db().name == "graph.db" and r.graph_db().parent.name == "graph"
    assert r.events_log().parent.name == "events"
    assert r.audit_log().parent.name == "audit"
    assert r.lock_file("repo").parent.name == "locks"
    r.ensure()
    assert r.graph_db().parent.is_dir() and r.events_log().parent.is_dir() and r.lock_file("repo").parent.is_dir()


def test_repo_refuses_names_that_escape_the_store(tmp_path):
    r = Repo(tmp_path)
    for bad in ("..", "../x", "a/b", "a\\b", "", "."):
        with pytest.raises(ValueError):
            r.path(bad)


def test_write_sqlite_atomic_builds_reads_and_replaces(tmp_path):
    import sqlite3
    db = tmp_path / ".dancr" / "graph" / "graph.db"

    def build(conn):
        conn.execute("create table t(a text)")
        conn.execute("insert into t values ('x')")

    write_sqlite_atomic(db, build)
    conn = sqlite3.connect(db)
    assert conn.execute("select a from t").fetchall() == [("x",)]
    conn.close()

    def rebuild(conn):
        conn.execute("create table t(a text)")
        conn.execute("insert into t values ('y')")

    write_sqlite_atomic(db, rebuild)
    conn = sqlite3.connect(db)
    assert conn.execute("select a from t").fetchall() == [("y",)]
    conn.close()
    assert not list(db.parent.glob(".*.tmp")), "no temp file is left behind"


def test_append_and_read_jsonl_skips_damaged_lines(tmp_path):
    log = tmp_path / "events.jsonl"
    append_jsonl(log, {"seq": 1})
    append_jsonl(log, {"seq": 2})
    with open(log, "a") as f:
        f.write("not json\n\n")
    assert [r["seq"] for r in read_jsonl(log)] == [1, 2]
    assert list(read_jsonl(tmp_path / "none.jsonl")) == []


def test_write_and_read_json_atomic(tmp_path):
    f = tmp_path / "meta.json"
    write_json_atomic(f, {"a": 1})
    assert read_json(f) == {"a": 1}
    assert read_json(tmp_path / "missing.json") is None
    (tmp_path / "broken.json").write_text("{not json")
    assert read_json(tmp_path / "broken.json") is None


def test_repo_lock_is_reentrant(tmp_path):
    from dancr.headless import repo_lock
    with repo_lock(tmp_path):
        with repo_lock(tmp_path):
            pass


def test_project_locks_are_reentrant_and_order_independent(tmp_path):
    from dancr.headless import project_locks
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    with project_locks([p2, p1]):
        with project_locks([p1, p2]):
            pass


def test_project_locks_acquire_in_sorted_order_so_two_callers_do_not_deadlock(tmp_path):
    """Thread A holds both locks (as [a, b]); thread B asks for them the other way ([b, a]) and must be refused
    rather than deadlock (the sorted acquisition means B never holds one A wants while waiting for the other)."""
    from dancr.headless import ProjectBusy, project_locks
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with project_locks([p1, p2]):
            holding.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert holding.wait(5)
        with pytest.raises(ProjectBusy):
            with project_locks([p2, p1], wait=0.2):
                pass
    finally:
        release.set()
        t.join(5)
