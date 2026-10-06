"""The repository event log and watcher (A2): typed events, ripple invalidation, opt-in rerun."""
import json
import threading
import time
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.events import EventLog, KINDS, make_event
from dancr.headless import append_event, build_graph, invalidate, read_events, watch_repo


def test_event_schema_is_closed_and_ordered():
    assert make_event("source_changed", project="p.json")["kind"] == "dancr.event"
    with pytest.raises(ValueError, match="Unknown event kind"):
        make_event("nonsense")


def test_event_log_append_read_since_and_limit(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    assert log.append("source_changed", project="p.json")["seq"] == 1
    assert log.append("graph_updated", root=".")["seq"] == 2
    assert log.append("source_changed", project="q.json")["seq"] == 3
    assert [e["seq"] for e in log.read()] == [1, 2, 3]
    assert [e["seq"] for e in log.read(since=1)] == [2, 3]
    assert [e["seq"] for e in log.read(type="source_changed")] == [1, 3]
    assert [e["seq"] for e in log.read(limit=1)] == [3]


def make_repo(root: Path) -> None:
    (root / "a").mkdir()
    (root / "b").mkdir()
    pl.DataFrame({"id": [1, 2, 3], "region": ["N", "S", "N"]}).write_csv(root / "a" / "customers.csv")
    pl.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, 1]}).write_csv(root / "a" / "orders.csv")
    pl.DataFrame({"customer_id": [1, 2, 3], "email": ["a", "b", "c"]}).write_csv(root / "b" / "contacts.csv")
    pl.DataFrame({"sale_id": [1, 2, 3], "customer_id": [1, 2, 3]}).write_csv(root / "b" / "sales.csv")
    for folder, name, nodes in (("a", "a.json", [("customers", "customers.csv"), ("orders", "orders.csv")]),
                                ("b", "b.json", [("contacts", "contacts.csv"), ("sales", "sales.csv")])):
        p = Pipeline(name[:-5])
        p.path = root / folder / name
        for nid, f in nodes:
            p.add_node("load_file", nid.title(), {"path": f}, id=nid)
        p.save()


def test_invalidate_ripples_across_projects(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    events = invalidate(tmp_path, [str(tmp_path / "a" / "a.json")], files=["customers.csv"])
    types = [e["type"] for e in events]
    assert types[0] == "source_changed"
    invalidated = {e["dataset"] for e in events if e["type"] == "dataset_invalidated"}
    # the changed project's datasets and the cross-project dependents (b) are invalidated
    assert {"a/a.json#customers", "a/a.json#orders"} <= invalidated
    assert any(d.startswith("b/b.json#") for d in invalidated)


def test_append_and_read_events_roundtrip(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    append_event(tmp_path, "graph_updated", counts={"projects": 1})
    out = read_events(tmp_path)
    assert out["count"] >= 1 and out["events"][-1]["type"] == "graph_updated"
    assert read_events(tmp_path, type="source_changed")["count"] == 0


def test_watch_repo_records_and_reruns_on_a_change(tmp_path):
    make_repo(tmp_path)
    build_graph(tmp_path)
    stop = threading.Event()
    done = threading.Event()

    def on_event(e):
        if e.get("type") == "project_recomputed":
            done.set()

    t = threading.Thread(target=watch_repo, kwargs=dict(root=tmp_path, interval=0.2, rerun=True,
                                                        on_event=on_event, stop=stop))
    t.start()
    try:
        time.sleep(0.6)                                   # let the first snapshot settle
        pl.DataFrame({"id": [1, 2, 3, 4], "region": ["N", "S", "N", "S"]}).write_csv(tmp_path / "a" / "customers.csv")
        assert done.wait(15), "the changed project was not recomputed"
    finally:
        stop.set()
        t.join(15)
    events = read_events(tmp_path)["events"]
    types = [e["type"] for e in events]
    assert "source_changed" in types and "dataset_invalidated" in types and "project_recomputed" in types


def test_mcp_get_events(tmp_path, mcp_root):
    import dancr.mcp_server as srv
    pj = tmp_path / "shop.json"
    srv.create_pipeline(str(pj))
    srv.add_node(str(pj), "enter_data", {"columns": [{"name": "x", "type": "number"}], "rows": [[1]]}, node_id="d")
    srv.graph_build()
    out = json.loads(srv.get_events())
    assert out["count"] >= 1 and out["events"][-1]["type"] == "graph_updated"


def test_cli_events(tmp_path):
    import subprocess
    import sys
    make_repo(tmp_path)
    build_graph(tmp_path)
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "events", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "graph_updated" in r.stdout
