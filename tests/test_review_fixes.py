"""Regression tests for the review fixes: the window's cache hold during a run, thread-safe provider
cancellation, atomic context export, and a race-free row count in the table pager."""
import json
import subprocess
import sys
import threading
from pathlib import Path

import polars as pl
import pytest

from dancr import headless as hl
from dancr.core import Pipeline
from dancr.views.table import TablePager


# ---------------------------------------------------------------- 1. cache hold is not clobbered mid-run
def test_apply_states_does_not_rewrite_the_hold_while_running(tmp_path, monkeypatch):
    """A states poll taken before a run must not replace the window's cache lease while the run is in flight
    (its hashes are stale, and overwriting would let another process's sweep delete results the run is using)."""
    from PySide6.QtWidgets import QApplication
    import dancr.ui.document as doc_mod

    doc = doc_mod.Document()
    doc.pipeline.add_node("enter_data", id="d", params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    doc.pipeline.path = tmp_path / "p.json"

    calls = []
    monkeypatch.setattr(doc.executor, "hold", lambda hashes: calls.append(dict(hashes)))

    # simulate a run in flight, then a stale background read arriving
    doc._run = object()          # any non-None thread stand-in
    doc._run_settled = False
    assert doc.running is True
    from dancr.core.executor import NodeState
    doc._apply_states({"d": NodeState("d", status="done", hash="stale")})
    assert calls == []           # nothing was held while running

    # once the run settles, the next apply records the hold again
    doc._run = None
    doc._run_settled = True
    doc._apply_states({"d": NodeState("d", status="done", hash="fresh")})
    assert calls == [{"d": "fresh"}]


# ---------------------------------------------------------------- 2. provider cancellation is thread-safe
def test_provider_cancel_sets_a_thread_event():
    from dancr.core.assistant.client import FakeProvider, ProviderError, ModelSettings

    p = FakeProvider(["hi"])
    assert p.cancelled is False
    p.cancel()
    assert p.cancelled is True
    with pytest.raises(ProviderError, match="stopped"):
        p.chat([], [], ModelSettings(api_key="k"))


def test_provider_cancel_seen_from_another_thread():
    from dancr.core.assistant.client import FakeProvider, ProviderError, ModelSettings

    p = FakeProvider(["a"])
    seen = {}

    def worker():
        try:
            p.chat([], [], ModelSettings(api_key="k"))
            seen["raised"] = False
        except ProviderError:
            seen["raised"] = True

    p.cancel()                                   # set before the thread starts, as the GUI thread would
    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert seen["raised"] is True


# ---------------------------------------------------------------- 3. atomic context export
def test_write_text_atomic_replaces_and_leaves_no_temp(tmp_path):
    target = tmp_path / "kb.jsonl"
    target.write_text("old\n", encoding="utf-8")
    out = hl.write_text_atomic(target, "new\n")
    assert out == target and target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob("*.tmp")) == [] and list(tmp_path.glob(".*.tmp")) == []


def test_cli_context_output_is_written_atomically(project_factory, tmp_path):
    pj = project_factory()
    out = tmp_path / "kb.jsonl"
    r = subprocess.run([sys.executable, "-m", "dancr.cli", "context", str(pj), "--jsonl", "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.read_text(encoding="utf-8").strip()
    assert list(tmp_path.glob(".*.tmp")) == [] and list(tmp_path.glob("*.tmp")) == []


@pytest.fixture
def project_factory(tmp_path):
    def make():
        pl.DataFrame({"region": ["N", "S"], "amount": [1.0, 2.0]}).write_csv(tmp_path / "o.csv")
        p = Pipeline("shop"); p.path = tmp_path / "shop.json"
        p.add_node("load_file", params={"path": "o.csv"}, id="s")
        p.save()
        return tmp_path / "shop.json"
    return make


# ---------------------------------------------------------------- 4. TablePager.rows counted once
def test_pager_rows_counted_once_under_concurrency():
    lf = pl.DataFrame({"a": list(range(1000))}).lazy()
    pager = TablePager(lf, rows=None)
    results = []
    errors = []

    def worker():
        try:
            results.append(pager.rows)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [1000] * 16


def test_pager_rows_shares_one_count(monkeypatch):
    """The streaming count must run once however many callers ask (it is over the whole table)."""
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    pager = TablePager(lf, rows=None)
    calls = {"n": 0}
    real = pl.LazyFrame.select

    def counting_select(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(pl.LazyFrame, "select", counting_select)
    assert pager.rows == 3
    assert pager.rows == 3
    assert pager.rows == 3
    assert calls["n"] == 1                       # counted once, then cached


def test_pager_rows_passed_in_is_never_counted(monkeypatch):
    """A pager told its row count (the GUI passes one for on-disk results) must not touch the table at all."""
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    pager = TablePager(lf, rows=3)

    def boom(*a, **k):
        raise AssertionError("should not scan for the count")

    monkeypatch.setattr(pl.LazyFrame, "select", boom)
    assert pager.rows == 3
