"""The window's document: undo, runs, autosave and recovery, and the project file changing under it."""
import json
import subprocess
import sys
import threading

import polars as pl
import pytest
from PySide6.QtWidgets import QApplication

from dancr import headless as hl
from dancr.core import Pipeline, PipelineError
from dancr.ui import document as docmod
from dancr.ui.document import Document
from helpers import settle


@pytest.fixture
def project(tmp_path):
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(tmp_path / "data.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "data.csv"}, id="src")
    p.meta["auto_run"] = False
    p.save()
    return tmp_path / "p.json"


def _edit_elsewhere(pj, fn):
    p = Pipeline.load(pj)
    fn(p)
    p.save()


@pytest.fixture
def doc(app, tmp_path):
    from dancr.ui.document import Document, recovery_path
    d = Document()
    yield d
    d.undo.setClean()
    d.shutdown()
    recovery_path().unlink(missing_ok=True)


@pytest.fixture
def stacked(doc, tmp_path):
    """Three tables feeding one stack step, in the order 1, 2, 3."""
    for i in (1, 2, 3):
        pl.DataFrame({"v": [float(i)]}).write_csv(tmp_path / f"t{i}.csv")
    ids = [doc.add_node("load_file", 0, i * 100, params={"path": str(tmp_path / f"t{i}.csv")}) for i in (1, 2, 3)]
    st = doc.add_node("stack", 300, 100)
    for s in ids:
        doc.connect(s, st, "tables")
    assert doc.pipeline.inputs_of(st)["tables"] == ids
    return ids, st


def _saved(doc, tmp_path, small_csv):
    doc.add_node("load_file", 0, 0, params={"path": str(small_csv)})
    doc.save(tmp_path / "p.json")
    return tmp_path / "p.json"


def test_reload_records_the_text_it_read_not_a_later_one(app, project, monkeypatch):
    doc = Document()
    doc.load(project)
    _edit_elsewhere(project, lambda p: p.rename_node("src", "T1"))
    real = docmod.Pipeline.from_dict
    written = []

    def parse_then_someone_writes(data, path=None):
        out = real(data, path)
        if not written:                                  # another program writes right after the read
            written.append(1)
            _edit_elsewhere(project, lambda p: p.rename_node("src", "T2"))
        return out
    monkeypatch.setattr(docmod.Pipeline, "from_dict", staticmethod(parse_then_someone_writes))
    doc._maybe_reload()
    monkeypatch.setattr(docmod.Pipeline, "from_dict", real)
    assert doc.pipeline.nodes["src"].title == "T1"
    assert doc.changed_elsewhere()                       # T2 is someone else's: a save would ask first
    doc._maybe_reload()                                  # the watcher's next look picks T2 up
    assert doc.pipeline.nodes["src"].title == "T2"
    doc.shutdown()


def test_the_same_project_written_differently_is_not_a_change(app, project):
    doc = Document()
    doc.load(project)
    project.write_text(project.read_text().replace("\n", "\r\n"))
    doc._maybe_reload()
    assert not doc.changed_elsewhere()
    doc.shutdown()


def test_steps_changed_elsewhere_that_write_outside_are_held(app, project, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-leak.csv"
    doc = Document()
    doc.load(project)
    doc.set_auto_run(True)

    def add(p):
        p.add_node("export", params={"path": str(outside)}, id="leak"); p.connect("src", "leak")
        p.add_node("export", params={"path": "data.csv"}, id="over"); p.connect("src", "over")
        p.add_node("export", params={"path": "out.csv"}, id="fine"); p.connect("src", "fine")
    _edit_elsewhere(project, add)
    doc._maybe_reload()
    assert doc.held == {"leak", "over"}
    doc._auto_pending = True
    doc._auto_run_now()
    settle(app, lambda: not (doc._auto_pending or doc.running))
    assert (tmp_path / "out.csv").exists() and not outside.exists()
    assert pl.read_csv(tmp_path / "data.csv")["a"].to_list() == [1, 2, 3]
    doc.run(["leak"])                                    # the person asks for it: it runs
    settle(app, lambda: not doc.running)
    assert outside.exists() and doc.held == {"over"}
    outside.unlink()
    doc.shutdown()



def test_a_held_step_stays_held_through_later_edits_elsewhere_and_revert(app, project, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-leak2.csv"
    doc = Document()
    doc.load(project)
    doc.set_auto_run(True)
    _edit_elsewhere(project, lambda p: (p.add_node("export", params={"path": str(outside)}, id="leak"), p.connect("src", "leak")))
    doc._maybe_reload()
    assert doc.held == {"leak"}
    _edit_elsewhere(project, lambda p: p.rename_node("src", "renamed elsewhere"))     # an unrelated change
    doc._maybe_reload()
    assert doc.held == {"leak"}
    doc.load(project)                                    # File → Revert
    assert doc.held == {"leak"}
    doc._auto_pending = True
    doc._auto_run_now()
    settle(app, lambda: not (doc._auto_pending or doc.running))
    assert not outside.exists()
    doc.shutdown()

def test_workers_get_a_copy_of_the_project(app, project):
    doc = Document()
    doc.load(project)
    ex = doc.snapshot_executor()
    doc.rename("src", "changed")
    assert ex.pipeline is not doc.pipeline and ex.pipeline.nodes["src"].title != "changed"
    assert ex.cache_dir == doc.executor.cache_dir
    doc.shutdown()


def test_saving_waits_for_another_program_holding_the_file(app, project, monkeypatch):
    doc = Document()
    doc.load(project)
    doc.rename("src", "mine")
    hold = subprocess.Popen([sys.executable, "-c", "import sys\nfrom dancr.headless import project_lock\n"
                             "with project_lock(sys.argv[1]):\n    print('held', flush=True); sys.stdin.readline()",
                             str(project)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert hold.stdout.readline().strip() == "held"
    try:
        with pytest.raises(hl.ProjectBusy):
            doc.save()
        doc.autosave_now()                               # quietly kept in the recovery copy instead
        assert docmod.recovery_path().exists()
        assert Pipeline.load(project).nodes["src"].title != "mine"
    finally:
        hold.stdin.write("\n"); hold.stdin.flush(); hold.wait()
    doc.save()
    assert Pipeline.load(project).nodes["src"].title == "mine"
    doc.shutdown()


def test_an_unsaved_project_never_writes_into_the_folder_it_was_started_in(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pl.DataFrame({"a": [1, 2]}).write_csv(tmp_path / "d.csv")
    p = Pipeline("u")
    p.add_node("load_file", params={"path": str(tmp_path / "d.csv")}, id="src")
    p.add_node("export", params={"path": "out.csv"}, id="x"); p.connect("src", "x")
    doc = Document(p)
    doc.run()
    settle(app, lambda: not doc.running)
    assert not (tmp_path / "out.csv").exists()
    assert doc.state("src").status == "done"
    doc.shutdown()


def test_a_group_that_fails_half_way_leaves_nothing(doc):
    doc.add_node("enter_data", 0, 0)
    doc.undo.setClean()
    before = set(doc.pipeline.nodes)
    with pytest.raises(RuntimeError):
        with doc.macro("Two steps"):
            doc.add_node("sort", 0, 0)
            raise RuntimeError("the second part failed")
    assert set(doc.pipeline.nodes) == before and not doc.dirty


def test_a_group_that_changed_nothing_leaves_no_undo_step(doc):
    doc.undo.setClean()
    doc.duplicate_nodes([])
    with doc.macro("Nothing"):
        pass
    assert not doc.dirty and not doc.undo.canUndo()


def test_saving_never_silently_overwrites_another_programs_change(app, tmp_path):
    from dancr.ui.document import Document, ChangedOnDisk
    pj = tmp_path / "p.json"
    Pipeline("p").save(pj)
    doc = Document(); doc.load(pj)
    doc.add_node("enter_data", 0, 0)                   # unsaved edit in the window
    other = Pipeline.load(pj); other.add_node("enter_data", id="agents"); other.save()     # an agent's edit
    with pytest.raises(ChangedOnDisk):
        doc.save()
    assert "agents" in Pipeline.load(pj).nodes
    doc.save(overwrite=True)
    assert "agents" not in Pipeline.load(pj).nodes and doc.versions()      # theirs is kept as an earlier version
    doc.shutdown()


def test_events_of_a_replaced_run_are_ignored(app, tmp_path):
    from dancr.ui.document import Document
    from dancr.core.executor import NodeState
    doc = Document()
    doc.add_node("enter_data", 0, 0, params={"columns": [{"name": "a", "type": "number"}], "rows": [[1]]})
    doc.run()
    old = doc._run
    doc.stop(wait=True)
    doc._run = None                                          # the project was replaced; that run is over
    seen = []
    doc.nodeState.connect(lambda *a: seen.append(a))
    old.event.emit({"type": "node_finished", "state": NodeState("ghost", status="failed")})
    app.processEvents()
    assert seen == [] and "ghost" not in doc._states_cache
    doc.undo.setClean(); doc.shutdown()


def test_moving_a_connection_is_checked_without_copying_the_project(app, tmp_path, monkeypatch):
    from dancr.ui.document import Document
    from dancr.core import Pipeline
    doc = Document()
    a = doc.add_node("enter_data", 0, 0)
    s1 = doc.add_node("sort", 300, 0); s2 = doc.add_node("sort", 300, 100)
    doc.connect(a, s1)
    monkeypatch.setattr(Pipeline, "from_dict", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(AssertionError("copied"))))
    doc.move_edge(doc.pipeline.edges[0], s2, "in")
    doc.set_input("rate", 2)
    doc.set_column_meta("v", "Flow", "L/s")
    assert doc.pipeline.inputs_of(s2) == {"in": [a]}
    monkeypatch.undo()
    doc.undo.setClean(); doc.shutdown()


def test_stale_run_thread_is_ignored(pipe):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from dancr.ui.document import Document
    from dancr.ui.workers import run_gate
    doc = Document(pipe)
    finished = []
    doc.runFinished.connect(lambda ok, r: finished.append(ok))
    doc.run()
    first = doc._run
    doc.stop(wait=True)                                      # settles the run right away
    assert not doc.running and run_gate.is_set() and len(finished) == 1
    doc.run(force=True)
    assert doc.running and doc._run is not first
    doc._on_run_done(first)                                  # the old thread's queued finish must change nothing
    assert doc.running and len(finished) == 1
    doc.stop(wait=True)
    assert len(finished) == 2
    doc.shutdown()


def test_undoing_a_delete_puts_inputs_back_in_order(doc, stacked):
    ids, st = stacked
    doc.remove_nodes([ids[0]])
    assert doc.pipeline.inputs_of(st)["tables"] == ids[1:]
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids
    doc.undo.redo(); doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


def test_undoing_a_multi_delete_puts_inputs_back_in_order(doc, stacked):
    ids, st = stacked
    doc.remove_nodes([ids[2], ids[0]])
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


def test_undoing_a_disconnect_puts_the_input_back_in_place(doc, stacked):
    ids, st = stacked
    edge = next(e for e in doc.pipeline.edges if e.source == ids[1])
    doc.disconnect(edge)
    doc.undo.undo()
    assert doc.pipeline.inputs_of(st)["tables"] == ids


def test_moving_a_connection_onto_a_taken_input_changes_nothing(doc, tmp_path):
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    a = doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    b = doc.add_node("load_file", 0, 100, params={"path": str(tmp_path / "a.csv")})
    s1 = doc.add_node("sort", 300, 0)
    s2 = doc.add_node("sort", 300, 100)
    doc.connect(a, s1); doc.connect(b, s2)
    edges_before = [e.key() for e in doc.pipeline.edges]
    depth = doc.undo.index()
    edge = next(e for e in doc.pipeline.edges if e.source == a)
    with pytest.raises(PipelineError):
        doc.move_edge(edge, s2, "in")                  # s2.in is already fed by b
    assert [e.key() for e in doc.pipeline.edges] == edges_before
    assert doc.undo.index() == depth                   # nothing to undo either
    doc.disconnect(next(e for e in doc.pipeline.edges if e.source == b))
    doc.move_edge(edge, s2, "in")
    assert doc.pipeline.inputs_of(s2) == {"in": [a]} and doc.pipeline.inputs_of(s1) == {}
    doc.undo.undo()                                    # one step back: the move
    assert doc.pipeline.inputs_of(s1) == {"in": [a]} and doc.pipeline.inputs_of(s2) == {}


def test_the_periodic_state_check_runs_off_the_gui_thread_and_drops_stale_answers(doc, tmp_path, app, monkeypatch):
    from dancr.core.executor import Executor
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    doc.save(tmp_path / "p.json")
    threads = []
    orig = Executor.states
    monkeypatch.setattr(Executor, "states", lambda self: (threads.append(threading.current_thread()), orig(self))[1])
    doc._poll_states()
    settle(app, lambda: doc._poll_task is None)
    assert threads and threads[-1] is not threading.main_thread()
    doc._poll_states()
    doc.refresh_states()                                 # an edit refreshed meanwhile: the poll's answer is old
    before = dict(doc._states_cache)
    settle(app, lambda: doc._poll_task is None)
    assert doc._states_cache == before


def test_undo_after_save_as_elsewhere_still_points_at_the_same_files(doc, tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir(); b.mkdir()
    pl.DataFrame({"v": [1.0]}).write_csv(a / "data.csv")
    pl.DataFrame({"v": [1.0]}).write_csv(a / "other.csv")
    doc.save(a / "p.json")
    keep = doc.add_node("load_file", 0, 0, params={"path": "data.csv"})
    gone = doc.add_node("load_file", 0, 100, params={"path": "other.csv"})
    doc.set_params(keep, {"path": "other.csv"})
    doc.remove_nodes([gone])
    doc.save(b / "p.json")
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "other.csv")
    doc.undo.undo()                                     # the delete
    assert doc.pipeline.nodes[gone].params["path"] == str(a / "other.csv")
    doc.undo.undo()                                     # the setting
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "data.csv")
    doc.undo.redo()
    assert doc.pipeline.nodes[keep].params["path"] == str(a / "other.csv")


def test_save_as_leaves_no_lease_behind(doc, tmp_path, app):
    from dancr.core.executor import LIVE_DIR
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    doc.save(tmp_path / "one.json")
    doc.executor.run()
    doc.refresh_states()                                # holds its results
    settle(app, lambda: doc._poll_task is None)
    assert list((doc.executor.cache_dir / LIVE_DIR).glob("*.json"))
    doc.save(tmp_path / "two.json")
    doc.executor.release()
    assert not list((doc.executor.cache_dir / LIVE_DIR).glob("*.json"))


def test_autosave_never_writes_while_a_dialog_about_the_edits_is_open(doc, tmp_path, small_csv, monkeypatch):
    from dancr.ui.document import recovery_path
    doc.set_autosave(True)
    path = _saved(doc, tmp_path, small_csv)
    before = path.read_text()
    doc.add_node("sort", 0, 0)
    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: object()))
    doc.autosave_now()
    assert path.read_text() == before                      # "Discard changes?" is still on screen
    data = json.loads(recovery_path().read_text())         # but a crash now would lose nothing
    assert data["path"] == str(path) and len(data["pipeline"]["nodes"]) == 2
    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: None))
    doc.autosave_now()
    assert len(json.loads(path.read_text())["nodes"]) == 2
    assert not recovery_path().exists()                    # saved: nothing left to recover


def test_autosave_is_off_until_turned_on(doc, tmp_path, small_csv):
    from dancr.ui.document import recovery_path
    assert doc.autosave is False
    path = _saved(doc, tmp_path, small_csv)
    before = path.read_text()
    doc.add_node("sort", 0, 0)
    doc.autosave_now()
    assert path.read_text() == before                      # the file changes only when the person saves
    assert len(json.loads(recovery_path().read_text())["pipeline"]["nodes"]) == 2     # a crash still loses nothing
    doc.set_autosave(True)
    doc.autosave_now()
    assert len(json.loads(path.read_text())["nodes"]) == 2


def test_recovered_edits_stay_protected_until_saved(doc, tmp_path, small_csv):
    from dancr.ui.document import Document, recovery_path
    doc.add_node("load_file", 0, 0, params={"path": str(small_csv)})
    doc.write_recovery()
    dead = recovery_path(2 ** 22 + 7)
    recovery_path().replace(dead)
    pipe, rp = Document.pending_recovery()
    doc.recover(pipe, rp)
    assert doc.autosave_paused and doc.dirty
    doc.add_node("sort", 0, 0)
    doc.autosave_now()                                     # paused: the recovery copy follows the edits
    assert len(json.loads(recovery_path().read_text())["pipeline"]["nodes"]) == 2


def test_unsaved_changes_to_a_saved_project_survive_a_forced_quit(doc, tmp_path, small_csv):
    from dancr.ui.document import Document, recovery_path
    path = _saved(doc, tmp_path, small_csv)
    doc.add_node("sort", 0, 0)
    doc.write_recovery()                                   # what SIGTERM does
    recovery_path().replace(recovery_path(2 ** 22 + 9))
    pipe, rp = Document.pending_recovery()
    assert pipe.path == path and len(pipe.nodes) == 2
    assert len(json.loads(path.read_text())["nodes"]) == 1  # the file itself was not touched
    rp.unlink()


def test_old_style_recovery_files_are_discarded(doc):
    from dancr.ui.document import Document, recovery_path
    stale = recovery_path(2 ** 22 + 11)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps(Pipeline("x").to_dict()))
    assert Document.pending_recovery() is None and not stale.exists()


def test_stopping_a_run_mid_step_never_freezes_the_window(doc, app, monkeypatch):
    # a step writing its result cannot be interrupted: waiting for it must keep the window's events flowing
    from PySide6.QtCore import QTimer
    from dancr.core.executor import Executor
    from dancr.ui.workers import run_gate
    writing, release = threading.Event(), threading.Event()

    def long_step(self, targets=None, on_event=None, cancel=None, force=False):
        writing.set()
        release.wait(20)                                    # ignores cancel, as sink_parquet does
        return {}
    monkeypatch.setattr(Executor, "run", long_step)
    doc.add_node("enter_data", 0, 0)
    doc.run()
    settle(app, writing.is_set)
    busy, ticks = [], []
    doc.busy.connect(busy.append)
    tick = QTimer(); tick.setInterval(20); tick.timeout.connect(lambda: ticks.append(1)); tick.start()
    QTimer.singleShot(300, release.set)                     # only an event loop that keeps running gets here
    threading.Timer(15, release.set).start()                # never hang the suite if it does freeze
    doc.stop(wait=True)
    tick.stop()
    assert release.is_set() and len(ticks) >= 5
    assert busy == ["Stopping the current step…", None]
    assert not doc.running and run_gate.is_set()


def test_a_stopped_step_is_not_left_marked_running(doc, app):
    from dancr.core.executor import NodeState
    nid = doc.add_node("enter_data", 0, 0)
    doc.run()
    t = doc._run
    doc._states_cache[nid] = NodeState(nid, status="running")   # it was being computed when the run stopped
    doc.stop(wait=True)
    assert t is not None and doc.state(nid).status != "running"


def test_refreshing_states_reads_off_the_gui_thread_once_per_burst(doc, tmp_path, app, monkeypatch):
    from dancr.core.executor import Executor
    pl.DataFrame({"v": [1.0]}).write_csv(tmp_path / "a.csv")
    nid = doc.add_node("load_file", 0, 0, params={"path": str(tmp_path / "a.csv")})
    settle(app, lambda: doc._poll_task is None)
    threads = []
    orig = Executor.states
    monkeypatch.setattr(Executor, "states", lambda self: (threads.append(threading.current_thread()), orig(self))[1])
    for _ in range(5):
        doc.refresh_states()                                # a burst of edits
    assert threads == [] or threads[0] is not threading.main_thread()
    settle(app, lambda: doc._poll_task is None and not doc._states_again)
    assert 1 <= len(threads) <= 2 and threading.main_thread() not in threads
    assert doc.state(nid).status in ("idle", "stale")


def test_a_project_file_deleted_then_written_again_later_is_still_watched(app, project):
    import time
    doc = Document()
    doc.load(project)
    text = project.read_text()
    project.unlink()
    settle(app, lambda: str(project.parent) in doc._watcher.directories())
    t = time.time()
    while time.time() - t < 0.6:                            # longer than the one retry the watcher used to make
        app.processEvents()
    data = json.loads(text)
    data["nodes"].append({"id": "later", "type": "sort", "title": "Later", "params": {}, "x": 0, "y": 0})
    project.write_text(json.dumps(data), encoding="utf-8")
    settle(app, lambda: "later" in doc.pipeline.nodes)
    assert str(project) in doc._watcher.files()
    doc.shutdown()


def test_a_data_file_missing_at_open_is_noticed_when_it_appears(app, project, monkeypatch):
    monkeypatch.setattr(docmod, "SOURCE_SETTLE_MS", 50)
    (project.parent / "data.csv").unlink()
    doc = Document()
    doc.load(project)
    assert str(project.parent) in doc._src_watcher.directories()
    got = []
    doc.message.connect(got.append)
    pl.DataFrame({"a": [1, 2, 3]}).write_csv(project.parent / "data.csv")
    settle(app, lambda: "A data file changed on disk" in got)
    assert str(project.parent / "data.csv") in doc._src_watcher.files()
    doc.shutdown()
