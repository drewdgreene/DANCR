"""Projects in the window: the start page, opening, templates, saving, versions, recovery and changes made
elsewhere."""
import json
import os
import time

import pytest
from PySide6.QtCore import QPointF

from dancr.core import Pipeline
from dancr.core.samples import write_sample, EXAMPLES, example
from helpers import pump, wait_run


def test_file_drop_creates_loader(opened_window, app, probe_dir):
    opened_window._file_dropped(str(probe_dir / "probe_B.csv"), QPointF(10, 300))
    loaders = [n for n in opened_window.doc.pipeline.nodes.values() if n.type == "load_file"]
    assert len(loaders) == 2
    import time
    t = time.time()
    while opened_window.doc.running and time.time() - t < 30:
        app.processEvents()


def test_external_edit_reloads(opened_window, app, tmp_path):
    p = Pipeline.load(opened_window.doc.path)
    p.add_node("summarize", id="ext", x=0, y=500)
    p.save()
    import time
    t = time.time()
    while "ext" not in opened_window.scene.nodes and time.time() - t < 5:
        app.processEvents()
    assert "ext" in opened_window.scene.nodes


def test_save_as_keeps_cache(opened_window, app, tmp_path):
    opened_window.run()
    import time
    t = time.time()
    while opened_window.doc.running and time.time() - t < 30:
        app.processEvents()
    assert opened_window.doc.state("tb").status == "done"
    opened_window.doc.save(tmp_path / "renamed.json")
    assert opened_window.doc.state("tb").status == "done"


def test_start_page_then_table(window, app, sample):
    assert window.pages.currentWidget() is window.start
    assert not window.rail.isVisible() and not window.inspector.isVisible()      # nothing to list or set yet
    window._add_load_node(str(sample), None)
    pump(app, 100)
    assert window.pages.currentWidget() is window.table
    assert window.rail.isVisible() and window.inspector.isVisible()
    assert window.doc.auto_run                     # small file: runs by itself
    wait_run(window, app)
    nid = window.current_table()
    assert window.doc.state(nid).status == "done"
    pump(app, 300)
    assert window.table.grid.model.rowCount() == window.doc.state(nid).rows
    assert not window.a_run.isVisible()            # auto-run hides the Run button


def test_start_template_in_window(window, app, tmp_path, monkeypatch):
    monkeypatch.setattr("dancr.ui.mainwindow.write_sample", lambda base: write_sample(tmp_path, rows=5000))
    window._start_template("limits")
    assert window.pages.currentWidget() is window.table
    assert [i.name for i in window.doc.pipeline.inputs] == ["upper limit"]
    wait_run(window, app)
    assert all(s.status == "done" for s in window.doc.executor.states().values())


def test_a_template_report_is_saved_next_to_the_project(window, app, tmp_path, monkeypatch):
    monkeypatch.setattr("dancr.ui.mainwindow.write_sample", lambda base: write_sample(tmp_path, rows=5000))
    window._start_template("report")
    wait_run(window, app)
    pipe = window.doc.pipeline
    savers = {n for n in pipe.nodes if window.doc._needs_folder(pipe, n)}
    assert savers                                         # its report waits for the project to have a folder
    states = window.doc.executor.states()
    assert all(states[n].status == "done" for n in pipe.nodes if n not in savers)
    window.doc.save(tmp_path / "proj" / "limits.json")
    window.run(); wait_run(window, app)
    assert all(s.status == "done" for s in window.doc.executor.states().values())
    assert [f.name for f in (tmp_path / "proj").glob("*.html")]    # saved next to the project, not the sample data


def test_versions_and_restore(window, app, sample, tmp_path, monkeypatch):
    window._add_load_node(str(sample), None); wait_run(window, app)
    path = tmp_path / "proj.json"
    window.doc.save(path)
    window.add_node("sort", None, params={"columns": ["time"]})
    window.doc.save()                                     # keeps a version of the one-step project
    versions = window.doc.pipeline.versions()
    assert len(versions) == 1
    from dancr.ui.dialogs import VersionsDialog
    dlg = VersionsDialog(window, window.doc)
    assert dlg.list.count() == 1
    dlg.list.setCurrentRow(0)
    assert dlg.chosen() == versions[0]
    data = json.loads(versions[0].read_text())
    assert len(data["nodes"]) == 1


def test_source_change_marks_stale(window, app, sample):
    window._add_load_node(str(sample), None); wait_run(window, app)
    nid = window.current_table()
    assert window.doc.state(nid).status == "done"
    with open(sample, "a") as f:
        f.write("2030-01-01 00:00:00,1,1,1,x\n")
    t = time.time()
    while window.doc.state(nid).status == "done" and time.time() - t < 8:
        app.processEvents()
    assert window.doc.state(nid).status != "done"


def test_recovery_of_unsaved_project(window, app, sample, monkeypatch):
    from dancr.ui.document import recovery_path
    from dancr.ui.mainwindow import MainWindow
    from PySide6.QtWidgets import QMessageBox
    window._add_load_node(str(sample), None); wait_run(window, app)
    assert window.doc.path is None and window.doc.dirty
    window.doc.autosave_now()
    mine = recovery_path()
    assert mine.exists() and str(os.getpid()) in mine.name          # per process
    assert window.doc.pending_recovery() is None                     # our own copy is never offered back
    dead = recovery_path(2**22 + 7)                                  # a pid that cannot be running
    mine.replace(dead)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    w2 = MainWindow(); w2.show()
    pump(app, 400)
    assert len(w2.doc.pipeline.nodes) == 1 and w2.doc.path is None
    assert w2.doc.autosave_paused and "recovered" in w2.doc.autosave_paused
    assert not dead.exists() and recovery_path().exists()            # taken over by this process
    w2.doc.undo.setClean(); w2.close()
    assert not recovery_path().exists()          # a deliberate close clears it


def test_autosave_pauses_after_external_change_and_restore(window, app, sample, tmp_path):
    window._add_load_node(str(sample), None); wait_run(window, app)
    path = tmp_path / "paused.json"
    window.doc.save(path)
    window.add_node("sort", None, params={"columns": ["time"]})
    assert window.doc.dirty and window.doc.autosave_paused is None
    other = Pipeline.load(path); other.add_node("summarize", id="ext", x=0, y=500); other.save()
    t = time.time()
    while window.doc.autosave_paused is None and time.time() - t < 5:
        app.processEvents()
    assert window.doc.autosave_paused and "changed on disk" in window.doc.autosave_paused
    assert "autosave paused" in window.windowTitle()
    window.doc.autosave_now()
    assert "ext" in Pipeline.load(path).nodes                        # not overwritten
    window.doc.save(overwrite=True)                                  # the person chose "Keep mine": saving resumes it
    assert window.doc.autosave_paused is None
    versions = window.doc.versions()
    assert versions
    window.doc.restore_version(versions[0])
    assert window.doc.dirty and "earlier version" in window.doc.autosave_paused
    before = path.read_text()
    window.doc.autosave_now()
    assert path.read_text() == before
    window.open_path(path)                                           # Revert
    assert window.doc.autosave_paused is None


@pytest.mark.parametrize("key", [e["key"] for e in EXAMPLES])
def test_each_example_opens_and_runs(window, app, tmp_path, monkeypatch, key):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    window.start.cards[key].clicked.emit()
    wait_run(window, app)
    p = window.doc.pipeline
    ex = example(key)
    assert p.path == (tmp_path / "DANCR samples" / ex["title"] / f"{ex['title']}.json").resolve()
    assert len(p.answers) == len(ex["questions"])
    window.run(); wait_run(window, app)
    assert all(s.status == "done" for s in window.doc.executor.states().values())
    assert (p.path.parent / f"{ex['title']} report.html").exists()


def test_recent_projects_can_be_taken_off_the_list(window, app, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    Pipeline().save(a); Pipeline().save(b)
    window.settings.setValue("recent", [str(a), str(b), str(tmp_path / "gone.json")])
    window._show_page()
    rows = [window.start.recent_box.itemAt(i).widget() for i in range(window.start.recent_box.count())]
    assert [r.path for r in rows] == [a, b]                   # a project that no longer exists is left out
    rows[0].remove.emit()
    assert window._recent() == [str(b), str(tmp_path / "gone.json")]
