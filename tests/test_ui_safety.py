"""Edits are never lost and never land where the person did not want them (review 2026-09-24, phase 4)."""
import json
import os

import pytest
from PySide6.QtWidgets import QApplication

from dancr.core import Pipeline


@pytest.fixture
def doc(app, tmp_path):
    from dancr.ui.document import Document, recovery_path
    d = Document()
    yield d
    d.undo.setClean()
    d.shutdown()
    recovery_path().unlink(missing_ok=True)


def _saved(doc, tmp_path, small_csv):
    doc.add_node("load_file", 0, 0, params={"path": str(small_csv)})
    doc.save(tmp_path / "p.json")
    return tmp_path / "p.json"


def test_autosave_never_writes_while_a_dialog_about_the_edits_is_open(doc, tmp_path, small_csv, monkeypatch):
    from dancr.ui.document import recovery_path
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


def test_autosave_versions_never_push_out_saved_versions(tmp_path):
    p = Pipeline("v")
    path = tmp_path / "v.json"
    t = 1_700_000_000

    def save(auto: bool, i: int):
        p.meta["i"] = i                                    # a change, so a version is kept
        p.save(path, auto=auto)
        os.utime(path, (t + i, t + i))                     # versions are named by the time of the file they keep

    save(False, 0)
    for i in range(1, 6):
        save(False, i)
    for i in range(6, 6 + Pipeline.MAX_AUTO_VERSIONS + 40):
        save(True, i)
    names = [v.name for v in p.versions()]
    saved = [n for n in names if not n.endswith(".auto.json")]
    autos = [n for n in names if n.endswith(".auto.json")]
    assert len(saved) == 6                                 # every saved version is still there
    assert len(autos) == Pipeline.MAX_AUTO_VERSIONS
