"""MainWindow session methods."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QPoint, QSettings, QTimer, QSize, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (QMainWindow, QMenu, QFileDialog, QMessageBox, QSplitter, QToolBar, QStatusBar, QInputDialog, QLabel,
                               QProgressDialog, QApplication, QToolButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFrame, QProgressBar, QDialog, QSizePolicy, QCheckBox)

from ..core import registry, PipelineError
from ..core.model import Pipeline
from ..core.samples import write_sample, build_template
from .document import Document
from .common import listen
from .dialogs import Toast, VersionsDialog
from .stepfactory import StepFactory
from .workers import view_pool
from .canvas import CanvasScene, CanvasView, NODE_W, NODE_H
from .inspector import InspectorPanel
from .steppicker import StepPicker
from .rail import Rail, VIEW_TYPES, REPORT_TYPES
from .tableview import TableView
from .chartview import ChartView
from .mapview import MapView
from .reportview import ReportView
from .inputsview import InputsView
from .enterdata import EnterDataView
from .answering import Understanding, AskBar, AnswerPanel
from .assistant import AssistantPanel, SideDock
from .startpage import StartPage
from .insight import InsightBar
from .sources import SourcesTray
from .theme import T
from .icons import icon

from .window_common import DATA_FILTER, FILE_FILTER, log  # noqa: F401


class WindowSession:
    # ------------------------------------------------------------ file ops
    def _set_autosave(self, on: bool) -> None:
        """Autosave is the person's choice for every project (remembered between sessions); off to begin with."""
        self.settings.setValue("autosave", on)
        self.doc.set_autosave(on)
        self.status.showMessage("Autosave on: the project is saved every minute" if on
                                else "Autosave off: the project file changes only when you save", 4000)

    def _update_title(self) -> None:
        name = self.doc.path.stem if self.doc.path else "Untitled"
        paused = self.doc.autosave_paused if self.doc.autosave else None
        self.setWindowTitle(f"{'• ' if self.doc.dirty else ''}{name}{' (autosave paused)' if paused else ''} — DANCR")
        self.autosave_label.setText(f"Autosave paused: {paused}" if paused else "")
        self.a_revert.setEnabled(self.doc.path is not None and self.doc.dirty)

    def _on_reloaded(self) -> None:
        self._entered = True                    # a project was opened or replaced: past the start page for good
        self._building.clear()
        self._update_title()
        self._current = None
        self._current_answer = None
        self.scene.clear_highlight()
        self.answer_bar.set_answer(None)
        self.inspector.set_node(None)
        QTimer.singleShot(0, self.view.fit_all)
        self.doc.schedule_auto_run()
        if self.doc.path:
            self._push_recent(self.doc.path)
        first = next(iter(self.doc.pipeline.topological_order()), None) if self.doc.pipeline.nodes else None
        if first:
            self.show_node(first)
        else:
            self._show_page()
        self._refresh_mode()

    def maybe_save(self) -> bool:
        self.doc.flush_edits()
        if not self.doc.dirty:
            return True
        if self.doc.autosave_paused:
            why = f"Autosave is paused ({self.doc.autosave_paused})" if self.doc.autosave else self.doc.autosave_paused.capitalize()
            what = f"{why}. Save replaces {self.doc.path.name if self.doc.path else 'nothing'} with what you see now; Discard keeps the file on disk as it is."
        else:
            what = "Save changes to this project?"
        r = QMessageBox.question(self, "Unsaved changes", what, QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def _confirm_stop_run(self, what: str) -> bool:
        if not self.doc.running:
            return True
        r = QMessageBox.question(self, "A run is in progress", f"Stop the run and {what}?", QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return False
        self.doc.stop(wait=True)
        return True

    def _on_busy(self, why: str | None) -> None:
        """While a run is torn down, a non-blocking banner says why and the structural actions are paused, so
        nothing opens, saves or quits mid-teardown. No modal dialog: the window stays visible and animated."""
        if why:
            self.busy_label.setText(str(why)); self.busy_bar.show()
            self.menuBar().setEnabled(False); self.toolbar.setEnabled(False)
        else:
            self.busy_bar.hide()
            self.menuBar().setEnabled(True); self.toolbar.setEnabled(True)

    def new_pipeline(self) -> None:
        if self._confirm_stop_run("start a new project") and self.maybe_save():
            self._entered = True                # a new project starts on the empty canvas, not the start page
            self.doc.new()

    def open_dialog(self) -> None:
        if not self._confirm_stop_run("open another project") or not self.maybe_save():
            return
        start = str(self.doc.path.parent) if self.doc.path else str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, "Open project", start, FILE_FILTER)
        if f:
            self.open_path(f)

    def open_path(self, path: str | Path) -> None:
        try:
            self.doc.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open", str(e))
            return
        self.status.showMessage(f"Opened {path}", 5000)

    def save(self) -> bool:
        if self.doc.path is None:
            return self.save_as()
        from .document import ChangedOnDisk
        try:
            self.doc.save()
        except ChangedOnDisk:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("The project changed on disk")
            box.setText(f"{self.doc.path.name} was changed by another program (an agent, the command line or another "
                        "window) since it was opened here.")
            box.setInformativeText("Keep mine overwrites their change (it stays under File → Earlier versions). "
                                   "Load theirs throws away the unsaved changes in this window.")
            mine = box.addButton("Keep mine", QMessageBox.AcceptRole)
            theirs = box.addButton("Load theirs", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            chosen = box.clickedButton()
            box.deleteLater()
            if chosen is mine:
                try:
                    self.doc.save(overwrite=True)
                except (OSError, PipelineError) as e:
                    QMessageBox.critical(self, "Cannot save", str(e)); return False
            elif chosen is theirs:
                self.doc.load(self.doc.path)
                return True
            else:
                return False
        except (OSError, PipelineError) as e:
            QMessageBox.critical(self, "Cannot save", str(e)); return False
        self.status.showMessage(f"Saved {self.doc.path.name}", 3000)
        self._update_title()
        return True

    def save_as(self) -> bool:
        if self.doc.running:
            QMessageBox.information(self, "Save as", "Wait for the run to finish before saving under a new name.")
            return False
        start = str(self.doc.path) if self.doc.path else str(Path.home() / "project.json")
        f, _ = QFileDialog.getSaveFileName(self, "Save project as", start, FILE_FILTER)
        if not f:
            return False
        if not f.lower().endswith(".json"):
            f += ".json"
        try:
            self.doc.save(f)
        except (OSError, PipelineError) as e:
            QMessageBox.critical(self, "Cannot save", str(e)); return False
        self._push_recent(Path(f))
        self._update_title()
        self.status.showMessage(f"Saved {f}", 3000)
        return True

    def _maybe_recover(self) -> None:
        if self._disposed or self.doc.pipeline.nodes:
            return
        found = self.doc.pending_recovery()
        if found is None:
            return
        pipe, rp = found
        if pipe.path is not None:
            text = (f"Last time DANCR closed with unsaved changes to {pipe.path.name}. Bring them back? "
                    "The file itself is unchanged until you save.")
        else:
            text = f"Last time DANCR closed with an unsaved project of {len(pipe.nodes)} steps. Bring it back?"
        r = QMessageBox.question(self, "Unsaved changes", text, QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            self.doc.recover(pipe, rp)
            self.status.showMessage("Recovered. Save it to keep it", 8000)
        else:
            rp.unlink(missing_ok=True)

    def revert(self) -> None:
        if self.doc.path and self._confirm_stop_run("reload from disk") and \
                QMessageBox.question(self, "Revert", "Discard your changes and reload from disk?") == QMessageBox.Yes:
            self.open_path(self.doc.path)

    def show_versions(self) -> None:
        if self.doc.path is None:
            QMessageBox.information(self, "Earlier versions", "Save the project first. From then on every save keeps an earlier version you can go back to.")
            return
        dlg = VersionsDialog(self, self.doc)
        if dlg.exec() == QDialog.Accepted and dlg.chosen() and self._confirm_stop_run("restore an earlier version"):
            try:
                self.doc.restore_version(dlg.chosen())
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Cannot restore", str(e)); return
            self.status.showMessage("Restored. Save to keep it, or Revert to go back."
                                    + (" Autosave is paused until then." if self.doc.autosave else ""), 8000)

    # ------------------------------------------------------------ dataset metadata (FAIR)
    def dataset_details(self) -> None:
        from .dialogs import DatasetDialog
        dlg = DatasetDialog(self, self.doc.pipeline.dataset_meta())
        if dlg.exec() == QDialog.Accepted:
            self.doc.set_dataset_meta(**dlg.fields())

    def export_metadata(self, fmt: str) -> None:
        from ..headless import export_fair
        from ..core.executor import Executor
        p = self.doc.pipeline
        default = {"schema.org": "dataset.jsonld", "frictionless": "datapackage.json",
                   "manifest": "dancr-manifest.json", "rocrate": "ro-crate-metadata.json"}.get(fmt, "metadata.json")
        f, _ = QFileDialog.getSaveFileName(self, "Export metadata", str(p.directory / default),
                                           "JSON (*.json);;JSON-LD (*.jsonld);;All files (*)")
        if not f:
            return
        try:
            export_fair(p, Executor(p), fmt=fmt, out=f)
        except Exception as e:  # noqa: BLE001 - shown to the person, never a traceback
            QMessageBox.warning(self, "Export metadata", str(e)); return
        self.status.showMessage(f"Wrote {f}", 8000)

    def browse_catalog(self) -> None:
        from .dialogs import CatalogDialog
        start = self.doc.path.parent if self.doc.path else None
        dlg = CatalogDialog(self, start)
        dlg.openProject.connect(self.open_path)
        dlg.exec()

    def show_relations(self) -> None:
        """How the tables fit together, from the engine's own understanding, with one-click builds."""
        m = self.understanding.model
        if m is None:
            self.understanding.when_full(self._open_relations)
            return
        self._open_relations(m)

    def _open_relations(self, model) -> None:
        from .dialogs import RelationsDialog
        rels = [r for r in model.relations if len([t for t in r.tables if t in self.doc.pipeline.nodes]) >= 2]
        if not rels:
            self.status.showMessage("No relations found between the tables yet", 6000)
            return
        RelationsDialog(self, rels, on_build=self.build_relation).exec()

    def build_relation(self, rel) -> None:
        """Add the step a relation implies: a join, a stack, or a nearest-time alignment."""
        tables = [t for t in rel.tables if t in self.doc.pipeline.nodes]
        if len(tables) < 2:
            return
        try:
            if rel.kind == "stack":
                nid = self.add_node("stack", None, connect_from=tables[0], port="tables")
                for t in tables[1:]:
                    self.doc.connect(t, nid, "tables")
            elif rel.kind == "align" and rel.pairs:
                left_time, right_time = next(iter(rel.pairs.items()))
                nid = self.add_node("combine", None,
                                    params={"method": "nearest_time", "left_time": left_time,
                                            "right_time": right_time, "tolerance": rel.tolerance or ""},
                                    connect_from=tables[0], port="left")
                self.doc.connect(tables[1], nid, "right")
            else:                                       # a link: join on the key
                params: dict = {"method": "match", "on": rel.left_on}
                if rel.right_on and rel.right_on != rel.left_on:
                    params["right_on"] = rel.right_on
                nid = self.add_node("combine", None, params=params, connect_from=tables[0], port="left")
                self.doc.connect(tables[1], nid, "right")
        except (PipelineError, ValueError, KeyError) as e:
            self.status.showMessage(f"Could not build that: {e}", 8000); return
        self.show_node(nid)
        self.view.focus_node(nid)
        self.doc.schedule_auto_run()
        self.status.showMessage(f"Added {self.doc.pipeline.nodes[nid].title}", 5000)

    def package_crate(self) -> None:
        from ..headless import package_rocrate
        from ..core.executor import Executor
        p = self.doc.pipeline
        default = f"{(p.path.stem if p.path else 'dataset')}.rocrate.zip"
        f, _ = QFileDialog.getSaveFileName(self, "Package as RO-Crate", str(p.directory / default),
                                           "RO-Crate (*.zip);;All files (*)")
        if not f:
            return
        if not f.lower().endswith(".zip"):
            f += ".zip"
        try:
            rec = package_rocrate(p, Executor(p), out=f, zip=True)
        except Exception as e:  # noqa: BLE001 - shown to the person, never a traceback
            QMessageBox.warning(self, "Package as RO-Crate", str(e)); return
        self.status.showMessage(f"Wrote {rec['path']}", 8000)

    def terminate(self) -> None:
        """The system asked us to quit (SIGTERM): no questions, keep an unsaved project recoverable, close cleanly."""
        self._terminating = True
        self.close()
        if self.isVisible():
            QApplication.quit()

    def closeEvent(self, e: QCloseEvent) -> None:
        if self._terminating:
            self.doc.stop(wait=True, why="Stopping the current step before quitting…")
            self.doc.write_recovery()             # unsaved edits of any project; offered back on the next start
        else:
            if self.doc.running:
                r = QMessageBox.question(self, "A run is in progress", "Stop the run and quit?", QMessageBox.Yes | QMessageBox.No)
                if r != QMessageBox.Yes:
                    e.ignore(); return
                self.doc.stop(wait=True, why="Stopping the current step before quitting…")
            if not self.maybe_save():
                e.ignore(); return
            self.doc.clear_recovery()             # saved or deliberately discarded
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("top_split", self.top_split.saveState())
        self.settings.setValue("outer_split", self.outer_split.saveState())
        self.settings.setValue("result", self.a_result.isChecked())
        view_pool().shutdown(5000)                # nothing may still read the results when the cache is removed
        self.doc.shutdown()
        e.accept()

    def _restore_layout(self) -> None:
        g = self.settings.value("geometry")
        if g:
            self.restoreGeometry(g)
        t = self.settings.value("top_split")
        if t:
            self.top_split.restoreState(t)
        o = self.settings.value("outer_split")
        if o:
            self.outer_split.restoreState(o)
        if self.inspector.width() < 260:                    # a saved layout that squeezed it too far
            sizes = self.top_split.sizes()
            if len(sizes) == 3:
                self.top_split.setSizes([sizes[0], max(300, sizes[1] - 120), 360])
        m = self.settings.value("result")
        if m is not None:
            self.a_result.setChecked(m in (True, "true", "True", 1))

    def _recent(self) -> list[str]:
        v = self.settings.value("recent")
        if isinstance(v, str):
            return [v] if v else []
        return [str(x) for x in (v or [])]

    def _remove_recent(self, p: str) -> None:
        self.settings.setValue("recent", [r for r in self._recent() if r != p])
        self._refresh_recent()
        if self._on_start_page():
            self.start.set_recent(self._recent())

    def _push_recent(self, p: Path) -> None:
        rec = [str(p)] + [r for r in self._recent() if r != str(p)]
        self.settings.setValue("recent", rec[:10])
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        for r in self._recent():
            self.recent_menu.addAction(QAction(r, self, triggered=lambda checked=False, p=r: self._confirm_stop_run("open another project") and self.maybe_save() and self.open_path(p)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.toast.reposition()

    # ------------------------------------------------------------ drag and drop onto the whole window
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        projects = [p for p in paths if p.lower().endswith(".json")]
        data = [p for p in paths if not p.lower().endswith(".json")]
        if projects:
            if self._confirm_stop_run("open another project") and self.maybe_save():
                self.open_path(projects[0])
            e.acceptProposedAction(); return
        self._add_files(data)
        e.acceptProposedAction()

