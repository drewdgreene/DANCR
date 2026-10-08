"""MainWindow answers methods."""
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


class WindowAnswers:
    # ------------------------------------------------------------ answers
    def focus_ask(self) -> None:
        """Ask a question: opens the ask bar (or closes it when it is already open)."""
        if not self.doc.pipeline.nodes:
            self.add_data_files(); return
        if self._ask_open and self.askbar.isVisible():
            self.close_ask(); return
        self.open_ask(focus=True)

    def _on_side_tab(self, assistant_on: bool) -> None:
        """The person switched the dock tab; keep the menu action in step so it is not flipped back."""
        self.a_assistant.setChecked(assistant_on)
        self._apply_side_panels()

    def _on_assistant_toggled(self, on: bool) -> None:
        """The Assistant action's checked state changed: show or hide the side column to match. The action
        already holds the new state, so this only applies it and never flips it back (which cancelled out)."""
        if on and not self.doc.pipeline.nodes:
            self.add_data_files()
            self.a_assistant.setChecked(False)     # nothing to chat about yet; asking for data is the useful thing
            return
        self._apply_side_panels()
        if on:
            self.assistant.edit.setFocus()

    def _preview_proposal(self, proposal: dict) -> None:
        """Ghost a proposed chain of steps onto the canvas, so the person sees what would be built before it is."""
        steps = proposal.get("steps") or []
        if steps:
            self.a_assistant.setChecked(True)       # bring the plan card (with Approve/Discard) into view
            self._apply_side_panels()
            self.scene.show_ghost(steps)
        else:
            self.scene.clear_ghost()

    def apply_assistant_proposal(self, proposal: dict) -> None:
        """Build what the Assistant proposed: real steps, through the undo stack, then run them."""
        from ..core.recipes import PlanError
        self.scene.clear_ghost()
        model = self.understanding.model
        if model is None:
            self.assistant.built(message="Still reading your tables; try again in a moment.")
            return
        terminals: list[str] = []
        answers: list[str] = []
        try:
            if proposal.get("kind") in ("answer", "answers"):
                specs = proposal.get("specs") or ([proposal["spec"]] if proposal.get("spec") else [])
                with self.doc.macro("Assistant answers" if len(specs) > 1 else "Assistant answer"):
                    for spec in specs:            # the built steps and their AI marking undo together
                        aid = self.doc.build_answer(model, spec)
                        a = self.doc.pipeline.answer(aid)
                        if a is not None:
                            self.doc.add_ai_steps(a.nodes)
                            terminals.append(a.terminal)
                            answers.append(aid)
            else:
                terminal = self._apply_assistant_steps(proposal.get("steps") or [])
                if terminal:
                    terminals.append(terminal)
        except (PlanError, KeyError, ValueError) as e:
            self.assistant.built(message=f"The engine could not build that: {str(e).strip(chr(39) + chr(34))}")
            return
        terminals = [t for t in terminals if t and t in self.doc.pipeline.nodes]
        if not terminals:
            self.assistant.built(message="Built, but there is nothing to run.")
            return
        # stay in the chat: do not move the view or the side panel. The result lands on the card; the person
        # clicks "Canvas" when they want to look at the steps.
        self._assistant_terminals = terminals
        self._assistant_answers = answers
        self.run(terminals)

    def _apply_assistant_steps(self, steps: list) -> str | None:
        """Add a hand-built plan as ordinary, undoable steps. Returns the last step's id."""
        from .. import headless as hl
        last = None
        made: dict[str, str] = {}
        with self.doc.macro("Assistant"):
            for s in steps:
                after = s.get("after")
                after = made.get(after, after)
                x, y = hl.place(self.doc.pipeline, after)
                nid = self.doc.add_node(s["type"], x, y, params=s.get("params") or {}, title=s.get("title"),
                                        connect_from=after if after in self.doc.pipeline.nodes else None,
                                        port=s.get("port"))
                made[s.get("id")] = nid
                last = nid
            self.doc.add_ai_steps(list(made.values()))     # so they can be marked and removed as a set
        return last

    def _on_assistant_run_finished(self, ok: bool, states: dict) -> None:
        """The Assistant's proposed steps finished: give the card the engine's own finding(s)."""
        terminals = getattr(self, "_assistant_terminals", None) or []
        if not terminals:
            return
        self._assistant_terminals = None
        answer_ids, self._assistant_answers = getattr(self, "_assistant_answers", []) or [], None
        findings: list[str] = []
        for t in terminals:
            st = states.get(t) or self.doc.state(t)
            if st.status == "failed":
                self.assistant.built(message=f"That step failed: {st.error}", terminal=t)
                return
            if st.status == "done":
                f = ((st.report or {}).get("finding") or {}).get("statement") or ""
                if f:
                    findings.append(f)
        multi = len(terminals) > 1
        self.assistant.built(finding=" ".join(findings) or "Ran over every row.",
                             terminal=terminals[0],
                             answer_id=(None if multi else (answer_ids[0] if answer_ids else None)),
                             multi=multi)

    def assistant_apply_edits(self, edits: list) -> None:
        """Apply the Assistant's project changes (renames, settings, column labels, inputs) as one undo step."""
        if not edits:
            return
        try:
            with self.doc.macro("Assistant changes"):
                for e in edits:
                    op = e.get("op")
                    if op == "rename":
                        self.doc.rename(str(e["node"]), str(e["title"]))
                    elif op == "set_params":
                        self.doc.set_params(str(e["node"]), e.get("params") or {})
                    elif op == "set_input":
                        self.doc.set_input(str(e["name"]), e.get("value"), e.get("unit"), e.get("note"))
                    elif op == "column_label":
                        self.doc.set_column_meta(str(e["column"]), e.get("label"), e.get("unit"))
        except (PipelineError, KeyError, ValueError) as e:
            self.assistant.built(message=f"Could not apply the changes: {str(e).strip(chr(39) + chr(34))}")
            return
        n = len(edits)
        self.status.showMessage(f"Applied {n} change{'s' if n != 1 else ''} from the Assistant", 6000)
        self.assistant.edits_applied(n)

    def _assistant_reveal(self, terminal: str) -> None:
        """Show the built steps on the canvas, centred on the answer, with its branch lit up."""
        if terminal not in self.doc.pipeline.nodes:
            return
        self.a_result.setChecked(True)
        self._apply_result_visibility()
        self.show_node(terminal)
        branch = self.doc.pipeline.upstream_closure(terminal) | {terminal}

        def show() -> None:
            if terminal in self.scene.nodes:
                self.view.reveal(terminal)
                self.scene.highlight_branch({n for n in branch if n in self.scene.nodes})
        QTimer.singleShot(0, show)

    def assistant_save_as_project(self, terminal: str) -> None:
        """Save just the steps behind this answer to a new project file (paths kept pointing at the same files)."""
        from ..core.answers import save_as_project
        base = self.doc.pipeline.path
        start = str(base.with_name(f"{base.stem} (assistant).json")) if base else str(Path.home() / "assistant.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save the Assistant's steps as a project", start, FILE_FILTER)
        if not path:
            return
        try:
            out = save_as_project(self.doc.pipeline, terminal, path)
        except (OSError, PipelineError, ValueError) as e:
            self.assistant.built(message=f"Could not save a project: {e}")
            return
        self.status.showMessage(f"Saved a new project: {out}", 8000)

    def assistant_replace_canvas(self, terminal: str, confirm: bool = True) -> None:
        """Leave only the Assistant's steps on the canvas (one undo step), then run them."""
        from ..core.answers import project_from_answer
        if terminal not in self.doc.pipeline.nodes:
            return
        from PySide6.QtWidgets import QMessageBox
        if confirm and QMessageBox.question(
                self, "Replace the canvas?",
                "Leave only the Assistant's steps for this result on the canvas?\n\n"
                "Your other steps are hidden, not deleted — Ctrl+Z brings everything back.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        title = self.doc.pipeline.nodes[terminal].title
        a = next((x for x in self.doc.pipeline.answers if x.terminal == terminal), None)
        built = list(a.nodes) if a is not None else []
        new = project_from_answer(self.doc.pipeline, terminal, name=self.doc.pipeline.name)
        with self.doc.macro(f"Assistant: {title}"):        # one undo step: the replace and the marking together
            self.doc.replace_canvas(new.to_dict(), text=f"Assistant: {title}")
            self.doc.add_ai_steps(built)                   # the Assistant's steps stay marked after a replace
        self.show_node(terminal)
        self._assistant_terminals = [terminal]
        self._assistant_answers = []
        self.run([terminal])

    def open_ask(self, focus: bool = False) -> None:
        self._ask_open = True
        if self.doc.pipeline.nodes:
            self.askbar.setVisible(True)
        if focus:
            self.askbar.focus_edit()

    def close_ask(self) -> None:
        self._ask_open = False
        self.askbar.setVisible(False)

    def build_answer(self, spec: dict, answer_id: str | None = None) -> None:
        """Build (or change) an answer once every row of the tables has been read: answers are never planned
        from a sample. Shows the answer when it is built."""
        from ..core.recipes import PlanError
        same = self._same_answer(spec) if answer_id is None else None
        if same is not None:                       # asked again (or a double click): the answer is already here
            self.show_answer(same)
            return
        key = json.dumps([spec, answer_id], sort_keys=True, default=str)
        if key in self._building:
            return
        self._building.add(key)
        if not self.understanding.full:
            self.askbar.show_status("Reading every row of your tables first…")

        def go(model) -> None:
            self._building.discard(key)
            if self._disposed:
                return
            if answer_id is None and self._same_answer(spec) is not None:
                self.show_answer(self._same_answer(spec)); return
            try:
                aid = self.doc.build_answer(model, spec, answer_id)
            except (PlanError, KeyError) as e:
                self.askbar.show_status(str(e).strip("'\""), error=True)     # inline, never a dialog out of the blue
                return
            self.askbar.show_status("")
            self.doc.schedule_auto_run()
            self.show_answer(aid)
            a = self.doc.pipeline.answer(aid)
            if a is not None and a.terminal in self.scene.nodes:
                QTimer.singleShot(0, lambda: self.view.reveal(a.terminal) if not self._disposed else None)
            a = self.doc.pipeline.answer(aid)
            if a is not None and not self.doc.auto_run and not self.doc.running:
                self.run([a.terminal])
            from ..core.answers import set_aside_note
            note = set_aside_note(self.doc.last_set_aside)
            if note:
                self.toast.show_message(note, "OK", lambda: None)
            self.status.showMessage(note or "Built the answer. Change its choices above, or ask another question", 8000)
        self.understanding.when_full(go)

    def _same_answer(self, spec: dict) -> str | None:
        want = json.dumps({k: v for k, v in spec.items() if k != "title"}, sort_keys=True, default=str)
        for a in self.doc.pipeline.answers:
            if a.terminal in self.doc.pipeline.nodes and json.dumps({k: v for k, v in a.spec.items() if k != "title"},
                                                                     sort_keys=True, default=str) == want:
                return a.id
        return None

    def change_answer(self, aid: str, key: str, value) -> None:
        from ..core.recipes import apply_choice
        a = self.doc.pipeline.answer(aid)
        if a is not None:
            self.build_answer(apply_choice(a.spec, key, value), aid)

    def _show_answer_steps(self) -> None:
        a = self.doc.pipeline.answer(self._current_answer) if self._current_answer else None
        if not self.a_result.isChecked():
            self.a_result.setChecked(True)
        if a is not None and a.terminal in self.doc.pipeline.nodes:
            self.view.focus_node(a.terminal)
        else:
            self.view.fit_all()

    def _focus_answers(self, nid: str | None) -> None:
        """The tray offers answers about the table being looked at; a chart or report is not a table to ask about."""
        n = self.doc.pipeline.nodes.get(nid) if nid else None
        self.understanding.set_focus(nid if n is not None and registry.get(n.type).kind != "sink" and n.type not in VIEW_TYPES else None)

    def show_answer(self, aid: str) -> None:
        """Select an Answer: show its result in the centre and highlight its branch on the map."""
        answer = self.doc.pipeline.answer(aid)
        if answer is None:
            return
        self.understanding.set_focus(None)
        self.open_ask()
        self._entered = True
        self.a_result.setChecked(True)
        self._current_answer = aid
        self._current = answer.terminal if answer.terminal in self.doc.pipeline.nodes else None
        self.rail.select("answer", aid, emit=False)
        self.scene.select_answer(aid)
        if self._current:
            self.scene.highlight_branch(self.doc.pipeline.upstream_closure(self._current) | {self._current})
            self.inspector.set_node(self._current)
        else:
            self.scene.clear_highlight()
        self._sync_answer_bar()
        self._show_page()

    def delete_answer_dialog(self, aid: str) -> None:
        answer = self.doc.pipeline.answer(aid)
        if answer is None:
            return
        exclusive = self.doc.answer_exclusive_nodes(answer)
        box = QMessageBox(self)
        box.setWindowTitle("Delete answer")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Delete “{answer.title}”?")
        cb = QCheckBox("Also remove the steps it built")
        if exclusive:
            cb.setText(f"Also remove the {len(exclusive)} step(s) it built")
        else:
            cb.setText("Its steps are shared with other answers, so they are kept")
            cb.setEnabled(False)
        box.setCheckBox(cb)
        box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        box.setDefaultButton(QMessageBox.Cancel)
        answer = box.exec()
        remove_steps = cb.isChecked()
        box.deleteLater()
        if answer != QMessageBox.Yes:
            return
        self.doc.delete_answer(aid, remove_steps=remove_steps)
        if self._current_answer == aid:
            self._current_answer = None
            self.scene.clear_highlight()
            self.answer_bar.set_answer(None)
            self._current = None
            self._show_page()

    def _sync_answer_bar(self) -> None:
        answer = self.doc.pipeline.answer(self._current_answer) if self._current_answer else None
        self.answer_bar.set_answer(answer.id if answer is not None else None)

    def current_table(self) -> str | None:
        """The table the person is looking at (charts and reports resolve to their input)."""
        nid = self._current
        if nid in (None, "inputs") or nid not in self.doc.pipeline.nodes:
            return None
        n = self.doc.pipeline.nodes[nid]
        if n.type in VIEW_TYPES or n.type in REPORT_TYPES:
            ins = self.doc.pipeline.inputs_of(nid)
            for lst in ins.values():
                if lst:
                    return lst[0]
            return None
        return nid

    def _on_start_page(self) -> bool:
        return not self.doc.pipeline.nodes and self._current is None and not self._entered

    def _apply_side_panels(self) -> None:
        """The project list and the side column have nothing to show until the project has a step."""
        start = self._on_start_page()
        self.rail.setVisible(not start)
        show = not start and (self.a_settings.isChecked() or self.a_assistant.isChecked())
        self.side.setVisible(show)
        self.side.set_assistant(self.a_assistant.isChecked())    # keep the tab in step even while hidden
        if show:
            sizes = self.top_split.sizes()
            if len(sizes) == 3 and sizes[2] < 120:               # opened into a collapsed slot: give it room
                total = sum(sizes) or self.top_split.width()
                self.top_split.setSizes([sizes[0], max(320, total - sizes[0] - 360), 360])
        self.inspector.setVisible(not start and self.a_settings.isChecked() and not self.a_assistant.isChecked())
        for a in (self.a_settings, self.a_result, self.a_add, self.a_ask, self.a_assistant, self.a_run):
            a.setEnabled(not start and (a is not self.a_assistant or bool(self.doc.pipeline.nodes)))

    def _toggle_result(self, on: bool) -> None:
        self._apply_result_visibility()
        if on:
            QTimer.singleShot(0, self.view.fit_all)

    def _apply_result_visibility(self) -> None:
        have = bool(self.doc.pipeline.nodes) or self._current is not None
        on = self.a_result.isChecked()
        self.result_box.setVisible(on and have)
        self.result_handle.setVisible(have and not on)
        if not on and self.result_expand.isChecked():        # closing clears the expanded state
            self.result_expand.blockSignals(True); self.result_expand.setChecked(False); self.result_expand.blockSignals(False)
            self.top_split.setMaximumHeight(16777215)
        if on and have:
            sizes = self.outer_split.sizes()
            if len(sizes) == 2 and sizes[1] < 140 and not self.result_expand.isChecked():
                total = sum(sizes) or self.outer_split.height()
                self.outer_split.setSizes([max(220, total - 280), 280])

    def _toggle_result_expand(self, on: bool) -> None:
        """Give the result most of the window (or restore the canvas to most of it)."""
        if on:
            self._result_sizes = self.outer_split.sizes()
            self.top_split.setMaximumHeight(90)          # the canvas keeps a sliver; the result takes the rest
            h = self.outer_split.height() or max(300, self.height() - 120)
            self.outer_split.setSizes([90, max(200, h - 90)])
        else:
            self.top_split.setMaximumHeight(16777215)
            sizes = getattr(self, "_result_sizes", None)
            if sizes and len(sizes) == 2:
                self.outer_split.setSizes(sizes)
            else:
                h = self.outer_split.height() or self.height()
                self.outer_split.setSizes([max(220, h - 260), 260])

    def _refresh_mode(self) -> None:
        auto = self.doc.auto_run and bool(self.doc.pipeline.nodes)
        self.a_auto.blockSignals(True); self.a_auto.setChecked(self.doc.auto_run); self.a_auto.blockSignals(False)
        self.a_run.setVisible(not auto or self.doc.running)
        self.a_run_sel.setEnabled(not auto)
        if not self.doc.pipeline.nodes:
            self.mode_label.setText("")
        elif auto:
            self.mode_label.setText("runs automatically")
        else:
            mb = self.doc.source_bytes() / 1e6
            self.mode_label.setText(f"large data ({mb:,.0f} MB), press Run to compute")
        if self.table.nid:
            self.table._refresh_header()

    def remove_ai_steps(self) -> None:
        """Delete every step the Assistant built, as one undoable action."""
        ids = self.doc.ai_steps()
        if not ids:
            self.status.showMessage("No Assistant-built steps to remove", 4000); return
        with self.doc.macro("Remove AI steps"):
            self.doc.remove_nodes(ids)
            self.doc.clear_ai_steps()
        self.status.showMessage(f"Removed {len(ids)} Assistant-built step(s). Ctrl+Z undoes it.", 6000)

    def _refresh_insight(self) -> None:
        """Publish the one sentence worth reading: the selected step's finding, else the best finding in the
        project. Shown in the insight bar above the canvas; clicking it jumps to the step that produced it."""
        nid = self._current if self._current in self.doc.pipeline.nodes else None
        if nid:
            st = self.doc.cached_state(nid)
            fnd = ((st.report or {}).get("finding") or {}).get("statement") if st else None
            if fnd:
                self.insight.set_insight(fnd, nid); return
        best = None
        for k in self.doc.pipeline.nodes:
            st = self.doc.cached_state(k)
            fnd = ((st.report or {}).get("finding") or {}).get("statement") if st else None
            if fnd:
                best = (fnd, k)
        if best:
            self.insight.set_insight(best[0], best[1])
        else:
            self.insight.clear()

    def delete_current(self) -> None:
        aids = self.scene.selected_answer_ids()
        if aids:
            self.delete_answer_dialog(aids[0])
            return
        if not self.scene.selected_node_ids() and self.scene.selectedItems():
            self.scene.delete_selection()          # an arrow or a note is selected on the map: that is what goes
            return
        ids = self.scene.selected_node_ids() or ([self._current] if self._current and self._current in self.doc.pipeline.nodes else [])
        if not ids:
            return
        titles = [self.doc.pipeline.nodes[i].title for i in ids]
        if self.scene.selected_node_ids():
            self.scene.delete_selection()          # nodes plus any selected arrows and notes, one undo entry
        else:
            self.doc.remove_nodes(ids)
        idx = self._toast_index = self.doc.undo.index()
        self.toast.show_message(f"Deleted {titles[0] if len(titles) == 1 else f'{len(titles)} steps'}", "Undo",
                                lambda: self.doc.undo.undo() if self.doc.undo.index() == idx else None)

