"""Right dock: settings for the selected step, generated from its Param specs."""
from __future__ import annotations

import html
from collections import OrderedDict
from pathlib import Path
from typing import Any

import polars as pl
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QFormLayout,
                               QPushButton, QToolButton, QFrame, QSizePolicy)

from ..core import registry
from ..core.nodes.load import EXCEL_EXT, list_sheets
from .document import Document
from .widgets import make_widget, ParamWidget, SheetWidget, PathWidget, BoolWidget, kind_of
from .finding import FindingCard
from .workers import Serial, Task, view_pool
from .theme import T, category_color
from .common import status_dot
from .icons import icon, node_icon_name


class InspectorDock(QDockWidget):
    runRequested = Signal(str)
    SUGGEST_CACHE_SIZE = 64

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__("Settings", parent)
        self.setObjectName("inspector")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        self.setTitleBarWidget(self._title_bar())
        self.doc = doc
        self.nid: str | None = None
        self.widgets: dict[str, ParamWidget] = {}
        self._labels: dict[str, QWidget] = {}
        self._committing = False
        self._pending: dict[str, Any] = {}
        self._schema_serial = Serial()
        self._suggest_cache: OrderedDict[tuple, list[str]] = OrderedDict()     # (output file, column) -> values, newest last
        self._suggest_tasks: set[Task] = set()
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(350); self._timer.timeout.connect(self._commit)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.scroll)
        self.setMinimumWidth(320)
        self.scroll.setMinimumWidth(0)
        doc.nodeChanged.connect(self._on_node_changed)
        doc.nodeRemoved.connect(lambda nid: self.set_node(None) if nid == self.nid else None)
        doc.reloaded.connect(lambda: self.set_node(None))
        doc.statesChanged.connect(self._refresh_status)
        doc.nodeState.connect(lambda nid, st: self._refresh_status() if nid == self.nid else None)
        doc.runStarted.connect(self._refresh_status)
        doc.runFinished.connect(lambda ok, res: self._refresh_status())
        doc.edgeAdded.connect(lambda e: self._refresh_schema() if self.nid in (e.source, e.target) else None)
        doc.edgeRemoved.connect(lambda e: self._refresh_schema() if self.nid in (e.source, e.target) else None)
        self.set_node(None)

    def _title_bar(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(12, 8, 8, 6)
        lab = QLabel("Settings"); lab.setObjectName("section")
        h.addWidget(lab); h.addStretch()
        return w

    # ------------------------------------------------------------ build
    def set_node(self, nid: str | None) -> None:
        if self._timer.isActive():
            self._commit()
        self._pending.clear()
        self.nid = nid if nid in self.doc.pipeline.nodes else None
        self.widgets.clear(); self._labels.clear()
        body = QWidget(); lay = QVBoxLayout(body); lay.setContentsMargins(14, 6, 14, 14); lay.setSpacing(8)
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # never wider than the dock
        if self.nid is None:
            lay.addWidget(self._empty_state()); lay.addStretch()
            self.scroll.setWidget(body)
            return
        node = self.doc.pipeline.nodes[self.nid]
        nt = registry.get(node.type)
        # header: icon + editable title
        head = QHBoxLayout(); head.setSpacing(8)
        ic = QLabel(); ic.setPixmap(icon(node_icon_name(nt.key), category_color(nt.category).name(), 22).pixmap(22, 22))
        self.title_edit = QLineEdit(node.title)
        f = QFont(T.ui_font); f.setPointSizeF(11.5); f.setBold(True); self.title_edit.setFont(f)
        self.title_edit.setStyleSheet(f"QLineEdit {{ border: none; border-bottom: 1px solid transparent; border-radius: 0; background: transparent; padding: 2px 0; }} QLineEdit:hover, QLineEdit:focus {{ border-bottom: 1px solid {T.border}; }}")
        self.title_edit.setToolTip("Click to rename this step")
        self.title_edit.editingFinished.connect(self._rename)
        head.addWidget(ic); head.addWidget(self.title_edit, 1)
        lay.addLayout(head)
        from ..core.examples import EXAMPLES
        kind = QLabel(f"{nt.label} · {nt.description}"); kind.setObjectName("muted"); kind.setWordWrap(True)
        lay.addWidget(kind)
        if EXAMPLES.get(nt.key):
            ex = QLabel(f"<i>e.g.</i> {EXAMPLES[nt.key]}"); ex.setObjectName("faint"); ex.setWordWrap(True); lay.addWidget(ex)
        # status line
        srow = QHBoxLayout(); srow.setSpacing(8)
        self.status_label = QLabel(); self.status_label.setWordWrap(True); self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.run_btn = QPushButton("Run to here"); self.run_btn.setObjectName("primary"); self.run_btn.setIcon(icon("play", "#ffffff", 14))
        self.run_btn.setToolTip("Run the pipeline up to and including this step (Ctrl+Shift+R)")
        self.run_btn.clicked.connect(lambda: self.runRequested.emit(self.nid))
        srow.addWidget(self.status_label, 1); srow.addWidget(self.run_btn, 0, Qt.AlignTop)
        lay.addLayout(srow)
        self.finding = FindingCard()
        lay.addWidget(self.finding)
        line = QFrame(); line.setObjectName("hline"); lay.addWidget(line)
        # form
        self.form_widget = QWidget(); self.form = QFormLayout(self.form_widget)
        self.form.setContentsMargins(0, 2, 0, 0); self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.form.setRowWrapPolicy(QFormLayout.WrapAllRows); self.form.setVerticalSpacing(10)
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay.addWidget(self.form_widget)
        self.adv_btn = QToolButton(); self.adv_btn.setObjectName("quiet"); self.adv_btn.setText("More options"); self.adv_btn.setCheckable(True)
        self.adv_btn.setIcon(icon("dots-three", T.muted, 14)); self.adv_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.adv_btn.toggled.connect(self._toggle_advanced)
        self.adv_widget = QWidget(); self.adv_form = QFormLayout(self.adv_widget); self.adv_form.setContentsMargins(0, 0, 0, 0)
        self.adv_form.setRowWrapPolicy(QFormLayout.WrapAllRows); self.adv_form.setVerticalSpacing(10)
        self.adv_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.adv_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.adv_widget.hide()
        has_adv = False
        for p in nt.params:
            if p.kind in ("table_columns", "table_rows", "blocks"):     # edited on the main page
                continue
            w = make_widget(p, nt.key, suggest=self._suggest)
            self.widgets[p.name] = w
            w.set_value(node.params.get(p.name, p.default))
            w.changed.connect(lambda name=p.name, w=w: self._on_edit(name, w))
            target = self.adv_form if p.advanced else self.form
            if isinstance(w, BoolWidget):
                target.addRow(w); self._labels[p.name] = w
            else:
                lab = QLabel(p.label + (" *" if p.required else ""))
                if p.help:
                    tip = f"<p style='margin:0'>{html.escape(p.help)}</p>"      # rich text wraps; plain text is clipped
                    lab.setToolTip(tip); w.setToolTip(tip)
                    lab.setText(lab.text() + f"  <span style='color:{T.faint}'>ⓘ</span>")
                target.addRow(lab, w)
                self._labels[p.name] = lab
            if p.advanced:
                has_adv = True
            if isinstance(w, PathWidget):
                w.base_dir = self.doc.pipeline.directory
        if has_adv:
            lay.addWidget(self.adv_btn); lay.addWidget(self.adv_widget)
        self.problems_label = QLabel(); self.problems_label.setWordWrap(True); self.problems_label.setObjectName("error"); self.problems_label.hide()
        lay.addWidget(self.problems_label)
        lay.addStretch()
        self.scroll.setWidget(body)
        self._refresh_schema()
        self._refresh_status()
        self._apply_visibility()

    def _empty_state(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 8, 0, 0); v.setSpacing(6)
        t = QLabel("Nothing selected"); t.setObjectName("heading"); v.addWidget(t)
        body = QLabel("Pick a table, chart or report on the left and its settings appear here.<br><br>"
                      "To start, open a data file, or drop one anywhere on this window. "
                      "Right-click a column header in the table to filter, sort, chart or check it.")
        body.setWordWrap(True); body.setObjectName("muted"); v.addWidget(body)
        return w

    def _rename(self) -> None:
        if self.nid and self.nid in self.doc.pipeline.nodes:
            nt = registry.get(self.doc.pipeline.nodes[self.nid].type)
            self.doc.rename(self.nid, self.title_edit.text().strip() or nt.label)

    def focus_first_field(self) -> None:
        if self.nid is None:
            return
        nt = registry.get(self.doc.pipeline.nodes[self.nid].type)
        shown = [p for p in nt.params if not p.advanced and p.name in self.widgets and self._labels.get(p.name) is not None
                 and self._labels[p.name].isVisibleTo(self)]
        for p in shown:                              # first field still waiting for a value
            w = self.widgets[p.name]
            if w.is_blank() and p.kind != "bool":
                w.focus_entry(); return
        for p in shown:
            w = self.widgets[p.name]
            if p.required or p.kind in ("path", "columns", "formulas", "conditions", "column", "series"):
                w.focus_entry(); return

    # ------------------------------------------------------------ schema / suggestions
    def _refresh_schema(self) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            return
        nid = self.nid
        node = self.doc.pipeline.nodes[nid]
        nt = registry.get(node.type)
        executor = self.doc.executor
        first = nt.inputs[0].name if nt.inputs else None
        path = self.widgets["path"].value() if "path" in self.widgets else ""
        pipeline_dir = self.doc.pipeline.directory

        def work():
            if first is not None:
                schemas = executor.input_schemas(nid)
                own = schemas.get(first)
            else:
                schemas, own = {}, executor.schema(nid)
            sheets = list_sheets(pipeline_dir / path) if path and Path(path).suffix.lower() in EXCEL_EXT else None
            return nid, own, schemas, sheets

        def done(r):
            rnid, own, schemas, sheets = r
            if rnid != self.nid:
                return
            for name, w in self.widgets.items():
                w.set_schema(own, schemas)
                if isinstance(w, SheetWidget):
                    if sheets is not None:
                        w.set_sheets(sheets); self._labels[name].show(); w.show()
                    else:
                        self._labels[name].hide(); w.hide()
            self._apply_visibility()

        self._schema_serial.submit(work, done)          # a failure is logged with its traceback by the task

    def _suggest(self, column: str) -> list[str]:
        """Distinct text values for the filter dropdown. Served from a cache filled on a worker;
        the first call for a column returns [] and refills the row when the values arrive."""
        try:
            ups = [s for srcs in self.doc.pipeline.inputs_of(self.nid).values() for s in srcs]
            if not ups:
                return []
            st = self.doc.state(ups[0])
            if st.status != "done" or not st.output:
                return []
            key = (st.output, column)
            cache = self._suggest_cache
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            cache[key] = []
            while len(cache) > self.SUGGEST_CACHE_SIZE:
                cache.popitem(last=False)
            output, nid = st.output, self.nid

            def work():
                lf = pl.scan_parquet(output)
                schema = dict(lf.collect_schema())
                if column not in schema or kind_of(schema, column) not in ("text", "true/false"):
                    return key, []
                vals = lf.select(pl.col(column).cast(pl.Utf8)).head(50_000).unique().head(40).collect(engine="streaming")[column].drop_nulls().to_list()
                return key, sorted(vals)

            def done(r):
                k, vals = r
                if k in cache:
                    cache[k] = vals
                if nid == self.nid and "conditions" in self.widgets:
                    for row in self.widgets["conditions"].rows:
                        if row.col.column() == column and vals:
                            cur = row.val.currentText()
                            row.val.blockSignals(True); row.val.clear(); row.val.addItems(vals); row.val.setCurrentText(cur); row.val.blockSignals(False)
            t = Task(work); t.waits_for_run = False; t.signals.done.connect(done)
            self._suggest_tasks.add(t)
            t.signals.finished.connect(lambda t=t: self._suggest_tasks.discard(t))
            view_pool().start(t)
            return []
        except Exception:
            return []

    # ------------------------------------------------------------ editing
    def _on_edit(self, name: str, w: ParamWidget) -> None:
        self._pending[name] = w.value()
        self._apply_visibility()
        if w.immediate:
            self._commit()
        else:
            self._timer.start()

    def _commit(self) -> None:
        if not self._pending or self.nid is None or self.nid not in self.doc.pipeline.nodes:
            self._pending.clear()
            return
        changes, self._pending = self._pending, {}
        self._committing = True
        try:
            self.doc.set_params(self.nid, changes)
        except ValueError as e:
            self.problems_label.setText(str(e)); self.problems_label.show()
        finally:
            self._committing = False
        if "path" in changes:
            self._refresh_schema()

    def _on_node_changed(self, nid: str) -> None:
        if nid != self.nid or nid not in self.doc.pipeline.nodes:
            return
        node = self.doc.pipeline.nodes[nid]
        if self.title_edit.text() != node.title:
            self.title_edit.setText(node.title)
        if not self._committing:
            for name, w in self.widgets.items():
                w.blockSignals(True)
                try:
                    w.set_value(node.params.get(name))
                finally:
                    w.blockSignals(False)
        self._apply_visibility()
        self._refresh_status()

    def _apply_visibility(self) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            return
        node = self.doc.pipeline.nodes[self.nid]
        current = dict(node.params); current.update(self._pending)
        nt = registry.get(node.type)
        for p in nt.params:
            vis = p.is_visible(current)
            w = self.widgets.get(p.name)
            if w is None or isinstance(w, SheetWidget):
                continue
            w.setVisible(vis); self._labels[p.name].setVisible(vis)
        probs = [x for x in (w.problem() for w in self.widgets.values()) if x]
        self.problems_label.setText("\n".join(probs)); self.problems_label.setVisible(bool(probs))

    def _toggle_advanced(self, on: bool) -> None:
        self.adv_widget.setVisible(on)
        self.adv_btn.setText("Fewer options" if on else "More options")

    # ------------------------------------------------------------ status
    def _refresh_status(self) -> None:
        if self.nid is None or not hasattr(self, "status_label") or self.nid not in self.doc.pipeline.nodes:
            return
        st = self.doc.state(self.nid)
        if st.status == "done":
            txt = f"{st.rows:,} rows × {len(st.columns)} columns" + (f" · {st.elapsed:.1f} s" if st.elapsed and not st.from_cache else "")
        elif st.status == "failed":
            txt = f"<b style='color:{T.danger}'>Failed</b><br>{st.error}"
        elif st.status == "running":
            txt = "Running…"
        elif st.status == "stale":
            txt = "Settings changed — run again to update"
        else:
            txt = "Not run yet"
        self.status_label.setText(status_dot(st.status, txt))
        self.finding.set_finding(self.doc.pipeline.nodes[self.nid].type, st, self.doc.pipeline.column_title)
        self.run_btn.setEnabled(not self.doc.running)

    def show_problems(self, probs: list[str]) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            return
        mine = [p.split(":", 1)[1].strip() for p in probs if p.startswith(self.doc.pipeline.nodes[self.nid].title + ":")]
        if mine:
            self.problems_label.setText("\n".join(mine)); self.problems_label.show()
