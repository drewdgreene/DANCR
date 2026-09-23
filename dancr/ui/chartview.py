"""Centre page for a chart: chips above, an interactive plot below. Edits a chart node's settings.
With "Split by" the chart becomes one panel per category value, stacked with a shared X axis."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QToolButton, QFileDialog, QStackedLayout, QMenu, QLineEdit, QInputDialog

from ..core.expr import _kind_of_dtype, NUM, TIME, STR, BOOL
from ..core.executor import Executor
from ..core.fits import KINDS as FIT_KINDS
from ..views import lod
from ..views.chartquery import ChartData, ChartError, query_panels, limit_values, MAX_PANELS
from .document import Document
from .theme import T, SERIES_COLORS
from .common import page_header
from .workers import Serial
from .icons import icon
from .flow import FlowLayout

pg.setConfigOptions(antialias=False, useOpenGL=False)
KIND_LABELS = [("line", "Line"), ("scatter", "Scatter"), ("histogram", "Histogram"), ("bar", "Bar")]


class Chip(QToolButton):
    def __init__(self, icon_name: str, text: str) -> None:
        super().__init__()
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setIcon(icon(icon_name, T.muted, 14)); self.setText(text)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setStyleSheet(f"QToolButton {{ border: 1px solid {T.border}; border-radius: 12px; padding: 3px 10px 3px 8px; background: {T.panel}; }}"
                           f"QToolButton:hover {{ border-color: {T.accent}; }} QToolButton::menu-indicator {{ image: none; }}")


@dataclass
class Panel:
    """One plot area: the PlotItem plus what is drawn in it."""
    plot: pg.PlotItem
    legend: Any
    hover: pg.TextItem
    vline: pg.InfiniteLine
    items: list = field(default_factory=list)
    series: list = field(default_factory=list)      # (label, xs, ys) for hover readouts
    date_axis: bool = False
    utc_offset: int = 0                             # seconds east of UTC shown on the axis (the column's own zone)
    tz_aware: bool = False


def _log10(v: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.asarray(v, dtype=float))


def _zone_offset(tz: str, at: float | None) -> int:
    """Seconds east of UTC for zone ``tz`` at epoch second ``at`` (now when unknown)."""
    try:
        when = datetime.fromtimestamp(at if at is not None and math.isfinite(at) else time.time(), timezone.utc)
        return int(when.astimezone(ZoneInfo(tz)).utcoffset().total_seconds())
    except (ValueError, OSError, OverflowError, KeyError):
        return 0


class ChartView(QWidget):
    addToReport = Signal(str)

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nid: str | None = None          # the chart node
        self.src: str | None = None          # its input table
        self.lf: pl.LazyFrame | None = None
        self.schema: dict[str, pl.DataType] = {}
        self._schema_src: str | None = None  # the table ``schema`` describes
        self.rows = 0
        self.preview = False
        self._serial = Serial()
        self._suppress = False
        self._last_range: tuple[float, float] | None = None
        self._bounds: dict[str, tuple[float, float, int]] = {}
        self._src_hash: str | None = None
        self.panels: list[Panel] = []
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header(spacing=6)
        self.title = QLineEdit(); self.title.setPlaceholderText("Chart title"); self.title.setObjectName("heading")
        self.title.setStyleSheet("QLineEdit { border: none; background: transparent; font-weight: 600; font-size: 12pt; padding: 0; }")
        self.title.editingFinished.connect(self._title_changed)
        self.source = QLabel(""); self.source.setObjectName("muted")
        self.copy_btn = QToolButton(); self.copy_btn.setObjectName("quiet"); self.copy_btn.setIcon(icon("copy", T.muted, 16)); self.copy_btn.setToolTip("Copy chart image"); self.copy_btn.clicked.connect(self.copy_image)
        self.save_btn = QToolButton(); self.save_btn.setObjectName("quiet"); self.save_btn.setIcon(icon("image", T.muted, 16)); self.save_btn.setToolTip("Save chart as image…"); self.save_btn.clicked.connect(self.save_image)
        self.report_btn = QToolButton(); self.report_btn.setObjectName("quiet"); self.report_btn.setIcon(icon("article", T.muted, 16)); self.report_btn.setToolTip("Add this chart to a report"); self.report_btn.clicked.connect(lambda: self.addToReport.emit(self.nid))
        h.addWidget(self.title, 1); h.addWidget(self.source); h.addWidget(self.copy_btn); h.addWidget(self.save_btn); h.addWidget(self.report_btn)
        lay.addWidget(head)
        chips = QWidget(); c = FlowLayout(chips, margin=0, h_space=6, v_space=4); c.setContentsMargins(10, 6, 10, 6)
        self.kind_chip = Chip("chart-line", "Line"); self.x_chip = Chip("arrows-out-line-horizontal", "X"); self.y_chip = Chip("trend-up", "Y")
        self.color_chip = Chip("dots-three", "Colour by"); self.split_chip = Chip("rows", "Split by")
        self.limit_chip = Chip("check-circle", "Limit line"); self.fit_chip = Chip("chart-scatter", "Fitted curve")
        self.mean_chip = Chip("wave-sine", "Average line"); self.mean_chip.setCheckable(True); self.mean_chip.setPopupMode(QToolButton.DelayedPopup)
        self.mean_chip.clicked.connect(lambda on: self._set({"mean_line": bool(on)}))
        self.fit_btn = QToolButton(); self.fit_btn.setObjectName("quiet"); self.fit_btn.setIcon(icon("arrows-out", T.muted, 16)); self.fit_btn.setToolTip("Zoom out to everything"); self.fit_btn.clicked.connect(self.fit)
        self.info = QLabel(""); self.info.setObjectName("muted")
        for w in (self.kind_chip, self.x_chip, self.y_chip, self.color_chip, self.split_chip, self.limit_chip, self.fit_chip, self.mean_chip):
            c.addWidget(w)
        c.addWidget(self.fit_btn); c.addWidget(self.info)
        lay.addWidget(chips)
        holder = QWidget(); self.stack = QStackedLayout(holder); self.stack.setStackingMode(QStackedLayout.StackAll)
        self.gl = pg.GraphicsLayoutWidget(); self.gl.setBackground(T.panel)
        self.gl.ci.layout.setSpacing(4)
        self.title_item = self.gl.addLabel("", row=0, col=0, color=T.text, size="11pt")
        self.overlay = QLabel(""); self.overlay.setAlignment(Qt.AlignCenter); self.overlay.setWordWrap(True)
        self.overlay.setStyleSheet(f"QLabel {{ color: {T.muted}; background: transparent; font-size: 11pt; }}"); self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.stack.addWidget(self.gl); self.stack.addWidget(self.overlay); self.stack.setCurrentWidget(self.overlay)
        lay.addWidget(holder, 1)
        self._ensure_panels(1)
        self.gl.scene().sigMouseMoved.connect(self._mouse_moved)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(90); self._timer.timeout.connect(lambda: self._query(full=False))
        self.hint = QLabel("Drag to pan · scroll to zoom · right-drag to zoom into a box · double-click to fit · click a legend entry to hide it"); self.hint.setObjectName("faint"); self.hint.setWordWrap(True)
        self.hint.setContentsMargins(10, 2, 10, 3); lay.addWidget(self.hint)
        doc.nodeChanged.connect(lambda nid: self.refresh() if nid == self.nid else None)
        doc.statesChanged.connect(self._maybe_refresh)
        doc.inputsChanged.connect(self.refresh)
        doc.columnsChanged.connect(self.refresh)
        doc.runFinished.connect(lambda ok, r: self.refresh())
        doc.reloaded.connect(self.clear)

    # ------------------------------------------------------------ panels
    @property
    def plot(self) -> pg.PlotItem:
        return self.panels[0].plot

    def _ensure_panels(self, n: int) -> None:
        n = max(1, min(n, MAX_PANELS))
        while len(self.panels) > n:
            p = self.panels.pop()
            self.gl.removeItem(p.plot)
        while len(self.panels) < n:
            i = len(self.panels)
            plot = self.gl.addPlot(row=i + 1, col=0)
            plot.showGrid(x=True, y=True, alpha=0.12); plot.setClipToView(True)
            for ax in ("left", "bottom"):
                plot.getAxis(ax).setTextPen(QColor(T.muted)); plot.getAxis(ax).setPen(QColor(T.border))
            legend = plot.addLegend(offset=(10, 10), brush=pg.mkBrush(QColor(T.panel).name() + "e6"), pen=pg.mkPen(T.border), labelTextColor=QColor(T.text))
            hover = pg.TextItem("", anchor=(0, 1), color=T.text, fill=pg.mkBrush(T.panel)); hover.setZValue(100); plot.addItem(hover, ignoreBounds=True); hover.hide()
            vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(T.faint, width=1, style=Qt.DashLine)); vline.setZValue(90); plot.addItem(vline, ignoreBounds=True); vline.hide()
            if i == 0:
                plot.getViewBox().sigXRangeChanged.connect(self._range_changed)
            else:
                plot.setXLink(self.panels[0].plot)
            self.panels.append(Panel(plot, legend, hover, vline))

    def _clear_items(self) -> None:
        for p in self.panels:
            for it in p.items:
                p.plot.removeItem(it)
            p.items = []; p.series = []; p.legend.clear()

    def _set_date_axis(self, p: Panel, on: bool, tz: str | None = None, at: float | None = None) -> None:
        """A date axis in the column's own time zone (naive columns are shown as they are)."""
        offset = _zone_offset(tz, at) if (on and tz) else 0
        if on != p.date_axis or offset != p.utc_offset:
            p.date_axis, p.utc_offset, p.tz_aware = on, offset, bool(on and tz)
            axis = pg.DateAxisItem(orientation="bottom", utcOffset=-offset) if on else pg.AxisItem(orientation="bottom")
            axis.setTextPen(QColor(T.muted)); axis.setPen(QColor(T.border))
            p.plot.setAxisItems({"bottom": axis})

    def _add(self, p: Panel, item) -> None:
        p.plot.addItem(item); p.items.append(item)

    def _add_curve(self, p: Panel, xs, ys, color, name, symbols=False, width=1.0):
        if symbols:
            item = pg.PlotDataItem(xs, ys, pen=pg.mkPen(color, width=1.3), symbol="o", symbolSize=3.5, symbolPen=None, symbolBrush=color, name=name, connect="finite")
        else:
            item = pg.PlotDataItem(xs, ys, pen=pg.mkPen(color, width=width), name=name, connect="finite")
        self._add(p, item)
        return item

    def _add_hline(self, p: Panel, value: float, log_y: bool, pen, label: str, color: str, position: float, notes: list[str]) -> None:
        """A horizontal reference line; on a log axis it sits at log10(value), or is left out when value <= 0."""
        if log_y:
            if value <= 0:
                notes.append(f"{label} is not on the log axis (≤ 0)"); return
            value = math.log10(value)
        self._add(p, pg.InfiniteLine(pos=value, angle=0, pen=pen, label=label, labelOpts={"color": color, "position": position}))

    # ------------------------------------------------------------ node binding
    def set_node(self, nid: str | None) -> None:
        self.nid = nid if nid in self.doc.pipeline.nodes else None
        self.refresh()

    def _params(self) -> dict[str, Any]:
        return dict(self.doc.pipeline.nodes[self.nid].params) if self.nid else {}

    def _set(self, changes: dict[str, Any]) -> None:
        if self.nid:
            self.doc.set_params(self.nid, changes)

    def _title_changed(self) -> None:
        if self.nid is None:
            return
        text = self.title.text().strip()
        node = self.doc.pipeline.nodes[self.nid]
        if text != (node.params.get("title") or "") and text != node.title:
            self._set({"title": text})
        if text and text != node.title:
            self.doc.rename(self.nid, text)

    def _maybe_refresh(self) -> None:
        if self.nid and self.src:
            st = self.doc.state(self.src)
            if st.hash != self._src_hash:
                self.refresh()

    def refresh(self) -> None:
        if self.nid is None or self.nid not in self.doc.pipeline.nodes:
            self.clear(); return
        node = self.doc.pipeline.nodes[self.nid]
        ins = self.doc.pipeline.inputs_of(self.nid).get("in") or []
        self.src = ins[0] if ins else None
        if self.src != self._schema_src:            # never show another table's columns in the chips
            self.schema = {}; self.lf = None; self._schema_src = self.src
        self.title.setText(node.params.get("title") or node.title); self.title.setCursorPosition(0)
        self._fill_chips()
        if not self.src:
            self._set_overlay("Connect a table to this chart (use 'Chart this' on a column)"); return
        st = self.doc.state(self.src)
        self._src_hash = st.hash
        self.source.setText(f"from {self.doc.pipeline.nodes[self.src].title}" + (" · preview" if st.status != "done" else ""))
        nid, src = self.nid, self.src
        if st.status == "done" and st.output:
            output, rows = st.output, st.rows or 0
            self._set_overlay("")

            def open_result():                      # Parquet footer read: off the GUI thread
                lf = pl.scan_parquet(output)
                return lf, dict(lf.collect_schema())

            def ready(r):
                if nid != self.nid or src != self.src or self.doc.state(src).output != output:
                    return
                lf, schema = r
                self._got_frame(lf, schema, rows, preview=False)
            self._serial.submit(open_result, ready, self._set_overlay)
        else:
            self._set_overlay("Waiting for the run to finish…" if self.doc.running else "Building a preview…")

            def work():
                df, _res, _kind = self.doc.executor.preview(src, Executor.PREVIEW_ROWS)
                return df

            def done(df):
                if nid != self.nid or src != self.src:
                    return
                self._got_frame(df.lazy(), dict(df.schema), len(df), preview=True)
            self._serial.submit(work, done, self._set_overlay)

    def _got_frame(self, lf: pl.LazyFrame, schema: dict, rows: int, preview: bool) -> None:
        self.lf = lf; self.rows = rows; self.preview = preview; self.schema = schema; self._schema_src = self.src
        self._bounds = {}; self._last_range = None
        self._fill_chips()
        self._query(full=True)

    def clear(self) -> None:
        self.nid = None; self.src = None; self.lf = None; self.schema = {}; self._schema_src = None
        self.rows = 0; self._bounds = {}; self._last_range = None; self._src_hash = None
        self._serial.cancel()
        self._clear_items(); self._ensure_panels(1); self.info.setText(""); self.source.setText(""); self._set_overlay("")

    def _set_overlay(self, text: str) -> None:
        self.overlay.setText(text); self.overlay.setVisible(bool(text))

    # ------------------------------------------------------------ chips
    def _fill_chips(self) -> None:
        p = self._params()
        kind = p.get("kind", "line")
        self.kind_chip.setText(dict(KIND_LABELS).get(kind, "Line")); self.kind_chip.setIcon(icon({"line": "chart-line", "scatter": "chart-scatter", "histogram": "rows", "bar": "table"}.get(kind, "chart-line"), T.muted, 14))
        m = QMenu(self)
        for k, label in KIND_LABELS:
            a = m.addAction(label); a.setCheckable(True); a.setChecked(k == kind); a.triggered.connect(lambda _=False, k=k: self._set({"kind": k}))
        self.kind_chip.setMenu(m)
        schema = self.schema                         # filled by the worker; empty until the table has been read
        nums = [c for c, dt in schema.items() if _kind_of_dtype(dt) == NUM]
        axes = [c for c, dt in schema.items() if _kind_of_dtype(dt) in (NUM, TIME)]
        cats = [c for c, dt in schema.items() if _kind_of_dtype(dt) in (STR, BOOL)] or list(schema)
        title = self.doc.pipeline.column_title
        # X
        xcol = p.get("x") or p.get("category") or ""
        self.x_chip.setText("X: " + (title(xcol) if xcol else "automatic"))
        xm = QMenu(self)
        a = xm.addAction("automatic"); a.triggered.connect(lambda: self._set({"x": "", "category": ""}))
        for c in (axes if kind != "bar" else list(schema)):
            a = xm.addAction(title(c)); a.setCheckable(True); a.setChecked(c == xcol)
            a.triggered.connect(lambda _=False, c=c: self._set({"category": c} if kind == "bar" else {"x": c}))
        self.x_chip.setMenu(xm)
        # Y
        if kind == "histogram":
            ycol = p.get("column") or ""
            self.y_chip.setText("Column: " + (title(ycol) if ycol else "choose"))
            ym = QMenu(self)
            for c in nums:
                a = ym.addAction(title(c)); a.setCheckable(True); a.setChecked(c == ycol); a.triggered.connect(lambda _=False, c=c: self._set({"column": c}))
            self.y_chip.setMenu(ym)
        elif kind == "bar":
            vcol = p.get("value") or ""
            self.y_chip.setText(f"{p.get('stat', 'mean')} of " + (title(vcol) if vcol else "rows"))
            ym = QMenu(self)
            for c in nums:
                a = ym.addAction(title(c)); a.setCheckable(True); a.setChecked(c == vcol); a.triggered.connect(lambda _=False, c=c: self._set({"value": c}))
            ym.addSeparator()
            for stt in ("mean", "sum", "count", "min", "max", "median"):
                a = ym.addAction(f"statistic: {stt}"); a.setCheckable(True); a.setChecked(stt == p.get("stat", "mean")); a.triggered.connect(lambda _=False, s=stt: self._set({"stat": s}))
            self.y_chip.setMenu(ym)
        else:
            series = [s for s in (p.get("series") or []) if s.get("column")]
            chosen = [s["column"] for s in series]
            self.y_chip.setText("Y: " + (", ".join(title(c) for c in chosen[:3]) + (f" +{len(chosen) - 3}" if len(chosen) > 3 else "") if chosen else "choose"))
            ym = QMenu(self)
            for c in nums:
                a = ym.addAction(title(c)); a.setCheckable(True); a.setChecked(c in chosen)
                a.triggered.connect(lambda on, c=c: self._toggle_series(c, on))
            if series:
                ym.addSeparator()
                a = ym.addAction("Rename series for the legend…"); a.triggered.connect(self._rename_series)
            self.y_chip.setMenu(ym)
        # colour by / split by
        cb = p.get("color_by") or ""
        self.color_chip.setText("Colour by: " + (title(cb) if cb else "none")); self.color_chip.setVisible(kind in ("line", "scatter"))
        cm = QMenu(self)
        a = cm.addAction("none"); a.triggered.connect(lambda: self._set({"color_by": ""}))
        for c in cats:
            a = cm.addAction(title(c)); a.setCheckable(True); a.setChecked(c == cb); a.triggered.connect(lambda _=False, c=c: self._set({"color_by": c}))
        self.color_chip.setMenu(cm)
        sb = p.get("split_by") or ""
        self.split_chip.setText("Split by: " + (title(sb) if sb else "none")); self.split_chip.setVisible(kind != "bar")
        self.split_chip.setToolTip("One panel per value of a category column, stacked with a shared X axis")
        sm = QMenu(self)
        a = sm.addAction("none"); a.triggered.connect(lambda: self._set({"split_by": ""}))
        for c in cats:
            a = sm.addAction(title(c)); a.setCheckable(True); a.setChecked(c == sb); a.triggered.connect(lambda _=False, c=c: self._set({"split_by": c}))
        self.split_chip.setMenu(sm)
        # limits
        lims = p.get("limits") or []
        self.limit_chip.setText("Limit line" + (f": {', '.join(str(l.get('value')) for l in lims)}" if lims else "")); self.limit_chip.setVisible(kind != "bar")
        lm = QMenu(self)
        a = lm.addAction("Add a limit line…"); a.triggered.connect(self._add_limit)
        for i, l in enumerate(lims):
            a = lm.addAction(f"Remove {l.get('label') or l.get('value')}"); a.triggered.connect(lambda _=False, i=i: self._set({"limits": [x for j, x in enumerate(lims) if j != i]}))
        self.limit_chip.setMenu(lm)
        # fit
        fk = p.get("fit") or ""
        self.fit_chip.setText("Fitted curve" + (f": {dict(FIT_KINDS).get(fk, fk)}" if fk else "")); self.fit_chip.setVisible(kind == "scatter")
        fm = QMenu(self)
        a = fm.addAction("none"); a.triggered.connect(lambda: self._set({"fit": ""}))
        for k, label in FIT_KINDS:
            a = fm.addAction(label); a.setCheckable(True); a.setChecked(k == fk); a.triggered.connect(lambda _=False, k=k: self._set({"fit": k}))
        self.fit_chip.setMenu(fm)
        self.mean_chip.setChecked(bool(p.get("mean_line"))); self.mean_chip.setVisible(kind in ("line", "scatter"))

    def _toggle_series(self, col: str, on: bool) -> None:
        series = [s for s in (self._params().get("series") or []) if s.get("column")]
        if on and col not in [s["column"] for s in series]:
            series.append({"column": col, "color": SERIES_COLORS[len(series) % len(SERIES_COLORS)]})
        elif not on:
            series = [s for s in series if s["column"] != col]
        self._set({"series": series})

    def _rename_series(self) -> None:
        series = [dict(s) for s in (self._params().get("series") or []) if s.get("column")]
        for s in series:
            text, ok = QInputDialog.getText(self, "Legend name", f"Name shown for {s['column']}:", text=s.get("label") or self.doc.pipeline.column_title(s["column"]))
            if ok:
                s["label"] = text.strip()
        self._set({"series": series})

    def _add_limit(self) -> None:
        value, ok = QInputDialog.getText(self, "Limit line", "Value (a number or the name of an input):")
        if not ok or not value.strip():
            return
        label, ok = QInputDialog.getText(self, "Limit line", "Label:", text=f"limit {value.strip()}")
        lims = list(self._params().get("limits") or []) + [{"value": value.strip(), "label": label.strip() if ok else value.strip()}]
        self._set({"limits": lims})

    # ------------------------------------------------------------ querying
    def _range_changed(self, *_: Any) -> None:
        if self._suppress or self.lf is None or self._params().get("kind", "line") == "bar":
            return
        self._timer.start()

    def fit(self) -> None:
        self._last_range = None; self._query(full=True)

    def _query(self, full: bool) -> None:
        if self.lf is None or self.nid is None:
            return
        spec = self._params()
        vb = self.plot.getViewBox()
        width = max(200, vb.width() or 800); height = max(150, vb.height() or 400)
        x_range = None
        if not full:
            (x0, x1), _ = vb.viewRange()
            x_range = (float(x0), float(x1))
            if self._last_range and abs(x0 - self._last_range[0]) < 1e-9 and abs(x1 - self._last_range[1]) < 1e-9:
                return
        self._last_range = x_range
        lf, nid, schema, bounds = self.lf, self.nid, self.schema, self._bounds
        inputs = self.doc.pipeline.input_values()
        t0 = time.perf_counter()
        self.info.setText("waiting for the run…" if self.doc.running else f"drawing {self.rows:,} rows…")
        height = height * max(1, len(self.panels))          # the view box is one panel; queries are sized per panel

        def work():
            try:
                return query_panels(lf, schema, spec, x_range=x_range, width_px=int(width * 1.5), height_px=int(height), bounds=bounds), None
            except ChartError as e:
                return [], str(e)

        def done(r):
            if nid != self.nid:
                return
            panels, error = r
            self._render(panels, error, full, time.perf_counter() - t0, spec, inputs)

        self._serial.submit(work, done, lambda m: (self.info.setText(""), self._set_overlay(m)) if nid == self.nid else None)

    # ------------------------------------------------------------ render
    def _render(self, panels: list[tuple[str | None, ChartData]], error: str | None, full: bool, secs: float, spec: dict, inputs: dict) -> None:
        self._suppress = True
        try:
            self._clear_items()
            if error:
                self._ensure_panels(1)
                self.plot.setLabel("bottom", "")
                self.info.setText(""); self._set_overlay(error); return
            self._set_overlay("")
            self._ensure_panels(len(panels))
            infos, notes = [], []
            for i, ((label, cd), p) in enumerate(zip(panels, self.panels)):
                infos.append(self._render_panel(p, cd, full, spec, inputs, label, show_x_label=(i == len(panels) - 1), notes=notes))
            info = infos[-1] if len(infos) == 1 else f"{len(panels)} panels · " + infos[0].split(" · ")[0]
            info += f" · {secs * 1000:.0f} ms" + (" · preview" if self.preview else "")
            if notes:
                info += " · " + "; ".join(dict.fromkeys(notes))
            self.info.setText(info)
            self.title_item.setText(spec.get("title") or self.doc.pipeline.nodes[self.nid].title)
        finally:
            self._suppress = False

    def _render_panel(self, p: Panel, cd: ChartData, full: bool, spec: dict, inputs: dict, panel_label: str | None, show_x_label: bool, notes: list[str]) -> str:
        """Draw one queried panel; returns the info text. In log mode pyqtgraph only transforms curves
        (PlotDataItem), so scatter points, bars and reference lines are placed at log10 here."""
        plot = p.plot
        title = self.doc.pipeline.column_title
        log_y = bool(spec.get("log_y"))
        plot.setTitle(panel_label if panel_label is not None else None, color=T.muted, size="9.5pt")
        plot.setLogMode(y=log_y)
        mean_pen = pg.mkPen("#555", width=1, style=Qt.DashLine)
        limit_pen = pg.mkPen(T.danger, width=1.2, style=Qt.DotLine)
        x_tz = self.schema[cd.x].time_zone if (cd.x and isinstance(self.schema.get(cd.x), pl.Datetime)) else None
        if cd.kind == "line":
            if cd.groups:
                for gi, (g, d) in enumerate(cd.groups):
                    for s in d.series:
                        if len(s.x) == 0:
                            continue
                        xs, ys = lod.break_gaps(s.x, s.y) if spec.get("break_gaps", True) else (s.x, s.y)
                        self._add_curve(p, xs, ys, SERIES_COLORS[gi % len(SERIES_COLORS)], g, symbols=(s.mode == "raw" and len(xs) <= 1500))
                        p.series.append((g, s.x, s.y))
            else:
                specs = spec.get("series") or []
                for i, s in enumerate(cd.line.series):
                    sp = specs[i] if i < len(specs) else {}
                    color = sp.get("color") or SERIES_COLORS[i % len(SERIES_COLORS)]
                    label = sp.get("label") or title(s.name)
                    if len(s.x) == 0:
                        continue
                    xs, ys = lod.break_gaps(s.x, s.y) if spec.get("break_gaps", True) else (s.x, s.y)
                    self._add_curve(p, xs, ys, color, label, symbols=(s.mode == "raw" and len(xs) <= 1500))
                    p.series.append((label, s.x, s.y))
            if cd.mean is not None:
                self._add_hline(p, cd.mean, log_y, mean_pen, f"average {cd.mean:.4g}", T.muted, 0.05, notes)
            axis = cd.axis
            self._set_date_axis(p, axis.kind == "time", x_tz, axis.lo)
            plot.setLabel("bottom", (title(axis.column) if axis.column else "row") if show_x_label else "")
            plot.setLabel("left", spec.get("y_label") or (title(cd.ys[0]) if len(cd.ys) == 1 else ""))
            if full and axis.lo is not None and axis.hi is not None and axis.hi > axis.lo and p is self.panels[0]:
                plot.setXRange(axis.lo, axis.hi, padding=0.01)
            plot.enableAutoRange(axis="y")
        elif cd.kind == "scatter":
            d = cd.scatter
            at = (d.x[0] if d.mode == "raw" and d.x is not None and len(d.x) else (d.extent[0] if d.extent else None))
            self._set_date_axis(p, d.x_kind == "time", x_tz, at)
            yname = (spec.get("series") or [{}])[0]
            plot.setLabel("bottom", title(cd.x) if show_x_label else ""); plot.setLabel("left", spec.get("y_label") or yname.get("label") or title(cd.ys[0]))
            if cd.groups:
                for gi, (g, gd) in enumerate(cd.groups):
                    if gd.mode == "raw":
                        self._add_points(p, gd.x, gd.y, 5, SERIES_COLORS[gi % len(SERIES_COLORS)], g, log_y, notes)
            elif d.mode == "raw":
                self._add_points(p, d.x, d.y, 4, SERIES_COLORS[0], None, log_y, notes)
            elif log_y:
                notes.append("the density view cannot be drawn on a log axis: zoom in to see the points")
            else:
                x0, x1, y0, y1 = d.extent
                img = pg.ImageItem(np.log1p(d.density))
                lut = pg.colormap.get("viridis").getLookupTable(nPts=256, alpha=True); lut[0, 3] = 0
                img.setLookupTable(lut)
                img.setRect(pg.QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)); self._add(p, img)
            for gi, (f, cx, cy) in enumerate(cd.fits):
                if cx is None:
                    self._set_overlay(f"Fit: {f}")
                else:
                    lab = (f"{f.group}: " if f.group is not None else "") + f.equation + (f"   R² = {f.r2:.3f}" if f.r2 is not None else "")
                    self._add_curve(p, cx, cy, SERIES_COLORS[gi % len(SERIES_COLORS)] if cd.groups else T.text, lab, width=2.0)
            if cd.mean is not None:
                self._add_hline(p, cd.mean, log_y, mean_pen, f"average {cd.mean:.4g}", T.muted, 0.05, notes)
            if full:
                plot.autoRange()
        elif cd.kind == "hist":
            h = cd.hist
            self._set_date_axis(p, False)
            plot.setLabel("bottom", title(cd.ys[0]) if show_x_label else ""); plot.setLabel("left", "count")
            self._add_bars(p, h.edges[:-1], np.diff(h.edges), h.counts, log_y, notes, pen=pg.mkPen(T.panel))
            for lv, lab in limit_values(spec, inputs):
                self._add(p, pg.InfiniteLine(pos=lv, angle=90, pen=limit_pen, label=lab, labelOpts={"color": T.danger, "position": 0.9}))
            if full:
                if p is self.panels[0]:
                    plot.setXRange(h.lo, h.hi, padding=0.02)
                plot.enableAutoRange(axis="y")
        else:
            b = cd.bar
            self._set_date_axis(p, False)
            xs = np.arange(len(b.labels))
            self._add_bars(p, xs - 0.4, np.full(len(xs), 0.8), b.values, log_y, notes)
            plot.getAxis("bottom").setTicks([[(i, lab[:18]) for i, lab in enumerate(b.labels)]])
            plot.setLabel("left", f"{b.stat} of {title(cd.ys[0]) if cd.ys else 'rows'}"); plot.setLabel("bottom", title(cd.x))
            plot.autoRange()
        if cd.kind in ("line", "scatter"):
            for lv, lab in limit_values(spec, inputs):
                self._add_hline(p, lv, log_y, limit_pen, lab, T.danger, 0.95, notes)
        return cd.summary()

    def _add_points(self, p: Panel, x, y, size: int, color: str, name: str | None, log_y: bool, notes: list[str]) -> None:
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if log_y:
            keep = y > 0
            if not keep.all():
                notes.append(f"{int((~keep).sum()):,} points ≤ 0 are not on the log axis")
            x, y = x[keep], _log10(y[keep])
        self._add(p, pg.ScatterPlotItem(x, y, size=size, pen=None, brush=pg.mkBrush(color), name=name))

    def _add_bars(self, p: Panel, x0, width, values, log_y: bool, notes: list[str], pen=None) -> None:
        """Bars from 0 to the value, or on a log axis from half a decade below the smallest positive value."""
        values = np.asarray(values, dtype=float)
        if not log_y:
            self._add(p, pg.BarGraphItem(x0=x0, width=width, height=values, brush=SERIES_COLORS[0], pen=pen)); return
        pos = values > 0
        if not pos.any():
            notes.append("nothing to show on the log axis (no value above 0)"); return
        if not pos.all():
            notes.append(f"{int((~pos).sum()):,} bars ≤ 0 are not on the log axis")
        base = math.log10(values[pos].min()) - 0.5
        self._add(p, pg.BarGraphItem(x0=np.asarray(x0)[pos], width=np.asarray(width)[pos], y0=base, height=_log10(values[pos]) - base, brush=SERIES_COLORS[0], pen=pen))

    # ------------------------------------------------------------ hover
    def _mouse_moved(self, pos) -> None:
        target = next((p for p in self.panels if p.series and p.plot.sceneBoundingRect().contains(pos)), None)
        if target is None:
            self._hide_hover(); return
        vp = target.plot.getViewBox().mapSceneToView(pos)
        x = vp.x()
        for p in self.panels:
            lines = []
            for label, xs, ys in p.series:
                if len(xs) == 0:
                    continue
                i = int(np.searchsorted(xs, x)); i = min(max(i, 0), len(xs) - 1)
                if i > 0 and abs(xs[i - 1] - x) < abs(xs[i] - x):
                    i -= 1
                lines.append(f"{label}: {ys[i]:.6g}")
            if not lines:
                p.hover.hide(); p.vline.hide(); continue
            when = self._format_time(p, x) if p.date_axis else f"{x:.6g}"
            y = vp.y() if p is target else p.plot.getViewBox().viewRange()[1][1]
            p.hover.setText(when + "\n" + "\n".join(lines[:6])); p.hover.setPos(x, y); p.hover.show()
            p.vline.setPos(x); p.vline.show()

    @staticmethod
    def _format_time(p: Panel, x: float) -> str:
        """Wall time in the column's zone, with its offset when the column carries one (as the grid shows it)."""
        try:
            text = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(x + p.utc_offset))
        except (ValueError, OSError, OverflowError):
            return f"{x:.6g}"
        if p.tz_aware:
            sign = "+" if p.utc_offset >= 0 else "-"
            text += f"{sign}{abs(p.utc_offset) // 3600:02d}{abs(p.utc_offset) % 3600 // 60:02d}"
        return text

    def _hide_hover(self) -> None:
        for p in self.panels:
            p.hover.hide(); p.vline.hide()

    # ------------------------------------------------------------ export
    def _grab(self):
        from pyqtgraph.exporters import ImageExporter
        ex = ImageExporter(self.gl.scene())
        ex.parameters()["width"] = max(1400, self.gl.width() * 2)
        return ex

    def save_image(self) -> None:
        if self.nid is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save chart as image", str(self.doc.pipeline.directory / f"{self.nid}.png"), "PNG image (*.png);;SVG (*.svg)")
        if not f:
            return
        self._hide_hover()
        if f.lower().endswith(".svg"):
            from pyqtgraph.exporters import SVGExporter
            SVGExporter(self.gl.scene()).export(f)
        else:
            self._grab().export(f)
        self.doc.message.emit(f"Saved {f}")

    def copy_image(self) -> None:
        if self.nid is None:
            return
        self._hide_hover()
        self._grab().export(copy=True)
        self.doc.message.emit("Chart image copied — paste it into Word or an email")
