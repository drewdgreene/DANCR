"""Centre page for a map: chips above, an interactive map below. Edits a map node's settings.

Country outlines come from the basemap bundled with the app, so the view never reaches the network. The
data is read and shaped off the GUI thread (like the chart) and only drawn here.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.executor import Executor
from ..core.expr import NUM, kind_of_dtype
from ..core.geo import world_outlines
from ..views.geo_draw import project
from ..views.mapquery import OTHER_COLOR, MapData, MapError, query_map
from ..views.palette import SERIES_COLORS
from .chartview import Chip
from .common import listen, page_header
from .document import Document
from .flow import FlowLayout
from .icons import icon
from .theme import T
from .workers import Serial


class MapView(QWidget):
    addToReport = Signal(str)

    def __init__(self, doc: Document, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.nid: str | None = None
        self.src: str | None = None
        self.lf: pl.LazyFrame | None = None
        self.schema: dict[str, pl.DataType] = {}
        self._schema_src: str | None = None
        self.rows = 0
        self.preview = False
        self._serial = Serial(self)
        self._src_hash: str | None = None
        self._items: list[Any] = []
        self._basemap = None                     # a PolyLineROI-free combined outline item, built once

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head, h = page_header(spacing=6)
        self.title = QLineEdit(); self.title.setPlaceholderText("Map title"); self.title.setObjectName("heading")
        self.title.setStyleSheet("QLineEdit { border: none; background: transparent; font-weight: 600; font-size: 12pt; padding: 0; }")
        self.title.editingFinished.connect(self._title_changed)
        self.source = QLabel(""); self.source.setObjectName("muted")
        self.copy_btn = QToolButton(); self.copy_btn.setObjectName("quiet"); self.copy_btn.setIcon(icon("copy", T.muted, 16)); self.copy_btn.setToolTip("Copy map image"); self.copy_btn.clicked.connect(self.copy_image)
        self.save_btn = QToolButton(); self.save_btn.setObjectName("quiet"); self.save_btn.setIcon(icon("image", T.muted, 16)); self.save_btn.setToolTip("Save map as image…"); self.save_btn.clicked.connect(self.save_image)
        self.report_btn = QToolButton(); self.report_btn.setObjectName("quiet"); self.report_btn.setIcon(icon("article", T.muted, 16)); self.report_btn.setToolTip("Add this map to a report"); self.report_btn.clicked.connect(lambda: self.addToReport.emit(self.nid))
        h.addWidget(self.title, 1); h.addWidget(self.source); h.addWidget(self.copy_btn); h.addWidget(self.save_btn); h.addWidget(self.report_btn)
        lay.addWidget(head)

        chips = QWidget(); c = FlowLayout(chips, margin=0, h_space=6, v_space=4); c.setContentsMargins(10, 6, 10, 6)
        self.lat_chip = Chip("map-pin", "Latitude"); self.lon_chip = Chip("grid-four", "Longitude")
        self.color_chip = Chip("dots-three", "Colour by"); self.size_chip = Chip("trend-up", "Size by")
        self.cell_chip = Chip("grid-four", "Points"); self.basemap_chip = Chip("map-trifold", "Outlines")
        self.basemap_chip.setCheckable(True); self.basemap_chip.setPopupMode(QToolButton.DelayedPopup)
        self.basemap_chip.clicked.connect(lambda on: self._set({"basemap": bool(on)}))
        self.fit_btn = QToolButton(); self.fit_btn.setObjectName("quiet"); self.fit_btn.setIcon(icon("arrows-out", T.muted, 16)); self.fit_btn.setToolTip("Zoom out to everything"); self.fit_btn.clicked.connect(self.fit)
        self.info = QLabel(""); self.info.setObjectName("muted")
        self.lat_menu = QMenu(self); self.lat_chip.setMenu(self.lat_menu)
        self.lon_menu = QMenu(self); self.lon_chip.setMenu(self.lon_menu)
        self.color_menu = QMenu(self); self.color_chip.setMenu(self.color_menu)
        self.size_menu = QMenu(self); self.size_chip.setMenu(self.size_menu)
        self.cell_menu = QMenu(self); self.cell_chip.setMenu(self.cell_menu)
        for w in (self.lat_chip, self.lon_chip, self.color_chip, self.size_chip, self.cell_chip, self.basemap_chip, self.fit_btn, self.info):
            c.addWidget(w)
        lay.addWidget(chips)

        holder = QWidget(); self.stack = QStackedLayout(holder); self.stack.setStackingMode(QStackedLayout.StackAll)
        self.gl = pg.GraphicsLayoutWidget(); self.gl.setBackground(T.panel)
        self.plot = self.gl.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True); self.plot.showGrid(x=True, y=True, alpha=0.12)
        for ax in ("left", "bottom"):
            self.plot.getAxis(ax).setTextPen(T.muted); self.plot.getAxis(ax).setPen(T.border)
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.overlay = QLabel(""); self.overlay.setAlignment(Qt.AlignCenter); self.overlay.setWordWrap(True)
        self.overlay.setStyleSheet(f"QLabel {{ color: {T.muted}; background: transparent; font-size: 11pt; }}")
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hover = pg.TextItem("", anchor=(0, 1), color=T.text, fill=pg.mkBrush(T.panel))
        self._hover.setZValue(100); self.plot.addItem(self._hover, ignoreBounds=True); self._hover.hide()
        self._pts: list[tuple[float, float, str]] = []      # (lon, lat, tooltip) actually drawn, for hover
        self._proj = "equirectangular"
        self.stack.addWidget(self.gl); self.stack.addWidget(self.overlay); self.stack.setCurrentWidget(self.overlay)
        lay.addWidget(holder, 1)
        self.hint = QLabel("Drag to pan · scroll to zoom · double-click to fit · hover a point to read it"); self.hint.setObjectName("faint")
        self.hint.setContentsMargins(10, 2, 10, 3); lay.addWidget(self.hint)
        self.gl.scene().sigMouseMoved.connect(self._mouse_moved)

        listen(self, doc.nodeChanged, lambda nid: self.refresh() if nid == self.nid else None)
        listen(self, doc.statesChanged, self._maybe_refresh)
        listen(self, doc.inputsChanged, self.refresh)
        listen(self, doc.columnsChanged, self.refresh)
        listen(self, doc.runFinished, lambda ok, r: self.refresh())
        listen(self, doc.reloaded, self.clear)
        listen(self, doc.edgeAdded, lambda e: self.refresh() if e.target == self.nid else None)
        listen(self, doc.edgeRemoved, lambda e: self.refresh() if e.target == self.nid else None)

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
        if not text or (text == (node.params.get("title") or "") and text == node.title):
            return
        with self.doc.macro("Map title"):
            if text != (node.params.get("title") or ""):
                self._set({"title": text})
            if text != node.title:
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
        if self.src != self._schema_src:
            self.schema = {}; self.lf = None; self._schema_src = self.src
        self.title.setText(node.params.get("title") or node.title); self.title.setCursorPosition(0)
        self._fill_chips()
        if not self.src:
            self._set_overlay("Connect a table to this map (use 'Map these points' on a location step)"); return
        st = self.doc.state(self.src)
        self._src_hash = st.hash
        self.source.setText(f"from {self.doc.pipeline.nodes[self.src].title}" + (" · preview" if st.status != "done" else ""))
        nid, src = self.nid, self.src
        if st.status == "done" and st.output:
            output, rows = st.output, st.rows or 0
            self._set_overlay("")

            def open_result():
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
            executor = self.doc.snapshot_executor()

            def work():
                try:
                    df, _res, _kind = executor.preview(src, Executor.PREVIEW_ROWS)
                    return df, None
                except Exception as e:  # noqa: BLE001
                    from ..core.executor import friendly_error
                    return None, friendly_error(e)

            def done(r):
                if nid != self.nid or src != self.src:
                    return
                df, err = r
                if err is not None:
                    self._set_overlay(f"Couldn't build a preview: {err}"); return
                self._got_frame(df.lazy(), dict(df.schema), len(df), preview=True)
            self._serial.submit(work, done, self._set_overlay)

    def _got_frame(self, lf: pl.LazyFrame, schema: dict, rows: int, preview: bool) -> None:
        self.lf = lf; self.rows = rows; self.preview = preview; self.schema = schema; self._schema_src = self.src
        self._fill_chips()
        self._query()

    def clear(self) -> None:
        self.nid = None; self.src = None; self.lf = None; self.schema = {}; self._schema_src = None
        self.rows = 0; self._src_hash = None
        self._serial.cancel()
        self._clear_items(); self.info.setText(""); self.source.setText(""); self._set_overlay("")

    def _set_overlay(self, text: str) -> None:
        self.overlay.setText(text); self.overlay.setVisible(bool(text))

    # ------------------------------------------------------------ chips
    def _fill_chips(self) -> None:
        p = self._params()
        schema = self.schema
        nums = [c for c, dt in schema.items() if kind_of_dtype(dt) == NUM]
        any_cols = list(schema)
        title = self.doc.pipeline.column_title
        for chip, menu, key, extra in (
            (self.lat_chip, self.lat_menu, "lat", None),
            (self.lon_chip, self.lon_menu, "lon", None),
            (self.color_chip, self.color_menu, "color_by", None),
        ):
            cur = p.get(key) or ""
            chip.setText({"lat": "Latitude: ", "lon": "Longitude: ", "color_by": "Colour by: "}[key] + (title(cur) if cur else "none"))
            menu.clear()
            if key in ("color_by", "size_by"):
                a = menu.addAction("none"); a.triggered.connect(lambda _=False, k=key: self._set({k: ""}))
            for col in (any_cols if key == "color_by" else nums):
                a = menu.addAction(title(col)); a.setCheckable(True); a.setChecked(col == cur)
                a.triggered.connect(lambda _=False, k=key, c=col: self._set({k: c}))
            chip.setMenu(menu)
        cur = p.get("size_by") or ""
        self.size_chip.setText("Size by: " + (title(cur) if cur else "none"))
        self.size_menu.clear()
        a = self.size_menu.addAction("none"); a.triggered.connect(lambda: self._set({"size_by": ""}))
        for col in nums:
            a = self.size_menu.addAction(title(col)); a.setCheckable(True); a.setChecked(col == cur)
            a.triggered.connect(lambda _=False, c=col: self._set({"size_by": c}))
        self.size_chip.setMenu(self.size_menu)
        cell = str(p.get("cell_size") or "")
        self.cell_chip.setText("Squares of " + cell if cell else "Points")
        self.cell_menu.clear()
        a = self.cell_menu.addAction("Points"); a.setCheckable(True); a.setChecked(not cell); a.triggered.connect(lambda: self._set({"cell_size": ""}))
        for size in ("0.1", "0.5", "1", "5km", "10km", "25km"):
            a = self.cell_menu.addAction(f"Squares of {size}"); a.setCheckable(True); a.setChecked(cell == size)
            a.triggered.connect(lambda _=False, s=size: self._set({"cell_size": s}))
        self.cell_chip.setMenu(self.cell_menu)
        self.basemap_chip.setChecked(bool(p.get("basemap", True)))

    # ------------------------------------------------------------ querying
    def fit(self) -> None:
        self._query()

    def _query(self) -> None:
        if self.lf is None or self.nid is None:
            return
        spec, schema, lf, nid = self._params(), self.schema, self.lf, self.nid
        t0 = time.perf_counter()
        self._set_overlay("")
        self.info.setText("waiting for the run…" if self.doc.running else f"drawing {self.rows:,} rows…")

        def work():
            try:
                return query_map(lf, schema, spec), None
            except MapError as e:
                return None, str(e)

        def done(r):
            if nid != self.nid:
                return
            md, error = r
            self._render(md, error, time.perf_counter() - t0, spec)

        self._serial.submit(work, done, lambda m: (self.info.setText(""), self._set_overlay(m)) if nid == self.nid else None)

    # ------------------------------------------------------------ render
    def _clear_items(self) -> None:
        for it in self._items:
            try:
                self.plot.removeItem(it)
            except Exception:  # noqa: BLE001 - already gone
                pass
        self._items = []
        self._pts = []
        self.legend.clear()
        self._hover.hide()

    def _add(self, item):
        self.plot.addItem(item); self._items.append(item)
        return item

    def _render(self, md: MapData | None, error: str | None, secs: float, spec: dict) -> None:
        self._clear_items()
        if error or md is None:
            self.info.setText(""); self._set_overlay(error or "Nothing to draw"); return
        self._set_overlay("")
        projection = spec.get("projection") or "equirectangular"
        if md.basemap:
            self._draw_basemap(projection)
        if md.plotted:
            x, y = project(md.xs, md.ys, projection)
            if md.kind == "cells" and md.cell_size:
                self._draw_cells(md, x, y)
            else:
                self._draw_points(md, x, y)
        x0, x1, y0, y1 = md.extent
        px0, py1 = project(x0, y1, projection); px1, py0 = project(x1, y0, projection)
        self.plot.setXRange(float(px0), float(px1), padding=0)
        self.plot.setYRange(float(py0), float(py1), padding=0)
        info = md.summary() + f" · {secs * 1000:.0f} ms" + (" · preview" if self.preview else "")
        self.info.setText(info)
        self.plot.setTitle(spec.get("title") or (self.doc.pipeline.nodes[self.nid].title if self.nid else ""))

    def _draw_basemap(self, projection: str) -> None:
        lons: list[float] = []; lats: list[float] = []
        for ring in world_outlines():
            arr = np.asarray(ring, dtype=float)
            lons.extend(arr[:, 0].tolist()); lats.extend(arr[:, 1].tolist())
            lons.append(float("nan")); lats.append(float("nan"))
        if not lons:
            return
        x, y = project(np.array(lons), np.array(lats), projection)
        self._add(pg.PlotDataItem(x, y, connect="finite", pen=pg.mkPen("#cbd5e1", width=1)))

    def _brushes(self, md: MapData) -> list:
        if md.codes is not None:
            palette = [c for _, c in md.legend]
            return [palette[c] if 0 <= c < len(palette) else OTHER_COLOR for c in md.codes]
        if md.color is not None:
            v = np.asarray(md.color, dtype=float)
            finite = np.isfinite(v)
            lo = float(np.nanmin(v[finite])) if finite.any() else 0.0
            hi = float(np.nanmax(v[finite])) if finite.any() else 1.0
            span = (hi - lo) or 1.0
            cmap = pg.colormap.get("viridis")
            out = []
            for val, ok in zip(v, finite):
                if not ok:
                    out.append(OTHER_COLOR); continue
                out.append(cmap.mapToQColor(float(min(1.0, max(0.0, (val - lo) / span)))))
            return out
        return [SERIES_COLORS[0]] * len(md.xs)

    def _sizes(self, md: MapData, base: float) -> Any:
        if md.size is None or not len(md.size):
            return base
        v = np.asarray(md.size, dtype=float)
        finite = np.isfinite(v)
        lo = float(np.nanmin(v[finite])) if finite.any() else 0.0
        hi = float(np.nanmax(v[finite])) if finite.any() else 1.0
        span = (hi - lo) or 1.0
        return np.where(finite, base + 12.0 * np.clip((v - lo) / span, 0.0, 1.0), base)

    def _draw_points(self, md: MapData, x, y) -> None:
        brushes = self._brushes(md)
        sizes = self._sizes(md, 7.0)
        item = pg.ScatterPlotItem(x=x, y=y, size=sizes, brush=brushes, pen=None)
        self._add(item)
        self._remember_points(md)
        if md.legend:
            for label, col in md.legend:
                proxy = pg.ScatterPlotItem(x=[], y=[], size=8, brush=col, pen=None); proxy.setData([], [])
                self.legend.addItem(proxy, str(label))
        elif md.color is not None and len(md.color):
            v = np.asarray(md.color, dtype=float); v = v[np.isfinite(v)]
            if len(v):
                self.info.setText((self.info.text() + f" · colour {self.doc.pipeline.column_title(md.color_column or '')} "
                                   f"{np.nanmin(v):g}–{np.nanmax(v):g}").strip(" ·"))

    def _draw_cells(self, md: MapData, x, y) -> None:
        brushes = self._brushes(md)
        # a square sized to the cell: fraction of the view width, in pixels, mapped to a marker size
        frac = md.cell_size / max(1e-9, md.extent[1] - md.extent[0])
        size = max(3.0, min(80.0, frac * max(600, self.plot.getViewBox().width())))
        item = pg.ScatterPlotItem(x=x, y=y, size=size, symbol="s", brush=brushes, pen=None)
        self._add(item)
        if md.legend:
            for label, col in md.legend:
                proxy = pg.ScatterPlotItem(x=[], y=[], size=8, symbol="s", brush=col, pen=None)
                self.legend.addItem(proxy, str(label))
        self._remember_points(md)

    def _remember_points(self, md: MapData) -> None:
        """Keep the drawn points and what each stands for, so hovering can read one out. Only when there are
        few enough to hover meaningfully."""
        self._proj = self._params().get("projection") or "equirectangular"
        self._pts = []
        if md.plotted > 5000:
            return
        label_col = self._params().get("label")
        for i in range(md.plotted):
            bits = [f"{self._params().get('lat') or 'lat'} {md.ys[i]:.4f}, {self._params().get('lon') or 'lon'} {md.xs[i]:.4f}"]
            if label_col and md.labels and i < len(md.labels) and md.labels[i]:
                bits.insert(0, md.labels[i])
            self._pts.append((float(md.xs[i]), float(md.ys[i]), "\n".join(bits)))

    def _mouse_moved(self, pos) -> None:
        if not self._pts or not self.plot.sceneBoundingRect().contains(pos):
            self._hover.hide(); return
        vb = self.plot.getViewBox()
        vp = vb.mapSceneToView(pos)
        # a small tolerance in view units, scaled to the current zoom
        (x0, x1), _ = vb.viewRange()
        tol = max(1e-9, (x1 - x0) * 0.02)
        px, py = project([vp.x()], [vp.y()], self._proj)
        best = None; best_d = None
        for lon, lat, tip in self._pts:
            rx, ry = project([lon], [lat], self._proj)
            d = (rx[0] - px[0]) ** 2 + (ry[0] - py[0]) ** 2
            if d <= tol ** 2 and (best_d is None or d < best_d):
                best, best_d = (rx[0], ry[0], tip), d
        if best is None:
            self._hover.hide(); return
        self._hover.setText(best[2]); self._hover.setPos(best[0], best[1]); self._hover.show()

    # ------------------------------------------------------------ images
    def _grab(self):
        from pyqtgraph.exporters import ImageExporter
        ex = ImageExporter(self.gl.scene())
        ex.parameters()["width"] = max(1400, self.gl.width() * 2)
        return ex

    def save_image(self) -> None:
        if self.nid is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save map as image", str(self.doc.pipeline.directory / f"{self.nid}.png"), "PNG image (*.png);;SVG (*.svg)")
        if not f:
            return
        if f.lower().endswith(".svg"):
            from pyqtgraph.exporters import SVGExporter
            SVGExporter(self.gl.scene()).export(f)
        else:
            self._grab().export(f)
        self.doc.message.emit(f"Saved {f}")

    def copy_image(self) -> None:
        if self.nid is None:
            return
        self._grab().export(copy=True)
        self.doc.message.emit("Map image copied. Paste it into Word or an email")
