"""Table-first actions: turn a column or cell action from the grid into a step after the current table."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QInputDialog

if TYPE_CHECKING:
    from .mainwindow import MainWindow


class StepFactory:
    def __init__(self, win: MainWindow) -> None:
        self.win = win
        self.doc = win.doc

    def after(self, type_key: str, params: dict, title: str | None = None, show: bool = True, src: str | None = None) -> str | None:
        src = src or self.win.current_table()
        if src is None:
            self.win.status.showMessage("Open a table first", 4000); return None
        return self.win.add_node(type_key, None, params=params, title=title, connect_from=src, show=show)

    def column_action(self, action: str, column: str) -> None:
        src = self.win.current_table()
        if src is None:
            return
        title = self.doc.pipeline.column_title(column)
        if action.startswith("fit:"):
            _, x, y = action.split(":", 2)
            self.after("chart", {"kind": "scatter", "x": x, "series": [{"column": y}], "fit": "linear", "title": f"{self.doc.pipeline.column_title(y)} vs {self.doc.pipeline.column_title(x)}"}, title=f"{y} vs {x}")
            return
        if action == "filter":
            self.after("keep_rows", {"mode": "keep", "conditions": {"match": "all", "rules": [{"column": column, "op": "gt", "value": ""}]}}, title=f"Filter by {title}")
            self.win.inspector.focus_first_field()
        elif action in ("sort_asc", "sort_desc"):
            self.after("sort", {"columns": [column], "descending": action == "sort_desc"}, title=f"Sort by {title}")
        elif action == "chart":
            self.after("chart", {"kind": "line", "series": [{"column": column}], "title": title}, title=title)
        elif action == "limit":
            self.after("check_limits", {"column": column}, title=f"Check {title}")
            self.win.inspector.focus_first_field()
        elif action == "formula":
            self.after("calculate", {"formulas": [{"name": f"{column} (new)", "expr": f"[{column}]"}]}, title="New column")
            self.win.inspector.focus_first_field()
        elif action == "describe":
            self.rename_column(column)
        elif action == "fixtype":
            self.after("change_type", {"columns": [column], "to": "number"}, title=f"Fix {title}")
        elif action == "hide":
            self.after("choose_columns", {"mode": "drop", "columns": [column]}, title=f"Hide {title}")
        elif action == "copyname":
            QGuiApplication.clipboard().setText(column); self.win.status.showMessage(f"Copied “{column}”", 3000)

    def rename_column(self, column: str) -> None:
        meta = self.doc.pipeline.columns.get(column, {})
        label, ok = QInputDialog.getText(self.win, "Column name and units", f"Name shown for “{column}” (leave blank to keep it):", text=meta.get("label") or "")
        if not ok:
            return
        unit, ok = QInputDialog.getText(self.win, "Column name and units", "Units (e.g. mm, °C, kg):", text=meta.get("unit") or "")
        if not ok:
            return
        # blank means "keep what is there", not "clear it"
        self.doc.set_column_meta(column, label.strip() or None, unit.strip() or None)

    def cell_action(self, action: str, row: int, column: str, value) -> None:
        if action != "fix":
            return
        src = self.win.current_table()
        if src is None:
            return
        new, ok = QInputDialog.getText(self.win, "Fix this value", f"Row {row:,}, {self.doc.pipeline.column_title(column)}\nCurrent value: {value!r}\n\nNew value (blank = empty):", text="" if value is None else str(value))
        if not ok:
            return
        note, ok2 = QInputDialog.getText(self.win, "Fix this value", "Why? (kept with the correction, shown in the report):")
        fix = {"row": int(row), "column": column, "value": new.strip(), "was": None if value is None else str(value), "note": note.strip() if ok2 else ""}
        # reuse a Fix values step directly after this table, else create one
        n = self.doc.pipeline.nodes[src]
        target = None
        if n.type == "fix_values":
            target = src
        else:
            for e in self.doc.pipeline.edges:
                if e.source == src and self.doc.pipeline.nodes[e.target].type == "fix_values":
                    target = e.target; break
        if target:
            fixes = list(self.doc.pipeline.nodes[target].params.get("fixes") or []) + [fix]
            self.doc.set_params(target, {"fixes": fixes}); self.win.show_node(target)
        else:
            self.after("fix_values", {"fixes": [fix]}, title="Fixed values")

    def chart_columns(self, columns: list[str]) -> None:
        if not columns:
            return
        t = ", ".join(self.doc.pipeline.column_title(c) for c in columns[:3])
        self.after("chart", {"kind": "line", "series": [{"column": c} for c in columns], "title": t}, title=t)

    def add_to_report(self, chart_id: str) -> None:
        if chart_id not in self.doc.pipeline.nodes:
            return
        reports = [n.id for n in self.doc.pipeline.nodes.values() if n.type == "report"]
        if reports:
            rid = reports[0]
            self.doc.connect(chart_id, rid, "items")
        else:
            rid = self.win.add_node("report", None, params={"title": "Report", "path": "report.html"}, title="Report", connect_from=chart_id, port="items", show=False)
        self.win.show_node(rid)
        self.win.status.showMessage("Added to the report. Press Build report when it is ready.", 6000)
