"""Sample data and starter templates for the start screen (and for AI agents wanting a quick demo)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SAMPLE_NAME = "sample_data.csv"


def write_sample(directory: Path | str, rows: int = 60_000) -> Path:
    """Two values recorded every 5 seconds for a few days, with drift, a spike, a gap and a daily cycle."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / SAMPLE_NAME
    if out.exists():
        return out
    rng = np.random.default_rng(7)
    t0 = datetime(2024, 6, 3, 8, 0, 0)
    secs = np.arange(rows) * 5.0
    tide = 0.35 * np.sin(2 * np.pi * secs / (12.42 * 3600)) + 0.12 * np.sin(2 * np.pi * secs / (24 * 3600))
    drift = 0.0000004 * secs
    probe_a = 101.3 + tide + drift + rng.normal(0, 0.012, rows)
    probe_b = 101.3 + tide + rng.normal(0, 0.004, rows)
    spike = rows // 3
    probe_a[spike:spike + 4] += 2.4
    temp = 18.5 + 0.9 * np.sin(2 * np.pi * secs / (24 * 3600) - 1.2) + rng.normal(0, 0.05, rows)
    location = np.where((secs // 86400) % 2 == 0, "north", "south")
    df = pl.DataFrame({
        "time": [t0 + timedelta(seconds=float(s)) for s in secs],
        "value A": np.round(probe_a, 4),
        "value B": np.round(probe_b, 4),
        "temperature": np.round(temp, 2),
        "location": location,
    })
    gap0, gap1 = rows // 2, min(rows, rows // 2 + 900)          # 75 minutes of missing data (never past the end)
    df = pl.concat([df[:gap0], df[gap1:]])
    df.write_csv(out)
    return out


TEMPLATES = [
    {"key": "compare", "title": "Compare two columns over time", "blurb": "Load a file, smooth two columns, chart them together and describe the difference."},
    {"key": "limits", "title": "Check a column against a limit", "blurb": "Flag every row outside a limit, chart it with the limit line and count how many."},
    {"key": "fit", "title": "Fit a curve and predict", "blurb": "Fit a straight line between two columns, see the equation and R², and predict new values."},
    {"key": "report", "title": "Hourly averages and a report", "blurb": "Average over each hour, chart the result and put table and chart on one report page."},
]


def build_template(key: str, pipe, data_path: Path) -> None:
    """Add a starter set of steps to a Pipeline for `data_path`. Positions are laid out left to right."""
    from .executor import Executor
    if key not in {t["key"] for t in TEMPLATES}:
        raise ValueError(f"Unknown template {key!r}. Known: {[t['key'] for t in TEMPLATES]}")
    x = [60.0]
    def col(dx: float = 300) -> float:
        x[0] += dx
        return x[0] - dx
    def add(type_key, params, title, y=200.0, after=None, port=None, dx=300):
        n = pipe.add_node(type_key, title=title, params=params, x=col(dx), y=y)
        if after:
            pipe.connect(after, n.id, port)
        return n.id
    load = add("load_file", {"path": str(data_path)}, data_path.stem)
    schema = Executor(pipe).schema(load) or {}
    nums = [c for c, dt in schema.items() if dt.is_numeric()]
    time_col = next((c for c, dt in schema.items() if isinstance(dt, (pl.Datetime, pl.Date)) or dt in (pl.Datetime, pl.Date)), None)
    a, b = (nums + [None, None])[:2]
    if key == "compare":
        smooth = add("rolling", {"columns": [c for c in (a, b) if c], "window": "5m", "stat": "mean", "time_column": time_col or ""}, "Smoothed", after=load)
        add("chart", {"kind": "line", "x": time_col or "", "series": [{"column": c} for c in (a, b) if c], "title": "Both columns"}, "Both columns", y=120, after=smooth, dx=0)
        if a and b:
            diff = add("calculate", {"formulas": [{"name": "difference", "expr": f"[{a}] - [{b}]"}]}, "Difference", y=300, after=smooth)
            add("summarize", {}, "Describe the difference", y=300, after=diff)
    elif key == "limits":
        pipe.set_input("upper limit", 101.9, "", "The value that must not be exceeded")
        chk = add("check_limits", {"column": a or "", "max": "upper limit", "action": "flag"}, "Check limits", after=load)
        add("chart", {"kind": "line", "x": time_col or "", "series": [{"column": a}] if a else [], "limits": [{"value": "upper limit", "label": "upper limit"}], "title": "Values and limit"}, "Values and limit", y=120, after=chk, dx=0)
        add("keep_rows", {"mode": "keep", "conditions": {"match": "all", "rules": [{"column": f"{a}_ok", "op": "false", "value": ""}]}}, "Only the rows outside", y=300, after=chk)
    elif key == "fit":
        fit = add("fit_curve", {"x": a or "", "y": b or "", "kind": "linear"}, "Fit a line", after=load)
        add("chart", {"kind": "scatter", "x": a or "", "series": [{"column": b}] if b else [], "fit": "linear", "title": f"{b} vs {a}"}, "Scatter with fit", y=120, after=load, dx=0)
        new = add("enter_data", {"columns": [{"name": a or "x", "type": "number"}], "rows": [[101.2], [101.5], [101.8]]}, "New values", y=340)
        pred = add("predict", {}, "Predict", y=300, after=fit, port="model")
        pipe.connect(new, pred, "data")
    elif key == "report":
        avg = add("time_buckets", {"every": "1h", "columns": nums[:2], "default_stats": ["mean"]}, "Hourly averages", after=load)
        ch = add("chart", {"kind": "line", "x": time_col or "", "series": [{"column": c} for c in nums[:2]], "title": "Hourly averages"}, "Hourly chart", y=120, after=avg)
        rep = add("report", {"title": "Hourly summary", "path": "report.html", "notes": "Hourly averages of both columns."}, "Report", after=ch, port="items")
        pipe.connect(avg, rep, "items")
