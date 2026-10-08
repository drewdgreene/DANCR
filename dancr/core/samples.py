"""Sample data and starter templates for the start screen (and for AI agents wanting a quick demo)."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SAMPLE_NAME = "sample_data.csv"


SAMPLE_COLUMNS = ["time", "value A", "value B", "temperature", "location"]
MIN_SAMPLE_ROWS = 2_000         # enough for the spike, the gap and a daily cycle to show


def write_sample(directory: Path | str, rows: int = 60_000) -> Path:
    """Two values recorded every 5 seconds for a few days, with drift, a spike, a gap and a daily cycle.
    An existing sample file of the same size is reused; a file of that name that is not DANCR's sample (or
    has another size) is never overwritten or used: the sample gets a numbered name instead."""
    if rows < MIN_SAMPLE_ROWS:
        raise ValueError(f"A sample needs at least {MIN_SAMPLE_ROWS:,} rows")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        out = directory / (SAMPLE_NAME if n == 1 else SAMPLE_NAME.replace(".csv", f"_{n}.csv"))
        if not out.exists():
            break
        if _is_sample(out, rows):
            return out
        n += 1
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
    tmp = out.with_name(f"{out.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")   # a crash mid-write never leaves half a sample
    df.write_csv(tmp)
    os.replace(tmp, out)
    return out


def _sample_rows(rows: int) -> int:
    return rows - (min(rows, rows // 2 + 900) - rows // 2)


def _is_sample(path: Path, rows: int) -> bool:
    """True when `path` is the sample DANCR writes for `rows` rows (its header and its number of lines)."""
    try:
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            lines = sum(1 for _ in f)
    except (OSError, UnicodeDecodeError):
        return False
    return header == SAMPLE_COLUMNS and lines == _sample_rows(rows)


TEMPLATES = [
    {"key": "compare", "title": "Compare two columns over time", "blurb": "Load a file, smooth two columns, chart them together and describe the difference."},
    {"key": "limits", "title": "Check a column against a limit", "blurb": "Flag every row outside a limit, chart it with the limit line and count how many."},
    {"key": "fit", "title": "Fit a curve and predict", "blurb": "Fit a straight line between two columns, see the equation and R², and predict new values."},
    {"key": "report", "title": "Hourly averages and a report", "blurb": "Average over each hour, chart the result and put table and chart on one report page."},
]


def check_template(key: str) -> None:
    if key not in {t["key"] for t in TEMPLATES}:
        raise ValueError(f"Unknown template {key!r}. Known: {[t['key'] for t in TEMPLATES]}")


def build_template(key: str, pipe, data_path: Path) -> None:
    """Add a starter set of steps to a Pipeline for `data_path`. Positions are laid out left to right."""
    from .executor import Executor
    from .model import portable_path
    check_template(key)
    x = [60.0]
    def col(dx: float = 300) -> float:
        x[0] += dx
        return x[0] - dx
    def add(type_key, params, title, y=200.0, after=None, port=None, dx=300):
        n = pipe.add_node(type_key, title=title, params=params, x=col(dx), y=y)
        if after:
            pipe.connect(after, n.id, port)
        return n.id
    load = add("load_file", {"path": portable_path(Path(data_path).resolve(), pipe.directory)}, Path(data_path).stem)
    schema = Executor(pipe).schema(load) or {}
    nums = [c for c, dt in schema.items() if dt.is_numeric()]
    time_col = next((c for c, dt in schema.items() if isinstance(dt, (pl.Datetime, pl.Date))), None)
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


# ---------------------------------------------------------------- example projects
# Each is a finished project on its own realistic data: the files, the questions it answers, and any steps
# added by hand. They open from the start page into ~/DANCR samples/<title>.
EXAMPLES = [
    {"key": "shop", "title": "Shop sales",
     "blurb": "A year of orders and a customer list, linked by customer. Sales by region, month, customer and product.",
     "questions": ["total sales by region", "monthly sales", "top 10 customers by sales", "total sales by product"]},
    {"key": "loggers", "title": "Two sensor logs",
     "blurb": "A week of pressure and temperature from two loggers. Compared with each other, averaged per hour, "
              "with the gaps and spikes found.",
     "questions": ["compare logger_A and logger_B", "average pressure per hour", "gaps in logger_A", "spikes in pressure"]},
    {"key": "budget", "title": "Department budget",
     "blurb": "A spreadsheet with a column for each month. Turned into rows, then totalled by department, month and cost.",
     "questions": ["total value by department", "total value per month", "total value by cost"]},
    {"key": "batches", "title": "Batch tests",
     "blurb": "Strength tests from three presses. Checked against a minimum strength, compared by press, and "
              "strength fitted against curing temperature.",
     "questions": ["average strength by machine", "strength against cure_temp", "strength per week"]},
]


def example(key: str) -> dict:
    for e in EXAMPLES:
        if e["key"] == key:
            return e
    raise ValueError(f"No example called {key!r}. Examples: {', '.join(e['key'] for e in EXAMPLES)}")


def write_example(key: str, directory: Path | str) -> Path:
    """Example ``key`` as a finished project in ``directory``: its files, its answers, and a report of their
    charts. An example already there is reused as it is (the person may have changed it)."""
    from .model import Pipeline
    from .answers import build, model_for
    from .ask import ask
    from .executor import Executor
    from .planner import place_near
    ex = example(key)
    directory = Path(directory)
    project = directory / f"{ex['title']}.json"
    if project.exists():
        return project
    directory.mkdir(parents=True, exist_ok=True)
    files = _EXAMPLE_DATA[key](directory)
    p = Pipeline(ex["title"])
    p.path = project
    for i, name in enumerate(files):                    # the files one below another, at the left
        p.add_node("load_file", title=Path(name).stem, params={"path": name}, id=Path(name).stem, y=i * 155.0)
    model = model_for(p, Executor(p))                   # the tables; the answers only add steps after them
    for q in ex["questions"]:
        asked = ask(model, q)
        if not asked.ok:
            raise RuntimeError(f"The example question {q!r} is no longer understood: {asked.message}")
        build(p, model, asked.spec)
    items = [a.terminal for a in p.answers]
    if key == "batches":
        items.append(_batch_limit(p))
    x, y = place_near(p, items)                         # after the last step it shows, so its inputs read left to right
    report = p.add_node("report", title="Report", id="report", x=x, y=y,
                        params={"title": ex["title"], "path": f"{ex['title']} report.html", "notes": ex["blurb"]})
    for nid in items:
        p.connect(nid, report.id, "items")
    p.save(project)
    return project


def _batch_limit(p) -> str:
    """A pass/fail check against a minimum kept on the Inputs page, so changing it reruns the check."""
    from .planner import place_near
    p.set_input("minimum strength", 30, "MPa", "from the product spec")
    x, y = place_near(p, ["batch_tests"])
    check = p.add_node("check_limits", title="Strength at least the minimum", id="strength_check", x=x, y=y,
                       params={"column": "strength", "min": "minimum strength", "action": "flag"})
    p.connect("batch_tests", check.id)
    return check.id


def _write_shop(d: Path) -> list[str]:
    rng = np.random.default_rng(7)
    first = ["Ava", "Ben", "Chloe", "Dan", "Ella", "Finn", "Grace", "Harry", "Isla", "Jack",
             "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Ruby", "Sam", "Tom", "Zoe"]
    last = ["Brown", "Clark", "Evans", "Green", "Hall", "Jones", "Khan", "Lewis", "Moore", "Patel",
            "Reed", "Shaw", "Taylor", "Walsh", "Wood"]
    nc = 60
    pl.DataFrame({"customer_id": list(range(1001, 1001 + nc)),
                  "customer": [f"{first[i % 20]} {last[(i * 7) % 15]}" for i in range(nc)],
                  "region": [["North", "South", "East", "West"][i % 4] for i in range(nc)]}).write_csv(d / "customers.csv")
    products = [("Hammer", 12.5), ("Screwdriver set", 18.0), ("Paint 1L", 9.75), ("Paint 5L", 38.0), ("Brush", 4.5),
                ("Garden hose", 24.0), ("Spade", 29.0), ("Gloves", 6.25), ("Drill", 79.0), ("Ladder", 95.0)]
    n = 3000
    t0 = datetime(2024, 1, 1)
    days = np.sort(rng.integers(0, 366, n))
    item = rng.integers(0, len(products), n)
    qty = rng.integers(1, 6, n)
    pl.DataFrame({
        "order_id": list(range(50001, 50001 + n)),
        "ordered_at": [t0 + timedelta(days=int(dd), hours=int(h), minutes=int(mm))
                       for dd, h, mm in zip(days, rng.integers(8, 19, n), rng.integers(0, 60, n))],
        "customer_id": rng.integers(1001, 1001 + nc, n).tolist(),
        "product": [products[i][0] for i in item],
        "quantity": qty.tolist(),
        "sales": [round(products[i][1] * int(q), 2) for i, q in zip(item, qty)],
    }).write_csv(d / "orders.csv")
    return ["orders.csv", "customers.csv"]


def _write_loggers(d: Path) -> list[str]:
    """A week at one reading a minute. B reads a little high, drifts and is noisier; A has a gap and a spike."""
    rng = np.random.default_rng(3)
    n = 7 * 24 * 60
    t0 = datetime(2024, 5, 6)
    secs = np.arange(n) * 60
    daily = np.sin(2 * np.pi * secs / 86400)
    pa = 4.2 + 0.15 * daily + rng.normal(0, 0.01, n)
    pa[3000] += 1.5
    keep = np.ones(n, bool)
    keep[6000:6180] = False
    pl.DataFrame({"time": [t0 + timedelta(seconds=int(s)) for s in secs], "pressure (bar)": np.round(pa, 4),
                  "temperature (°C)": np.round(18 + 3 * daily + rng.normal(0, 0.05, n), 2)}
                 ).filter(pl.Series(keep)).write_csv(d / "logger_A.csv")
    pl.DataFrame({"time": [t0 + timedelta(seconds=int(s) + 13) for s in secs],
                  "pressure (bar)": np.round(4.2 + 0.15 * daily + 0.0004 * np.arange(n) / 60 + rng.normal(0, 0.03, n), 4),
                  "temperature (°C)": np.round(18.4 + 3 * daily + rng.normal(0, 0.08, n), 2)}).write_csv(d / "logger_B.csv")
    return ["logger_A.csv", "logger_B.csv"]


def _write_budget(d: Path) -> list[str]:
    rng = np.random.default_rng(5)
    lines = [("Operations", "Staff"), ("Operations", "Equipment"), ("Operations", "Travel"), ("Sales", "Staff"),
             ("Sales", "Travel"), ("Marketing", "Staff"), ("Marketing", "Campaigns"), ("IT", "Staff"), ("IT", "Software"),
             ("IT", "Hardware"), ("HR", "Staff"), ("HR", "Training"), ("Finance", "Staff"), ("Finance", "Audit")]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base = {line: rng.uniform(4, 40) * 1000 for line in lines}
    rows: dict[str, list] = {"department": [a for a, _ in lines], "cost": [b for _, b in lines]}
    for m in months:
        busy = 1.4 if m in ("Nov", "Dec") else 1.0
        rows[m] = [int(round(base[line] * (busy if line[1] == "Campaigns" else 1) * rng.uniform(0.9, 1.1), -2)) for line in lines]
    pl.DataFrame(rows).write_csv(d / "spend_2024.csv")
    return ["spend_2024.csv"]


def _write_batches(d: Path) -> list[str]:
    rng = np.random.default_rng(11)
    n = 240
    machine = rng.choice(["Press 1", "Press 2", "Press 3"], n)
    temp = np.round(rng.uniform(160, 220, n), 1)
    strength = 31 + 0.08 * (temp - 160) - 0.0015 * (temp - 200) ** 2 + np.where(machine == "Press 3", -1.2, 0) + rng.normal(0, 0.6, n)
    pl.DataFrame({"batch": [f"B{2400 + i}" for i in range(n)],
                  "tested_at": [datetime(2024, 3, 1) + timedelta(hours=int(h)) for h in np.sort(rng.integers(0, 24 * 60, n))],
                  "machine": machine, "cure_temp": temp, "strength": np.round(strength, 2)}).write_csv(d / "batch_tests.csv")
    return ["batch_tests.csv"]


_EXAMPLE_DATA = {"shop": _write_shop, "loggers": _write_loggers, "budget": _write_budget, "batches": _write_batches}
