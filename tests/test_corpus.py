"""The answer rules against a corpus of realistic projects, compared with what they gave when last checked.

Each case generates its files (seeded, so they are the same every time), reads them, and records how the
tables were understood, what was suggested, and how a few questions were read. A change to the rules that
changes any of that fails here, on purpose: look at the difference, and if the new result is better, run

    DANCR_UPDATE_CORPUS=1 pytest tests/test_corpus.py

to store it (the stored results live in tests/corpus/). Every suggested answer is also built and run.
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.planner import instantiate
from dancr.core.recipes import suggest, plan
from dancr.core.understand import understand, deepen

HERE = Path(__file__).parent / "corpus"
UPDATE = os.environ.get("DANCR_UPDATE_CORPUS") == "1"


# ------------------------------------------------------------------- the cases
def shop(d: Path) -> list[str]:
    rng = np.random.default_rng(1)
    n = 400
    t0 = datetime(2024, 1, 1)
    pl.DataFrame({
        "order_id": list(range(1, n + 1)),
        "customer_id": rng.integers(1, 41, n).tolist(),
        "product_id": rng.integers(1, 9, n).tolist(),
        "quantity": rng.integers(1, 6, n).tolist(),
        "price": np.round(rng.uniform(2, 40, n), 2).tolist(),
        "ordered_at": [t0 + timedelta(hours=int(h)) for h in np.sort(rng.integers(0, 24 * 120, n))],
    }).write_csv(d / "orders.csv")
    pl.DataFrame({"customer_id": list(range(1, 41)), "customer_name": [f"Customer {i}" for i in range(1, 41)],
                  "region": (["North", "South", "East", "West"] * 10), "segment": (["Retail", "Trade"] * 20)}).write_csv(d / "customers.csv")
    pl.DataFrame({"product_id": list(range(1, 9)), "product": [f"P{i}" for i in range(1, 9)],
                  "category": ["Tools", "Tools", "Paint", "Paint", "Garden", "Garden", "Garden", "Tools"]}).write_excel(d / "products.xlsx")
    return ["orders.csv", "customers.csv", "products.xlsx"]


def probes(d: Path) -> list[str]:
    rng = np.random.default_rng(2)
    n = 7200
    for name, offset, noise in (("site_A", 0.0, 0.01), ("site_B", 0.3, 0.003)):
        t0 = datetime(2024, 6, 1) + timedelta(seconds=offset)
        secs = np.arange(n) * 1.0
        tide = np.sin(2 * np.pi * secs / 3600)
        pl.DataFrame({"time": [t0 + timedelta(seconds=float(s)) for s in secs],
                      "pressure (bar)": np.round(10 + tide + rng.normal(0, noise, n), 5),
                      "temperature": np.round(4 + 0.1 * tide + rng.normal(0, 0.01, n), 4)}).write_csv(d / f"logger_{name}.csv")
    return ["logger_site_A.csv", "logger_site_B.csv"]


def monthly(d: Path) -> list[str]:
    rng = np.random.default_rng(3)
    out = []
    for m in (1, 2, 3):
        days = [datetime(2024, m, 1) + timedelta(days=i) for i in range(28)]
        rows = [(day, store) for day in days for store in ("Leeds", "York", "Hull")]
        pl.DataFrame({"date": [r[0] for r in rows], "store": [r[1] for r in rows],
                      "sales": np.round(rng.uniform(100, 900, len(rows)), 2).tolist()}).write_csv(d / f"sales_2024-0{m}.csv")
        out.append(f"sales_2024-0{m}.csv")
    return out


def trap(d: Path) -> list[str]:
    """Two tables that share a key that repeats on both sides: linking them would multiply rows."""
    pl.DataFrame({"patient_id": [i % 30 for i in range(120)], "visit_cost": [float(i % 7 + 1) for i in range(120)]}).write_csv(d / "visits.csv")
    pl.DataFrame({"patient_id": [i % 30 for i in range(90)], "drug": [f"D{i % 9}" for i in range(90)]}).write_csv(d / "prescriptions.csv")
    return ["visits.csv", "prescriptions.csv"]


def stock(d: Path) -> list[str]:
    """Two small counting ids with different names (they must not be linked), and codes written as numbers."""
    pl.DataFrame({"store_id": list(range(1, 201)), "zip": [10000 + i * 7 for i in range(200)],
                  "city": [f"C{i % 30}" for i in range(200)]}).write_csv(d / "stores.csv")
    pl.DataFrame({"stock_id": [i % 150 + 1 for i in range(1000)], "units": [i % 9 for i in range(1000)]}).write_csv(d / "inventory.csv")
    return ["stores.csv", "inventory.csv"]


def plain(d: Path) -> list[str]:
    """No dates and no categories: just measurements."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 300)
    pl.DataFrame({"length": np.round(x, 3), "weight": np.round(2 * x + rng.normal(0, 0.5, 300), 3),
                  "width": np.round(rng.uniform(1, 2, 300), 3)}).write_csv(d / "parts.csv")
    return ["parts.csv"]


CASES = {
    "shop": (shop, ["total quantity by region", "top 5 customers by quantity", "average price per month",
                    "quantity by category for North", "orders where price above 30", "how many orders by segment",
                    "orders before March", "orders since March", "average price from 2024-02-01 to 2024-03-01",
                    "total quantity for North and South", "orders in North or South", "what is the average price",
                    "orders where price is above 30", "price is between 10 and 20", "orders not in North",
                    "total quantity last month", "orders in the last 7 days", "total quantity above 3",
                    "total quantity except North", "total quantity per product", "top 5 products by price", "biggest month"]),
    "probes": (probes, ["compare logger_site_A and logger_site_B", "average pressure per minute",
                        "temperature against pressure", "gaps in logger_site_B", "spikes in pressure",
                        "hottest hour in logger_site_A"]),
    "monthly": (monthly, ["total sales by store", "sales per week", "sales by store for 2024-02", "compare Leeds and York",
                          "sales in Leeds vs York", "highest sales day", "total sales since February"]),
    "trap": (trap, ["total visit_cost by drug", "visit_cost by patient"]),
    "plain": (plain, ["weight against length", "average weight", "spread of width"]),
    "stock": (stock, ["total units by city", "average units"]),
}


# ------------------------------------------------------------------- the check
def describe(p: Pipeline, questions: list[str]) -> dict:
    ex = Executor(p)
    m = deepen(p, ex, understand(p, ex))
    out = {
        "tables": {t.title: {"shape": t.shape, "rows": t.rows, "time": t.time,
                             "columns": {c.name: c.role for c in t.columns}} for t in m.tables.values()},
        "relations": sorted(f"{r.kind} {' / '.join(m.tables[x].title for x in r.tables)}"
                            + (f" on {r.left_on}={r.right_on} ({r.cardinality})" if r.kind == "link" else "")
                            + (f" labels {r.labels}" if r.kind == "stack" else "")
                            + (f" within {r.tolerance}" if r.kind == "align" else "") for r in m.relations),
        "suggestions": [f"{s.recipe}: {s.title}" for s in suggest(m)],
        "questions": {},
    }
    for q in questions:
        a = ask(m, q)
        out["questions"][q] = a.title if a.ok else f"(not answered) {a.message}"
    return out, m


@pytest.mark.parametrize("case", sorted(CASES))
def test_corpus(case, tmp_path):
    make, questions = CASES[case]
    files = make(tmp_path)
    p = Pipeline(case); p.path = tmp_path / f"{case}.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f})
    got, m = describe(p, questions)
    stored = HERE / f"{case}.json"
    if UPDATE or not stored.exists():
        HERE.mkdir(exist_ok=True)
        stored.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert got == json.loads(stored.read_text(encoding="utf-8"))
    # and every suggestion is an answer that runs
    for s in suggest(m):
        q = Pipeline.from_dict(p.to_dict(), p.path)
        pl_ = plan(m, s.spec)
        res = instantiate(q, pl_)
        st = Executor(q).run(targets=[res[pl_.terminal]])[res[pl_.terminal]]
        assert st.status == "done", (case, s.title, st.error)
