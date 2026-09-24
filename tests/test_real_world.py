"""Answers on messy real-world files: numbers written as text, codes with leading zeros, a date and a time of day,
a total row at the bottom, values spelled several ways, and the way people actually phrase questions."""
import csv
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core import answers as A
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.planner import instantiate
from dancr.core.recipes import plan, suggest
from dancr.core.understand import understand, deepen


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def project(folder: Path, *files: str) -> tuple[Pipeline, object]:
    p = Pipeline("p"); p.path = folder / "p.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f}, id=Path(f).stem)
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex))


def result(p: Pipeline, spec: dict, key: str | None = None) -> pl.DataFrame:
    pl_ = plan(model_of(p), spec)
    res = instantiate(p, pl_)
    ex = Executor(p); ex.run()
    node = res[key] if key else res[pl_.terminal]
    if p.nodes[node].type == "chart":
        node = p.inputs_of(node)["in"][0]
    return pl.read_parquet(ex.state(node).output)


def model_of(p):
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


@pytest.fixture
def shop(tmp_path):
    regions = ["North", "north", "NORTH", "South", "East", "West ", "South", "North"]
    rows = [[f"{1000 + i}", (datetime(2024, 1, 1) + timedelta(days=i * 3)).strftime("%d/%m/%Y"), f"C{i % 7:04d}",
             ["Widget", "Gadget", "Doohickey"][i % 3], i % 5 + 1, f"£{(i % 9 + 1) * 12.5:,.2f}", regions[i % 8]] for i in range(80)]
    total = sum(float(r[5].replace("£", "").replace(",", "")) for r in rows)
    rows.append(["TOTAL", "", "", "", sum(r[4] for r in rows), f"£{total:,.2f}", ""])
    write(tmp_path / "orders.csv", ["Order ID", "Order Date", "Customer ID", "Product", "Qty", "Amount (£)", "Region"], rows)
    return tmp_path, total


def test_numbers_codes_and_totals_are_read_as_people_mean_them(shop):
    d, total = shop
    p, m = project(d, "orders.csv")
    t = m.tables["orders"]
    assert t.column("Amount (£)").role == "measure" and t.column("Customer ID").kind == "text"   # C0001 keeps its zeros
    assert t.total_row is not None
    df = result(p, ask(m, "total amount by region").spec)
    assert df["Amount (£)"].sum() == pytest.approx(total)                                        # the total row is not counted twice
    assert set(df["Region"].to_list()) == {"North", "South", "East", "West"}                      # one spelling each


@pytest.mark.parametrize("question,title", [
    ("total amount by region in 2024", "Total Amount (£) by Region where Order Date in 2024"),
    ("sales in March", "Total Amount (£) per week where Order Date in March"),
    ("biggest orders", "Biggest 10 orders by Amount (£)"),
    ("top 5 orders by amount", "Biggest 5 orders by Amount (£)"),
    ("how many orders by product", "Rows by Product"),
    ("total amount where order date between 2024-03-01 and 2024-03-31",
     "Total Amount (£) per week where Order Date from 2024-03-01 to 2024-03-31"),
])
def test_everyday_questions(shop, question, title):
    d, _ = shop
    _, m = project(d, "orders.csv")
    a = ask(m, question)
    assert a.ok, a.message
    assert a.title == title


def test_a_word_that_cannot_be_used_is_never_dropped(shop):
    d, _ = shop
    _, m = project(d, "orders.csv")
    a = ask(m, "total amount by region by wednesday")
    assert not a.ok and "wednesday" in a.message


def test_a_month_filter_keeps_only_that_month(shop):
    d, _ = shop
    p, m = project(d, "orders.csv")
    df = result(p, ask(m, "orders in March").spec)
    assert df.height and all(x.month == 3 for x in df["Order Date"].to_list())


def test_a_date_and_a_time_of_day_become_one_time(tmp_path):
    rows = [[(datetime(2024, 5, 1) + timedelta(minutes=10 * i)).strftime("%Y-%m-%d"),
             (datetime(2024, 5, 1) + timedelta(minutes=10 * i)).strftime("%H:%M:%S"), 20 + i % 3] for i in range(300)]
    write(tmp_path / "log.csv", ["Date", "Time", "temperature"], rows)
    p, m = project(tmp_path, "log.csv")
    t = m.tables["log"]
    assert t.time == "Date Time" and t.shape == "series"
    a = ask(m, "average temperature per hour")
    assert a.ok and a.spec["every"] == "1h"
    assert result(p, a.spec).height == 50


def test_text_cannot_be_averaged_and_says_so(tmp_path):
    write(tmp_path / "g.csv", ["Student", "Subject", "Score"], [[f"S{i}", ["Maths", "Art"][i % 2], "absent" if i % 2 else "B"] for i in range(20)])
    _, m = project(tmp_path, "g.csv")
    a = ask(m, "average score by subject")
    assert not a.ok and "text" in a.message


def test_a_few_notes_among_numbers_become_blanks(tmp_path):
    write(tmp_path / "g.csv", ["Student", "Subject", "Score"],
          [[f"S{i}", ["Maths", "Art"][i % 2], "absent" if i % 25 == 0 else str(50 + i % 40)] for i in range(100)])
    p, m = project(tmp_path, "g.csv")
    df = result(p, ask(m, "average score by subject").spec)
    assert df.height == 2 and df["Score"].null_count() == 0


def test_rows_per_lookup_and_columns_compared(tmp_path):
    write(tmp_path / "inventory.csv", ["Item Code", "Supplier ID", "Stock Level", "Reorder Level"],
          [[f"I{i:03d}", f"S{i % 3}", i % 10, 5] for i in range(30)])
    write(tmp_path / "suppliers.csv", ["Supplier ID", "Supplier Name"], [[f"S{i}", f"Supplier {i}"] for i in range(3)])
    p, m = project(tmp_path, "inventory.csv", "suppliers.csv")
    a = ask(m, "which supplier has the most items")
    assert a.ok and a.spec["by"] == ["suppliers", "Supplier Name"]
    low = result(p, ask(m, "items where stock level below reorder level").spec)
    assert low.height == 15 and (low["Stock Level"] < 5).all()


def test_per_file_questions_on_stacked_logs(tmp_path):
    for n in (1, 2, 3):
        write(tmp_path / f"device_{n}.csv", ["time", "temperature_c"],
              [[(datetime(2024, 1, 1) + timedelta(minutes=i)).isoformat(), 20 + n + (85 if (n == 2 and i == 100) else 0)] for i in range(300)])
    p, m = project(tmp_path, "device_1.csv", "device_2.csv", "device_3.csv")
    a = ask(m, "max temperature by device")
    assert a.ok and a.title == "Highest temperature_c by device"
    spikes = result(p, ask(m, "spikes in temperature").spec)
    assert spikes.height == 1 and spikes["source"].to_list() == ["device 2"]


def test_a_workbook_brings_in_every_sheet(tmp_path):
    import xlsxwriter
    from dancr import headless as hl
    wb = xlsxwriter.Workbook(tmp_path / "shop.xlsx")
    o = wb.add_worksheet("Orders"); o.write_row(0, 0, ["order_id", "customer_id", "amount"])
    for i in range(30):
        o.write_row(i + 1, 0, [i + 1, i % 5 + 1, 10.0 + i])
    c = wb.add_worksheet("Customers"); c.write_row(0, 0, ["customer_id", "name", "region"])
    for i in range(5):
        c.write_row(i + 1, 0, [i + 1, f"C{i}", ["N", "S"][i % 2]])
    wb.close()
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    ids = hl.add_files(p, [str(tmp_path / "shop.xlsx")])
    assert [p.nodes[i].title for i in ids] == ["Orders", "Customers"]
    out = hl.ask_question(p, "total amount by region")
    assert out["answer"]["title"] == "Total amount by region"


# ------------------------------------------------------------------ round two
def test_a_title_line_above_the_header_is_skipped(tmp_path):
    (tmp_path / "rep.csv").write_text("Monthly report — generated 2024-10-01\n\nRegion,Units,Revenue\n"
                                      + "".join(f"{r},{i},{i * 2.5}\n" for i, r in enumerate(["N", "S", "E", "W"] * 3)))
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "rep.xlsx"); ws = wb.add_worksheet()
    ws.write_row(0, 0, ["Report"]); ws.write_row(2, 0, ["Region", "Units"])
    for i in range(6):
        ws.write_row(3 + i, 0, [["N", "S"][i % 2], i])
    wb.close()
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    for f in ("rep.csv", "rep.xlsx"):
        p.add_node("load_file", params={"path": f}, id=f.replace(".", "_"))
    assert Executor(p).preview("rep_csv")[0].columns == ["Region", "Units", "Revenue"]
    df = Executor(p).preview("rep_xlsx")[0]
    assert df.columns == ["Region", "Units"] and df.height == 6


def test_excel_skip_rows_keeps_the_header_after_them(tmp_path):
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "b.xlsx"); ws = wb.add_worksheet()
    ws.write_row(0, 0, ["title"]); ws.write_row(1, 0, ["a", "b"]); ws.write_row(2, 0, [1, 2]); ws.write_row(3, 0, [3, 4])
    wb.close()
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "b.xlsx", "skip_rows": 1}, id="l")
    df = Executor(p).preview("l")[0]
    assert df.columns == ["a", "b"] and df["a"].to_list() == [1, 3]


@pytest.fixture
def pos(tmp_path):
    write(tmp_path / "tickets.csv", ["ticket_id", "opened_at", "table", "server", "subtotal", "tip"],
          [[1000 + i, (datetime(2024, 5, 1, 11) + timedelta(minutes=37 * i)).isoformat(), i % 12 + 1, ["Ana", "Bob", "IT"][i % 3],
            round(20 + i % 17 * 3.5, 2), round(2 + i % 5, 2)] for i in range(120)])
    return project(tmp_path, "tickets.csv")


@pytest.mark.parametrize("question,expect", [
    ("average tip per ticket", {"recipe": "single"}),                       # one row per ticket: the plain average
    ("average subtotal per table", {"recipe": "breakdown", "by": ["tickets", "table"]}),
    ("total tip for IT", {"filters": [{"column": ["tickets", "server"], "op": "eq", "value": "IT"}]}),   # a value spelled like a word
])
def test_round_two_questions(pos, question, expect):
    _, m = pos
    a = ask(m, question)
    assert a.ok, a.message
    for k, v in expect.items():
        assert a.spec.get(k) == v, a.spec


@pytest.mark.parametrize("question,words", [
    ("total by server", "which number"),
    ("tips per hour per day", "two time steps"),
])
def test_round_two_refusals(pos, question, words):
    _, m = pos
    a = ask(m, question)
    assert not a.ok and words in a.message


def test_weather_readings_are_averaged_and_the_hottest_day_is_one_day(tmp_path):
    write(tmp_path / "station.csv", ["date", "station", "tmax_F", "rain_mm"],
          [[(datetime(2024, 7, 1) + timedelta(days=i // 2)).date().isoformat(), ["Coast", "Hill"][i % 2], 70 + (i * 7) % 20, i % 4]
           for i in range(60)])
    p, m = project(tmp_path, "station.csv")
    assert suggest(m)[0].spec.get("stat") == "mean"
    a = ask(m, "hottest day")
    assert a.ok and a.title == "Hottest day (by tmax_F)"
    assert result(p, a.spec).height == 1


def test_a_column_named_with_its_sheet_uses_that_sheet(tmp_path):
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "budget.xlsx")
    for name, base in (("Budget", 100), ("Actual", 90)):
        ws = wb.add_worksheet(name); ws.write_row(0, 0, ["Department", "Jan", "Feb"])
        for i, dep in enumerate(["HR", "IT", "Ops"]):
            ws.write_row(i + 1, 0, [dep, base + i, base + 2 * i])
    wb.close()
    from dancr import headless as hl
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    hl.add_files(p, [str(tmp_path / "budget.xlsx")])
    m = hl.data_model(p)
    a = ask(m, "total Actual Jan by Department")
    assert a.ok and a.spec["together"] is False and a.title == "Total Jan by Department in Actual"
    df = result(p, a.spec)
    assert sorted(df["Jan"].to_list()) == [90, 91, 92]


def test_a_named_table_that_cannot_be_reached_is_refused(tmp_path):
    write(tmp_path / "orders.csv", ["order_id", "category", "total"], [[i, ["A", "B"][i % 2], i * 2.0] for i in range(40)])
    write(tmp_path / "returns.csv", ["return_id", "order_ref", "reason"], [[i, f"#{i}", ["size", "late"][i % 2]] for i in range(10)])
    _, m = project(tmp_path, "orders.csv", "returns.csv")
    a = ask(m, "returns by category")
    assert not a.ok and "not linked" in a.message


def test_a_wide_budget_is_answered_by_month(tmp_path):
    import xlsxwriter
    from dancr import headless as hl
    months = ["Jan", "Feb", "Mar", "Apr"]
    wb = xlsxwriter.Workbook(tmp_path / "budget_2024.xlsx")
    for name, base in (("Budget", 100), ("Actual", 90)):
        ws = wb.add_worksheet(name); ws.write_row(0, 0, ["Department", *months])
        for i, dep in enumerate(["HR", "IT", "Ops"]):
            ws.write_row(i + 1, 0, [dep, *[base + i + k for k in range(4)]])
    wb.close()
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    hl.add_files(p, [str(tmp_path / "budget_2024.xlsx")])
    m = hl.data_model(p)
    t = next(iter(m.tables.values()))
    assert t.wide and t.time == "date" and {c.name for c in t.columns} == {"Department", "month", "value", "date"}
    a = ask(m, "total value per month in Actual")
    assert a.ok, a.message
    df = result(p, a.spec)
    assert df.height == 4 and df["value"].to_list() == [sum(90 + i + k for i in range(3)) for k in range(4)]
    c = ask(m, "compare budget and actual")
    assert c.ok and c.spec["recipe"] == "breakdown" and c.spec["stat"] == "sum"


def test_shares_and_parts_of_the_day(pos):
    p, m = pos
    a = ask(m, "tips by hour of day")
    assert a.ok and a.spec["by_part"] == "hour"
    df = result(p, a.spec, "order")
    assert df["hour of the day"].to_list() == sorted(df["hour of the day"].to_list())
    b = ask(m, "share of tips by server")
    assert b.ok and b.title == "Share of tip by server (%)"
    shares = result(p, b.spec, "order")
    assert shares["share (%)"].sum() == pytest.approx(100.0)
    assert ask(m, "busiest day of the week").spec["by_part"] == "weekday"
