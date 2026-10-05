"""Sheets laid out for people to read: banners over tables side by side, blocks one below another, section lines,
labels written once per run, headers in two rows, several tables on a sheet, and summary rows under the data."""
import csv
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.layout import group_labels, summary_stat, find_tables, Grid
from dancr.core.nodes.load import tables_in

from sheets import lab_sheet, write_rows


def load(tmp_path: Path, name: str, **params):
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": name, **params}, id="l")
    ex = Executor(p)
    st = ex.run()["l"]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output), st


def test_a_lab_sheet_of_two_tables_side_by_side_is_one_table_in_two_groups(tmp_path):
    data = lab_sheet(tmp_path / "lab.xlsx")
    df, st = load(tmp_path, "lab.xlsx")
    assert df.columns == ["N - Sample", "group", "inner area (IA) cm2", "outer area (OA) cm2", "gap", "mass (m) g", "mass/area x 10000 (g/m2)"]
    assert df.height == 30                                           # not the template row 16, not the summary rows
    assert df["group"].to_list() == ["Treated"] * 15 + ["Control"] * 15
    assert df["N - Sample"].to_list() == list(range(1, 16)) * 2 and df["N - Sample"].dtype == pl.Int64
    assert df.filter(pl.col("group") == "Control")["inner area (IA) cm2"].to_list() == [r[0] for r in data["control"]]
    lay = st.report["layout"]
    assert lay["kind"] == "groups" and lay["groups"] == ["Treated", "Control"]
    assert [s["label"] for s in lay["summary_rows"]] == ["AVERAGE", "STANDARD DEV", "MEDIAN"]
    assert lay["checks"] and all(c["ok"] for c in lay["checks"])
    text = " ".join(st.messages)
    assert "TREATED" in text and "They agree with the data" in text and "sheet row 18" in text


def test_a_stale_summary_value_is_named(tmp_path):
    lab_sheet(tmp_path / "lab.xlsx", stale_average=True)
    _, st = load(tmp_path, "lab.xlsx")
    bad = [c for c in st.report["layout"]["checks"] if not c["ok"]]
    assert len(bad) == 1 and bad[0]["column"] == "inner area (IA) cm2" and bad[0]["group"] == "Treated"
    assert any("AVERAGE of inner area (IA) cm2 for Treated is" in m and "but the data gives" in m for m in st.messages)


def test_read_as_it_is_keeps_every_row(tmp_path):
    lab_sheet(tmp_path / "lab.xlsx")
    df, _ = load(tmp_path, "lab.xlsx", layout="as_is")
    assert df.height >= 20 and "group" not in df.columns and "AVERAGE" in df[df.columns[0]].to_list()


def test_blocks_one_below_another_under_titles(tmp_path):
    rows = [(0, 0, ["Water survey 2024"])]
    r = 2
    for site in ("Site A", "Site B", "Site C"):
        rows += [(r, 0, [site]), (r + 1, 0, ["Date", "pH", "Temp (°C)"])]
        rows += [(r + 2 + i, 0, [f"2024-06-0{i + 1}", 7 + i / 10, 15 + i]) for i in range(5)]
        rows += [(r + 7, 0, ["Average", 7.2, 17])]
        r += 10
    write_rows(tmp_path / "b.xlsx", rows)
    df, st = load(tmp_path, "b.xlsx")
    assert df.columns[0] == "group" and df.height == 15
    assert df["group"].unique(maintain_order=True).to_list() == ["Site A", "Site B", "Site C"]
    assert df["Date"].dtype == pl.Datetime("us")                      # the Average rows no longer make it text
    assert len(st.report["layout"]["summary_rows"]) == 3


def test_section_lines_become_a_group_column(tmp_path):
    rows = [(0, 0, ["Item", "Qty", "Price"]), (1, 0, ["Fruit"])] + [(2 + i, 0, [f"F{i}", i + 1, 1.5]) for i in range(4)]
    rows += [(6, 0, ["Veg"])] + [(7 + i, 0, [f"V{i}", i + 1, 2.5]) for i in range(4)]
    write_rows(tmp_path / "s.xlsx", rows)
    df, _ = load(tmp_path, "s.xlsx")
    assert df.height == 8 and df["group"].to_list() == ["Fruit"] * 4 + ["Veg"] * 4


def test_a_label_written_once_per_run_is_filled_down(tmp_path):
    rows = [(0, 0, ["Region", "Store", "Sales"])]
    k = 1
    for reg in ("North", "South", "East"):
        for i in range(4):
            rows.append((k, 0, [reg if i == 0 else None, f"S{k}", 100 + k])); k += 1
    rows.append((k, 0, ["Total", None, sum(100 + j for j in range(1, k))]))
    write_rows(tmp_path / "f.xlsx", rows)
    df, st = load(tmp_path, "f.xlsx")
    assert df.height == 12 and df["Region"].to_list() == ["North"] * 4 + ["South"] * 4 + ["East"] * 4
    assert any("Filled “Region” down" in m for m in st.messages)


def test_a_header_in_two_rows(tmp_path):
    rows = [(0, 1, ["Pressure"]), (0, 3, ["Temperature"]), (1, 0, ["Day", "min", "max", "min", "max"])]
    rows += [(2 + i, 0, [i + 1, 1.0 + i, 2.0 + i, 10 + i, 20 + i]) for i in range(6)]
    write_rows(tmp_path / "h.xlsx", rows)
    df, _ = load(tmp_path, "h.xlsx")
    assert df.columns == ["Day", "Pressure min", "Pressure max", "Temperature min", "Temperature max"] and df.height == 6


def test_several_tables_on_one_sheet(tmp_path):
    rows = [(0, 0, ["Order", "Customer", "Amount"]), (0, 4, ["Customer", "Region"])]
    rows += [(1 + i, 0, [100 + i, f"C{i % 3}", 10 * i + 5]) for i in range(6)] + [(1 + i, 4, [f"C{i}", "NSE"[i]]) for i in range(3)]
    write_rows(tmp_path / "side.xlsx", rows)
    rows = [(0, 0, ["Orders"]), (1, 0, ["Order", "Customer", "Amount"])] + [(2 + i, 0, [100 + i, f"C{i % 3}", 10 * i + 5]) for i in range(6)]
    rows += [(10, 0, ["Customers"]), (11, 0, ["Customer", "Region", "Since"])] + [(12 + i, 0, [f"C{i}", "NSE"[i], 2020 + i]) for i in range(3)]
    write_rows(tmp_path / "below.xlsx", rows)
    a, st = load(tmp_path, "side.xlsx")
    assert a.columns == ["Order", "Customer", "Amount"] and a.height == 6 and a["Amount"].dtype == pl.Int64
    assert "This sheet holds 2 tables" in st.messages[0]
    b, _ = load(tmp_path, "side.xlsx", table=2)
    assert b.columns == ["Customer", "Region"] and b.height == 3
    c, _ = load(tmp_path, "below.xlsx")
    assert c.columns == ["Order", "Customer", "Amount"] and c.height == 6 and c["Order"].dtype == pl.Int64
    d, _ = load(tmp_path, "below.xlsx", table=2)
    assert d.columns == ["Customer", "Region", "Since"] and d.height == 3
    assert tables_in(tmp_path / "below.xlsx") == [("Orders", {"table": 1}), ("Customers", {"table": 2})]
    assert [t for t, _ in tables_in(tmp_path / "side.xlsx")] == ["side table 1", "side table 2"]
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "side.xlsx", "table": 3}, id="l")
    st = Executor(p).run()["l"]
    assert st.status == "failed" and "has 2 tables" in st.error


def test_summary_rows_under_a_csv_and_a_big_one(tmp_path):
    with open(tmp_path / "t.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Region", "Units", "Revenue"])
        w.writerows([["NS"[i % 2], i, i * 2.5] for i in range(10)]); w.writerow([]); w.writerow(["Total", 45, 112.5])
    df, st = load(tmp_path, "t.csv")
    assert df.height == 10 and st.report["layout"]["checks"] and all(c["ok"] for c in st.report["layout"]["checks"])
    with open(tmp_path / "big.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "value", "group"])
        w.writerows([[i, i % 10, "abc"[i % 3]] for i in range(25_000)]); w.writerow(["Average", 4.5, None]); w.writerow(["Total", 1, None])
    df, st = load(tmp_path, "big.csv")
    assert df.height == 25_000 and df["id"].dtype == pl.Int64 and df["id"].max() == 24_999
    assert [s["label"] for s in st.report["layout"]["summary_rows"]] == ["Average", "Total"]


def test_a_big_sheet_with_a_total_at_the_bottom(tmp_path):
    rows = [(0, 0, ["id", "value"])] + [(1 + i, 0, [i, i % 97]) for i in range(21_000)] + [(21_002, 0, ["Total", 123])]
    write_rows(tmp_path / "big.xlsx", rows)
    df, _ = load(tmp_path, "big.xlsx")
    assert df.height == 21_000 and df["id"].dtype == pl.Int64


def test_a_plain_table_reads_as_before(tmp_path):
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_csv(tmp_path / "p.csv")
    df, st = load(tmp_path, "p.csv")
    assert df.to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    assert st.report["layout"]["kind"] == "plain" and not st.report["layout"]["summary_rows"] and not st.messages


@pytest.mark.parametrize("text,stat", [("AVERAGE", ("mean", True)), ("Standard Dev.", ("std", True)), ("S.D.", ("std", False)),
                                       ("Grand Total:", ("sum", True)), ("Total sales", ("sum", True)), ("North", None),
                                       ("Mean ± SD", ("mean", True)), ("Northern", None)])
def test_summary_labels(text, stat):
    assert summary_stat(text) == stat


def test_group_labels_from_banners():
    assert group_labels(["TREATED PLOTS", "CONTROL PLOTS"]) == ["Treated", "Control"]
    assert group_labels(["Site A", "Site B"]) == ["Site A", "Site B"]
    assert group_labels(["Control", "Treated"]) == ["Control", "Treated"]
    assert group_labels(["UK sales", "US sales"]) == ["UK", "US"]


def test_a_sheet_of_words_only_is_not_a_table_of_numbers():
    g = Grid([["Notes"], ["call Bob"], ["buy milk"]], width=1)
    t = find_tables(g)
    assert len(t) == 1 and t[0].plain


def test_a_title_row_of_two_cells_is_not_banners(tmp_path):
    write_rows(tmp_path / "t.xlsx", [(0, 0, ["Sales report", "Q1"]), (1, 0, ["Region", "Units", "Revenue"])]
               + [(2 + i, 0, ["NS"[i % 2], i, i * 2.5]) for i in range(6)])
    df, st = load(tmp_path, "t.xlsx")
    assert df.columns == ["Region", "Units", "Revenue"] and df.height == 6 and st.report["layout"]["kind"] == "plain"
