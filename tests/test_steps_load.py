"""Loading data: files, sheets, headers, encodings, dates, and tables typed in by hand."""
from datetime import datetime

import polars as pl

from conftest import run_one
from dancr.core import Pipeline
from dancr.core.executor import Executor


def pipe_from_csv(path, tmp_path):
    p = Pipeline()
    p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(path)}, id="src")
    return p


def failed(p, node_id):
    st = Executor(p).run(targets=[node_id])[node_id]
    assert st.status == "failed", "expected a failure"
    return st.error


def _minute_log(tmp_path, fmt: str) -> Pipeline:
    ts = pl.datetime_range(datetime(2024, 1, 5), datetime(2024, 1, 20), "1m", eager=True)
    df = pl.DataFrame({"when": ts.dt.strftime(fmt), "v": range(len(ts))})
    df.write_csv(tmp_path / "log.csv")
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": "log.csv"}, id="src")
    p.path = tmp_path / "p.json"
    return p


def pipe_with(tmp_path, df: pl.DataFrame, name="t.parquet") -> Pipeline:
    f = tmp_path / name
    df.write_parquet(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    return p


def test_load_detects_dates_and_numbers(small_csv, tmp_path):
    p = pipe_from_csv(small_csv, tmp_path)
    df = run_one(p, "src")
    assert df.schema["t"] == pl.Datetime("us")
    assert df.schema["x"] == pl.Float64
    assert df["name"][9] is None


def test_load_separator_sniff_and_skip_rows(tmp_path):
    f = tmp_path / "semi.txt"
    f.write_text("junk line\nanother junk\na;b;c\n1;2;x\n3;4;y\n")
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f), "skip_rows": 2}, id="src")
    df = run_one(p, "src")
    assert df.columns == ["a", "b", "c"] and df["b"].to_list() == [2, 4]


def test_load_excel(tmp_path, small_df):
    f = tmp_path / "book.xlsx"
    small_df.write_excel(f)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    df = run_one(p, "src")
    assert len(df) == 10 and df.schema["t"] == pl.Datetime("us")


def test_load_missing_file_is_friendly(tmp_path):
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "nope.csv"}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "failed" and "File not found" in st.error


def test_a_forced_date_format_only_touches_date_columns(tmp_path):
    pl.DataFrame({"when": ["05/01/2024", "06/01/2024"], "name": ["alpha", "beta"]}).write_csv(tmp_path / "x.csv")
    p = Pipeline("t"); p.add_node("load_file", params={"path": "x.csv", "date_format": "%d/%m/%Y"}, id="src"); p.path = tmp_path / "p.json"
    df = run_one(p, "src")
    assert df["name"].to_list() == ["alpha", "beta"] and df["when"][0] == datetime(2024, 1, 5)


def test_typed_in_dates_are_read_cell_by_cell(tmp_path):
    p = Pipeline("t")
    p.add_node("enter_data", params={"columns": [{"name": "t", "type": "datetime"}],
                                     "rows": [["2024-06-03 09:00"], ["2024-06-10"], ["junk"]]}, id="e")
    p.path = tmp_path / "p.json"
    st = Executor(p).run()["e"]
    assert pl.read_parquet(st.output)["t"].to_list() == [datetime(2024, 6, 3, 9), datetime(2024, 6, 10), None]
    assert any("'junk'" in m for m in st.messages)


def test_a_sheet_named_like_a_number_is_that_sheet(tmp_path):
    import xlsxwriter
    wb = xlsxwriter.Workbook(tmp_path / "b.xlsx")
    for name in ("2023", "2024"):
        ws = wb.add_worksheet(name); ws.write_row(0, 0, ["year"]); ws.write_row(1, 0, [int(name)])
    wb.close()
    p = Pipeline("t"); p.path = tmp_path / "t.json"
    p.add_node("load_file", params={"path": "b.xlsx", "sheet": "2024"}, id="l")
    assert Executor(p).preview("l")[0]["year"].to_list() == [2024]


def test_load_sheet_zero_header_dupes_and_dash_values(tmp_path):
    f = tmp_path / "d.csv"
    good = "".join(f"{i},{i},2024-01-{i + 1:02d}\n" for i in range(9))
    f.write_text("a, a ,when\n1,-,2024-01-01\n2,3,not a date\n" + good)
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    assert df.columns == ["a", "a_2", "when"] and df["a_2"].to_list()[:2] == [None, 3.0]      # "-" is a blank among numbers
    assert any("Read 'a_2' as numbers" in m for m in st.messages)
    assert any("only differed by spaces" in m for m in st.messages)
    assert df.schema["when"] == pl.Datetime("us") and any("don't match and are left blank" in m for m in st.messages)
    xl = tmp_path / "b.xlsx"; pl.DataFrame({"x": [1]}).write_excel(xl)
    p.set_params("src", path=str(xl), sheet="0")
    assert "numbered from 1" in failed(p, "src")


def test_us_log_whose_first_rows_are_ambiguous_is_read_month_first(tmp_path):
    p = _minute_log(tmp_path, "%m/%d/%Y %H:%M")
    st = Executor(p).run()["src"]
    df = pl.read_parquet(st.output)
    assert df["when"].null_count() == 0
    assert df["when"][0] == datetime(2024, 1, 5)
    assert any("whole file fits month/day" in m or "month/day" in m for m in st.messages)


def test_european_log_whose_first_rows_are_ambiguous_is_read_day_first_over_the_whole_file(tmp_path):
    p = _minute_log(tmp_path, "%d/%m/%Y %H:%M")     # the first 2000 rows are all 05/01/2024 (5 January)
    st = Executor(p).run()["src"]
    df = pl.read_parquet(st.output)
    assert df["when"].null_count() == 0
    assert df["when"][0] == datetime(2024, 1, 5)
    assert any("whole file fits day/month" in m for m in st.messages)


def test_genuinely_ambiguous_dates_say_how_they_were_read(tmp_path):
    pl.DataFrame({"d": ["01/05/2024", "02/06/2024"], "v": [1, 2]}).write_csv(tmp_path / "x.csv")
    p = Pipeline("t"); p.add_node("load_file", "L", {"path": "x.csv"}, id="src"); p.path = tmp_path / "p.json"
    st = Executor(p).run()["src"]
    assert pl.read_parquet(st.output)["d"][0] == datetime(2024, 1, 5)
    assert any("Read as month/day" in m and "Day comes before month" in m for m in st.messages)
    p.set_params("src", day_first=True)
    assert pl.read_parquet(Executor(p).run()["src"].output)["d"][0] == datetime(2024, 5, 1)


def test_source_blank_report(tmp_path):
    f = tmp_path / "b.csv"
    rows = "".join(f"2024-01-01 00:00:{i:02d},{i}\n" for i in range(20))
    f.write_text("time,v\n" + rows + "not a time,2\n2024-01-01 00:00:59,\n")
    p = Pipeline(); p.path = tmp_path / "p.json"; p.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p).run()["src"]
    assert st.status == "done" and any("Blank or unreadable" in m and "time: 1" in m for m in st.messages)
    tb = p.add_node("time_buckets", params={"every": "1s"}); p.connect("src", tb.id)
    assert run_one(p, tb.id)["time"].null_count() == 0     # blank time rows skipped


def test_loader_keeps_time_zone_and_reads_latin1(tmp_path):
    df = pl.DataFrame({"t": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 2), "1d", eager=True, time_unit="ns")}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Oslo"))
    p = pipe_with(tmp_path, df)
    out = run_one(p, "src")
    assert out.schema["t"] == pl.Datetime("us", "Europe/Oslo") and out["t"][0].hour == 0
    f = tmp_path / "l1.csv"; f.write_bytes("name,v\nété,1\n".encode("latin-1"))
    p2 = Pipeline(); p2.path = tmp_path / "p2.json"
    p2.add_node("load_file", params={"path": str(f)}, id="src")
    st = Executor(p2).run()["src"]
    assert st.status == "failed" and "Latin-1" in st.error
    p2.set_params("src", encoding="latin1")
    assert run_one(p2, "src")["name"][0] == "été"


def _daily(tmp_path, rows, **params):
    (tmp_path / "t.csv").write_text("t,v\n" + "\n".join(f"{t},{v}" for t, v in rows))
    p = Pipeline(); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": str(tmp_path / "t.csv"), **params}, id="src")
    p.add_node("time_buckets", params={"every": "1d", "default_stats": ["sum"]}, id="day"); p.connect("src", "day")
    st = Executor(p).run()
    return st, [(r["t"].day, r["v"]) for r in pl.read_parquet(st["day"].output).iter_rows(named=True)] if st["day"].output else None


def test_times_with_one_offset_are_bucketed_in_that_offset(tmp_path):
    st, days = _daily(tmp_path, [("2024-03-01T23:30:00+02:00", 1), ("2024-03-02T00:30:00+02:00", 2)])
    assert days == [(1, 1), (2, 2)]                     # local midnight, not UTC midnight
    st, days = _daily(tmp_path, [("2024-03-01T23:30:00+05:30", 1), ("2024-03-02T00:30:00+05:30", 2)])
    assert days == [(1, 1), (2, 2)]


def test_times_at_several_offsets_stay_in_utc_unless_a_zone_is_chosen(tmp_path):
    rows = [("2024-03-30T23:30:00+00:00", 1), ("2024-03-31T23:30:00+01:00", 2), ("2024-04-01T00:30:00+01:00", 4)]
    st, days = _daily(tmp_path, rows)
    assert days == [(30, 1), (31, 6)] and any("several UTC offsets" in m for m in st["src"].messages)
    st, days = _daily(tmp_path, rows, time_zone="Europe/London")
    assert days == [(30, 1), (31, 2), (1, 4)]
    st, _ = _daily(tmp_path, rows, time_zone="Mars/Olympus")
    assert st["src"].status == "failed" and "not a time zone name" in st["src"].error
