"""The Load folder source: every file matching a folder/glob as one table, with a file column, and a cache
fingerprint that notices a change to any member."""
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from dancr.core.nodes.load_folder import matched_files


@pytest.fixture()
def folder(tmp_path):
    d = tmp_path / "exports"; d.mkdir()
    pl.DataFrame({"site": ["A", "A"], "yield": [3.1, 3.4]}).write_csv(d / "site_a.csv")
    pl.DataFrame({"site": ["B"], "yield": [2.9]}).write_csv(d / "site_b.csv")
    pl.DataFrame({"site": ["C"], "yield": [4.0], "rain": [12.0]}).write_csv(d / "site_c.csv")
    return d


def _project(tmp_path, d, **params):
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_folder", params={"path": str(d), **params}, id="trials")
    p.save()
    return p


def _out(p, node="trials"):
    ex = Executor(p)
    res = ex.run([node])
    assert res[node].status == "done", res[node].error
    return pl.read_parquet(res[node].output)


def test_reads_every_file_with_a_source_column(folder, tmp_path):
    p = _project(tmp_path, folder)
    df = _out(p)
    assert df.height == 4
    assert set(df["source_file"]) == {"site_a.csv", "site_b.csv", "site_c.csv"}
    # union of the columns: only site_c has rain, the rest are blank there
    assert "rain" in df.columns
    assert df.filter(pl.col("source_file") == "site_a.csv")["rain"].is_null().all()
    assert df.filter(pl.col("source_file") == "site_c.csv")["rain"].to_list() == [12.0]


def test_strict_refuses_different_columns(folder, tmp_path):
    p = _project(tmp_path, folder, unify="strict")
    ex = Executor(p)
    st = ex.run(["trials"])["trials"]
    assert st.status == "failed"
    assert "same columns" in st.error


def test_text_mode_reads_everything_as_text(folder, tmp_path):
    p = _project(tmp_path, folder, unify="text")
    df = _out(p)
    assert all(str(dt) in ("String", "Utf8") for dt in df.schema.values())


def test_pattern_filters_files(folder, tmp_path):
    p = _project(tmp_path, folder, pattern="site_[ab].csv")
    df = _out(p)
    assert set(df["source_file"]) == {"site_a.csv", "site_b.csv"}


def test_source_column_can_be_off(folder, tmp_path):
    p = _project(tmp_path, folder, source_column="")
    df = _out(p)
    assert "source_file" not in df.columns


def test_empty_folder_fails_with_a_plain_message(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    p = _project(tmp_path, empty)
    st = Executor(p).run(["trials"])["trials"]
    assert st.status == "failed" and "No data files matched" in st.error


def test_skip_unreadable_file(folder, tmp_path):
    (folder / "broken.csv").write_text("")            # an empty file cannot be read
    p = _project(tmp_path, folder, on_error="skip")
    df = _out(p)
    assert df.height == 4                              # the three good files, the empty one skipped


def test_order_is_deterministic(folder):
    a = [f.name for f in matched_files(folder, {"path": str(folder)})]
    b = [f.name for f in matched_files(folder, {"path": str(folder)})]
    assert a == sorted(a) and a == b


def test_every_table_of_a_workbook(tmp_path):
    import xlsxwriter
    d = tmp_path / "books"; d.mkdir()
    wb = xlsxwriter.Workbook(d / "site.xlsx")
    ws = wb.add_worksheet("yield"); ws.write_row(0, 0, ["plot", "kg"]); ws.write_row(1, 0, ["p1", 10])
    ws2 = wb.add_worksheet("soil"); ws2.write_row(0, 0, ["plot", "ph"]); ws2.write_row(1, 0, ["p1", 6.5])
    wb.close()
    p = _project(tmp_path, d, tables="all")
    df = _out(p).sort("source_table")
    assert set(df["source_table"]) == {"soil", "yield"}
    assert {"plot", "kg", "ph"} <= set(df.columns)


def test_match_table_by_name(tmp_path):
    import xlsxwriter
    d = tmp_path / "books"; d.mkdir()
    wb = xlsxwriter.Workbook(d / "site.xlsx")
    ws = wb.add_worksheet("yield"); ws.write_row(0, 0, ["plot", "kg"]); ws.write_row(1, 0, ["p1", 10])
    ws2 = wb.add_worksheet("soil"); ws2.write_row(0, 0, ["plot", "ph"]); ws2.write_row(1, 0, ["p1", 6.5])
    wb.close()
    p = _project(tmp_path, d, tables="match", table_match="soil")
    df = _out(p)
    assert set(df["source_table"]) == {"soil"} and "ph" in df.columns


def test_first_mode_stays_one_table_per_file(tmp_path):
    import xlsxwriter
    d = tmp_path / "books"; d.mkdir()
    wb = xlsxwriter.Workbook(d / "site.xlsx")
    ws = wb.add_worksheet("yield"); ws.write_row(0, 0, ["plot", "kg"]); ws.write_row(1, 0, ["p1", 10])
    ws2 = wb.add_worksheet("soil"); ws2.write_row(0, 0, ["plot", "ph"]); ws2.write_row(1, 0, ["p1", 6.5])
    wb.close()
    df = _out(_project(tmp_path, d, tables="first"))
    assert "source_table" not in df.columns and "ph" not in df.columns


def test_folder_path_rebases_on_save_as(tmp_path):
    from dancr.core.registry import resolve_path
    d = tmp_path / "exports"; d.mkdir()
    pl.DataFrame({"x": [1]}).write_csv(d / "a.csv")
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_folder", params={"path": "exports"}, id="f")
    p.save()
    p.save(tmp_path / "sub" / "q.json")
    # moving the project leaves the folder setting pointing at the same folder (relative when inside the new
    # project folder, absolute otherwise — as for a file path)
    assert resolve_path(p.directory, p.nodes["f"].params["path"]).resolve() == d.resolve()


def test_folder_members_count_as_source_files(tmp_path):
    from dancr import headless as hl
    d = tmp_path / "exports"; d.mkdir()
    f = d / "a.csv"
    pl.DataFrame({"x": [1]}).write_csv(f)
    p = Pipeline("p"); p.path = tmp_path / "p.json"
    p.add_node("load_folder", params={"path": "exports"}, id="f")
    assert f.resolve() in hl.source_files(p)          # so an export may not overwrite it


def test_cache_invalidates_when_a_member_changes(folder, tmp_path):
    p = _project(tmp_path, folder)
    ex = Executor(p)
    h1 = ex.plan_hash("trials")
    (folder / "site_a.csv").write_text("site,yield\nA,9.9\nA,9.8\n")
    assert ex.plan_hash("trials") != h1


def test_cache_invalidates_when_a_file_is_added(folder, tmp_path):
    p = _project(tmp_path, folder)
    ex = Executor(p)
    h1 = ex.plan_hash("trials")
    pl.DataFrame({"site": ["D"], "yield": [1.0]}).write_csv(folder / "site_d.csv")
    assert ex.plan_hash("trials") != h1
