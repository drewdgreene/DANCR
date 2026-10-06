"""Reading questions: comparisons, several values of one column, dates and ranges of dates, words that name the
rows, and superlatives over time. Each answer is built and run on the corpus data and checked against the rows
it should hold, not just its title."""
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core import ask as A
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.planner import instantiate
from dancr.core.recipes import plan
from dancr.core.understand import understand, deepen

import test_corpus as corpus


def _case(tmp_path_factory, name: str):
    d = tmp_path_factory.mktemp(name)
    files = corpus.CASES[name][0](d)
    p = Pipeline(name); p.path = d / f"{name}.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f}, id=Path(f).stem.replace("-", "_"))
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex)), d


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    p, m, d = _case(tmp_path_factory, "shop")
    orders = pl.read_csv(d / "orders.csv", try_parse_dates=True)
    customers = pl.read_csv(d / "customers.csv")
    products = pl.DataFrame({"product_id": list(range(1, 9)), "product": [f"P{i}" for i in range(1, 9)]})   # as in products.xlsx
    return p, m, orders.join(customers, on="customer_id").join(products, on="product_id")


@pytest.fixture(scope="module")
def monthly(tmp_path_factory):
    p, m, d = _case(tmp_path_factory, "monthly")
    return p, m, pl.concat([pl.read_csv(d / f, try_parse_dates=True) for f in sorted(p.nodes[n].params["path"] for n in p.nodes)])


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    p, m, d = _case(tmp_path_factory, "probes")
    return p, m, pl.read_csv(d / "logger_site_A.csv", try_parse_dates=True)


def answer(p: Pipeline, m, question: str) -> tuple[dict, pl.DataFrame]:
    """The spec a question is read as, and the table its answer computes (a chart's input for a chart)."""
    a = ask(m, question)
    assert a.ok, (question, a.message)
    q = Pipeline.from_dict(p.to_dict(), p.path)
    pl_ = plan(m, a.spec)
    res = instantiate(q, pl_)
    ex = Executor(q)
    node = res[pl_.terminal]
    st = ex.run(targets=[node])[node]
    assert st.status == "done", st.error
    if q.nodes[node].type == "chart":
        node = q.inputs_of(node)["in"][0]
    return a.spec, pl.read_parquet(ex.state(node).output)


# ------------------------------------------------------------------- dates: comparisons, ranges, relative dates
@pytest.mark.parametrize("question,keep", [
    ("orders before March", lambda t: t < datetime(2024, 3, 1)),
    ("orders since March", lambda t: t >= datetime(2024, 3, 1)),
    ("orders after March 2024", lambda t: t >= datetime(2024, 4, 1)),
    ("orders until February", lambda t: t < datetime(2024, 3, 1)),
    ("orders from January to February", lambda t: t < datetime(2024, 3, 1)),
    ("orders between 2024-02-01 and 2024-02-10", lambda t: datetime(2024, 2, 1) <= t < datetime(2024, 2, 11)),
])
def test_a_month_or_date_keeps_its_comparison(shop, question, keep):
    p, m, rows = shop
    _, df = answer(p, m, question)
    want = sorted(t for t in rows["ordered_at"].to_list() if keep(t))
    assert want and sorted(df["ordered_at"].to_list()) == want


def test_from_one_date_to_another_is_a_range(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "average price from 2024-02-01 to 2024-03-01")
    assert spec["filters"][0]["op"] == "between" and len(spec["filters"]) == 1
    want = rows.filter((pl.col("ordered_at") >= datetime(2024, 2, 1)) & (pl.col("ordered_at") < datetime(2024, 3, 2)))["price"].mean()
    assert df["price"].to_list() == [pytest.approx(want)]


def test_relative_dates_are_taken_from_the_latest_date_in_the_data(shop):
    p, m, rows = shop
    latest = rows["ordered_at"].max()                   # 2024-04-29: "last month" is March 2024 whatever today is
    spec, df = answer(p, m, "total quantity last month")
    assert "March 2024" in spec["filters"][0]["text"]
    assert df["quantity"].to_list() == [rows.filter(pl.col("ordered_at").dt.month() == 3)["quantity"].sum()]
    _, df = answer(p, m, "orders in the last 7 days")
    assert sorted(df["ordered_at"].to_list()) == sorted(t for t in rows["ordered_at"].to_list() if t > latest - timedelta(days=7))
    _, df = answer(p, m, "orders this month")
    assert df.height and all(t.month == 4 for t in df["ordered_at"].to_list())


def test_two_separate_periods_are_refused_not_emptied(shop):
    _, m, _ = shop
    a = ask(m, "orders in March and April")
    assert not a.ok and "from March to April" in a.message


# ------------------------------------------------------------------- several values of one column
def test_values_of_one_column_are_either(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "total quantity for North and South")
    assert spec["filters"] == [{"column": ["customers", "region"], "op": "in", "value": ["North", "South"]}]
    assert df["quantity"].to_list() == [rows.filter(pl.col("region").is_in(["North", "South"]))["quantity"].sum()]
    _, df = answer(p, m, "orders in North or South")
    assert df.height == rows.filter(pl.col("region").is_in(["North", "South"])).height


def test_or_between_two_different_conditions_is_refused(shop):
    _, m, _ = shop
    a = ask(m, "orders in North or price above 30")
    assert not a.ok and "“or”" in a.message


@pytest.mark.parametrize("question", ["compare Leeds and York", "sales in Leeds vs York"])
def test_comparing_two_values_compares_those_groups(monthly, question):
    p, m, rows = monthly
    spec, df = answer(p, m, question)
    assert spec["recipe"] == "breakdown" and spec["by"][1] == "store" and spec.get("together")
    want = rows.filter(pl.col("store").is_in(["Leeds", "York"])).group_by("store").agg(pl.col("sales").sum())
    assert dict(zip(df["store"], df["sales"])) == pytest.approx(dict(zip(want["store"], want["sales"])))


# ------------------------------------------------------------------- everyday wordings
def test_is_and_not_are_read_as_people_mean_them(shop):
    p, m, rows = shop
    a = ask(m, "what is the average price")
    assert a.ok and a.spec["recipe"] == "single" and a.spec["stat"] == "mean"
    _, df = answer(p, m, "orders where price is above 30")
    assert df.height == rows.filter(pl.col("price") > 30).height
    _, df = answer(p, m, "price is between 10 and 20")
    assert df.height == rows.filter(pl.col("price").is_between(10, 20)).height
    _, df = answer(p, m, "price is not above 30")
    assert df.height == rows.filter(pl.col("price") <= 30).height
    _, df = answer(p, m, "orders not in North")
    assert df.height == rows.filter(pl.col("region") != "North").height


def test_a_value_binds_to_the_column_that_holds_it(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "total quantity above 3")
    assert spec["measure"] == ["orders", "quantity"]
    assert df["quantity"].to_list() == [rows.filter(pl.col("quantity") > 3)["quantity"].sum()]
    spec, df = answer(p, m, "total quantity except North")
    assert spec["filters"] == [{"column": ["customers", "region"], "op": "ne", "value": "North"}]
    assert df["quantity"].to_list() == [rows.filter(pl.col("region") != "North")["quantity"].sum()]


def test_a_group_named_by_its_id_is_shown_by_its_name(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "total quantity per product")
    assert spec["by"] == ["products", "product"]
    assert sorted(df["product"].to_list()) == sorted(rows["product"].unique().to_list())
    a = ask(m, "top 5 products by price")
    assert a.ok and a.spec["by"] == ["products", "product"] and a.title.startswith("Top 5 products")


# ------------------------------------------------------------------- superlatives over time
def test_the_highest_day_is_the_day_with_the_highest_total(monthly):
    p, m, rows = monthly
    spec, df = answer(p, m, "highest sales day")
    assert spec["recipe"] == "top" and spec["every"] == "1d"
    daily = rows.group_by("date").agg(pl.col("sales").sum()).sort("sales", descending=True)
    assert df.height == 1 and df["sales"].to_list() == [pytest.approx(daily["sales"][0])]


def test_the_hottest_hour_averages_each_hour_first(probes):
    p, m, rows = probes
    a = ask(m, "hottest hour in logger_site_A")
    assert a.ok, a.message
    q = Pipeline.from_dict(p.to_dict(), p.path)
    pl_ = plan(m, a.spec)
    res = instantiate(q, pl_)
    ex = Executor(q); ex.run(targets=[res[pl_.terminal]])
    df = pl.read_parquet(ex.state(res[pl_.terminal]).output)
    hourly = rows.group_by_dynamic("time", every="1h").agg(pl.col("temperature").mean()).sort("temperature", descending=True)
    assert df.height == 1 and df["temperature"].to_list() == [pytest.approx(hourly["temperature"][0])]


# ------------------------------------------------------------------- the word lists
def test_everyday_words_do_not_shadow_the_vocabulary():
    fixed = (set(A.STATS) | set(A.RECIPE_WORDS) | A.BY_WORDS | set(A.OPS) | set(A.TIME_UNITS) | set(A.TOP_WORDS)
             | set(A.MONTH_WORDS) | set(A.PARTS) | set(A.SHARE_WORDS) | set(A.ADVERBS) | set(A.ADJECTIVES)
             | A.COPULA | set(A.RELATIVE) | set(A.DAY_WORDS) | {"and", "or"})
    assert not A.STOP & fixed


def test_day_of_the_week_labels_the_right_days(shop):
    p, m, rows = shop
    _, df = answer(p, m, "total quantity by day of the week")
    want = rows.group_by(pl.col("ordered_at").dt.weekday().alias("d")).agg(pl.col("quantity").sum())   # Monday = 1
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    label = next(c for c in df.columns if c != "quantity")
    got = dict(zip(df[label].to_list(), df["quantity"].to_list()))
    assert got == {f"{d} {names[d - 1]}": q for d, q in want.iter_rows()}


# ------------------------------------------------------------------- dates written every way, and dates that cannot meet
def test_from_one_month_and_another_is_both_months(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "total quantity from March and April")
    assert len(spec["filters"]) == 1 and spec["filters"][0]["op"] == "between"
    assert df["quantity"].to_list() == [rows.filter(pl.col("ordered_at").dt.month().is_in([3, 4]))["quantity"].sum()]


@pytest.mark.parametrize("question,words", [
    ("total quantity before 2024-02-01 and after 2024-03-01", "No row is both"),
    ("orders from 2024-03-01 to 2024-02-01", "Put the earlier one first"),
    ("orders in the last 0 days", "no time at all"),
    ("orders before 01/02/2024", "day/month or month/day"),
])
def test_dates_no_row_can_be_in_are_refused_not_emptied(shop, question, words):
    _, m, _ = shop
    a = ask(m, question)
    assert not a.ok and words in a.message, a.message


def test_a_month_without_its_year_is_the_latest_one_up_to_the_data(shop):
    p, m, rows = shop                                   # January to April 2024: December is December 2023
    spec, df = answer(p, m, "orders since December")
    assert "December 2023" in spec["filters"][0]["text"] and df.height == rows.height


@pytest.mark.parametrize("question,day", [
    ("orders on 1 Feb 2024", (2, 1)), ("orders on Feb 1", (2, 1)), ("orders on 2024/02/01", (2, 1)),
    ("orders on 25/03/2024", (3, 25)), ("orders in 2024-02", (2, None)),
])
def test_dates_written_as_people_write_them(shop, question, day):
    p, m, rows = shop
    _, df = answer(p, m, question)
    want = rows.filter((pl.col("ordered_at").dt.month() == day[0])
                       & ((pl.col("ordered_at").dt.day() == day[1]) if day[1] else pl.lit(True)))
    assert want.height and sorted(df["order_id"].to_list()) == sorted(want["order_id"].to_list())


def test_a_range_of_days_named_by_month(monthly):
    p, m, rows = monthly
    _, df = answer(p, m, "total sales from 1 Feb to 10 Feb")
    want = rows.filter(pl.col("date").is_between(datetime(2024, 2, 1), datetime(2024, 2, 10)))["sales"].sum()
    assert df["sales"].to_list() == [pytest.approx(want)]


# ------------------------------------------------------------------- numbers: lists, ranges, thousands
def test_a_comma_between_numbers_is_a_list_unless_it_groups_thousands(shop):
    p, m, rows = shop
    spec, df = answer(p, m, "orders where customer_id is 1,2")
    assert spec["filters"] == [{"column": ["orders", "customer_id"], "op": "in", "value": [1, 2]}]
    assert sorted(df["order_id"].to_list()) == sorted(rows.filter(pl.col("customer_id").is_in([1, 2]))["order_id"].to_list())
    spec, _ = answer(p, m, "orders where price above 1,000")
    assert spec["filters"][0]["value"] == 1000


def test_number_ranges_in_everyday_words(shop):
    p, m, rows = shop
    _, df = answer(p, m, "orders where price from 10 to 20")
    assert df.height == rows.filter(pl.col("price").is_between(10, 20)).height
    _, df = answer(p, m, "orders where price not between 10 and 20")
    assert df.height == rows.filter((pl.col("price") < 10) | (pl.col("price") > 20)).height > 0


# ------------------------------------------------------------------- what "top" and "biggest" rank
@pytest.mark.parametrize("question,n,bottom", [("top 3 days by sales", 3, False), ("worst 3 days by sales", 3, True),
                                                ("best 3 days", 3, False), ("best day", 1, False)])
def test_top_days_add_each_day_up_first(monthly, question, n, bottom):
    p, m, rows = monthly
    spec, df = answer(p, m, question)
    assert spec["recipe"] == "top" and spec["every"] == "1d"
    daily = rows.group_by("date").agg(pl.col("sales").sum()).sort("sales", descending=not bottom)
    assert df["sales"].to_list() == pytest.approx(daily["sales"].head(n).to_list())


def test_top_months_by_a_number(shop):
    p, m, rows = shop
    _, df = answer(p, m, "top 3 months by quantity")
    monthly = rows.group_by(pl.col("ordered_at").dt.month()).agg(pl.col("quantity").sum()).sort("quantity", descending=True)
    assert df["quantity"].to_list() == monthly["quantity"].head(3).to_list()


def test_which_day_had_the_highest_is_one_day(monthly):
    p, m, rows = monthly
    _, df = answer(p, m, "which day had the highest sales")
    daily = rows.group_by("date").agg(pl.col("sales").sum()).sort("sales", descending=True)
    assert df.height == 1 and df["sales"].to_list() == [pytest.approx(daily["sales"][0])]


@pytest.mark.parametrize("question,n", [("top 5 orders by price", 5), ("biggest orders by price", 10)])
def test_the_number_after_by_is_what_rows_are_ranked_by(shop, question, n):
    p, m, rows = shop
    _, df = answer(p, m, question)
    assert df["price"].to_list() == rows.sort("price", descending=True)["price"].head(n).to_list()


def test_a_superlative_of_a_named_number_is_one_value(probes):
    p, m, rows = probes
    spec, df = answer(p, m, "lowest temperature in logger_site_A")
    assert spec["recipe"] == "single" and df["temperature"].to_list() == [rows["temperature"].min()]


def test_top_lookup_rows_in_a_period_rank_by_the_rows_that_point_at_them(shop):
    p, m, rows = shop
    _, df = answer(p, m, "top 3 customers in March")
    counts = rows.filter(pl.col("ordered_at").dt.month() == 3).group_by("customer_name").len()
    assert df["rows"].to_list() == counts.sort("len", descending=True)["len"].head(3).to_list()


def test_comparing_values_of_a_lookup_compares_the_rows_that_point_at_them(shop):
    p, m, rows = shop
    _, df = answer(p, m, "compare North and South")
    want = rows.filter(pl.col("region").is_in(["North", "South"])).group_by("region").agg(pl.col("quantity").sum())
    assert dict(zip(df["region"], df["quantity"])) == dict(zip(want["region"], want["quantity"]))


@pytest.mark.parametrize("question", ["total quantity by region by segment", "total quantity by region and segment"])
def test_two_groups_at_once_are_refused(shop, question):
    _, m, _ = shop
    a = ask(m, question)
    assert not a.ok and "one thing at a time" in a.message


# ------------------------------------------------------------------- refused rather than answered with something else
def _refused(m, question: str) -> str:
    from dancr.core.bank import Bank
    a = ask(m, question, bank=Bank([], []))            # the grammar's own answer, no recall behind it
    assert not a.ok, (question, a.spec)
    return a.message


@pytest.mark.parametrize("question", ["top 0 orders by price", "top 2.5 orders by price", "top -3 orders by price",
                                      "bottom 0 customers by quantity"])
def test_top_needs_a_whole_number_of_one_or_more(shop, question):
    _, m, _ = shop
    assert "whole number of 1 or more" in _refused(m, question)


def test_a_spec_with_no_rows_to_keep_is_refused_and_one_without_n_keeps_ten(shop):
    from dancr.core.recipes import PlanError
    p, m, rows = shop
    spec = ask(m, "top 3 orders by price").spec
    for bad in (0, -3, 2.5, True, "5"):
        with pytest.raises(PlanError, match="whole number of 1 or more"):
            plan(m, {**spec, "n": bad})
    q = Pipeline.from_dict(p.to_dict(), p.path)
    pl_ = plan(m, {k: v for k, v in spec.items() if k != "n"})
    node = instantiate(q, pl_)[pl_.terminal]
    ex = Executor(q); ex.run(targets=[node])
    node = q.inputs_of(node)["in"][0] if q.nodes[node].type == "chart" else node
    assert pl.read_parquet(ex.state(node).output).height == 10


@pytest.mark.parametrize("question", ["orders where price between 20 and 10", "orders where price from 20 to 10",
                                      "orders where price not between 20 and 10"])
def test_a_range_of_numbers_written_backwards_is_refused_as_dates_are(shop, question):
    _, m, _ = shop
    assert "Put the smaller one first" in _refused(m, question)
    assert ask(m, question.replace("20", "x").replace("10", "20").replace("x", "10")).ok


@pytest.mark.parametrize("question", ["total quantity on 2024-02-30", "total quantity in 2024-13-01",
                                      "total quantity in 2024-13", "orders from 2024-02-31 to 2024-03-05",
                                      "orders on 31/02/2024"])
def test_a_day_that_does_not_exist_is_refused_when_the_question_is_read(shop, question):
    _, m, _ = shop
    assert "is not a date" in _refused(m, question)


@pytest.mark.parametrize("question", ["total quantity by order id", "count per order id", "share of quantity by order id",
                                      "quantity per order id"])
def test_a_total_per_unique_id_is_refused_not_answered_for_every_row(shop, question):
    _, m, _ = shop
    msg = _refused(m, question)
    assert "order_id is different on every row" in msg and "“order id”" in msg


@pytest.mark.parametrize("question,stat", [("average price per order", "mean"), ("highest price per order id", "max")])
def test_an_average_per_unique_id_is_the_plain_average(shop, question, stat):
    p, m, rows = shop
    spec, df = answer(p, m, question)
    assert spec["recipe"] == "single" and spec["stat"] == stat
    assert df["price"].to_list() == [pytest.approx(rows["price"].mean() if stat == "mean" else rows["price"].max())]


@pytest.mark.parametrize("question", ["total quantity by region region", "total quantity quantity by region"])
def test_a_column_named_twice_is_refused(shop, question):
    _, m, _ = shop
    assert "is named twice" in _refused(m, question)


# --------------------------------------------------- shares, bare readings and "biggest <period> by <measure>"
def _frame_model(tmp_path, df: pl.DataFrame, title: str = "sales"):
    p = Pipeline(title); p.path = tmp_path / "p.json"
    df.write_csv(tmp_path / f"{title}.csv")
    p.add_node("load_file", title=title, params={"path": f"{title}.csv"}, id=title)
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex))


def test_a_share_of_a_reading_is_the_part_of_the_total(tmp_path):
    p, m = _frame_model(tmp_path, pl.DataFrame({"region": ["North", "South", "South", "South"],
                                                "price": [100.0, 10.0, 10.0, 10.0]}))
    spec, df = answer(p, m, "share of price by region")
    assert df["share (%)"].sum() == pytest.approx(100.0)                 # a share of the whole, not of the sum of averages
    north = df.filter(pl.col("region") == "North")["share (%)"][0]
    assert north == pytest.approx(100 * 100 / 130)


def test_a_share_of_an_average_is_refused_not_silently_wrong(tmp_path):
    _, m = _frame_model(tmp_path, pl.DataFrame({"region": ["North", "South", "South", "South"],
                                                "price": [100.0, 10.0, 10.0, 10.0]}))
    assert "only defined for totals" in _refused(m, "share of average price by region")


def test_a_bare_reading_is_averaged_not_summed(tmp_path):
    p, m = _frame_model(tmp_path, pl.DataFrame({"site": ["A", "B", "C"], "temperature": [10.0, 3.0, 50.0]}), "sensors")
    spec, df = answer(p, m, "temperature")
    assert spec["recipe"] == "single" and spec["stat"] == "mean"
    assert df["temperature"][0] == pytest.approx(21.0)                 # mean of 10, 3, 50 — not the sum 63


def test_biggest_period_by_a_measure_is_the_top_period(tmp_path):
    when = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 3, 31), "1d", eager=True)
    df = pl.DataFrame({"when": when, "quantity": [1] * len(when)}).with_columns(
        pl.when(pl.col("when").dt.month() == 2).then(100).otherwise(pl.col("quantity")).alias("quantity"))
    p, m = _frame_model(tmp_path, df, "orders")
    spec, _ = answer(p, m, "biggest month by quantity")
    assert spec["recipe"] == "top" and spec["every"] == "1mo" and spec.get("n") == 1
    # "highest quantity per month" is a line of monthly maxima, not the single biggest month
    spec2, _ = answer(p, m, "highest quantity per month")
    assert spec2["recipe"] == "trend"
