"""A small study in a spreadsheet: does DANCR find the question it was made to answer, answer it with the right
statistics, notice what a student in a hurry would miss, and read the questions students ask?"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from dancr.core import Pipeline
from dancr.core.ask import ask
from dancr.core.derived import find
from dancr.core.executor import Executor
from dancr.core.planner import instantiate
from dancr.core.recipes import plan, suggest, apply_choice
from dancr.core.registry import registry, Ctx
from dancr.core.units import header_parts, quantity_of
from dancr.core.understand import understand, deepen
from dancr.views.lod import bar_data

from sheets import lab_sheet

MEASURES = ["Leaf area (LA) cm2", "polygon area (PA) cm2", "D", "m (g)", "m/LA x 10000 (g/m2)"]


@pytest.fixture
def lab(tmp_path):
    data = lab_sheet(tmp_path / "Leaves & Light Data Sheet.xlsx")
    p = Pipeline("lab"); p.path = tmp_path / "lab.json"
    p.add_node("load_file", title="leaves", params={"path": "Leaves & Light Data Sheet.xlsx"}, id="leaves")
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex)), data


def run(p: Pipeline, spec: dict):
    pl_ = plan(model_of(p), spec)
    ids = instantiate(p, pl_)
    ex = Executor(p)
    ex.run()
    return ex.state(ids[pl_.terminal]), pl_


def model_of(p):
    ex = Executor(p)
    return deepen(p, ex, understand(p, ex))


def compare(df: pl.DataFrame, **params):
    res = registry.get("compare_groups").apply(Ctx(Path("."), "g", "g"), {"in": [df.lazy()]}, params)
    return res.frame.collect(), res


# ------------------------------------------------------------------ what DANCR understands
def test_the_study_is_understood(lab):
    _, m, _ = lab
    t = m.tables["leaves"]
    assert t.rows == 30
    roles = {c.name: c.role for c in t.columns}
    assert roles["N - Leaf"] == "id" and roles["group"] == "category" and all(roles[c] == "measure" for c in MEASURES)
    la = t.column("Leaf area (LA) cm2")
    assert (la.abbrev, la.unit, la.quantity) == ("LA", "cm2", "area")
    assert t.column("m (g)").quantity == "mass" and t.column("m/LA x 10000 (g/m2)").quantity == "mass per area"
    d = t.column("D").derived
    assert d["formula"] == "[polygon area (PA) cm2] - [Leaf area (LA) cm2]" and d["holds"] == 29
    assert d["breaks"] == [{"row": 15, "value": 1.26, "expected": pytest.approx(d["breaks"][0]["expected"]),
                            "where": {"N - Leaf": 1, "group": "Shade"}}]
    assert t.column("m/LA x 10000 (g/m2)").derived["k"] == 10000
    # related by definition, so not offered as a finding
    assert not any({p_["x"], p_["y"]} in ({"D", "polygon area (PA) cm2"}, {"D", "Leaf area (LA) cm2"}) for p_ in t.pairs)


def test_the_first_suggestion_compares_the_groups_and_the_second_catches_the_slip(lab):
    _, m, _ = lab
    s = suggest(m)
    assert [x.recipe for x in s[:2]] == ["groups", "quality"]
    assert "D" in s[1].why and "break" in s[1].why


def test_the_groups_answer(lab):
    p, m, data = lab
    st, pl_ = run(p, suggest(m)[0].spec)
    assert st.status == "done", st.error
    df = pl.read_parquet(st.output)
    row = df.filter(pl.col("measure") == "Leaf area").row(0, named=True)
    assert row["unit"] == "cm2"
    sun = [r[0] for r in data["sun"]]; shade = [r[0] for r in data["shade"]]
    assert row["Sun mean"] == pytest.approx(np.mean(sun)) and row["Shade SD"] == pytest.approx(np.std(shade, ddof=1))
    assert row["Sun SE"] == pytest.approx(np.std(sun, ddof=1) / np.sqrt(15)) and row["Sun n"] == 15
    assert row["p value"] == pytest.approx(stats.ttest_ind(sun, shade, equal_var=False).pvalue)
    assert row["test"] == "Welch's t-test" and row["result"].startswith("Shade higher")
    statement = st.report["finding"]["statement"]
    assert statement.startswith("Shade has") and "Leaf area" in statement
    # the size check: D is bigger in shade leaves only because they are bigger; for their size, sun leaves are more lobed
    rel = [r for r in st.report["relative"] if r["turned"]]
    assert rel and rel[0]["measure"].startswith("D per") and "Sun" in rel[0]["sentence"]
    assert "the other way round" in statement
    assert any("Welch's t-test, which does not assume the groups vary alike" in msg for msg in st.messages)
    texts = [a["text"] for a in pl_.assumptions]
    assert any("Welch" in t for t in texts) and any("Kept as typed" in t and "1.26" in t for t in texts)


def test_using_the_calculated_value_puts_the_slip_right(lab):
    p, m, _ = lab
    spec = suggest(m)[0].spec
    pl_ = plan(m, spec)
    choice = next(a for a in pl_.assumptions if a["id"] == "breaks")["choices"][0]
    st, pl2 = run(p, apply_choice(spec, "set", choice["set"]))
    df = pl.read_parquet(st.output)
    fix = next(s for s in pl2.steps if s.type == "fix_values")
    assert fix.params["fixes"][0]["row"] == 16 and fix.params["fixes"][0]["was"] == 1.26
    d = df.filter(pl.col("measure") == "D").row(0, named=True)
    assert d["p value"] < 0.05                                         # with the slip put right, D clearly differs
    assert any("Used the calculated D" in a["text"] for a in pl2.assumptions)


def test_the_check_names_the_slip(lab):
    p, m, _ = lab
    st, _ = run(p, {"recipe": "quality", "table": "leaves"})
    s = st.report["finding"]["statement"]
    assert s.startswith("A calculated value looks mistyped") and "row 16 (N - Leaf 1, group Shade) says 1.26" in s


# ------------------------------------------------------------------ the questions students ask
@pytest.mark.parametrize("question,expect", [
    ("compare sun and shade leaves", {"recipe": "groups", "by": ["leaves", "group"]}),
    ("is there a difference between sun and shade", {"recipe": "groups"}),
    ("compare sun and shade leaf area", {"recipe": "groups", "measures": [["leaves", "Leaf area (LA) cm2"]]}),
    ("is leaf mass per area higher in sun leaves", {"recipe": "groups", "measures": [["leaves", "m/LA x 10000 (g/m2)"]]}),
    ("t test leaf area", {"recipe": "groups", "test": "welch"}),
    ("mann whitney mass", {"recipe": "groups", "test": "rank", "measures": [["leaves", "m (g)"]]}),
    ("what is the standard deviation of leaf area", {"recipe": "single", "stat": "std"}),
    ("average LA", {"recipe": "single", "measure": ["leaves", "Leaf area (LA) cm2"]}),
    ("average D", {"recipe": "single", "measure": ["leaves", "D"]}),
    ("median mass", {"recipe": "single", "measure": ["leaves", "m (g)"], "stat": "median"}),
    ("average m/LA", {"recipe": "single", "measure": ["leaves", "m/LA x 10000 (g/m2)"]}),
    ("average leaf area of sun leaves", {"recipe": "single", "filters": [{"column": ["leaves", "group"], "op": "eq", "value": "Sun"}]}),
    ("relationship between mass and leaf area", {"recipe": "relationship", "y": ["leaves", "m (g)"]}),
    ("which leaf is biggest", {"recipe": "toprows", "n": 1}),
    ("top 5 leaves by leaf area", {"recipe": "toprows", "n": 5}),
    ("what drives mass", {"recipe": "drivers", "target": ["leaves", "m (g)"]}),
    ("unusual leaves", {"recipe": "outliers"}),
])
def test_student_questions(lab, question, expect):
    _, m, _ = lab
    a = ask(m, question)
    assert a.ok, a.message
    for k, v in expect.items():
        assert a.spec.get(k) == v, (question, a.spec)


def test_unusual_leaves_are_judged_against_their_own_group(lab):
    p, m, data = lab
    a = ask(m, "unusual leaves")
    st, pl_ = run(p, a.spec)
    assert pl_.title == "Unusual leaves" and st.status == "done"
    flag = next(s for s in pl_.steps if s.type == "remove_outliers")
    assert flag.params["by"] == ["group"] and flag.params["method"] == "iqr" and len(flag.params["columns"]) == 5


def test_a_word_inside_a_column_name_is_not_corrected_into_another(lab):
    _, m, _ = lab
    a = ask(m, "average polygon")
    assert a.ok and a.spec["measure"] == ["leaves", "polygon area (PA) cm2"]
    b = ask(m, "average blade")                                        # not a word of this project
    assert not b.ok and "blade" in b.unknown


# ------------------------------------------------------------------ the step on its own
def test_compare_groups_matches_scipy():
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(10, 1, 20), rng.normal(11, 3, 25), rng.normal(10.5, 2, 30)
    df = pl.DataFrame({"g": ["a"] * 20 + ["b"] * 25 + ["c"] * 30, "x": np.concatenate([a, b, c])})
    two = df.filter(pl.col("g") != "c")
    out, _ = compare(two, by="g", columns=["x"], test="student")
    assert out["p value"][0] == pytest.approx(stats.ttest_ind(a, b).pvalue)
    out, _ = compare(two, by="g", columns=["x"], test="rank")
    assert out["p value"][0] == pytest.approx(stats.mannwhitneyu(a, b).pvalue)
    out, _ = compare(df, by="g", columns=["x"], test="student")
    assert out["p value"][0] == pytest.approx(stats.f_oneway(a, b, c).pvalue) and out["test"][0] == "one-way ANOVA"
    out, _ = compare(df, by="g", columns=["x"], test="welch")
    assert out["test"][0] == "Welch's ANOVA" and 0 < out["p value"][0] < 1
    out, _ = compare(df, by="g", columns=["x"], test="rank")
    assert out["p value"][0] == pytest.approx(stats.kruskal(a, b, c).pvalue)
    g = out["effect"][0]
    assert "eta²" in g


def test_compare_groups_on_a_big_table():
    n = 1_000_000
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"g": rng.choice(["web", "store"], n), "amount": rng.normal(50, 10, n)})
    df = df.with_columns(pl.when(pl.col("g") == "web").then(pl.col("amount") + 0.1).otherwise(pl.col("amount")))
    out, res = compare(df, by="g", columns=["amount"])
    assert out["web n"][0] + out["store n"][0] == n
    assert "negligible" in out["effect"][0]                    # with a million rows a tiny difference is "clear" but small


def test_compare_groups_refuses_what_it_cannot_do():
    df = pl.DataFrame({"g": [str(i) for i in range(40)], "x": list(range(40))})
    with pytest.raises(ValueError, match="at most 12"):
        compare(df, by="g", columns=["x"])
    with pytest.raises(ValueError, match="only one group"):
        compare(pl.DataFrame({"g": ["a"] * 5, "x": [1, 2, 3, 4, 5]}), by="g", columns=["x"])
    with pytest.raises(ValueError, match="text"):
        compare(pl.DataFrame({"g": ["a", "b"] * 3, "x": ["p"] * 6}), by="g", columns=["x"])


# ------------------------------------------------------------------ the pieces
def test_calculated_columns():
    df = pl.DataFrame({"a": [1.5, 2.0, 3.25, 4.0, 5.5, 6.0], "b": [0.5, 1.0, 1.25, 2.0, 2.5, 3.0]})
    df = df.with_columns((pl.col("a") - pl.col("b")).alias("diff"), (pl.col("a") * pl.col("b") * 100).alias("prod"),
                         (pl.col("a") * 2.54).round(4).alias("cm"))
    got = {d.target: (d.op, d.operands) for d in find(df)}
    assert got == {"diff": ("diff", ["a", "b"]), "prod": ("product", ["a", "b"]), "cm": ("scale", ["a"])}
    zeros = pl.DataFrame({"a": [0.0] * 8, "b": [0.0] * 8, "c": [0.0] * 8})
    assert find(zeros) == []


@pytest.mark.parametrize("name,parts,quantity", [
    ("Leaf area (LA) cm2", ("Leaf area", "LA", "cm2"), "area"), ("m (g)", ("m", "", "g"), "mass"),
    ("m/LA x 10000 (g/m2)", ("m/LA x 10000", "", "g/m2"), "mass per area"), ("Pressure (bar)", ("Pressure", "", "bar"), "pressure"),
    ("polygon area (PA)", ("polygon area", "PA", ""), ""), ("Amount (£)", ("Amount", "", "£"), ""), ("Region", ("Region", "", ""), ""),
])
def test_headers(name, parts, quantity):
    assert header_parts(name) == parts and quantity_of(parts[2]) == quantity


def test_bars_with_error_bars_keep_the_groups_in_order():
    df = pl.DataFrame({"g": ["sun"] * 4 + ["shade"] * 4, "x": [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]})
    b = bar_data(df.lazy(), "g", "x", "mean", error="se")
    assert b.labels == ["sun", "shade"] and list(b.values) == [2.5, 13.0]
    assert b.errors[0] == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2) and b.error == "se"
    plain = bar_data(df.lazy(), "g", "x", "mean")
    assert plain.labels == ["shade", "sun"] and plain.errors is None


# ------------------------------------------------------------------ groups kept in columns
def column_sheet(tmp_path: Path, name: str, header: list[str], rows: list[list]) -> tuple[Pipeline, object]:
    from sheets import write_rows
    write_rows(tmp_path / f"{name}.xlsx", [(0, 0, header)] + [(1 + i, 0, r) for i, r in enumerate(rows)])
    p = Pipeline(name); p.path = tmp_path / f"{name}.json"
    p.add_node("load_file", title=name, params={"path": f"{name}.xlsx"}, id=name)
    ex = Executor(p)
    return p, deepen(p, ex, understand(p, ex))


def test_before_and_after_columns_are_a_paired_comparison(tmp_path):
    rng = np.random.default_rng(4)
    before = np.round(rng.normal(12, 2, 10), 1)
    after = np.round(before + rng.normal(1.5, 0.6, 10), 1)
    p, m = column_sheet(tmp_path, "growth", ["Plant", "Before (cm)", "After (cm)"],
                        [[i + 1, float(b), float(a)] for i, (b, a) in enumerate(zip(before, after))])
    s = suggest(m)[0]
    assert s.recipe == "groups" and s.spec["columns"] == [["growth", "Before (cm)"], ["growth", "After (cm)"]]
    for q in ("compare before and after", "difference between before and after", "t test before after"):
        a = ask(m, q)
        assert a.ok and a.spec["recipe"] == "groups" and a.spec.get("columns"), (q, a.message)
    st, pl_ = run(p, s.spec)
    out = pl.read_parquet(st.output).row(0, named=True)
    assert out["test"] == "paired t-test" and out["p value"] == pytest.approx(stats.ttest_rel(after, before).pvalue)
    assert st.report["finding"]["statement"].startswith("After is") and "cm" in st.report["finding"]["statement"]
    assert any(a["id"] == "paired" for a in pl_.assumptions)
    st2, _ = run(p, apply_choice(s.spec, "set", {"paired": False}))
    assert pl.read_parquet(st2.output)["test"][0] == "Welch's t-test"


def test_control_and_treated_columns_are_groups(tmp_path):
    rng = np.random.default_rng(5)
    rows = [[i + 1, round(float(rng.normal(20, 2)), 1), round(float(rng.normal(26, 2)), 1)] for i in range(10)]
    p, m = column_sheet(tmp_path, "pots", ["Replicate", "Control", "Fertilised"], rows)
    a = ask(m, "compare control and fertilised")
    assert a.ok and a.spec["columns"] == [["pots", "Control"], ["pots", "Fertilised"]]
    st, pl_ = run(p, a.spec)
    out = pl.read_parquet(st.output).row(0, named=True)
    assert out["test"] == "Welch's t-test" and out["p value"] < 0.001
    assert st.report["finding"]["statement"].startswith("Fertilised is")
    assert suggest(m)[0].recipe == "groups"
    # two unrelated numbers stay a relationship
    q, m2 = column_sheet(tmp_path, "parts", ["Part", "length (mm)", "weight (g)"], [[i, 10 + i, 3 * i + 1] for i in range(10)])
    assert ask(m2, "compare length and weight").spec["recipe"] == "relationship"


def test_a_study_kept_in_a_file_per_group(tmp_path):
    from sheets import leaves, HEADERS
    data = leaves()
    p = Pipeline("two"); p.path = tmp_path / "two.json"
    for g in ("sun", "shade"):
        pl.DataFrame({"Leaf": list(range(1, 16)), **{h: [r[k] for r in data[g]] for k, h in enumerate(HEADERS)}}) \
            .write_csv(tmp_path / f"{g}_leaves.csv")
        p.add_node("load_file", title=f"{g}_leaves", params={"path": f"{g}_leaves.csv"}, id=g)
    ex = Executor(p)
    m = deepen(p, ex, understand(p, ex))
    s = suggest(m)[0]
    assert s.recipe == "groups" and s.spec["by"][1] == "source" and s.title == "sun and shade compared"
    a = ask(m, "compare sun and shade")
    assert a.ok and a.spec["recipe"] == "groups"
    pl_ = plan(m, s.spec)
    assert any(x["id"] == "breaks" and "1.26" in x["text"] for x in pl_.assumptions)     # the slip is in the shade file
    st, _ = run(p, s.spec)
    assert st.status == "done" and st.report["groups"] == ["sun", "shade"]


# ------------------------------------------------------------------ the same words on a table of sales
@pytest.mark.parametrize("question,recipe,title", [
    ("is price higher in North", "breakdown", "Average price by region"),
    ("which region has higher quantity", "breakdown", "Total quantity by region"),
    ("difference in quantity between North and South", "groups", "North and South compared on quantity"),
    ("is price different by region", "groups", "each region compared on price"),
    ("compare North and South", "breakdown", "Total quantity by region where region is North or South"),
    ("price vs quantity", "relationship", "price against quantity"),
    ("quantity between 2 and 4", "rows", "orders where quantity between 2 and 4"),
])
def test_the_new_words_on_a_table_of_sales(tmp_path, question, recipe, title):
    from test_corpus import shop
    files = shop(tmp_path)
    p = Pipeline("shop"); p.path = tmp_path / "shop.json"
    for f in files:
        p.add_node("load_file", title=Path(f).stem, params={"path": f})
    ex = Executor(p)
    m = deepen(p, ex, understand(p, ex))
    a = ask(m, question)
    assert a.ok, a.message
    assert (a.spec["recipe"], a.title) == (recipe, title)
