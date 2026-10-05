"""Questions about places: map, density and nearest-place, read by the grammar and built into steps."""
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.ask import ask
from dancr.core.executor import Executor
from dancr.core.planner import instantiate
from dancr.core.recipes import plan
from dancr.core.understand import understand


@pytest.fixture()
def places(tmp_path):
    d = tmp_path / "geo"; d.mkdir()
    pl.DataFrame({
        "lat": [-1.29, -1.25, -1.40, -1.20, -1.35],
        "lon": [36.82, 36.90, 36.70, 37.10, 36.60],
        "village": ["a", "b", "c", "d", "e"],
        "status": ["on", "on", "off", "on", "off"],
    }).write_csv(d / "villages.csv")
    pl.DataFrame({
        "clat": [-1.29, -1.20],
        "clon": [36.82, 37.10],
        "clinic": ["A", "B"],
    }).write_csv(d / "clinics.csv")
    p = Pipeline("geo"); p.path = d / "geo.json"
    p.add_node("load_file", title="villages", params={"path": str(d / "villages.csv")}, id="villages")
    p.add_node("load_file", title="clinics", params={"path": str(d / "clinics.csv")}, id="clinics")
    m = understand(p, Executor(p))
    return p, m


def _build(p, m, question):
    a = ask(m, question)
    assert a.ok, (question, a.message)
    q = Pipeline.from_dict(p.to_dict(), p.path)
    pl_ = plan(m, a.spec)
    res = instantiate(q, pl_)
    ex = Executor(q)
    st = ex.run(targets=[res[pl_.terminal]])[res[pl_.terminal]]
    assert st.status == "done", st.error
    return a, pl.read_parquet(st.output)


def test_map_question_builds_a_map(places):
    p, m = places
    a, _ = _build(p, m, "map the villages")
    assert a.spec["recipe"] == "map"


def test_map_question_can_colour_by(places):
    p, m = places
    a, _ = _build(p, m, "map the villages by status")
    assert a.spec["recipe"] == "map" and a.spec.get("color_by") == "status"


def test_density_question_counts_points(places):
    p, m = places
    a, df = _build(p, m, "density of villages")
    assert a.spec["recipe"] == "density"
    assert {"cell_lat", "cell_lon", "points"}.issubset(df.columns)
    assert df["points"].sum() == 5


def test_nearest_matches_each_row_to_a_place(places):
    p, m = places
    a, df = _build(p, m, "nearest clinic to each village")
    assert a.spec["recipe"] == "nearest"
    assert "clinic" in df.columns and "distance" in df.columns
    assert df.height == 5
    # the first village is at the same spot as clinic A
    first = df.filter(pl.col("village") == "a")
    assert first["clinic"][0] == "A" and first["distance"][0] == pytest.approx(0.0, abs=0.01)


def test_nearest_is_suggested(places):
    from dancr.core.recipes import _candidates
    _p, m = places
    recipes = {c["recipe"] for t in m.tables.values() for c in _candidates(m, t)}
    assert "nearest" in recipes and "map" in recipes and "density" in recipes


@pytest.fixture()
def regions(tmp_path):
    import json
    d = tmp_path / "geo2"; d.mkdir()
    # a and c fall in the Rift polygon; b (at 10°N) falls outside it
    pl.DataFrame({"lat": [-1.29, 10.0, -1.1], "lon": [36.82, 36.82, 36.70],
                  "name": ["a", "b", "c"]}).write_csv(d / "water_points.csv")
    (d / "regions.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"region": "Rift"},
         "geometry": {"type": "Polygon", "coordinates": [[[36.0, -2.0], [38.0, -2.0], [38.0, -1.0], [36.0, -1.0], [36.0, -2.0]]]}}]}))
    p = Pipeline("geo2"); p.path = d / "geo2.json"
    p.add_node("load_file", title="Water points", params={"path": str(d / "water_points.csv")}, id="pts")
    p.add_node("load_file", title="Regions", params={"path": str(d / "regions.geojson")}, id="reg")
    return p, understand(p, Executor(p))


def test_place_question_finds_the_region(regions):
    p, m = regions
    a, df = _build(p, m, "which region are the water points inside")
    assert a.spec["recipe"] == "place"
    region_col = next(c for c in df.columns if c.startswith("region"))
    got = dict(zip(df["name"].to_list(), df[region_col].to_list()))
    assert got["a"] == "Rift" and got["c"] == "Rift" and got["b"] is None


def test_place_question_over_a_geopackage(tmp_path):
    pytest.importorskip("pyogrio")
    import numpy as np
    from pyogrio import raw
    from shapely.geometry import Polygon
    d = tmp_path / "gpkg"; d.mkdir()
    pl.DataFrame({"lat": [-1.29, 10.0, -1.1], "lon": [36.82, 36.82, 36.70],
                  "name": ["a", "b", "c"]}).write_csv(d / "water_points.csv")
    poly = Polygon([(36.0, -2.0), (38.0, -2.0), (38.0, -1.0), (36.0, -1.0), (36.0, -2.0)])
    raw.write(str(d / "regions.gpkg"), np.array([poly.wkb], dtype=object), np.array([["Rift"]], dtype=object),
              np.array(["region"], dtype=object), layer="regions", driver="GPKG", crs="EPSG:4326",
              geometry_type="Polygon")
    p = Pipeline("gpkg"); p.path = d / "gpkg.json"
    p.add_node("load_file", title="Water points", params={"path": str(d / "water_points.csv")}, id="pts")
    p.add_node("load_file", title="Regions", params={"path": str(d / "regions.gpkg")}, id="reg")
    m = understand(p, Executor(p))
    a, df = _build(p, m, "which region are the water points inside")
    assert a.spec["recipe"] == "place"
    region_col = next(c for c in df.columns if c.startswith("region"))
    got = dict(zip(df["name"].to_list(), df[region_col].to_list()))
    assert got["a"] == "Rift" and got["c"] == "Rift" and got["b"] is None


def test_place_is_suggested(regions):
    from dancr.core.recipes import _candidates
    _p, m = regions
    recipes = {c["recipe"] for t in m.tables.values() for c in _candidates(m, t)}
    assert "place" in recipes
