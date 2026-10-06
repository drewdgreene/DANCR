"""Geospatial steps and maths: distances, coordinate cleanup and point grids."""

import polars as pl
import pytest
from conftest import run_one

from dancr.core import Pipeline, geo


# ---------------------------------------------------------------- maths
def test_haversine_known_distances():
    # London (51.5074, -0.1276) to Paris (48.8566, 2.3522): about 343.5 km great circle
    lf = pl.LazyFrame({"a_lat": [51.5074], "a_lon": [-0.1276], "b_lat": [48.8566], "b_lon": [2.3522]})
    d = lf.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().item()
    assert abs(d - 343_500) < 1_000


def test_haversine_one_degree_at_equator():
    lf = pl.LazyFrame({"a_lat": [0.0], "a_lon": [0.0], "b_lat": [0.0], "b_lon": [1.0]})
    d = lf.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().item()
    assert abs(d - geo.METRES_PER_DEGREE_LAT) < 1.0


def test_haversine_zero_and_symmetry():
    lf = pl.LazyFrame({"a_lat": [10.0, 10.0], "a_lon": [20.0, -30.0], "b_lat": [10.0, 10.0], "b_lon": [20.0, -30.0]})
    d = lf.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().to_series()
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.0)
    lf2 = pl.LazyFrame({"a_lat": [1.0], "a_lon": [2.0], "b_lat": [3.0], "b_lon": [4.0]})
    fwd = lf2.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().item()
    back = lf2.select(geo.distance_m_expr("b_lat", "b_lon", "a_lat", "a_lon")).collect().item()
    assert fwd == pytest.approx(back)


def test_haversine_across_antimeridian_is_short_way():
    # 179.9E to 179.9W is 0.2 degrees apart, not almost the whole globe
    lf = pl.LazyFrame({"a_lat": [0.0], "a_lon": [179.9], "b_lat": [0.0], "b_lon": [-179.9]})
    d = lf.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().item()
    assert d < 100_000


def test_haversine_null_in_null_out():
    lf = pl.LazyFrame({"a_lat": [None, 1.0], "a_lon": [2.0, 3.0], "b_lat": [4.0, 5.0], "b_lon": [6.0, 7.0]})
    d = lf.select(geo.distance_m_expr("a_lat", "a_lon", "b_lat", "b_lon")).collect().to_series()
    assert d[0] is None and d[1] is not None


def test_parse_distance_units():
    assert geo.parse_distance("5km") == ("5km", 5000.0)
    assert geo.parse_distance("3 miles")[1] == pytest.approx(4828.032)
    assert geo.parse_distance("500 m")[1] == 500.0
    assert geo.parse_distance("2") == ("2km", 2000.0)          # bare number takes the default unit
    with pytest.raises(ValueError):
        geo.parse_distance("soon")
    with pytest.raises(ValueError):
        geo.parse_distance("0km")


def test_grid_degrees():
    assert geo.grid_degrees("0.1") == pytest.approx(0.1)
    assert geo.grid_degrees("5km") == pytest.approx(5_000 / geo.METRES_PER_DEGREE_LAT, rel=1e-6)
    with pytest.raises(ValueError):
        geo.grid_degrees("200")


def test_lat_lon_pair_detection():
    cols = [("station", 0, 0), ("lat", -33.5, -33.4), ("lon", 18.3, 18.5)]
    assert geo.lat_lon_pair(cols) == ("lat", "lon")
    # plain x/y accepted only when the ranges fit
    assert geo.lat_lon_pair([("x", 18.3, 18.5), ("y", -33.5, -33.4)]) == ("y", "x")
    # a scatter of percentages is not a location
    assert geo.lat_lon_pair([("x", 0, 100), ("y", 0, 100)]) is None


# ---------------------------------------------------------------- nodes
def _pipe(tmp_path, df: pl.DataFrame, name="in.parquet") -> Pipeline:
    df.write_parquet(tmp_path / name)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": name}, id="src")
    p.path = tmp_path / "p.json"
    return p


def test_make_point_blanks_impossible(tmp_path):
    df = pl.DataFrame({"lat": [51.5, 200.0, None], "lon": [-0.1, 2.3, 4.0], "v": [1.0, 2.0, 3.0]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("make_point", params={"lat": "lat", "lon": "lon"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert out["latitude"].to_list() == [51.5, None, None]
    assert out["longitude"].to_list() == [-0.1, None, None]


def test_make_point_drops_missing(tmp_path):
    df = pl.DataFrame({"lat": [51.5, 200.0, None], "lon": [-0.1, 2.3, 4.0]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("make_point", params={"lat": "lat", "lon": "lon", "drop_invalid": True}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert out.height == 1
    assert out["latitude"].to_list() == [51.5]


def test_distance_between_rows(tmp_path):
    df = pl.DataFrame({
        "lat_a": [51.5074], "lon_a": [-0.1276], "lat_b": [48.8566], "lon_b": [2.3522],
    })
    p = _pipe(tmp_path, df)
    nid = p.add_node("distance", params={"method": "between", "lat1": "lat_a", "lon1": "lon_a",
                                         "lat2": "lat_b", "lon2": "lon_b", "units": "km"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert "distance_km" in out.columns
    assert abs(out["distance_km"][0] - 343.5) < 1.0


def test_distance_from_fixed_point_and_units(tmp_path):
    df = pl.DataFrame({"lat": [0.0], "lon": [0.0]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("distance", params={"method": "from", "lat": "lat", "lon": "lon",
                                         "to_lat": 0.0, "to_lon": 1.0, "units": "m"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert "distance_m" in out.columns
    assert out["distance_m"][0] == pytest.approx(geo.METRES_PER_DEGREE_LAT, rel=1e-4)


def test_points_grid_counts_and_centres(tmp_path):
    # two points in the (0,0) cell, one in the (0.1, 0.1) cell
    df = pl.DataFrame({"lat": [0.01, 0.02, 0.11], "lon": [0.01, 0.05, 0.11], "v": [1.0, 3.0, 5.0]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("points_grid", params={"lat": "lat", "lon": "lon", "size": "0.1",
                                            "count_column": "points"}).id
    p.connect("src", nid)
    out = run_one(p, nid).sort("cell_lat")
    assert out["cell_lat"].to_list() == pytest.approx([0.05, 0.15])
    assert out["cell_lon"].to_list() == pytest.approx([0.05, 0.15])
    assert out["points"].to_list() == [2, 1]
    assert out["v"].to_list() == pytest.approx([2.0, 5.0])


def test_points_grid_uses_a_length(tmp_path):
    df = pl.DataFrame({"lat": [0.0, 0.5], "lon": [0.0, 0.5]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("points_grid", params={"lat": "lat", "lon": "lon", "size": "10km",
                                            "count_column": "points"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    # 10 km is about 0.09 degrees; 0.0 and 0.5 fall in cells 0 and 5
    assert out.height == 2


def test_check_data_flags_impossible_latitude(tmp_path):
    df = pl.DataFrame({"lat": [51.5, 200.0], "lon": [-0.1, 2.3]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("check_data", params={"columns": ["lat", "lon"]}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    row = out.filter(pl.col("column") == "lat").row(0, named=True)
    assert row["severity"] == 3
    assert "latitude" in row["issue"].lower()


def _cities(tmp_path) -> str:
    df = pl.DataFrame({
        "lat": [-1.29, -6.79, 6.31], "lon": [36.82, 39.21, -10.80],
        "city": ["Nairobi", "Dar es Salaam", "Monrovia"], "people": [4.4, 6.7, 1.3],
    })
    f = tmp_path / "cities.csv"
    df.write_csv(f)
    return str(f)


def _two_tables(tmp_path):
    pl.DataFrame({"vlat": [-1.30, -1.25, -1.40, 40.0], "vlon": [36.80, 36.90, 36.70, -3.0],
                  "village": ["a", "b", "c", "far"]}).write_parquet(tmp_path / "v.parquet")
    pl.DataFrame({"clat": [-1.29, -1.20], "clon": [36.82, 37.10], "clinic": ["A", "B"]}).write_parquet(tmp_path / "c.parquet")
    p = Pipeline("t")
    p.add_node("load_file", "V", {"path": "v.parquet"}, id="v")
    p.add_node("load_file", "C", {"path": "c.parquet"}, id="c")
    p.path = tmp_path / "p.json"
    return p


def _nearest(p, **extra):
    params = {"method": "nearest_feature", "left_lat": "vlat", "left_lon": "vlon",
              "right_lat": "clat", "right_lon": "clon", "max_distance": "50km", "units": "km", **extra}
    nid = p.add_node("combine", params=params).id
    p.connect("v", nid, "left"); p.connect("c", nid, "right")
    return run_one(p, nid)


def test_nearest_feature_matches_brute_force(tmp_path):
    out = _nearest(_two_tables(tmp_path))
    assert out["village"].to_list() == ["a", "b", "c", "far"]     # first table's order kept
    assert out["clinic"].to_list() == ["A", "A", "A", None]       # "far" is beyond 50km
    assert out["distance"].to_list()[:3] == pytest.approx([2.49, 9.94, 18.10], abs=0.05)
    assert out["distance"].to_list()[3] is None


def test_nearest_feature_inner_keeps_only_matches(tmp_path):
    out = _nearest(_two_tables(tmp_path), near_how="inner")
    assert set(out["village"].to_list()) == {"a", "b", "c"}
    assert out["distance"].to_list() == sorted(out["distance"].to_list())


def test_nearest_feature_respects_a_tight_distance(tmp_path):
    out = _nearest(_two_tables(tmp_path), max_distance="5km")
    got = dict(zip(out["village"].to_list(), out["clinic"].to_list()))
    assert got["a"] == "A" and got["b"] is None and got["c"] is None


def test_nearest_feature_keeps_the_first_table_columns_once(tmp_path):
    """The pairing carries the first table back; joining it again must not duplicate its columns."""
    out = _nearest(_two_tables(tmp_path))
    assert "vlat_right" not in out.columns and "village_right" not in out.columns
    assert out.columns.count("vlat") == 1 and out.columns.count("village") == 1


def test_nearest_feature_when_only_the_latitude_name_clashes(tmp_path):
    """A clash on one coordinate name says nothing about the other: the longitude keeps its own name."""
    pl.DataFrame({"lat": [-1.29], "lon": [36.82], "village": ["a"]}).write_parquet(tmp_path / "v.parquet")
    pl.DataFrame({"lat": [-1.29], "clon": [36.82], "clinic": ["A"]}).write_parquet(tmp_path / "c.parquet")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", "V", {"path": "v.parquet"}, id="v")
    p.add_node("load_file", "C", {"path": "c.parquet"}, id="c")
    nid = p.add_node("combine", params={"method": "nearest_feature", "left_lat": "lat", "left_lon": "lon",
                                        "right_lat": "lat", "right_lon": "clon", "max_distance": "50km",
                                        "units": "km"}).id
    p.connect("v", nid, "left"); p.connect("c", nid, "right")
    out = run_one(p, nid)
    assert out["clinic"].to_list() == ["A"] and out["lat_2"].to_list() == [-1.29]


def test_export_geojson(tmp_path):
    src = _cities(tmp_path)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": src}, id="src")
    p.path = tmp_path / "p.json"
    nid = p.add_node("export", params={"path": "out.geojson"}).id
    p.connect("src", nid)
    run_one(p, nid)
    data = (tmp_path / "out.geojson").read_text()
    assert "FeatureCollection" in data and "Nairobi" in data and "-1.29" in data


def test_render_map_writes_png(tmp_path):
    from dancr.views.render import render_map
    src = _cities(tmp_path)
    lf = pl.read_csv(src).lazy()
    out = render_map(lf, {"lat": "lat", "lon": "lon", "color_by": "people", "title": "People"},
                     tmp_path / "m.png", width=800, height=500)
    assert out.exists() and out.stat().st_size > 4000


def test_report_embeds_a_map(tmp_path):
    src = _cities(tmp_path)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": src}, id="src")
    p.path = tmp_path / "p.json"
    mid = p.add_node("map", params={"lat": "lat", "lon": "lon", "title": "Cities"}).id
    p.connect("src", mid)
    rid = p.add_node("report", params={"title": "Geo report", "path": "rep.html", "pdf": False}).id
    p.connect(mid, rid, "items")
    run_one(p, rid)
    html = (tmp_path / "rep.html").read_text()
    assert "data:image/png;base64" in html and "Geo report" in html


def _geojson(tmp_path) -> str:
    import json
    regions = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"region": "Rift Valley", "area_km2": 1200},
         "geometry": {"type": "Polygon", "coordinates": [[[36.0, -2.0], [38.0, -2.0], [38.0, -1.0], [36.0, -1.0], [36.0, -2.0]]]}},
        {"type": "Feature", "properties": {"region": "Somalia", "area_km2": 800},
         "geometry": {"type": "Polygon", "coordinates": [[[40.0, 2.0], [42.0, 2.0], [42.0, 4.0], [40.0, 4.0], [40.0, 2.0]]]}}]}
    f = tmp_path / "regions.geojson"
    f.write_text(json.dumps(regions))
    return str(f)


def test_load_geojson(tmp_path):
    src = _geojson(tmp_path)
    p = Pipeline("t")
    p.add_node("load_file", "Load", {"path": src}, id="src")
    p.path = tmp_path / "p.json"
    out = run_one(p, "src")
    assert set(out.columns) >= {"region", "area_km2", "geometry", "longitude", "latitude"}
    assert out.height == 2
    assert "POLYGON" in out["geometry"][0]
    assert abs(out["longitude"][0] - 37.0) < 0.1


def test_spatial_join_which_region(tmp_path):
    pl.DataFrame({"lat": [-1.29, -5.0, 10.0], "lon": [36.82, 36.82, 36.82],
                  "name": ["in-A", "edge", "far"]}).write_parquet(tmp_path / "pts.parquet")
    reg = _geojson(tmp_path)
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "pts.parquet"}, id="pts")
    p.add_node("load_file", params={"path": reg}, id="reg")
    nid = p.add_node("combine", params={"method": "within", "left_lat": "lat", "left_lon": "lon",
                                        "right_geometry": "geometry", "place_column": "region"}).id
    p.connect("pts", nid, "left"); p.connect("reg", nid, "right")
    out = run_one(p, nid).sort("name")
    got = dict(zip(out["name"].to_list(), out["region"].to_list()))
    assert got["in-A"] == "Rift Valley"
    assert got["edge"] is None and got["far"] is None


def test_project_utm_to_latlon(tmp_path):
    pl.DataFrame({"E": [263199.0], "N": [9857198.0]}).write_parquet(tmp_path / "utm.parquet")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "utm.parquet"}, id="src")
    nid = p.add_node("project", params={"easting": "E", "northing": "N", "utm_zone": "37", "south": True}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert abs(out["latitude"][0] - -1.29) < 0.01
    assert abs(out["longitude"][0] - 36.87) < 0.01


def test_utm_round_trip_within_a_few_metres():
    from dancr.core.geo import utm_to_latlon, latlon_to_utm
    lat, lon = -1.291, 36.872
    e, n = latlon_to_utm(lat, lon, 37, True)
    la, lo = utm_to_latlon(e, n, 37, True)
    assert abs(la - lat) < 3e-5 and abs(lo - lon) < 3e-5      # about 3 metres


def test_point_in_polygon_with_hole():
    from dancr.core.geo import parse_wkt_rings, point_in_polygon
    rings = parse_wkt_rings("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
    assert point_in_polygon(1, 1, rings)
    assert not point_in_polygon(3, 3, rings)                 # inside the hole


# ---------------------------------------------------------------- GeoPackage / shapefile loading
def _write_points_gpkg(path, layer="villages", crs="EPSG:4326"):
    """A GeoPackage of two points, written by pyogrio (the test is skipped when it is not installed)."""
    pytest.importorskip("pyogrio")
    import numpy as np
    from pyogrio import raw
    from shapely.geometry import Point
    if crs == "EPSG:3857":
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        pts = [tr.transform(36.82, -1.29), tr.transform(37.10, -1.20)]
    else:
        pts = [(36.82, -1.29), (37.10, -1.20)]
    geoms = np.array([Point(x, y).wkb for x, y in pts], dtype=object)
    raw.write(str(path), geoms, np.array([["a", "b"], [4.0, 2.0]], dtype=object),
              np.array(["village", "people"], dtype=object), layer=layer, driver="GPKG",
              crs=crs, geometry_type="Point")
    return str(path)


def test_load_geopackage(tmp_path):
    src = _write_points_gpkg(tmp_path / "villages.gpkg")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", "Load", {"path": src}, id="src")
    out = run_one(p, "src")
    assert set(out.columns) >= {"village", "people", "geometry", "longitude", "latitude"}
    assert out.height == 2
    assert out["village"].to_list() == ["a", "b"]
    assert "POINT" in out["geometry"][0]
    assert out["longitude"][0] == pytest.approx(36.82) and out["latitude"][0] == pytest.approx(-1.29)


def test_load_geopackage_reprojects_to_lonlat(tmp_path):
    pytest.importorskip("pyogrio"); pytest.importorskip("pyproj")
    src = _write_points_gpkg(tmp_path / "merc.gpkg", crs="EPSG:3857")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", "Load", {"path": src}, id="src")
    out = run_one(p, "src")
    # the file is Web Mercator (metres); the loader hands back degrees
    assert out["longitude"][0] == pytest.approx(36.82, abs=1e-4)
    assert out["latitude"][0] == pytest.approx(-1.29, abs=1e-4)
    assert out["geometry"][0].startswith("POINT (36.8")


def test_load_shapefile(tmp_path):
    pytest.importorskip("pyogrio")
    import numpy as np
    from pyogrio import raw
    from shapely.geometry import Point
    f = tmp_path / "pts.shp"
    raw.write(str(f), np.array([Point(36.82, -1.29).wkb], dtype=object), np.array([["a"]], dtype=object),
              np.array(["village"], dtype=object), driver="ESRI Shapefile", crs="EPSG:4326", geometry_type="Point")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", "Load", {"path": str(f)}, id="src")
    out = run_one(p, "src")
    assert out["village"].to_list() == ["a"]
    assert out["longitude"][0] == pytest.approx(36.82) and out["latitude"][0] == pytest.approx(-1.29)


def test_vector_load_without_pyogrio_explains(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "pyogrio" or name.startswith("pyogrio."):
            raise ImportError("blocked for the test")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    f = tmp_path / "x.gpkg"
    f.write_bytes(b"not really a geopackage")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", "Load", {"path": str(f)}, id="src")
    from dancr.core.executor import Executor
    st = Executor(p).run(targets=["src"])["src"]
    assert st.status == "failed"
    assert "pyogrio" in (st.error or "")


def test_tables_in_lists_every_gpkg_layer(tmp_path):
    pytest.importorskip("pyogrio")
    import numpy as np
    from pyogrio import raw
    from shapely.geometry import Point
    f = tmp_path / "multi.gpkg"
    for layer, x in (("alpha", 0), ("beta", 1)):
        raw.write(str(f), np.array([Point(x, x).wkb], dtype=object), np.array([[layer]], dtype=object),
                  np.array(["n"], dtype=object), layer=layer, driver="GPKG", crs="EPSG:4326",
                  geometry_type="Point", append=layer == "beta")
    from dancr.core.nodes.load import tables_in
    got = dict(tables_in(f))
    assert set(got) == {"alpha", "beta"} and got["beta"] == {"layer": "beta"}


def test_add_files_loads_every_gpkg_layer(tmp_path):
    pytest.importorskip("pyogrio")
    import numpy as np
    from pyogrio import raw
    from shapely.geometry import Point
    f = tmp_path / "multi.gpkg"
    for layer, x in (("alpha", 0), ("beta", 1)):
        raw.write(str(f), np.array([Point(x, x).wkb], dtype=object), np.array([[layer]], dtype=object),
                  np.array(["n"], dtype=object), layer=layer, driver="GPKG", crs="EPSG:4326",
                  geometry_type="Point", append=layer == "beta")
    from dancr.headless import add_files
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    ids = add_files(p, [str(f)])
    assert len(ids) == 2
    assert {p.nodes[i].params.get("layer") for i in ids} == {"alpha", "beta"}


def test_project_arbitrary_epsg_with_pyproj(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    lat, lon = 51.5074, -0.1276
    x, y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    pl.DataFrame({"X": [x], "Y": [y]}).write_parquet(tmp_path / "merc.parquet")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "merc.parquet"}, id="src")
    nid = p.add_node("project", params={"easting": "X", "northing": "Y", "crs": "EPSG:3857"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert out["longitude"][0] == pytest.approx(lon, abs=1e-6)
    assert out["latitude"][0] == pytest.approx(lat, abs=1e-6)


def test_project_blank_coordinates_stay_blank(tmp_path):
    pytest.importorskip("pyproj")
    pl.DataFrame({"E": [263199.0, None], "N": [9857198.0, None]}).write_parquet(tmp_path / "utm.parquet")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "utm.parquet"}, id="src")
    nid = p.add_node("project", params={"easting": "E", "northing": "N", "crs": "EPSG:32737"}).id
    p.connect("src", nid)
    out = run_one(p, nid)
    assert out["latitude"][0] == pytest.approx(-1.29, abs=0.01)
    assert out["longitude"][0] == pytest.approx(36.87, abs=0.01)
    assert out["latitude"][1] is None and out["longitude"][1] is None


def test_distance_needs_numeric_columns(tmp_path):
    df = pl.DataFrame({"lat": ["x"], "lon": ["y"]})
    p = _pipe(tmp_path, df)
    nid = p.add_node("make_point", params={"lat": "lat", "lon": "lon"}).id
    p.connect("src", nid)
    from dancr.core.executor import Executor
    st = Executor(p).run(targets=[nid])[nid]
    assert st.status == "failed"
    assert "number" in (st.error or "").lower()
