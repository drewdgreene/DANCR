"""The map page: chips, drawing from coordinates, and adding a map to a report."""
import polars as pl
from helpers import pump, wait_run
from PySide6.QtCore import QPointF


def _geo_csv(tmp_path) -> str:
    df = pl.DataFrame({
        "lat": [-1.29, -6.79, 6.31, 5.56, 0.35],
        "lon": [36.82, 39.21, -10.80, -0.19, 32.58],
        "city": ["Nairobi", "Dar es Salaam", "Monrovia", "Accra", "Kampala"],
        "people": [4.4, 6.7, 1.3, 2.3, 1.5],
        "status": ["on", "on", "off", "on", "off"],
    })
    f = tmp_path / "cities.csv"
    df.write_csv(f)
    return str(f)


def test_map_page_draws_and_adds_to_report(window, app, tmp_path):
    window._add_load_node(_geo_csv(tmp_path), None); wait_run(window, app)
    src = window.current_table()
    mid = window.add_node("map", QPointF(220, 200), params={"lat": "lat", "lon": "lon", "color_by": "status"},
                          connect_from=src)
    wait_run(window, app); pump(app, 1500)
    assert window.pages.currentWidget() is window.map
    assert window.map._items, "the map drew nothing"
    assert "Latitude: lat" in window.map.lat_chip.text()
    assert "Longitude: lon" in window.map.lon_chip.text()
    # switching to grid squares redraws
    window.map._set({"cell_size": "0.5"}); pump(app, 1200)
    assert window.map._items
    # add it to a report like a chart
    window.steps.add_to_report(mid)
    rid = window._current
    assert window.doc.pipeline.nodes[rid].type == "report"
    assert window.doc.pipeline.inputs_of(rid) == {"items": [mid]}


def test_map_page_without_coordinates_shows_a_message(window, app, tmp_path):
    pl.DataFrame({"v": [1.0, 2.0, 3.0]}).write_csv(tmp_path / "plain.csv")
    window._add_load_node(str(tmp_path / "plain.csv"), None); wait_run(window, app)
    src = window.current_table()
    window.add_node("map", QPointF(220, 200), params={"lat": "v", "lon": "v"}, connect_from=src)
    wait_run(window, app); pump(app, 1000)
    # the same column for both is refused by the validator; the overlay says so
    assert window.map.overlay.isVisible() or window.map.info.text()
