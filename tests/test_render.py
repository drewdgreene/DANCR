"""Charts and PDFs render without a window and without shared state, so parallel renders are safe."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import polars as pl

from dancr.views.render import render_chart


def test_charts_render_in_parallel_threads(tmp_path):
    lf = pl.DataFrame({"x": list(range(200)), "a": [i * 0.5 for i in range(200)], "b": [(-1) ** i for i in range(200)]}).lazy()
    kinds = [{"kind": "line", "x": "x", "series": [{"column": "a"}], "title": "t"},
             {"kind": "scatter", "x": "x", "series": [{"column": "b"}]},
             {"kind": "histogram", "column": "a"},
             {"kind": "line", "x": "x", "series": [{"column": "a"}, {"column": "b"}]}]
    jobs = [(kinds[i % len(kinds)], tmp_path / f"c{i}.png") for i in range(16)]
    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(lambda j: render_chart(lf, j[0], j[1], width=500, height=300), jobs))
    for out in outs:
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_does_not_use_pyplot():
    import inspect
    from dancr.views import render
    src = inspect.getsource(render)
    assert "import matplotlib.pyplot" not in src and "plt." not in src


def test_a_pdf_written_from_a_worker_thread_without_a_window(tmp_path):
    import threading
    from dancr.views.pdf import html_to_pdf
    out, got = tmp_path / "r.pdf", []
    t = threading.Thread(target=lambda: got.append(html_to_pdf("<h1>Report</h1>", out)))
    t.start(); t.join()
    assert got and out.stat().st_size > 500


def test_report_charts_use_the_columns_zone(tmp_path):
    from dancr.views.render import _wall_times
    x = np.array([datetime(2024, 7, 1, 10, tzinfo=__import__("zoneinfo").ZoneInfo("UTC")).timestamp()])
    assert str(_wall_times(x, "Europe/Oslo")[0]).startswith("2024-07-01T12:00")
    assert str(_wall_times(x)[0]).startswith("2024-07-01T10:00")


def test_headless_render_resolves_an_input_named_limit(tmp_path):
    from dancr.views.render import render_chart

    lf = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]}).lazy()
    out = tmp_path / "c.png"
    render_chart(lf, {"kind": "line", "x": "x", "series": [{"column": "y"}],
                      "limits": [{"value": "upper", "label": "upper"}]},
                 out, inputs={"upper": 4.0})
    assert out.exists() and out.stat().st_size > 0


def test_a_pdf_helper_that_fails_with_no_message_is_reported_plainly(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from dancr.views import pdf

    def fake(*a, **k):
        return subprocess.CompletedProcess(a, 1, "", "")     # non-zero exit, empty stderr (killed/aborted)

    monkeypatch.setattr(pdf.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="exit 1"):
        pdf._in_helper("<h1>x</h1>", tmp_path / "r.pdf")
