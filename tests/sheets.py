"""Workbooks laid out the way people lay them out, for the layout and study tests.

``lab_sheet`` is a two-group lab sheet: two tables side by side under the banners TREATED and CONTROL, sharing
a column of sample numbers, an empty template row numbered 16, and AVERAGE, STANDARD DEV and MEDIAN rows under
the data with blank rows between them. Its columns are an inner area, an outer area, the gap between them
(outer area minus inner area), a mass, and mass per area (mass / area × 10000), with one gap mistyped. Treated
samples are smaller, thicker for their size and heavier for their area than control samples.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import xlsxwriter

HEADERS = ["inner area (IA) cm2", "outer area (OA) cm2", "gap", "mass (m) g", "mass/area x 10000 (g/m2)"]


def measurements(seed: int = 7, n: int = 15) -> dict[str, list[list[float]]]:
    """Each group's samples as [inner area, outer area, gap, mass, mass/area x 10000]; control sample 1's gap is
    mistyped (1.26)."""
    rng = np.random.default_rng(seed)
    out = {}
    for group, area, gapfrac, lma in (("treated", 50.0, 0.26, 110.0), ("control", 125.0, 0.17, 80.0)):
        rows = []
        for i in range(n):
            a = round(float(area * rng.uniform(0.35, 1.6)), 1)
            oa = round(a / (1 - gapfrac * rng.uniform(0.8, 1.2)), 1)
            m = round(a * lma * rng.uniform(0.8, 1.2) / 10000, 3)
            rows.append([a, oa, round(oa - a, 1), m, m / a * 10000])
        out[group] = rows
    out["control"][0][2] = 1.26                     # the slip: outer area minus inner area is not 1.26
    return out


def lab_sheet(path: Path, data: dict | None = None, stale_average: bool = False) -> dict:
    data = data or measurements()
    wb = xlsxwriter.Workbook(path)
    ws = wb.add_worksheet("Sheet1")
    ws.merge_range(0, 1, 0, 5, "TREATED")
    ws.merge_range(0, 7, 0, 11, "CONTROL")
    ws.write(1, 0, "N - Sample")
    ws.write_row(1, 1, HEADERS)
    ws.write_row(1, 7, HEADERS)
    n = len(data["treated"])
    for i in range(n):
        ws.write(2 + i, 0, i + 1)
        ws.write_row(2 + i, 1, data["treated"][i])
        ws.write_row(2 + i, 7, data["control"][i])
    ws.write(2 + n, 0, n + 1)                     # the handout had room for one more sample
    for r, label, fn in ((3 + n, "AVERAGE", statistics.mean), (5 + n, "STANDARD DEV", statistics.stdev),
                         (7 + n, "MEDIAN", statistics.median)):
        ws.write(r, 0, label)
        for c0, g in ((1, "treated"), (7, "control")):
            for k in range(5):
                v = fn([row[k] for row in data[g]])
                if stale_average and label == "AVERAGE" and g == "treated" and k == 0:
                    v += 5                        # an average typed in before the last sample was added
                ws.write(r, c0 + k, v)
    wb.close()
    return data


def write_rows(path: Path, rows: list[tuple[int, int, list]], merges: list[tuple[int, int, int, int, str]] = ()) -> None:
    """A one-sheet workbook from (row, column, values) runs."""
    wb = xlsxwriter.Workbook(path)
    ws = wb.add_worksheet()
    for r0, c0, c1, c2, text in merges:
        ws.merge_range(r0, c0, c1, c2, text)
    for r, c, vals in rows:
        ws.write_row(r, c, vals)
    wb.close()
