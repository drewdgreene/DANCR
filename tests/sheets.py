"""Workbooks laid out the way people lay them out, for the layout and study tests.

``lab_sheet`` is a first-year lab data sheet: two tables side by side under the banners SUN LEAVES and SHADE
LEAVES, sharing a column of leaf numbers, an empty template row numbered 16, and AVERAGE, STANDARD DEV and MEDIAN
rows under the data with blank rows between them. Its columns are leaf area, polygon area, D (polygon area minus
leaf area), mass, and mass per area (mass / leaf area × 10000), with one D mistyped. Sun leaves are smaller, more
deeply lobed for their size and heavier for their area than shade leaves.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import xlsxwriter

HEADERS = ["Leaf area (LA) cm2", "polygon area (PA) cm2", "D", "m (g)", "m/LA x 10000 (g/m2)"]


def leaves(seed: int = 7, n: int = 15) -> dict[str, list[list[float]]]:
    """Each group's leaves as [LA, PA, D, m, m/LA x 10000]; shade leaf 1's D is mistyped (1.26)."""
    rng = np.random.default_rng(seed)
    out = {}
    for group, area, lobes, lma in (("sun", 50.0, 0.26, 110.0), ("shade", 125.0, 0.17, 80.0)):
        rows = []
        for i in range(n):
            la = round(float(area * rng.uniform(0.35, 1.6)), 1)
            pa = round(la / (1 - lobes * rng.uniform(0.8, 1.2)), 1)
            m = round(la * lma * rng.uniform(0.8, 1.2) / 10000, 3)
            rows.append([la, pa, round(pa - la, 1), m, m / la * 10000])
        out[group] = rows
    out["shade"][0][2] = 1.26                     # the slip: polygon area minus leaf area is not 1.26
    return out


def lab_sheet(path: Path, data: dict | None = None, stale_average: bool = False) -> dict:
    data = data or leaves()
    wb = xlsxwriter.Workbook(path)
    ws = wb.add_worksheet("Sheet1")
    ws.merge_range(0, 1, 0, 5, "SUN LEAVES")
    ws.merge_range(0, 7, 0, 11, "SHADE LEAVES")
    ws.write(1, 0, "N - Leaf")
    ws.write_row(1, 1, HEADERS)
    ws.write_row(1, 7, HEADERS)
    n = len(data["sun"])
    for i in range(n):
        ws.write(2 + i, 0, i + 1)
        ws.write_row(2 + i, 1, data["sun"][i])
        ws.write_row(2 + i, 7, data["shade"][i])
    ws.write(2 + n, 0, n + 1)                     # the handout had room for one more leaf
    for r, label, fn in ((3 + n, "AVERAGE", statistics.mean), (5 + n, "STANDARD DEV", statistics.stdev),
                         (7 + n, "MEDIAN", statistics.median)):
        ws.write(r, 0, label)
        for c0, g in ((1, "sun"), (7, "shade")):
            for k in range(5):
                v = fn([row[k] for row in data[g]])
                if stale_average and label == "AVERAGE" and g == "sun" and k == 0:
                    v += 5                        # an average typed in before the last leaf was added
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
