"""Time helpers: the main time column, its cadence, and human-readable durations."""
from __future__ import annotations

import math
from typing import Any

import polars as pl

from ._model import TIME_ROLE, Table


def _settle_time(t: Table, sample: pl.DataFrame, consecutive: pl.DataFrame) -> None:
    """The main time column (the first date/time column that varies) and how regularly rows arrive, measured on
    ``consecutive`` rows."""
    times = t.by_role(TIME_ROLE)
    if not times:
        return
    main = max(times, key=lambda c: (c.distinct, -t.columns.index(c)))     # the finest: Date Time rather than Date
    t.time = main.name
    s = consecutive[main.name].drop_nulls() if main.name in consecutive.columns else sample[main.name].drop_nulls()
    if s.len() >= 3:
        try:
            srt = s.sort()
            if isinstance(s.dtype, pl.Datetime):
                us = srt.dt.epoch("us").cast(pl.Float64)
            else:
                us = srt.cast(pl.Datetime("us")).dt.epoch("us").cast(pl.Float64)
            steps = us.diff().drop_nulls()
            steps = steps.filter(steps > 0)
            if steps.len():
                med = float(steps.median())
                main.cadence = med / 1e6
                main.regular = float(((steps - med).abs() <= 0.1 * med).mean()) >= 0.8
        except Exception:  # noqa: BLE001 - odd time types simply get no cadence
            pass
    t.start, t.end = main.minimum, main.maximum
    t.span_seconds = _span(t.start, t.end)


def _span(a: Any, b: Any) -> float | None:
    try:
        return float((b - a).total_seconds())
    except Exception:  # noqa: BLE001
        return None


# =================================================================== durations for people
NICE_STEPS = [(1e-3, "1ms"), (1e-2, "10ms"), (0.05, "50ms"), (0.1, "100ms"), (0.5, "500ms"), (1, "1s"), (5, "5s"),
              (10, "10s"), (30, "30s"), (60, "1m"), (300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"),
              (6 * 3600, "6h"), (86400, "1d"), (7 * 86400, "1w"), (30.44 * 86400, "1mo"), (91.31 * 86400, "1q"),
              (365.25 * 86400, "1y")]


def duration_text(secs: float) -> str:
    """A tolerance for pairing readings taken ``secs`` apart: that spacing, written the short way."""
    if secs <= 0:
        return ""
    for unit, scale in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1), ("ms", 1e-3)):
        v = secs / scale
        if v >= 1 and abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}{unit}"
    ms = secs * 1000
    if ms >= 1:
        return f"{int(math.ceil(ms))}ms"
    return f"{int(math.ceil(secs * 1e6))}us"


def bucket_for(span_seconds: float | None, cadence: float | None = None, target: int = 400) -> str:
    """The time bucket that turns a span into roughly ``target`` points (minutes for a day, days for months),
    never finer than the rows arrive (``cadence``): hourly rows are not bucketed per minute."""
    if not span_seconds or span_seconds <= 0:
        return "1d"
    ideal = span_seconds / target
    fits = [text for secs, text in NICE_STEPS if secs <= ideal * 2]
    best = fits[-1] if fits else NICE_STEPS[0][1]
    if cadence:
        floor = next((text for secs, text in NICE_STEPS if secs >= cadence * 0.999), NICE_STEPS[-1][1])
        if dict((t, s) for s, t in NICE_STEPS)[best] < dict((t, s) for s, t in NICE_STEPS)[floor]:
            best = floor
    return best
