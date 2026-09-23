"""Duration parsing and date-format sniffing."""
from __future__ import annotations

import re


import polars as pl

_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")
_UNITS = {
    "ns": ("ns", 1e-9), "us": ("us", 1e-6), "ms": ("ms", 1e-3),
    "s": ("s", 1), "sec": ("s", 1), "secs": ("s", 1), "second": ("s", 1), "seconds": ("s", 1),
    "m": ("m", 60), "min": ("m", 60), "mins": ("m", 60), "minute": ("m", 60), "minutes": ("m", 60),
    "h": ("h", 3600), "hr": ("h", 3600), "hrs": ("h", 3600), "hour": ("h", 3600), "hours": ("h", 3600),
    "d": ("d", 86400), "day": ("d", 86400), "days": ("d", 86400),
    "w": ("w", 604800), "wk": ("w", 604800), "week": ("w", 604800), "weeks": ("w", 604800),
}


def parse_duration(text: str) -> tuple[str, float]:
    """'5 min' -> ('5m', 300.0). Returns (polars duration string, seconds)."""
    if not text or not str(text).strip():
        raise ValueError("Enter a time span like 30s, 5m, 1h or 1d")
    s = str(text).strip().lower()
    m = _DUR_RE.match(s)
    if not m:
        # maybe already polars style like "1h30m"
        parts = re.findall(r"(\d+)\s*([a-z]+)", s)
        if parts and "".join(f"{n}{u}" for n, u in parts) == re.sub(r"\s+", "", s):
            total = 0.0
            out = ""
            for n, u in parts:
                if u not in _UNITS:
                    raise ValueError(f"Unknown time unit {u!r}")
                pu, secs = _UNITS[u]
                total += float(n) * secs
                out += f"{n}{pu}"
            if total <= 0:
                raise ValueError(f"{text!r} must be greater than zero")
            return out, total
        raise ValueError(f"Cannot read {text!r} as a time span. Try 30s, 5m, 1h or 1d")
    n, unit = m.groups()
    if unit not in _UNITS:
        raise ValueError(f"Unknown time unit {unit!r}. Use s, m, h, d or w")
    pu, secs = _UNITS[unit]
    value = float(n)
    if value <= 0:
        raise ValueError(f"{text!r} must be greater than zero")
    if value.is_integer():
        return f"{int(value)}{pu}", value * secs
    # fractional: express in a smaller unit
    total = value * secs
    for u in ("d", "h", "m", "s", "ms", "us"):
        _, us = _UNITS[u]
        if (total / us).is_integer():
            return f"{int(total / us)}{u}", total
    return f"{int(total * 1e6)}us", total


# Tried in order; the first format that parses every sampled value wins, else the one that parses most.
DATE_FORMATS = [
    "%+",                                   # RFC 3339 / ISO 8601 with offset or Z
    "%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S%.f", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S%.f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S%.f", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%d/%m/%y %H:%M", "%d/%m/%y", "%m/%d/%y %H:%M", "%m/%d/%y",
    "%d %b %Y %H:%M:%S", "%d %b %Y %H:%M", "%d %b %Y", "%b %d %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y",
    "%Y%m%d%H%M%S", "%Y%m%d %H%M%S", "%Y%m%d",
]


def detect_datetime_format(sample: pl.Series, min_fraction: float = 0.9) -> str | None:
    """Try known formats on a string sample; return the best one or None.

    Guards against false positives: separator-less formats (%Y%m%d) need a sane
    year range and more than one distinct day; version-like strings such as
    1.2.2024 are not dates unless every part is in range for every row.
    """
    s = sample.drop_nulls().cast(pl.Utf8).str.strip_chars()
    s = s.filter(s != "")
    if len(s) == 0:
        return None
    if s.head(50).str.contains(r"\d").sum() < min(len(s), 50) * 0.9:
        return None
    best: tuple[float, str] | None = None
    for fmt in DATE_FORMATS:
        try:
            parsed = s.str.to_datetime(fmt, strict=False)
        except Exception:
            continue
        frac = 1.0 - parsed.null_count() / len(s)
        if frac < min_fraction:
            continue
        ok = parsed.drop_nulls()
        years = ok.dt.year()
        if len(ok) and (years.min() < 1900 or years.max() > 2100):
            continue
        if fmt.startswith("%Y%m%d") and len(ok) > 3 and ok.dt.day().n_unique() < 2 and ok.dt.month().n_unique() < 2:
            continue
        if best is None or frac > best[0]:
            best = (frac, fmt)
            if frac == 1.0:
                break
    return best[1] if best else None


def format_seconds(secs: float) -> str:
    if secs < 1:
        return f"{secs * 1000:.0f} ms"
    if secs < 60:
        return f"{secs:.1f} s"
    if secs < 3600:
        return f"{secs / 60:.1f} min"
    if secs < 86400:
        return f"{secs / 3600:.1f} h"
    return f"{secs / 86400:.1f} d"
