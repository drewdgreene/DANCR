"""Duration parsing and date-format sniffing."""
from __future__ import annotations

import re
from decimal import Decimal

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
    """'5 min' -> ('5m', 300.0). Returns (polars duration string, seconds). Times are kept to the
    microsecond, so a span shorter than that is refused."""
    out, secs = _parse_duration(text)
    if secs < 1e-6:
        raise ValueError(f"{text!r} is shorter than a microsecond, the finest time DANCR keeps")
    return out, secs


_CALENDAR = {"mo": "mo", "mon": "mo", "month": "mo", "months": "mo", "q": "q", "quarter": "q", "quarters": "q",
             "y": "y", "yr": "y", "year": "y", "years": "y"}
_CALENDAR_SECS = {"mo": 30.436875 * 86400, "q": 91.310625 * 86400, "y": 365.2425 * 86400}


def parse_bucket(text: str) -> tuple[str, float]:
    """A time bucket: any duration, or calendar months, quarters and years ('1mo', '3 months', '1q', '1y').
    Calendar buckets follow the calendar (January, February …), so their seconds are only an average."""
    m = _DUR_RE.match(str(text or "").strip().lower())
    if m and m.group(2) in _CALENDAR:
        n = float(m.group(1))
        if not n.is_integer() or n <= 0:
            raise ValueError(f"{text!r}: calendar buckets are whole months, quarters or years")
        unit = _CALENDAR[m.group(2)]
        return f"{int(n)}{unit}", n * _CALENDAR_SECS[unit]
    return parse_duration(text)


def _parse_duration(text: str) -> tuple[str, float]:
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
    # fractional: count it exactly in nanoseconds (decimal arithmetic: 1.001 s is 1001 ms, not 1000.999…)
    total_ns = Decimal(n) * _NS[pu]
    if total_ns != total_ns.to_integral_value():
        total_ns = total_ns.to_integral_value()             # finer than a nanosecond: the nearest one
    for u in ("w", "d", "h", "m", "s", "ms", "us", "ns"):
        q = total_ns / _NS[u]
        if q == q.to_integral_value():
            return f"{int(q)}{u}", float(total_ns) / 1e9
    return f"{int(total_ns)}ns", float(total_ns) / 1e9


_NS = {k: Decimal(v) for k, v in {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000, "m": 60_000_000_000,
                                   "h": 3_600_000_000_000, "d": 86_400_000_000_000, "w": 604_800_000_000_000}.items()}


# Tried in order; the first format that parses every sampled value wins, else the one that parses most.
DATE_FORMATS = [
    "%+",                                   # RFC 3339 / ISO 8601 with offset or Z
    "%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M%z",
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


# Formats that read the same text two ways (01/05/2024: 1 May or 5 January). Month first is the default.
DAY_FIRST = {f: f.replace("%d/%m", "%m/%d") for f in DATE_FORMATS if f.startswith("%d/%m")}
_MONTH_FIRST = {v: k for k, v in DAY_FIRST.items()}


def swap_day_month(fmt: str) -> str | None:
    """The other reading of a day/month format, or None if the format is not one of a pair."""
    return DAY_FIRST.get(fmt) or _MONTH_FIRST.get(fmt)


def day_month_label(fmt: str) -> str:
    return "day/month" if fmt in DAY_FIRST else "month/day"


def detect_datetime_format(sample: pl.Series, min_fraction: float = 0.9, day_first: bool = False) -> str | None:
    """Try known formats on a string sample; return the best one or None.

    Guards against false positives: separator-less formats (%Y%m%d) need a sane
    year range and more than one distinct day; version-like strings such as
    1.2.2024 are not dates unless every part is in range for every row.
    When day/month and month/day read the sample equally well, ``day_first`` decides
    (see :func:`day_month_ambiguous` to tell the person).
    """
    s = sample.drop_nulls().cast(pl.Utf8).str.strip_chars()
    s = s.filter(s != "")
    if len(s) == 0:
        return None
    if s.head(50).str.contains(r"\d").sum() < min(len(s), 50) * 0.9:
        return None
    best: tuple[float, str] | None = None
    fracs: dict[str, float] = {}
    for fmt in DATE_FORMATS:
        frac = _fraction_parsed(s, fmt, min_fraction)
        if frac is None:
            continue
        fracs[fmt] = frac
        if best is None or frac > best[0]:
            best = (frac, fmt)
        if frac == 1.0 and fmt not in DAY_FIRST:      # a day-first match waits for its month-first twin
            break
    if best is None:
        return None
    fmt = best[1]
    twin = swap_day_month(fmt)
    if twin is not None and fracs.get(twin) == best[0]:
        return fmt if day_first == (fmt in DAY_FIRST) else twin
    return fmt


def _fraction_parsed(s: pl.Series, fmt: str, min_fraction: float) -> float | None:
    try:
        parsed = s.str.to_datetime(fmt, strict=False)
    except Exception:
        return None
    frac = 1.0 - parsed.null_count() / len(s)
    if frac < min_fraction:
        return None
    ok = parsed.drop_nulls()
    years = ok.dt.year()
    if len(ok) and (years.min() < 1900 or years.max() > 2100):
        return None
    if fmt.startswith("%Y%m%d") and len(ok) > 3 and ok.dt.day().n_unique() < 2 and ok.dt.month().n_unique() < 2:
        return None
    return frac


def day_month_ambiguous(sample: pl.Series, fmt: str) -> bool:
    """True when the other day/month reading parses the sample just as well as ``fmt``."""
    twin = swap_day_month(fmt)
    if twin is None:
        return False
    s = sample.drop_nulls().cast(pl.Utf8).str.strip_chars()
    s = s.filter(s != "")
    if len(s) == 0:
        return False
    return s.str.to_datetime(fmt, strict=False).null_count() == s.str.to_datetime(twin, strict=False).null_count()


def settle_day_month(lf: pl.LazyFrame, column: str, fmt: str, sample: pl.Series, whole: bool) -> tuple[str, str | None]:
    """Resolve a day/month format the sample cannot decide. With ``whole`` (a real run, not a preview) the
    whole column is read once and the reading that fits more of it wins, so a file whose first rows are all
    early in the month is still read the right way. Returns (format, message for the person or None)."""
    if not day_month_ambiguous(sample, fmt):
        return fmt, None
    twin = swap_day_month(fmt)
    assert twin is not None
    if whole:
        raw = pl.col(column).cast(pl.Utf8).str.strip_chars()
        a, b = lf.select(raw.str.to_datetime(fmt, strict=False).is_not_null().sum().alias("a"),
                         raw.str.to_datetime(twin, strict=False).is_not_null().sum().alias("b")
                         ).collect(engine="streaming").row(0)
        if b > a:
            return twin, (f"'{column}': the first rows could be read as day/month or month/day. The whole file fits "
                          f"{day_month_label(twin)}, so that is used")
        if a > b:
            return fmt, (f"'{column}': the first rows could be read as day/month or month/day. The whole file fits "
                         f"{day_month_label(fmt)}, so that is used")
    return fmt, (f"'{column}': every date could be read as day/month or month/day, like 01/05/2024. Read as "
                 f"{day_month_label(fmt)}. If that's wrong, tick 'Day comes before month' or set 'Date format'")


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


_OFFSET = re.compile(r"(Z|[+-]\d{2}:?\d{2})\s*$")
# offsets that are not whole hours have no Etc/GMT zone; these places have kept theirs without daylight saving
_FIXED_ZONES = {"+05:30": "Asia/Kolkata", "+05:45": "Asia/Kathmandu", "+04:30": "Asia/Kabul",
                "+06:30": "Asia/Yangon", "+09:30": "Australia/Darwin", "-09:30": "Pacific/Marquesas"}


def utc_offsets(sample: pl.Series) -> list[str]:
    """The distinct UTC offsets written at the end of the sample's times, as +HH:MM ('Z' is +00:00)."""
    seen: list[str] = []
    for v in sample.drop_nulls().to_list():
        m = _OFFSET.search(str(v))
        if not m:
            continue
        o = "+00:00" if m.group(1) == "Z" else m.group(1).replace(":", "")
        o = o if o == "+00:00" else f"{o[:3]}:{o[3:]}"
        if o not in seen:
            seen.append(o)
    return seen


def check_time_zone(name: str) -> str:
    """A time zone name Polars knows (Europe/London, America/New_York, UTC), or a plain-English error."""
    try:
        pl.Series([0], dtype=pl.Datetime("us", "UTC")).dt.convert_time_zone(name)
    except Exception:  # noqa: BLE001 - Polars raises several kinds for an unknown name
        raise ValueError(f"{name!r} is not a time zone name. Use a name such as Europe/London, America/New_York "
                         "or UTC") from None
    return name


def offset_time_zone(column: str, sample: pl.Series, chosen: str | None) -> tuple[str, str | None]:
    """The zone to show times written with a UTC offset in, and a note for the person. A chosen zone wins; else
    one offset throughout keeps the times in that offset (so days and hours are the file's own), and several
    offsets (daylight saving) keep UTC, since no one offset is right for every time."""
    if chosen:
        return check_time_zone(chosen), None
    offsets = utc_offsets(sample)
    if len(offsets) == 1:
        o = offsets[0]
        if o == "+00:00":
            return "UTC", None
        h, m = int(o[1:3]), int(o[4:6])
        if m == 0:
            return f"Etc/GMT{'-' if o[0] == '+' else '+'}{h}", None          # Etc/GMT signs are the other way round
        if o in _FIXED_ZONES:
            return _FIXED_ZONES[o], None
        return "UTC", (f"'{column}' is written at UTC{o}, which isn't tied to a time zone, so times are shown in UTC. "
                       "Set 'Time zone' to see them in local time")
    if len(offsets) > 1:
        return "UTC", (f"'{column}' has times at several UTC offsets ({', '.join(offsets[:4])}), as with daylight "
                       "saving, so they're shown in UTC. Set 'Time zone' (e.g. Europe/London) to see them in local time")
    return "UTC", None


def has_offset(fmt: str) -> bool:
    """True when a date format reads a UTC offset (%z, %:z, or %+, ISO 8601 with its offset)."""
    return "%z" in fmt or "%:z" in fmt or "%+" in fmt
