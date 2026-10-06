"""Cell classification: numbers, text, labels, summary statistics and column keys."""
from __future__ import annotations

import re
from typing import Any


# a summary row's label, as people write it -> the statistic it holds
SUMMARY = {"total": "sum", "totals": "sum", "grand total": "sum", "subtotal": "sum", "sub total": "sum", "sum": "sum",
           "overall": "sum",
           "average": "mean", "averages": "mean", "avg": "mean", "mean": "mean", "means": "mean", "ave": "mean",
           "median": "median", "med": "median",
           "standard deviation": "std", "standard dev": "std", "std dev": "std", "stdev": "std", "st dev": "std",
           "std": "std", "sd": "std", "stdev s": "std", "stddev": "std", "std deviation": "std",
           "standard error": "se", "std error": "se", "std err": "se", "se": "se", "sem": "se",
           "standard error of the mean": "se",
           "minimum": "min", "min": "min", "maximum": "max", "max": "max", "range": "range",
           "count": "count", "n": "count", "sample size": "count",
           "variance": "var", "var": "var", "mode": "mode", "cv": "cv", "coefficient of variation": "cv"}
STRONG = {"total", "grand total", "subtotal", "sub total", "average", "averages", "mean", "means", "median",
          "standard deviation", "standard dev", "std dev", "stdev", "st dev", "std deviation", "stddev",
          "standard error", "standard error of the mean", "minimum", "maximum", "variance", "sample size",
          "coefficient of variation"}
# names under banners that are parts of one quantity (Pressure min, Pressure max) rather than quantities
GENERIC = {"min", "max", "minimum", "maximum", "mean", "average", "avg", "median", "sd", "std", "stdev", "se", "n",
           "count", "total", "sum", "value", "values", "%", "percent", "pct", "share", "low", "high", "start", "end",
           "actual", "budget", "target", "plan", "forecast", "variance", "diff", "difference", "change", "rank"}

_NUMBER = re.compile(r"^[(\-+]?\s*[$€£¥]?\s*[+-]?(\d[\d,' ]*)?\.?\d+(e[+-]?\d+)?\s*%?\s*\)?$", re.IGNORECASE)
_DATEISH = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
# =================================================================== reading cells
def number(v: str | None) -> float | None:
    """A cell's number as a person writes it (1,234.5  (12)  31.5%  £9), or None."""
    if v is None or not _NUMBER.match(v):
        return None
    s = v.strip()
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[()$€£¥%\s']", "", s).replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return -f if neg else f


def is_text(v: str | None) -> bool:
    """A cell that holds words: not blank, not a number, not a date."""
    return v is not None and number(v) is None and not _DATEISH.match(v)


def norm_label(v: str | None) -> str:
    """'STANDARD DEV:' -> 'standard dev'; 'S.D.' -> 'sd'; 'Std. Dev.' -> 'std dev'."""
    if v is None:
        return ""
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", v.lower())      # s.d. -> sd.
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    return " ".join(s.split())


def summary_stat(v: str | None) -> tuple[str, bool] | None:
    """(statistic, unmistakable?) when a cell labels a summary row: 'AVERAGE' -> ('mean', True), 'SD' -> ('std',
    False); 'Total sales' counts too. None for anything else."""
    k = norm_label(v)
    if not k:
        return None
    if k in SUMMARY:
        return SUMMARY[k], k in STRONG
    for word in ("grand total", "subtotal", "total", "average", "mean", "median", "standard deviation"):
        if k.startswith(word + " "):
            return SUMMARY[word], True
    return None


def _filled(r: list[str | None], cols: list[int] | None = None) -> list[int]:
    return [j for j in (cols if cols is not None else range(len(r))) if j < len(r) and r[j] is not None]


def _key(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())
