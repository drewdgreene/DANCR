"""Paged access to a big table for the GUI grid and the CLI sample command."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, date
from typing import Any

import polars as pl

from ..core.expr import _kind_of_dtype


class TablePager:
    def __init__(self, lf: pl.LazyFrame, rows: int | None = None, page_size: int = 500, max_pages: int = 40) -> None:
        self.lf = lf
        self.page_size = page_size
        self.max_pages = max_pages
        self.schema = dict(lf.collect_schema())
        self.columns = list(self.schema)
        self.kinds = {c: _kind_of_dtype(dt) for c, dt in self.schema.items()}
        self._rows = rows
        self._pages: OrderedDict[int, pl.DataFrame] = OrderedDict()

    @property
    def rows(self) -> int:
        if self._rows is None:
            self._rows = int(self.lf.select(pl.len()).collect(engine="streaming")[0, 0])
        return self._rows

    def fetch_page(self, i: int) -> pl.DataFrame:
        """Read a page (safe to call from a worker thread)."""
        if i < 0:
            return self.lf.head(0).collect()
        return self.lf.slice(i * self.page_size, self.page_size).collect(engine="streaming")

    def store_page(self, i: int, df: pl.DataFrame) -> None:
        self._pages[i] = df
        self._pages.move_to_end(i)
        if len(self._pages) > self.max_pages:
            self._pages.popitem(last=False)

    def page(self, i: int) -> pl.DataFrame:
        if i in self._pages:
            self._pages.move_to_end(i)
            return self._pages[i]
        df = self.fetch_page(i)
        self.store_page(i, df)
        return df

    def value(self, row: int, col: int) -> Any:
        if row < 0:
            return None
        p, r = divmod(row, self.page_size)
        df = self.page(p)
        if r >= len(df):
            return None
        return df[r, col]

    def cached_value(self, row: int, col: int) -> tuple[bool, Any]:
        """(available, value) without any I/O."""
        if row < 0:
            return True, None
        p, r = divmod(row, self.page_size)
        df = self._pages.get(p)
        if df is None:
            return False, None
        return True, (df[r, col] if r < len(df) else None)


def format_value(v: Any) -> str:
    """One value as a person expects to read it: thousands separators, no float noise, times without a T."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:
            return ""
        if v in (float("inf"), float("-inf")):
            return "∞" if v > 0 else "-∞"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.6g}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, datetime):
        base = v.strftime("%Y-%m-%d %H:%M:%S")
        if v.microsecond:
            frac = f"{v.microsecond:06d}".rstrip("0")
            base += "." + (frac if len(frac) > 3 else f"{v.microsecond // 1000:03d}")
        if v.tzinfo is not None:
            base += v.strftime("%z")
        return base
    if isinstance(v, date):
        return v.isoformat()
    return str(v)
