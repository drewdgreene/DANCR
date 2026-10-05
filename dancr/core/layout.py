"""Read a sheet the way a person does: where its column names are, which rows are data, and how the page is laid
out around them.

People lay out spreadsheets for other people to read. A lab sheet puts two tables side by side under the
banners TREATED and CONTROL, sharing a column of sample numbers; a report puts a title above the column names,
AVERAGE and STANDARD DEV rows under the data, and an empty template row left over from the handout. Read as rows
and columns that is one wide table with words in its number columns. Read as a person reads it, it is thirty
samples in two groups.

This module looks at the cells of a sheet (an Excel sheet or a CSV file) and says how to read it:

- the **column names**: the first row as wide as the data under it, below any title lines;
- **banners** above them over runs of the same column names (TREATED and CONTROL, five columns each):
  the runs are one table in groups, read one under another with a ``group`` column. Generic names under banners
  (``min | max`` under Pressure and Temperature) are one header written in two rows instead (Pressure min …);
- the same column names again **further down**, each block under its own title line: one table, one block under
  another, with a ``group`` column from the titles;
- **section lines** inside the data (a lone ``Site B`` between runs of rows): a ``group`` column;
- **summary rows** under the data (AVERAGE, STANDARD DEV, MEDIAN, Total): left out, and checked against the data;
- **empty and template rows** (a row numbered 16 with nothing else in it): left out;
- a label written only on the **first row of each run** (merged cells, pivot exports): filled down;
- **several tables** on one sheet, side by side with an empty column between or one below another: each its own.

Deterministic, no Qt. Everything done is said in plain words (``notes``). A sheet that is already a plain table
is read exactly as before.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

HEAD_ROWS = 300            # rows read from the top of a big sheet to work out its layout
TAIL_ROWS = 60             # rows read from the bottom of a big sheet, for summary rows under the data
GRID_ROWS = 20_000         # a sheet up to this many rows is read whole, so every layout can be found
GRID_CELLS = 2_000_000     # ... and up to this many cells (a wide sheet is read from its top and bottom instead)
TAIL_BYTES = 64 * 1024     # bytes read from the end of a big CSV file
HEADER_SEARCH = 30         # the column names are within this many rows of the top of a table

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


# =================================================================== the grid
@dataclass
class Grid:
    """The cells of a sheet as text (None for an empty cell): all of it, or its top and bottom."""
    rows: list[list[str | None]]                    # the top of the sheet (all of it when ``complete``)
    tail: list[list[str | None]] = field(default_factory=list)   # the last rows, when not complete
    total: int = 0                                  # rows in the sheet (0: not known, a big CSV file)
    width: int = 0
    complete: bool = True
    col_offset: int = 0                             # the sheet column of grid column 0 (a sheet may start at column B)


def _cell(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _pad(rows: list, width: int) -> list[list[str | None]]:
    return [[_cell(v) for v in r] + [None] * (width - len(r)) for r in rows]


def _stamp(path: Path) -> tuple:
    st = path.stat()
    return (str(path.resolve()), st.st_size, st.st_mtime_ns)


def excel_grid(path: Path, sheet: int | str) -> Grid | None:
    """The cells of an Excel sheet as text: every row of a sheet up to ``GRID_ROWS``, else its top and bottom.
    Kept for the file as it is (its size and time), so previews and samples do not read it again."""
    return _excel_grid(_stamp(path), sheet)


@lru_cache(maxsize=8)
def _excel_grid(stamp: tuple, sheet: int | str) -> Grid | None:
    path = Path(stamp[0])
    import warnings
    import fastexcel
    try:
        reader = fastexcel.read_excel(str(path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skip_rows=0 keeps the empty rows at the top, so row numbers here are the sheet's own, as the typed
            # reads count them (without it, a sheet starting at row 3 would be read two rows off)
            head = reader.load_sheet(sheet, header_row=None, skip_rows=0, n_rows=GRID_ROWS + 1, dtypes="string")
            total = int(head.total_height)
            cols = head.available_columns()
            offset = int(cols[0].absolute_index - cols[0].index) if cols else 0
            df = _frame(head)
            tail: list = []
            complete = total <= GRID_ROWS and total * max(df.width, 1) <= GRID_CELLS
            if total > GRID_ROWS:
                df = df.head(HEAD_ROWS)
                tail = _frame(reader.load_sheet(sheet, header_row=None, skip_rows=total - TAIL_ROWS, dtypes="string")).rows()
            elif not complete:                     # all of it is here, but too many cells to keep: its top and bottom
                tail = df.tail(TAIL_ROWS).rows()
                df = df.head(HEAD_ROWS)
    except Exception:  # noqa: BLE001 - the real read reports any problem with the file
        return None
    width = df.width
    return Grid(_pad(df.rows(), width), _pad(tail, width), total, width, complete=complete, col_offset=offset)


def _frame(sheet):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)      # a Polars notice about its own interface
        return sheet.to_polars()


def csv_grid(path: Path, sep: str, encoding: str) -> Grid | None:
    """The cells of a CSV file as text: every row of a file up to ``GRID_ROWS`` rows, else its top and bottom."""
    return _csv_grid(_stamp(path), sep, encoding)


@lru_cache(maxsize=8)
def _csv_grid(stamp: tuple, sep: str, encoding: str) -> Grid | None:
    path = Path(stamp[0])
    enc = "utf-8-sig" if encoding == "utf8" else "latin-1"      # a byte-order mark is not part of the first name
    try:
        rows: list[list[str]] = []
        with open(path, encoding=enc, errors="replace", newline="") as f:
            for r in csv.reader(f, delimiter=sep):
                rows.append(r)
                if len(rows) > GRID_ROWS:
                    break
        complete = len(rows) <= GRID_ROWS
        tail: list[list[str]] = []
        wide = max((len(r) for r in rows[:HEAD_ROWS]), default=0)
        if complete and len(rows) * max(wide, 1) > GRID_CELLS:
            width = max(len(r) for r in rows)
            return Grid(_pad(rows[:HEAD_ROWS], width), _pad(rows[-TAIL_ROWS:], width), len(rows), width, complete=False)
        if not complete:
            rows = rows[:HEAD_ROWS]
            size = path.stat().st_size
            with open(path, "rb") as fb:
                fb.seek(max(0, size - TAIL_BYTES))
                text = fb.read().decode(enc, errors="replace")
            tail = list(csv.reader(text.splitlines()[1:][-TAIL_ROWS:], delimiter=sep))   # the first line may be cut
    except Exception:  # noqa: BLE001
        return None
    width = max((len(r) for r in rows + tail), default=0)
    return Grid(_pad(rows, width), _pad(tail, width), len(rows) if complete else 0, width, complete=complete)


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


# =================================================================== what a sheet holds
@dataclass
class Part:
    """One rectangle of a table: the row with its column names, the grid columns it uses (label columns every
    part shares first), and the rows of data under it."""
    header: int
    cols: list[int]
    names: list[str]
    first: int
    stop: int | None = None                  # one past the last data row; None: to the end of the sheet
    cut: int = 0                             # rows to leave off the bottom when the end is not known (a big CSV)
    drop: list[int] = field(default_factory=list)          # rows between first and stop that are left out
    label: str | None = None                 # the group this part is: its banner, or the title above it
    sections: dict[int, str] = field(default_factory=dict)  # a row -> the section line it is (rows below are in it)
    fill_down: list[str] = field(default_factory=list)      # label columns written only on the first row of a run
    empty: list[int] = field(default_factory=list)          # rows left out that had only a label or a number in them


@dataclass
class SheetTable:
    """A table found on a sheet: the parts to read and put one under another."""
    parts: list[Part]
    names: list[str]
    title: str = ""                                   # a title line above it
    group: str | None = None                          # the column that says which part (or section) a row is in
    summary: list[dict[str, Any]] = field(default_factory=list)   # {"row", "label", "stat"}
    checks: list[dict[str, Any]] = field(default_factory=list)    # summary values compared with the data
    notes: list[str] = field(default_factory=list)
    kind: str = "plain"                               # plain | groups | blocks | sections | two-row header
    col_offset: int = 0                               # the sheet column of grid column 0

    @property
    def plain(self) -> bool:
        """One rectangle from the first column, nothing left out between its rows, no groups: read the usual way
        (after its column-name row, and without the summary rows at the bottom)."""
        p = self.parts[0]
        return len(self.parts) == 1 and self.kind == "plain" and not p.drop and not p.sections and not p.fill_down \
            and p.cols == list(range(len(p.cols)))

    def describe(self) -> dict[str, Any]:
        groups = [p.label for p in self.parts if p.label] or list(dict.fromkeys(s for p in self.parts for s in p.sections.values()))
        return {"kind": self.kind, "title": self.title, "group_column": self.group, "groups": groups,
                "header_row": self.parts[0].header + 1,
                "summary_rows": [{"row": s["row"] + 1 if s["row"] >= 0 else None, "label": s["label"], "stat": s["stat"]}
                                 for s in self.summary],
                "checks": self.checks}


def find_tables(grid: Grid, header: int | None = None) -> list[SheetTable]:
    """The tables on a sheet, top to bottom and left to right. ``header`` fixes the row of the first table's
    column names (the loader's 'Skip rows at top')."""
    if grid.width == 0 or not grid.rows:
        return []
    out: list[SheetTable] = []
    start = 0
    for _ in range(12):                               # a sheet holds a few tables, not dozens
        h = header if (header is not None and not out) else _header_row(grid, start)
        if h is None or h >= len(grid.rows):
            break
        tables, end = _tables_at(grid, h, start)
        for t in tables:
            t.col_offset = grid.col_offset
        out += tables
        if not grid.complete or end >= len(grid.rows):
            break
        start = end
    return out


# =================================================================== the column names
def _typical_width(grid: Grid, start: int) -> int:
    widths = sorted(len(_filled(r)) for r in grid.rows[start:start + 200] if _filled(r))
    if not widths:
        return 0
    body = widths[len(widths) // 2:]
    return body[len(body) // 2]


def _header_row(grid: Grid, start: int) -> int | None:
    """The first row from ``start`` that names the columns: nearly as wide as the rows under it and mostly words.
    None when nothing below ``start`` looks like a table."""
    rows = grid.rows
    first = next((i for i in range(start, len(rows)) if _filled(rows[i])), None)
    if first is None:
        return None
    typical = _typical_width(grid, first)
    if typical < 2:
        return first
    lo = min((_filled(r)[0] for r in rows[first:first + 20] if _filled(r)), default=0)
    for i in range(first, min(len(rows), first + HEADER_SEARCH)):
        cells = _filled(rows[i])
        # a header's width is how far it reaches: pandas writes ",a,b" (no name over the index column)
        together = bool(cells) and len(cells) == cells[-1] - cells[0] + 1     # only its first cells blank
        reach = (cells[-1] - lo + 1) if together and len(cells) >= 2 else len(cells)
        if max(len(cells), reach if i == first and start == 0 else len(cells)) < max(2, typical - typical // 5):
            continue
        years = [number(rows[i][j]) for j in cells if not is_text(rows[i][j])]
        yearly = len(years) >= 2 and len(set(years)) == len(years) and all(y is not None and y == int(y) and 1900 <= y <= 2100 for y in years)
        if sum(is_text(rows[i][j]) for j in cells) * 2 >= len(cells) or (start == 0 and i == first) or \
                (yearly and any(is_text(rows[i][j]) for j in cells)):
            return i                                  # (numbers from the very first row: a file without names;
            # "Region, 2019, 2020, 2021": years are names too)
    return first


# =================================================================== tables at a header row
def _tables_at(grid: Grid, h: int, start: int) -> tuple[list[SheetTable], int]:
    """The tables whose column names are in row ``h``, and the row where the next table below may start."""
    head = grid.rows[h]
    body = grid.rows[h + 1:h + 201]
    # a column with neither a name nor data separates one table (or one group) from the next
    used = [j for j in range(grid.width) if head[j] is not None or any(r[j] is not None for r in body)]
    segments: list[list[int]] = []
    for j in used:
        if segments and j == segments[-1][-1] + 1:
            segments[-1].append(j)
        else:
            segments.append([j])
    banners = _banners(grid, h, start)
    tables = _split(grid, h, segments, banners)
    title = _title_above(grid, h, start, banners)
    end = h + 1
    for t in tables:
        t.title = t.title or title
        for p in t.parts:
            _extent(grid, t, p)
        end = max(end, *[(p.stop or len(grid.rows)) + p.cut for p in t.parts], *[i + 1 for p in t.parts for i in p.empty],
                  *[s["row"] + 1 for s in t.summary])
    if grid.complete and len(tables) == 1:
        end = _blocks_below(grid, tables[0], end)
    for t in tables:
        _finish(grid, t)
    while end < len(grid.rows) and (not _filled(grid.rows[end]) or _is_summary_row(grid.rows[end])):
        end += 1
    return tables, end


def _banners(grid: Grid, h: int, start: int) -> dict[int, str]:
    """Words in the row just above the column names (or the one above that) that head runs of columns:
    {column: banner}. A row with one cell is a title, not banners."""
    for i in (h - 1, h - 2):
        if i < start:
            break
        r = grid.rows[i]
        cells = _filled(r)
        if not cells:
            continue
        if len(cells) >= 2 and all(is_text(r[j]) for j in cells):
            return {j: r[j] for j in cells}
        break
    return {}


def _title_above(grid: Grid, h: int, start: int, banners: dict[int, str]) -> str:
    """A line of words above the column names that is not a row of banners: the table's title."""
    for i in range(h - 1, max(start, h - 6) - 1, -1):
        cells = _filled(grid.rows[i])
        if len(cells) == 1 and is_text(grid.rows[i][cells[0]]) and cells[0] not in banners:
            return grid.rows[i][cells[0]]
    return ""


def _letters(j: int) -> str:
    s, j = "", j + 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def _unique(names: list[str]) -> list[str]:
    out, taken = [], set()
    for base in names:
        name, k = base, 2
        while name in taken:
            name, k = f"{base}_{k}", k + 1
        taken.add(name)
        out.append(name)
    return out


def _names(head: list[str | None], cols: list[int]) -> list[str]:
    return _unique([head[j] if head[j] is not None else f"column {_letters(j)}" for j in cols])


def _split(grid: Grid, h: int, segments: list[list[int]], banners: dict[int, str]) -> list[SheetTable]:
    """The tables under one row of column names. Each run of columns between empty columns is a table, unless
    banners above say more: runs of the same names under different banners are one table in groups (they may be
    in different runs, an empty column between them); generic names under banners (min | max under Pressure and
    Temperature) are one header written in two rows. Banners over anything else are only a title."""
    head = grid.rows[h]
    runs: list[tuple[int, list[int], list[int], str | None]] = []   # (segment, label columns, columns, banner)
    for si, seg in enumerate(segments):
        starts = sorted(j for j in banners if seg[0] <= j <= seg[-1])
        if not starts:
            runs.append((si, [], seg, None))
            continue
        shared = [j for j in seg if j < starts[0]]
        for a, b in zip(starts, starts[1:] + [seg[-1] + 1]):
            runs.append((si, shared, [j for j in seg if a <= j < b], banners[a]))
    tables: list[SheetTable] = []
    grouped: set[int] = set()
    i = 0
    while i < len(runs):
        _, shared, cols, banner = runs[i]
        sig = [_key(head[j]) for j in cols]
        k = i + 1
        while k < len(runs) and banner is not None and runs[k][3] is not None and all(sig) and \
                [_key(head[j]) for j in runs[k][2]] == sig and (runs[k][1] == shared or not runs[k][1]):
            k += 1
        if k - i > 1:
            tables.append(_grouped(grid, h, [(r[1], r[2], r[3]) for r in runs[i:k]]))
            grouped.update(range(i, k))
        i = k
    for si, seg in enumerate(segments):                  # what is left, a segment at a time
        left = [r for n, r in enumerate(runs) if r[0] == si and n not in grouped]
        if not left:
            continue
        under = [r for r in left if r[3] is not None]
        if under and any(_key(head[j]) in {_key(g) for g in GENERIC} for r in under for j in r[2]):
            tables.append(_two_row(grid, h, [(r[1], r[2], r[3]) for r in left if r[3] is not None]))
            continue
        cols = sorted({j for r in left for j in r[1] + r[2]})
        names = _names(head, cols)
        title = " ".join(r[3] for r in under) if under else ""
        tables.append(SheetTable(parts=[Part(h, cols, names, h + 1)], names=names, title=title))
    order = {id(t): min(p.cols[0] for p in t.parts) for t in tables}
    return sorted(tables, key=lambda t: order[id(t)])


def _grouped(grid: Grid, h: int, runs: list[tuple[list[int], list[int], str | None]]) -> SheetTable:
    """Runs of the same columns under different banners: one table in groups (Treated samples, Control samples) — unless
    the names under them are parts of one quantity (min | max under Pressure and Temperature), which is one
    header written in two rows."""
    head = grid.rows[h]
    if all(_key(head[j]) in {_key(g) for g in GENERIC} for j in runs[0][1]):
        return _two_row(grid, h, runs)
    banners = [r[2] or "" for r in runs]
    labels = group_labels(banners)
    shared = runs[0][0]
    names = _names(head, shared + runs[0][1])
    parts = [Part(h, shared + cols, names, h + 1, label=lab) for (_, cols, _), lab in zip(runs, labels)]
    t = SheetTable(parts=parts, names=names, kind="groups")
    t.group = _free("group", names)
    shown = " and ".join(f"“{b}”" for b in banners) if len(banners) <= 3 else f"{len(banners)} banners"
    t.notes.append(f"Read the {len(runs)} side-by-side tables under {shown} as one table, one under another, with a "
                   f"“{t.group}” column saying which each row came from ({', '.join(labels)})")
    return t


def _two_row(grid: Grid, h: int, runs: list[tuple[list[int], list[int], str | None]]) -> SheetTable:
    head = grid.rows[h]
    shared = runs[0][0]
    names = [head[j] if head[j] is not None else f"column {_letters(j)}" for j in shared]
    cols = list(shared)
    for _, run_cols, banner in runs:
        for j in run_cols:
            n = head[j] if head[j] is not None else f"column {_letters(j)}"
            names.append(f"{banner} {n}" if banner else n)
            cols.append(j)
    names = _unique(names)
    t = SheetTable(parts=[Part(h, cols, names, h + 1)], names=names, kind="two-row header")
    shown = names[len(shared):len(shared) + 3]
    t.notes.append(f"Took the column names from two rows ({', '.join(shown)}{' …' if len(names) - len(shared) > 3 else ''})")
    return t


def group_labels(banners: list[str]) -> list[str]:
    """Short labels for groups named by banners: the words they all end with are left off (TREATED PLOTS, CONTROL
    PLOTS -> Treated, Control), and a word in capitals is written as a word. Kept whole when that would leave one
    empty or two the same."""
    words = [b.split() for b in banners]
    common = 0
    while all(len(w) > common + 1 for w in words) and len({w[-1 - common].lower() for w in words}) == 1:
        common += 1
    out = [" ".join(w[:len(w) - common]) for w in words]
    if any(not o for o in out) or len(set(out)) < len(out):
        out = list(banners)
    return [" ".join(x.capitalize() if x.isupper() and len(x) > 2 else x for x in o.split()) for o in out]


def _free(base: str, taken: list[str]) -> str:
    name, k = base, 2
    low = {t.lower() for t in taken}
    while name.lower() in low:
        name, k = f"{base}_{k}", k + 1
    return name


# =================================================================== the rows of each part
def _is_summary_row(r: list[str | None], cols: list[int] | None = None) -> bool:
    cells = _filled(r, cols)
    return bool(cells) and is_text(r[cells[0]]) and summary_stat(r[cells[0]]) is not None and \
        all(number(r[j]) is not None for j in cells[1:])


def _row_kind(r: list[str | None], p: Part, own: list[int], header_keys: list[str]) -> str:
    """blank | header (the column names again) | summary | label (nothing but the shared label or row number) |
    section (one line of words alone) | data."""
    cells = _filled(r, p.cols)
    if not cells:
        return "blank"
    if [_key(r[j]) for j in p.cols] == header_keys:
        return "header"
    if _is_summary_row(r, p.cols):
        return "summary"
    if not _filled(r, own):
        return "label"
    if len(cells) == 1 and is_text(r[cells[0]]) and len(p.cols) > 1:
        return "section"
    return "data"


def _own_cols(grid: Grid, t: SheetTable, p: Part) -> list[int]:
    """The columns that hold a part's own data: not the label columns every part shares, nor (for a table on its
    own) a first column of row numbers 1, 2, 3 …, so that a numbered row with nothing else in it is seen as empty."""
    if len(t.parts) > 1:
        shared = set.intersection(*[set(q.cols) for q in t.parts])
        return [j for j in p.cols if j not in shared]
    if len(p.cols) > 1:
        vals = [number(grid.rows[i][p.cols[0]]) for i in range(p.first, min(len(grid.rows), p.first + 200))]
        run = [v for v in vals if v is not None]
        if len(run) >= 3 and run[0] in (0.0, 1.0) and all(b - a == 1 for a, b in zip(run, run[1:])):
            return p.cols[1:]
    return list(p.cols)


def _extent(grid: Grid, t: SheetTable, p: Part) -> None:
    """Where a part's data ends, which rows in it are left out, and the summary rows under it."""
    own = _own_cols(grid, t, p)
    keys = [_key(grid.rows[p.header][j]) for j in p.cols]
    if not grid.complete:
        _tail(grid, t, p, own, keys)
        return
    kinds: list[tuple[int, str]] = []
    blanks = 0
    more = False                                           # something follows: another block or table
    for i in range(p.first, len(grid.rows)):
        k = _row_kind(grid.rows[i], p, own, keys)
        if k == "header":
            more = True
            break                                          # the same names again: a block below, read later
        if k == "blank":
            blanks += 1
            if blanks >= 3:
                more = any(_filled(r) for r in grid.rows[i + 1:])
                break                                      # three empty rows: this table has ended
        else:
            if (blanks or (kinds and kinds[-1][1] == "section")) and _starts_table(grid, i, p, kinds):
                more = True
                break                                      # the next table's names (under its title line)
            blanks = 0
        kinds.append((i, k))
    # a title line at the end belongs to what comes next (the next block's name), when something does come next;
    # then summary rows at the bottom (with blank and empty rows among them)
    while kinds and (kinds[-1][1] == "blank" or (more and kinds[-1][1] == "section")):
        kinds.pop()
    bottom: list[tuple[int, str]] = []
    while kinds and kinds[-1][1] in ("summary", "blank", "label"):
        bottom.append(kinds.pop())
    summ = sorted(i for i, k in bottom if k == "summary")
    left = sum(k == "data" for _, k in kinds)
    # summary rows sum up data above them: not a table of statistics itself (mean, median, sd …), nor most of one
    if summ and left >= 2 and len(summ) <= left and _is_summary(grid, summ, any(k == "blank" for _, k in bottom)):
        for i in summ:
            lead = _filled(grid.rows[i], p.cols)[0]
            if i not in {s["row"] for s in t.summary}:
                t.summary.append({"row": i, "label": grid.rows[i][lead], "stat": summary_stat(grid.rows[i][lead])[0]})
        p.empty += [i for i, k in bottom if k == "label"]
    else:
        kinds += list(reversed(bottom))
    while kinds and kinds[-1][1] in ("blank", "label"):
        i, k = kinds.pop()
        if k == "label":
            p.empty.append(i)                              # a numbered row with nothing else in it: a template row
    p.stop = kinds[-1][0] + 1 if kinds else p.first
    p.drop = [i for i, k in kinds if k == "blank" or (k == "label" and len(t.parts) > 1)]
    p.empty += [i for i, k in kinds if k == "label" and len(t.parts) > 1]
    secs = [(i, grid.rows[i][_filled(grid.rows[i], p.cols)[0]]) for i, k in kinds if k == "section"]
    if secs and (len(secs) >= 2 or secs[0][0] == p.first):
        seen = {grid.rows[i][j] for i, k in kinds if k == "data" for j in p.cols}
        rows = [i for i, _ in secs] + [kinds[-1][0] + 1]
        under = [sum(1 for i, k in kinds if k == "data" and a < i < b) for a, b in zip(rows, rows[1:])]
        before = sum(1 for i, k in kinds if k == "data" and i < secs[0][0])
        # a section heads a run of rows; a lone name with its other cells empty is a row with missing values
        if not any(s in seen for _, s in secs) and min(under) >= 2 and before == 0:
            p.sections = dict(secs)
            p.drop += list(p.sections)
    p.drop = sorted(set(p.drop))
    p.empty = sorted(set(p.empty))
    p.fill_down = _fill_down(grid, p, [i for i, k in kinds if k == "data"])


def _starts_table(grid: Grid, i: int, p: Part, kinds: list[tuple[int, str]] | None = None) -> bool:
    """After a blank row, a row of words as wide as the table, followed by values, with words where this table has
    numbers: the next table's names (a row of this table's own text, after a gap, is not)."""
    r = grid.rows[i]
    cells = _filled(r, p.cols)
    if len(cells) < max(2, len(p.cols) * 3 // 4) or not all(is_text(r[j]) for j in cells) or i + 1 >= len(grid.rows):
        return False
    nxt = grid.rows[i + 1]
    if not any(not is_text(nxt[j]) for j in _filled(nxt)):
        return False
    data = [k for k, kind in (kinds or []) if kind == "data"][-20:]
    if not data:
        return True
    numeric = [j for j in cells if sum(number(grid.rows[k][j]) is not None for k in data) * 2 > len(data)]
    return bool(numeric)


def _is_summary(grid: Grid, rows: list[int], blank_between: bool) -> bool:
    """Summary-like labels at the bottom are summary rows when one is unmistakable (AVERAGE, Total), when there are
    several, or when a blank row sets them apart from the data."""
    strong = any(summary_stat(grid.rows[i][_filled(grid.rows[i])[0]])[1] for i in rows)
    return strong or len(rows) >= 2 or blank_between


def _tail(grid: Grid, t: SheetTable, p: Part, own: list[int], keys: list[str]) -> None:
    """A sheet too big to read whole: summary rows are looked for among its last rows."""
    rows = grid.tail
    kinds = [(k, _row_kind(r, p, own, keys)) for k, r in enumerate(rows)]
    while kinds and kinds[-1][1] == "blank":
        kinds.pop()
    bottom = []
    while kinds and kinds[-1][1] in ("summary", "blank"):
        bottom.append(kinds.pop())
    summ = sorted(k for k, kind in bottom if kind == "summary")
    if not summ or not kinds or not _is_summary(Grid(rows), summ, any(kind == "blank" for _, kind in bottom)):
        return
    base = grid.total - len(rows) if grid.total else None
    for k in summ:
        lead = _filled(rows[k], p.cols)[0]
        t.summary.append({"row": base + k if base is not None else -1, "label": rows[k][lead],
                          "stat": summary_stat(rows[k][lead])[0]})
    cut = len(rows) - (kinds[-1][0] + 1)                   # rows under the last row of data
    if base is not None:
        p.stop = grid.total - cut
    else:
        p.cut = cut


def _fill_down(grid: Grid, p: Part, data: list[int]) -> list[str]:
    """Label columns written only on the first row of each run (merged cells, pivot exports): a column of words
    among the first three, filled on the first row of data, blank on at least a third of them, and never the same
    word twice in a row where it is written (a run is written once)."""
    if len(data) < 4:
        return []
    out = []
    for pos, j in enumerate(p.cols[:3]):
        if pos > len(out):
            break                  # labels written once per run sit leftmost (Region, then Store): not a notes column
        vals = [grid.rows[i][j] for i in data]
        filled = [v for v in vals if v is not None]
        if vals[0] is None or not all(is_text(v) for v in filled) or (len(vals) - len(filled)) * 3 < len(vals):
            continue
        if any(a == b for a, b in zip(filled, filled[1:])) or len(filled) * 2 > len(vals):
            continue
        others = [c for c in p.cols if c != j]
        if all(_filled(grid.rows[i], others) for i in data):
            out.append(p.names[p.cols.index(j)])
    return out


def _blocks_below(grid: Grid, t: SheetTable, end: int) -> int:
    """The same column names again further down (a block per site, per week): more parts of the same table, each
    named by the title line above it. Returns the row after the last block."""
    p0 = t.parts[0]
    keys = [_key(grid.rows[p0.header][j]) for j in p0.cols]
    found: list[Part] = []
    i = end
    while i < len(grid.rows):
        r = grid.rows[i]
        cells = _filled(r)
        if [_key(r[j]) for j in p0.cols] == keys:
            q = Part(i, list(p0.cols), list(p0.names), i + 1, label=_title_above(grid, i, max(0, i - 4), {}) or None)
            _extent(grid, t, q)
            found.append(q)
            i = max(q.stop or i + 1, i + 1)
            continue
        if len(cells) >= 2 and all(is_text(r[j]) for j in cells) and _starts_table(grid, i, p0):
            break                                          # another table's names
        if cells and not _is_summary_row(r) and len(cells) > 1:
            break                                          # something else below: not blocks of this table
        i += 1
    if not found:
        return end
    p0.label = p0.label or _title_above(grid, p0.header, 0, {}) or None
    t.parts = [p0] + found
    t.kind = "blocks"
    labels = [q.label for q in t.parts]
    if all(labels) and len(set(labels)) == len(labels):
        t.group = _free("group", t.names)
        t.notes.append(f"Read the {len(t.parts)} blocks with the same column names ({', '.join(labels)}) as one table, "
                       f"with a “{t.group}” column saying which block each row came from")
        t.title = ""
    else:
        for q in t.parts:
            q.label = None
        t.notes.append(f"Read the {len(t.parts)} blocks with the same column names as one table, one under another")
    return max((q.stop or q.first) for q in t.parts)


def _finish(grid: Grid, t: SheetTable) -> None:
    """Say what was done, and check the summary rows against the data."""
    secs = list(dict.fromkeys(s for p in t.parts for s in p.sections.values()))
    if secs and not t.group:
        t.group = _free("group", t.names)
        t.kind = "sections" if t.kind == "plain" else t.kind
        t.notes.append(f"Rows under the lines {', '.join(f'“{s}”' for s in secs[:4])}{' …' if len(secs) > 4 else ''} "
                       f"are marked with that name in a “{t.group}” column")
    fills = list(dict.fromkeys(c for p in t.parts for c in p.fill_down))
    if fills:
        t.notes.append(f"Filled {', '.join(f'“{c}”' for c in fills)} down: it is written only on the first row of each run")
    if t.summary:
        labels = list(dict.fromkeys(s["label"] for s in t.summary))
        t.notes.append(f"Left out {len(t.summary)} summary row{'s' if len(t.summary) != 1 else ''} under the data "
                       f"({', '.join(labels[:5])})")
        if grid.complete:
            t.checks = _check_summary(grid, t)
            bad = [c for c in t.checks if not c["ok"]]
            if t.checks and not bad:
                t.notes.append(f"They agree with the data ({len(t.checks)} value{'s' if len(t.checks) != 1 else ''} checked)")
            for c in bad[:4]:
                t.notes.append(f"The sheet's {c['label']} of {c['column']}{' for ' + c['group'] if c.get('group') else ''} "
                               f"is {c['sheet']:g}, but the data gives {c['data']:.6g}")
    empty = sorted({i for p in t.parts for i in p.empty})
    if empty:
        shown = ", ".join(str(i + 1) for i in empty[:5]) + (" …" if len(empty) > 5 else "")
        t.notes.append(f"Left out {len(empty)} row{'s' if len(empty) != 1 else ''} with nothing to read in "
                       f"{'them' if len(empty) != 1 else 'it'} but a label or number (sheet row{'s' if len(empty) != 1 else ''} {shown})")


def _check_summary(grid: Grid, t: SheetTable) -> list[dict[str, Any]]:
    """Each summary value compared with the same statistic of the data above it (for each group side by side, or
    the block just above). Excel's STDEV is the sample standard deviation; the population one is accepted too."""
    out = []
    for s in t.summary:
        r = grid.rows[s["row"]]
        above = [p for p in t.parts if p.stop is not None and p.stop <= s["row"]]
        if not above:
            continue
        last = max(p.stop for p in above)
        lead = _filled(r)[0]
        # side by side, every group ends above the summary row (a shorter group sooner); in blocks, the one just above
        for p in (above if t.kind == "groups" else [p for p in above if p.stop == last]):
            rows = [i for i in range(p.first, p.stop) if i not in p.drop]
            for j, name in zip(p.cols, p.names):
                sheet = number(r[j])
                if j == lead or sheet is None:
                    continue
                vals = [v for v in (number(grid.rows[i][j]) for i in rows) if v is not None]
                if len(vals) < 2:
                    continue
                data, alt = _stat(vals, s["stat"])
                if data is None:
                    continue
                ok = _close(sheet, data, r[j]) or (alt is not None and _close(sheet, alt, r[j]))
                out.append({"label": s["label"], "column": name, "group": p.label, "sheet": sheet,
                            "data": round(data, 10), "ok": ok})
    return out


def _stat(vals: list[float], stat: str) -> tuple[float | None, float | None]:
    """A statistic of the values, and a second reading of it that a sheet may mean instead (population spread)."""
    n = len(vals)
    mean = sum(vals) / n
    if stat in ("std", "var", "se", "cv"):
        ss = sum((v - mean) ** 2 for v in vals)
        var, pvar = ss / (n - 1), ss / n
        return {"std": (math.sqrt(var), math.sqrt(pvar)), "var": (var, pvar), "se": (math.sqrt(var / n), None),
                "cv": ((math.sqrt(var) / mean * 100) if mean else None, None)}[stat]
    s = sorted(vals)
    return {"mean": mean, "sum": sum(vals), "median": (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2),
            "min": s[0], "max": s[-1], "range": s[-1] - s[0], "count": float(n)}.get(stat), None


def _close(sheet: float, data: float, text: str | None) -> bool:
    """Equal to the precision the sheet shows (12.3 agrees with 12.34), or to a millionth."""
    body = (text or "").rstrip("%").strip()
    decimals = len(body.split(".")[1]) if "." in body and "e" not in body.lower() else 0
    tol = max(0.5 * 10 ** -decimals, 1e-6 * max(abs(sheet), abs(data)))
    return abs(sheet - data) <= tol + 1e-12
