"""Reading a sheet or a CSV file into a text grid, cached by the file's stamp."""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


HEAD_ROWS = 300            # rows read from the top of a big sheet to work out its layout
TAIL_ROWS = 60             # rows read from the bottom of a big sheet, for summary rows under the data
GRID_ROWS = 20_000         # a sheet up to this many rows is read whole, so every layout can be found
GRID_CELLS = 2_000_000     # ... and up to this many cells (a wide sheet is read from its top and bottom instead)
TAIL_BYTES = 64 * 1024     # bytes read from the end of a big CSV file
COMPRESS_SUFFIXES = (".gz", ".bgz")     # a compressed CSV is decompressed to read its grid


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


def _open_grid_text(path: Path, enc: str):
    """A text handle to a CSV file, decompressing ``.gz``/``.bgz`` so its layout is read, not its compressed bytes."""
    if path.suffix.lower() in COMPRESS_SUFFIXES:
        import gzip
        return gzip.open(path, "rt", encoding=enc, errors="replace", newline="")
    return open(path, encoding=enc, errors="replace", newline="")


@lru_cache(maxsize=8)
def _csv_grid(stamp: tuple, sep: str, encoding: str) -> Grid | None:
    path = Path(stamp[0])
    enc = "utf-8-sig" if encoding == "utf8" else "latin-1"      # a byte-order mark is not part of the first name
    try:
        if path.suffix.lower() in COMPRESS_SUFFIXES:
            return _csv_grid_compressed(path, sep, enc)
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


def _csv_grid_compressed(path: Path, sep: str, enc: str) -> Grid:
    """The grid of a compressed CSV: decompressed to the same top/bottom shape as a plain file (a gzip stream
    cannot be seeked to its tail, so the whole file is read, keeping only its head and a rolling tail)."""
    head: list[list[str]] = []
    tail: deque[list[str]] = deque(maxlen=TAIL_ROWS)
    total = 0
    with _open_grid_text(path, enc) as f:
        for r in csv.reader(f, delimiter=sep):
            total += 1
            if len(head) < GRID_ROWS:
                head.append(r)
            else:
                tail.append(r)
    complete = total <= GRID_ROWS
    if complete:
        width = max((len(r) for r in head), default=0)
        if total * max(width, 1) > GRID_CELLS:               # all of it is here, but too many cells to keep
            return Grid(_pad(head[:HEAD_ROWS], width), _pad(head[-TAIL_ROWS:], width), total, width, complete=False)
        return Grid(_pad(head, width), [], total, width, complete=True)
    width = max((len(r) for r in head + list(tail)), default=0)
    return Grid(_pad(head, width), _pad(list(tail), width), 0, width, complete=False)
