from __future__ import annotations

import math
import re
from typing import Any

import polars as pl

from ...params import Param
from ...registry import NodeType, Ctx, NodeResult, registry
from ...findings import finding, fmt_number, fmt_pct
from ...expr import NUM, TIME, STR, kind_of_dtype
from ...dtypes import resolve_number, temp_name
from ...timeutil import parse_bucket
from .._common import first_input, schema_of, require_column, number_param
from ._shared import (
    STAT_CHOICES,
    ASSOC_ROWS,
    ASSOC_MAX_COLS,
    CATEGORY_CAP,
    _looks_like_key,
    _label,
    _stat,
    _value_expr,
    _stat_label,
    _direction,
    _verb,
    _every,
    _truncate,
    _latest_two,
    _period_label,
    _signed,
)

def _check_data(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    from ...dtypes import text_to_number_expr
    lf = first_input(inputs)
    schema = schema_of(lf)
    cols = [require_column(schema, c, "column") for c in (params.get("columns") or list(schema))]
    exprs: list[pl.Expr] = [pl.len().alias("__n")]
    for i, c in enumerate(cols):
        dt = schema[c]
        exprs.append(pl.col(c).null_count().alias(f"miss{i}"))
        exprs.append(pl.col(c).approx_n_unique().alias(f"u{i}"))
        if dt.is_numeric():
            exprs += [pl.col(c).min().alias(f"lo{i}"), pl.col(c).max().alias(f"hi{i}")]
        if kind_of_dtype(dt) == STR:
            exprs.append(text_to_number_expr(pl.col(c)).is_not_null().sum().alias(f"num{i}"))
            s = pl.col(c).cast(pl.Utf8)
            exprs.append((s != s.str.strip_chars()).sum().alias(f"ws{i}"))
    row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
    rows = int(row["__n"])
    out_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for i, c in enumerate(cols):
        dt = schema[c]
        miss = int(row[f"miss{i}"] or 0)
        distinct = min(int(row[f"u{i}"] or 0), rows - int(row[f"miss{i}"] or 0))    # an estimate: never above the values
        pct = (100.0 * miss / rows) if rows else 0.0
        issues: list[tuple[int, str]] = []
        if rows and miss == rows:
            issues.append((3, "empty: every cell is blank"))
        elif pct >= 50:
            issues.append((3, f"mostly blank ({fmt_pct(pct)} missing)"))
        elif pct >= 5:
            issues.append((2, f"{fmt_pct(pct)} missing"))
        if rows > 1 and distinct <= 1:
            issues.append((2, "one value only"))
        if dt.is_numeric():
            from ...geo import name_suggests_lat, name_suggests_lon
            lo, hi = row.get(f"lo{i}"), row.get(f"hi{i}")
            if lo is not None and name_suggests_lat(c) and (float(lo) < -90.0 or float(hi) > 90.0):
                issues.append((3, "latitude values outside −90…90: are latitude and longitude the wrong way round?"))
            elif lo is not None and name_suggests_lon(c) and (float(lo) < -180.0 or float(hi) > 180.0):
                issues.append((2, "longitude values outside −180…180 (0…360 written as 180…540?)"))
        if kind_of_dtype(dt) == STR:
            num_like = int(row[f"num{i}"] or 0)
            filled = max(rows - miss, 1)
            # mostly numbers, but not all: the few that aren't (n/a, a typo) kept the column as text
            if num_like >= max(3, 0.5 * filled) and num_like < filled:
                issues.append((2, f"numbers stored as text ({num_like:,} look like numbers)"))
            if int(row[f"ws{i}"] or 0) > 0:
                issues.append((1, f"{int(row[f'ws{i}']):,} values have spaces around them"))
        issues.sort(reverse=True)
        sev, issue = issues[0] if issues else (0, "")
        if sev >= 2:
            problems.append(f"{_label(ctx, c)} ({issue})")
        entry = {"column": c, "type": kind_of_dtype(dt), "missing": miss, "missing_percent": round(pct, 2),
                 "distinct": distinct, "issue": issue, "severity": sev}
        if dt.is_numeric():
            entry["minimum"], entry["maximum"] = row.get(f"lo{i}"), row.get(f"hi{i}")
        out_rows.append(entry)
    # duplicate values in a column that looks like a key. A repeated id is almost always a data-integrity slip,
    # and it breaks joins silently — a whole-row duplicate check never sees it.
    key_dups: list[dict[str, Any]] = []
    if rows and len(cols) <= 30 and rows <= 5_000_000:
        for c in cols:
            dt = schema[c]
            if not (dt.is_integer() or _looks_like_key(c)):
                continue
            try:
                # count distinct and blanks together (one pass): a blank is not a repeated value, so only the
                # non-blank rows count towards a repeat
                nonblank, uniq = lf.select([pl.col(c).is_not_null().sum().alias("nb"),
                                            pl.col(c).drop_nulls().n_unique().alias("u")]) \
                                   .collect(engine="streaming").row(0)
            except Exception:  # noqa: BLE001 - a bonus check never fails the step
                continue
            if int(uniq) * 2 <= int(nonblank):
                continue                 # mostly repeated values: a category, not a key with a slip
            d = int(nonblank) - int(uniq)
            if d > 0:
                plural = "s" if d != 1 else ""
                key_dups.append({"column": c, "duplicates": d})
                for e in out_rows:
                    if e["column"] == c and e["severity"] < 2:
                        e["issue"], e["severity"] = f"{d:,} repeated value{plural} (a key should be unique)", 2
                problems.append(f"{_label(ctx, c)} ({d:,} repeated value{plural})")
    # a calculated column that one or two rows break: almost always a typing slip, and worth more than any blank
    broken: list[str] = []
    try:
        from ...derived import find
        num = [c for c in cols if schema[c].is_numeric()]
        derived = find(lf.select(num).head(5_000).collect(engine="streaming"), num) if len(num) >= 2 else []
    except Exception:  # noqa: BLE001 - a bonus check never fails the step
        derived = []
    naming = [c for c in cols if (kind_of_dtype(schema[c]) == STR or schema[c].is_integer())][:2]
    for d in derived:
        if not d.breaks:
            continue
        b = d.breaks[0]
        names = [c for c in naming if c not in d.operands and c != d.target]
        said = ""
        if names:
            vals = lf.select(names).slice(b["row"], 1).collect(engine="streaming").row(0, named=True)
            said = " (" + ", ".join(f"{_label(ctx, c)} {v}" for c, v in vals.items() if v is not None) + ")"
        where = f"row {b['row'] + 1}{said}" + (f" and {len(d.breaks) - 1} more" if len(d.breaks) > 1 else "")
        text = (f"{d.target} is {d.words} on {d.holds} of {d.rows} rows; {where} says {fmt_number(b['value'], 6)}, where the rule "
                f"gives {fmt_number(b['expected'], 6)}")
        broken.append(f"{_label(ctx, d.target)}: {where} says {fmt_number(b['value'], 6)} where {d.words} gives "
                      f"{fmt_number(b['expected'], 6)}")
        for e in out_rows:
            if e["column"] == d.target:
                e["issue"], e["severity"] = text, 3
    out = pl.DataFrame(out_rows).sort(["severity", "missing_percent"], descending=[True, True])
    dup = None
    if rows and len(cols) <= 30 and rows <= 5_000_000:
        try:
            distinct = int(lf.select(pl.struct(cols).n_unique()).collect(engine="streaming")[0, 0])
            dup = rows - distinct
        except Exception:  # noqa: BLE001 - duplicate detection is a bonus, never a failure
            dup = None
    if broken:
        statement = "A calculated value looks mistyped. " + "; ".join(broken[:2]) + (
            f". Also {len(problems)} column{'s' if len(problems) != 1 else ''} with blanks or misread values" if problems else "")
    elif problems:
        statement = (f"{len(problems)} of {len(cols)} columns need a look: " + "; ".join(problems[:3])
                     + (f" and {len(problems) - 3} more" if len(problems) > 3 else ""))
    elif dup:
        statement = f"The values look clean, but {dup:,} of {rows:,} rows are exact duplicates"
    else:
        statement = f"The data looks clean: no blanks, duplicates or misread numbers ({len(cols)} columns, {rows:,} rows)"
    report = {"rows": rows, "columns": len(cols), "problems": len(problems), "duplicate_rows": dup,
              "duplicate_keys": key_dups,
              "calculated": [d.to_dict() for d in derived],
              "finding": finding("quality", statement, magnitude=len(problems) + 3 * len(broken), direction="flat",
                                 exact=True, duplicate_rows=dup)}
    return NodeResult(out.lazy(), report=report,
                      messages=[f"{rows:,} rows, {len(cols)} columns; {len(problems)} need a look"])


registry.register(NodeType(
    key="check_data", uses_labels=True, label="Check the data", category="Clean up", icon="⚑",
    description="One pass over every column: blanks, duplicates, values that are really numbers stored as text, "
                "columns with a single value, and calculated columns (D = PA − LA) with a row that breaks the rule. "
                "Says what is worth a look.",
    apply=_check_data,
    summary=lambda p: f"{len(p['columns'])} columns" if p.get("columns") else "every column",
    params=[Param("columns", "Columns", "columns", default=[], help="Empty = all")],
))


# =================================================================== forecast
