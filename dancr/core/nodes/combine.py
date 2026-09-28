"""Combine two tables: match on a key, line up by nearest time, or side by side."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..timeutil import parse_duration
from ._common import schema_of, require_column, temporal_columns
from ..expr import TIME, NUM, STR, kind_of_dtype
from ..dtypes import align_time_column, temp_name, is_date
from ..findings import finding, fmt_pct


def _combine(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    left_frames = inputs.get("left") or []
    right_frames = inputs.get("right") or []
    if not left_frames or not right_frames:
        raise ValueError("Connect one table to 'First table' and one to 'Second table'")
    left, right = left_frames[0], right_frames[0]
    ls, rs = schema_of(left), schema_of(right)
    method = params.get("method", "match")
    suffix = params.get("suffix") or "_2"
    msgs: list[str] = []

    if method == "side_by_side":
        # join on row number
        rowc = temp_name("row", {**ls, **rs})
        l = left.with_row_index(rowc)
        r = right.with_row_index(rowc)
        out = l.join(r, on=rowc, how="full", suffix=suffix, coalesce=True, maintain_order="left").sort(rowc).drop(rowc)
        return NodeResult(out, messages=["Rows paired by position"])

    if method == "nearest_time":
        lt = params.get("left_time") or next(iter(temporal_columns(ls)), None)
        rt = params.get("right_time") or next(iter(temporal_columns(rs)), None)
        lt = require_column(ls, lt, "time column of the first table", TIME)
        rt = require_column(rs, rt, "time column of the second table", TIME)
        tol = (params.get("tolerance") or "").strip()
        strategy = params.get("direction") or "nearest"
        # make the right time column match the left one (unit, zone, Date vs Datetime)
        if rs[rt] != ls[lt]:
            r = right.with_columns(align_time_column(pl.col(rt), rs[rt], ls[lt]).alias(rt))
            if isinstance(ls[lt], pl.Date):
                l = left.with_columns(pl.col(lt).cast(pl.Datetime("us")))
            else:
                l = left
        else:
            l, r = left, right
        l = l.filter(pl.col(lt).is_not_null()).sort(lt)           # rows without a time cannot be lined up
        r = r.filter(pl.col(rt).is_not_null()).sort(rt)
        # avoid clashing non-key column names (including a right column named like the left key)
        clash = [c for c in rs if c in ls and c != rt]
        if clash:
            r = r.rename({c: f"{c}{suffix}" for c in clash})
        if rt != lt:
            r = r.rename({rt: lt})
        kwargs: dict[str, Any] = {"on": lt, "strategy": strategy}
        if tol:
            kwargs["tolerance"] = parse_duration(tol)[0]
        out = l.join_asof(r, **kwargs)
        msgs.append(f"Matched each row of the first table to the {strategy} row of the second by {lt}" + (f" within {tol}" if tol else ""))
        return NodeResult(out, messages=msgs)

    # method == match
    on = params.get("on") or []
    right_on = params.get("right_on") or on
    if not on:
        common = [c for c in ls if c in rs]
        if not common:
            raise ValueError("Choose the column(s) to match on. The tables have no column names in common.")
        on = common
        right_on = common
        msgs.append(f"Matching on shared columns: {', '.join(common)}")
    if len(on) != len(right_on):
        raise ValueError("Pick the same number of key columns on both sides")
    on = [require_column(ls, c, "key column") for c in on]
    right_on = [require_column(rs, c, "key column of the second table") for c in right_on]
    how = {"inner": "inner", "left": "left", "outer": "full", "right": "right"}.get(params.get("how") or "left", "left")
    restore: dict[str, pl.DataType] = {}
    for a, b in zip(on, right_on):
        if ls[a] != rs[b]:
            ka, kb = kind_of_dtype(ls[a]), kind_of_dtype(rs[b])
            if ka == TIME and kb == TIME:
                # a date against a date/time: both become date/times (midnight), then the key keeps the first table's type
                target = pl.Datetime("us") if is_date(ls[a]) else ls[a]
                if target != ls[a]:
                    left = left.with_columns(align_time_column(pl.col(a), ls[a], target).alias(a))
                    if how in ("left", "inner"):
                        restore[a] = ls[a]
                right = right.with_columns(align_time_column(pl.col(b), rs[b], target).alias(b))
            elif ka != kb:
                raise ValueError(f"Cannot match {a!r} ({ka}) with {b!r} ({kb}). Use 'Change type' so both are the same kind.")
            else:
                # same kind, different storage (Int32 vs UInt64, text vs category): compare in a type that holds both
                common = _common_key_type(ls[a], rs[b]) if ka == NUM else (pl.Utf8 if ka == STR else ls[a])
                left = left.with_columns(pl.col(a).cast(common).alias(a))
                right = right.with_columns(pl.col(b).cast(common).alias(b))
                if how in ("left", "inner"):
                    restore[a] = ls[a]           # every key comes from the first table, so its own type holds them
    # text keys match ignoring case, as VLOOKUP does: the join uses lower-case copies and the key column keeps
    # the values as written (the first table's, or the second's for rows only it has)
    taken = {*ls, *rs}
    lkeys, rkeys, folded = list(on), list(right_on), []
    for i, (a, b) in enumerate(zip(on, right_on)):
        if kind_of_dtype(ls[a]) == STR:
            k, orig = temp_name(f"key{i}", taken), temp_name(f"orig{i}", taken)
            taken |= {k, orig}
            left = left.with_columns(pl.col(a).cast(pl.Utf8).str.to_lowercase().alias(k))
            right = right.with_columns(pl.col(b).cast(pl.Utf8).str.to_lowercase().alias(k)).rename({b: orig})
            lkeys[i] = rkeys[i] = k
            folded.append((a, k, orig))
    out = left.join(right, left_on=lkeys, right_on=rkeys, how=how, suffix=suffix, coalesce=True,
                    maintain_order="left" if how in ("left", "inner") else ("right" if how == "right" else "none"))
    if folded:
        if how in ("full", "right"):
            out = out.with_columns([pl.coalesce(pl.col(a), pl.col(orig).cast(out.collect_schema()[a])).alias(a) for a, _, orig in folded])
        out = out.drop([c for _, k, orig in folded for c in (k, orig)])
    if restore:
        out = out.with_columns([pl.col(c).cast(dt) for c, dt in restore.items()])
    report: dict[str, Any] = {}
    if not ctx.preview:
        # like VLOOKUP people expect one match per row; say so when a key repeats in the second table
        dup = int(right.select(pl.struct(rkeys).is_duplicated().sum()).collect(engine="streaming")[0, 0])
        if dup:
            msgs.append(f"{dup:,} rows of the second table share their key with another row, so the rows of the first "
                        "table with those keys appear once per match. Use 'Remove duplicates' on the second table to keep one.")
        # how many of the first table's keys were actually found in the second, so a silent join is never believed blindly
        try:
            lk = left.select(pl.struct(lkeys).hash().alias("__k")).drop_nulls().unique()
            rk = right.select(pl.struct(rkeys).hash().alias("__k")).drop_nulls().unique().with_columns(pl.lit(1).alias("hit"))
            row = lk.join(rk, on="__k", how="left").select([pl.len().alias("n"), pl.col("hit").sum().alias("found")]) \
                    .collect(engine="streaming").row(0, named=True)
            total, found = int(row["n"] or 0), int(row["found"] or 0)
            pct = (100.0 * found / total) if total else 0.0
            if total:
                said = (f"{fmt_pct(pct)} of the first table's keys were found in the second ({found:,} of {total:,})")
                if found < total:
                    said += f"; {total - found:,} rows matched nothing"
                report = {"matched_keys": found, "left_keys": total, "match_percent": pct,
                          "finding": finding("summary", said, magnitude=100.0 - pct, exact=True)}
                msgs.append(said)
        except Exception:  # noqa: BLE001 - the join itself is the result; the match rate is a bonus
            report = {}
    return NodeResult(out, report=report, messages=msgs)


def _common_key_type(a: pl.DataType, b: pl.DataType) -> pl.DataType:
    """A number type that holds every value of both key columns exactly, where one exists."""
    if a.is_integer() and b.is_integer():
        if a.is_signed_integer() == b.is_signed_integer():
            return pl.Int64 if a.is_signed_integer() else pl.UInt64
        return pl.Int128                     # signed with unsigned: Int128 holds all of Int64 and UInt64
    return pl.Float64                        # a decimal key on either side: compare as decimals


def _summary(p: dict[str, Any]) -> str:
    m = p.get("method", "match")
    if m == "match":
        return f"match on {', '.join(p.get('on') or ['shared columns'])} ({p.get('how', 'left')})"
    if m == "nearest_time":
        return "nearest time" + (f" within {p['tolerance']}" if p.get("tolerance") else "")
    return "side by side"


registry.register(NodeType(
    key="combine", label="Combine two tables", category="Combine", icon="⋈",
    description="Match rows from two tables (like VLOOKUP), line up two time series by nearest time, or put tables side by side.",
    apply=_combine,
    inputs=[InputSpec("left", "First table"), InputSpec("right", "Second table")],
    summary=_summary,
    params=[
        Param("method", "How to combine", "choice", default="match", choices=[
            ("match", "Match rows that have the same value (like VLOOKUP)"),
            ("nearest_time", "Line up by nearest time"),
            ("side_by_side", "Side by side, row by row")]),
        Param("on", "Match on (first table)", "columns", default=[], port="left", visible_when={"method": "match"},
              help="Leave empty to use every column both tables share"),
        Param("right_on", "Match on (second table)", "columns", default=[], port="right", visible_when={"method": "match"},
              help="Only needed if the key has a different name in the second table"),
        Param("how", "Which rows to keep", "choice", default="left", visible_when={"method": "match"}, choices=[
            ("left", "All rows of the first table"), ("inner", "Only rows found in both"),
            ("outer", "All rows of both tables"), ("right", "All rows of the second table")]),
        Param("left_time", "Time column (first table)", "column", column_group="temporal", port="left", visible_when={"method": "nearest_time"}),
        Param("right_time", "Time column (second table)", "column", column_group="temporal", port="right", visible_when={"method": "nearest_time"}),
        Param("direction", "Pick the row that is", "choice", default="nearest", visible_when={"method": "nearest_time"},
              choices=[("nearest", "nearest in time"), ("backward", "at or before"), ("forward", "at or after")]),
        Param("tolerance", "Only match rows within", "duration", default="", visible_when={"method": "nearest_time"},
              placeholder="e.g. 500ms, 2s, 1m (blank = no limit)"),
        Param("suffix", "Added to column names that exist in both tables", "text", default="_2", advanced=True),
    ],
))
