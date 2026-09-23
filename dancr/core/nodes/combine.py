"""Combine two tables: match on a key, line up by nearest time, or side by side."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..timeutil import parse_duration
from ._common import schema_of, require_column, temporal_columns
from ..expr import TIME, NUM, STR, _kind_of_dtype
from ..dtypes import align_time_column, temp_name


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
    for a, b in zip(on, right_on):
        if ls[a] != rs[b]:
            ka, kb = _kind_of_dtype(ls[a]), _kind_of_dtype(rs[b])
            if ka == TIME and kb == TIME:
                right = right.with_columns(align_time_column(pl.col(b), rs[b], ls[a]).alias(b))
            elif ka != kb:
                raise ValueError(f"Cannot match {a!r} ({ka}) with {b!r} ({kb}). Use 'Change type' so both are the same kind.")
            else:
                # same kind, different storage (Int32 vs Float64, text vs category): compare in a common type, losing nothing
                common = pl.Float64 if ka == NUM else (pl.Utf8 if ka == STR else ls[a])
                left = left.with_columns(pl.col(a).cast(common).alias(a))
                right = right.with_columns(pl.col(b).cast(common).alias(b))
    how = {"inner": "inner", "left": "left", "outer": "full", "right": "right"}.get(params.get("how") or "left", "left")
    out = left.join(right, left_on=on, right_on=right_on, how=how, suffix=suffix, coalesce=True,
                    maintain_order="left" if how in ("left", "inner") else ("right" if how == "right" else "none"))
    return NodeResult(out, messages=msgs)


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
