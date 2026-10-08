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
from ..geo import distance_m_expr, parse_distance, UNITS_M, format_distance, METRES_PER_DEGREE_LAT

MAX_FUZZY_PAIRS = 50_000_000      # nearest-key matching compares every key against every other: a hard work ceiling


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

    if method == "nearest_feature":
        return _nearest_feature(ctx, left, right, ls, rs, params, suffix)

    if method == "within":
        return _within(ctx, left, right, ls, rs, params, suffix)

    if method == "fuzzy":
        return _fuzzy(ctx, left, right, ls, rs, params, suffix)

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
                    maintain_order="left")     # deterministic row order for every join kind, run to run
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


def _nearest_feature(ctx: Ctx, left: pl.LazyFrame, right: pl.LazyFrame, ls: dict[str, pl.DataType],
                     rs: dict[str, pl.DataType], params: dict[str, Any], suffix: str) -> NodeResult:
    """For each row of the first table, the nearest row of the second by great-circle distance.

    The candidate pairs are found by a lazy non-equi join pruned to a latitude band, then the closest per
    row is kept with a stable tie-break, so the same inputs always give the same pairing. With ``how`` left
    every row is kept (its distance and feature blank when none is within range); with inner only rows that
    found a feature within the distance are kept.
    """
    llat = require_column(ls, params.get("left_lat"), "latitude column of the first table", NUM)
    llon = require_column(ls, params.get("left_lon"), "longitude column of the first table", NUM)
    rlat = require_column(rs, params.get("right_lat"), "latitude column of the second table", NUM)
    rlon = require_column(rs, params.get("right_lon"), "longitude column of the second table", NUM)
    dist_text = str(params.get("max_distance") or "").strip()
    max_m = parse_distance(dist_text)[1] if dist_text else None
    unit = params.get("units") or "km"
    if unit not in UNITS_M:
        raise ValueError(f"Unknown distance unit {unit!r}")
    factor = UNITS_M[unit]
    dist_col = (params.get("distance_column") or "").strip() or "distance"
    if dist_col in ls or dist_col in rs:
        raise ValueError(f"There is already a column called {dist_col!r}; give the distance column another name")
    how = "inner" if (params.get("near_how") or "left") == "inner" else "left"

    # right columns that clash with the first table are suffixed, so the join condition is unambiguous.
    # Each of rlat/rlon is suffixed on its own: a clash on one says nothing about the other.
    clash = [c for c in rs if c in ls]
    rl = f"{rlat}{suffix}" if rlat in clash else rlat
    ro = f"{rlon}{suffix}" if rlon in clash else rlon

    left_i = left.with_row_index("__lrow")
    right_i = right.rename({c: f"{c}{suffix}" for c in clash}).with_row_index("__rrow")

    # Candidate pairs are found with a latitude-band non-equi join (a plain comparison, which every Polars
    # engine compiles the same way). The great-circle distance is computed afterwards, as a column on the
    # joined frame — computing it inside the join predicate crashes the default collect engine and only works
    # under streaming, so it is kept out of the join.
    if max_m is not None:
        band = max_m / METRES_PER_DEGREE_LAT + 1e-6
        cond: pl.Expr = (pl.col(rl) - pl.col(llat)).abs() <= band
        pairs = left_i.join_where(right_i, cond)
    else:
        # no limit: guard against a cross join nobody meant (a big feature table with no distance)
        n_left = int(left.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        n_right = int(right.select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        if n_left * n_right > 50_000_000:
            raise ValueError(f"With no distance limit every row would be compared with every feature "
                             f"({n_left:,} × {n_right:,}). Set 'Only match within' to prune the search")
        pairs = left_i.join(right_i, how="cross")
    pairs = pairs.with_columns((distance_m_expr(llat, llon, rl, ro) / factor).alias(dist_col))
    if max_m is not None:
        pairs = pairs.filter(pl.col(dist_col) <= max_m / factor)
    nearest = pairs.sort([dist_col, "__rrow"]).unique(subset=["__lrow"], keep="first", maintain_order=True)
    # keep only the pairing, the distance and the second table's own columns: the row index it carries would
    # otherwise be rejoined onto the first table and appear a second time, suffixed by Polars
    right_cols = [f"{c}{suffix}" if c in clash else c for c in rs]
    nearest = nearest.select(["__lrow", "__rrow", dist_col, *right_cols])
    out = left_i.join(nearest, on="__lrow", how=how).sort("__lrow").drop("__lrow", "__rrow")
    left_cols = list(ls)
    extra = [c for c in out.collect_schema().names() if c not in left_cols]
    out = out.select(left_cols + extra)

    msgs = ["Matched each row to the nearest place by latitude/longitude"
            + (f" within {dist_text}" if dist_text else "")]
    report: dict[str, Any] = {}
    if not ctx.preview:
        stats = out.select([pl.len().alias("n"), pl.col(dist_col).is_not_null().sum().alias("found"),
                            pl.col(dist_col).median().alias("med")]).collect(engine="streaming").row(0, named=True)
        total = int(stats["n"] or 0)
        found = int(stats["found"] or 0)
        pct = (100.0 * found / total) if total else 0.0
        if total:
            said = (f"{fmt_pct(pct)} of rows found a place" + (f" within {dist_text}" if dist_text else "")
                    + (f" (median {format_distance(stats['med'], unit)})" if found else ""))
            report = {"matched": found, "rows": total, "match_percent": pct, "median_distance": stats["med"],
                      "finding": finding("summary", said, magnitude=100.0 - pct, exact=True)}
            msgs.append(said)
    return NodeResult(out, report=report, messages=msgs)


def _within(ctx: Ctx, left: pl.LazyFrame, right: pl.LazyFrame, ls: dict[str, pl.DataType],
            rs: dict[str, pl.DataType], params: dict[str, Any], suffix: str) -> NodeResult:
    """Which places each point of the first table falls inside: a polygon (a WKT 'geometry' column of the second
    table). Each row is matched to the first place that contains it, so it appears once; ``near_how`` decides
    whether rows that fall inside no place are kept (blank) or dropped."""
    from ..geo import parse_wkt_rings, point_in_polygon, ring_bounds
    llat = require_column(ls, params.get("left_lat"), "latitude column of the first table", NUM)
    llon = require_column(ls, params.get("left_lon"), "longitude column of the first table", NUM)
    geom_col = params.get("right_geometry") or ("geometry" if "geometry" in rs else None)
    if not geom_col or geom_col not in rs:
        raise ValueError("Choose the column of the second table that holds the polygons (WKT text, as a GeoJSON file loads)")
    how = params.get("near_how") or "left"
    places = right.collect(engine="streaming")            # places are usually few: read them to test containment
    name_col = next((c for c in places.columns if c != geom_col and places[c].dtype == pl.Utf8), None)
    rings_by_place = [parse_wkt_rings(g) for g in places[geom_col].to_list()]
    label = (params.get("place_column") or "inside").strip() or "inside"

    left_i = left.with_row_index("__lrow")
    pts = left_i.select(["__lrow", llat, llon]).collect(engine="streaming")
    # A point can only be inside a polygon whose bounding box holds it, so prune to those pairs with a lazy
    # join first; the exact (and costly) containment test then runs on the few survivors, in place order.
    boxes = []
    for pi, rings in enumerate(rings_by_place):
        b = ring_bounds(rings) if rings else None
        if b:
            boxes.append({"__place": pi, "lon0": b[0], "lon1": b[1], "lat0": b[2], "lat1": b[3]})
    hits: dict[int, int] = {}
    if boxes:
        cand = (pts.lazy()
                .join_where(pl.DataFrame(boxes).lazy(),
                            (pl.col(llon) >= pl.col("lon0")) & (pl.col(llon) <= pl.col("lon1"))
                            & (pl.col(llat) >= pl.col("lat0")) & (pl.col(llat) <= pl.col("lat1")))
                .select(["__lrow", "__place", llat, llon])
                .collect(engine="streaming")
                .sort(["__lrow", "__place"]))
        for r in cand.iter_rows(named=True):
            lr = int(r["__lrow"])
            if lr in hits:                              # the first place (lowest index) that contains it wins
                continue
            pi = int(r["__place"])
            if point_in_polygon(float(r[llon]), float(r[llat]), rings_by_place[pi]):
                hits[lr] = pi
    match = pl.DataFrame({"__lrow": list(hits.keys()), "__place": list(hits.values())},
                         schema={"__lrow": pl.UInt32, "__place": pl.Int64}).lazy()
    out = left_i.join(match, on="__lrow", how="inner" if how == "inner" else "left")
    if name_col:
        lookup = pl.DataFrame({"__place": list(range(len(places))), label: places[name_col].to_list()},
                              schema={"__place": pl.Int64, label: pl.Utf8}).lazy()
        out = out.join(lookup, on="__place", how="left").drop("__place")
    else:
        out = out.with_columns(pl.col("__place").is_not_null().alias(label)).drop("__place")
    out = out.sort("__lrow").drop("__lrow")

    report: dict[str, Any] = {}
    msgs = [f"Matched each point to the place it falls inside ({label})"]
    if not ctx.preview:
        total = int(pts.height)
        matched = len(hits)
        pct = (100.0 * matched / total) if total else 0.0
        said = f"{fmt_pct(pct)} of points fell inside a place ({matched:,} of {total:,} rows)"
        report = {"matched": matched, "rows": total, "match_percent": pct,
                  "finding": finding("summary", said, magnitude=100.0 - pct, exact=True)}
        msgs.append(said)
    return NodeResult(out, report=report, messages=msgs)


def _common_key_type(a: pl.DataType, b: pl.DataType) -> pl.DataType:
    """A number type that holds every value of both key columns exactly, where one exists."""
    if a.is_integer() and b.is_integer():
        if a.is_signed_integer() == b.is_signed_integer():
            return pl.Int64 if a.is_signed_integer() else pl.UInt64
        return pl.Int128                     # signed with unsigned: Int128 holds all of Int64 and UInt64
    return pl.Float64                        # a decimal key on either side: compare as decimals


def _norm_expr(col: str) -> pl.Expr:
    """A text key tidied for matching: lower case, accents folded as far as a plain replace goes, runs of
    anything that is not a letter or digit collapsed to one space, trimmed."""
    return (pl.col(col).cast(pl.Utf8).str.to_lowercase()
            .str.replace_all(r"[^0-9a-z]+", " ").str.strip_chars())


def _fuzzy(ctx: Ctx, left: pl.LazyFrame, right: pl.LazyFrame, ls: dict[str, pl.DataType],
           rs: dict[str, pl.DataType], params: dict[str, Any], suffix: str) -> NodeResult:
    """Match two text keys that are not spelled the same: after tidying (case, spaces, punctuation), or by the
    nearest key within a similarity threshold. Never guesses silently: a row that found nothing keeps a blank
    key and score, and the report says how many matched."""
    import difflib
    lk = require_column(ls, params.get("left_key"), "key column of the first table", STR)
    rk = require_column(rs, params.get("right_key"), "key column of the second table", STR)
    how = {"inner": "inner", "left": "left", "outer": "full", "right": "right"}.get(params.get("how") or "left", "left")
    taken = {*ls, *rs}
    score_col = (params.get("score_column") or "match_score").strip() or "match_score"
    if score_col in taken:
        score_col = temp_name("score", taken)
    algorithm = params.get("algorithm") or "normalized"
    ltmp, rtmp = temp_name("lkey", taken), temp_name("rkey", taken)
    l = left.with_columns(_norm_expr(lk).alias(ltmp))
    r = right.with_columns(_norm_expr(rk).alias(rtmp))
    msgs: list[str] = []

    if algorithm == "nearest":
        cap = int(params.get("max_candidates") or 100000)
        threshold = float(params.get("threshold") if params.get("threshold") is not None else 0.9)
        rkeys = r.select(pl.col(rtmp)).drop_nulls().unique().collect(engine="streaming").get_column(rtmp).to_list()
        if len(rkeys) > cap:
            raise ValueError(f"The second table has {len(rkeys):,} distinct keys; nearest matching needs at most "
                             f"{cap:,}. Raise 'Most candidate keys', or use 'after tidying' instead.")
        lkeys = l.select(pl.col(ltmp)).drop_nulls().unique().collect(engine="streaming").get_column(ltmp).to_list()
        if len(lkeys) > cap:
            raise ValueError(f"The first table has {len(lkeys):,} distinct keys; nearest matching needs at most "
                             f"{cap:,}. Raise 'Most candidate keys', or use 'after tidying' instead.")
        if len(lkeys) * len(rkeys) > MAX_FUZZY_PAIRS:
            # every key of one table is compared with every key of the other: bound the total work as well as each side
            raise ValueError(f"Nearest matching would compare {len(lkeys):,} × {len(rkeys):,} keys. "
                             "Use 'after tidying', match on a shared key, or index fewer distinct keys.")
        lookup: dict[str, tuple[str, float]] = {}
        for a in lkeys:
            best, bs = None, 0.0
            for b in rkeys:
                s = difflib.SequenceMatcher(None, str(a), str(b)).ratio()
                if s > bs:
                    bs, best = s, b
            if best is not None and bs >= threshold:
                lookup[str(a)] = (best, round(float(bs), 4))
        if not lookup:
            msgs.append(f"No key in the first table reached a similarity of {threshold:g}")
        if lookup:
            mapping = pl.DataFrame({ltmp: list(lookup), "__rmatch": [v[0] for v in lookup.values()],
                                    score_col: [v[1] for v in lookup.values()]})
        else:
            # no key matched: still join *something* with the right dtypes, or an all-Null mapping would make the
            # key types disagree and the join below fail. Every left row keeps a blank key and score, as intended.
            mapping = pl.DataFrame(schema={ltmp: pl.Utf8, "__rmatch": pl.Utf8, score_col: pl.Float64})
        l = l.join(mapping.lazy(), on=ltmp, how="left")
        total = len(lkeys)
        matched = len(lookup)
        out = l.join(r, left_on="__rmatch", right_on=rtmp, how=how, suffix=suffix)
        present = out.collect_schema().names()
        out = out.drop([c for c in (ltmp, "__rmatch", rtmp) if c in present])
        msgs.append(f"Matched {lk} to the nearest {rk} within a similarity of {threshold:g}")
    else:
        l = l.rename({ltmp: "__lkey"})
        r = r.rename({rtmp: "__lkey"})
        total = int(l.select(pl.col("__lkey").drop_nulls().n_unique()).collect(engine="streaming")[0, 0] or 0)
        rk_unique = r.select(pl.col("__lkey")).drop_nulls().unique()
        found = int(l.select(pl.col("__lkey")).drop_nulls().unique().join(rk_unique, on="__lkey", how="semi")
                    .select(pl.len()).collect(engine="streaming")[0, 0] or 0)
        matched = found
        out = l.join(r, on="__lkey", how=how, suffix=suffix, coalesce=True,
                     maintain_order="left")     # deterministic row order, as for the keyed join
        rk_out = f"{rk}{suffix}" if rk in ls else rk
        out = out.with_columns(pl.when(pl.col(rk_out).is_not_null()).then(1.0).otherwise(None).alias(score_col))
        out = out.drop("__lkey")
        msgs.append(f"Matched {lk} to {rk} after tidying case, spaces and punctuation")

    report: dict[str, Any] = {}
    if not ctx.preview and total:
        pct = 100.0 * matched / total
        said = (f"{fmt_pct(pct)} of the first table's keys matched ({matched:,} of {total:,})"
                + (f"; {total - matched:,} matched nothing" if matched < total else ""))
        report = {"matched_keys": matched, "left_keys": total, "match_percent": pct,
                  "finding": finding("summary", said, magnitude=100.0 - pct, exact=True)}
        msgs.append(said)
    # keep the first table's columns, then the second's
    left_cols = list(ls)
    extra = [c for c in out.collect_schema().names() if c not in left_cols]
    return NodeResult(out.select([*left_cols, *extra]), report=report, messages=msgs)


def _summary(p: dict[str, Any]) -> str:
    m = p.get("method", "match")
    if m == "match":
        return f"match on {', '.join(p.get('on') or ['shared columns'])} ({p.get('how', 'left')})"
    if m == "nearest_time":
        return "nearest time" + (f" within {p['tolerance']}" if p.get("tolerance") else "")
    if m == "nearest_feature":
        return "nearest place" + (f" within {p['max_distance']}" if p.get("max_distance") else "")
    if m == "within":
        return "which place each point is inside"
    if m == "fuzzy":
        return f"fuzzy match {p.get('left_key') or '?'} → {p.get('right_key') or '?'}" + (
            f" (nearest ≥ {p.get('threshold')})" if p.get("algorithm") == "nearest" else " (after tidying)")
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
            ("nearest_feature", "Line up by nearest place (latitude/longitude)"),
            ("within", "Find which place each point is inside (polygons)"),
            ("fuzzy", "Match text keys that are almost the same (fuzzy)"),
            ("side_by_side", "Side by side, row by row")]),
        Param("on", "Match on (first table)", "columns", default=[], port="left", visible_when={"method": "match"},
              help="Leave empty to use every column both tables share"),
        Param("right_on", "Match on (second table)", "columns", default=[], port="right", visible_when={"method": "match"},
              help="Only needed if the key has a different name in the second table"),
        Param("how", "Which rows to keep", "choice", default="left", visible_when={"method": ["match", "fuzzy"]}, choices=[
            ("left", "All rows of the first table"), ("inner", "Only rows found in both"),
            ("outer", "All rows of both tables"), ("right", "All rows of the second table")]),
        Param("near_how", "Which rows to keep", "choice", default="left", visible_when={"method": ["nearest_feature", "within"]}, choices=[
            ("left", "Every row (the place may be blank)"), ("inner", "Only rows with a matching place")]),
        Param("left_time", "Time column (first table)", "column", column_group="temporal", port="left", visible_when={"method": "nearest_time"}),
        Param("right_time", "Time column (second table)", "column", column_group="temporal", port="right", visible_when={"method": "nearest_time"}),
        Param("direction", "Pick the row that is", "choice", default="nearest", visible_when={"method": "nearest_time"},
              choices=[("nearest", "nearest in time"), ("backward", "at or before"), ("forward", "at or after")]),
        Param("tolerance", "Only match rows within", "duration", default="", visible_when={"method": "nearest_time"},
              placeholder="e.g. 500ms, 2s, 1m (blank = no limit)"),
        Param("left_lat", "Latitude (first table)", "column", column_group="numeric", port="left", visible_when={"method": "nearest_feature"}),
        Param("left_lon", "Longitude (first table)", "column", column_group="numeric", port="left", visible_when={"method": "nearest_feature"}),
        Param("right_lat", "Latitude (second table)", "column", column_group="numeric", port="right", visible_when={"method": "nearest_feature"}),
        Param("right_lon", "Longitude (second table)", "column", column_group="numeric", port="right", visible_when={"method": "nearest_feature"}),
        Param("max_distance", "Only match within", "text", default="10km", visible_when={"method": "nearest_feature"},
              placeholder="e.g. 10km, 25 miles (blank = any distance)"),
        Param("units", "Distance unit", "choice", default="km", visible_when={"method": "nearest_feature"},
              choices=[("km", "kilometres (km)"), ("m", "metres (m)"), ("mi", "miles (mi)")]),
        Param("distance_column", "Name for the distance column", "text", default="", advanced=True, visible_when={"method": "nearest_feature"}),
        Param("right_geometry", "Polygon column (second table)", "column", column_group="string", port="right", visible_when={"method": "within"},
              help="The WKT text column of polygons (a GeoJSON file loads one as 'geometry')"),
        Param("place_column", "Name for the place column", "text", default="inside", visible_when={"method": "within"},
              help="The column added naming the place each point falls inside"),
        Param("left_key", "Text key (first table)", "column", column_group="string", port="left", visible_when={"method": "fuzzy"}),
        Param("right_key", "Text key (second table)", "column", column_group="string", port="right", visible_when={"method": "fuzzy"}),
        Param("algorithm", "How to match", "choice", default="normalized", visible_when={"method": "fuzzy"}, choices=[
            ("normalized", "After tidying case, spaces and punctuation"),
            ("nearest", "Nearest key, up to a similarity")]),
        Param("threshold", "Minimum similarity (nearest)", "float", default=0.9, min=0.0, max=1.0, advanced=True,
              visible_when={"method": "fuzzy", "algorithm": "nearest"},
              help="1.0 means the keys must be identical after tidying; 0.8 allows a small difference"),
        Param("max_candidates", "Most candidate keys (nearest)", "int", default=100000, min=1, advanced=True,
              visible_when={"method": "fuzzy", "algorithm": "nearest"}),
        Param("score_column", "Name for the match score", "text", default="match_score", advanced=True, visible_when={"method": "fuzzy"}),
        Param("suffix", "Added to column names that exist in both tables", "text", default="_2", advanced=True),
    ],
))
