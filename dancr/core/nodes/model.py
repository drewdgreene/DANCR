"""Analyse & model: fit a curve, predict from it, check against limits."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..fits import KINDS, fit_frame, predict_by_group, outside_range_by_group, Fit
from ._common import first_input, schema_of, require_column, number_param, column_title
from ..expr import NUM
from ..dtypes import resolve_number
from ..findings import finding, fmt_number, fmt_pct


# --------------------------------------------------------------- fit a curve
def _fit(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    x = require_column(schema, params.get("x"), "x column (the cause)", NUM)
    y = require_column(schema, params.get("y"), "y column (the effect)", NUM)
    kind = params.get("kind") or "linear"
    group = params.get("group") or None
    if group:
        group = require_column(schema, group, "group column")
    fits = fit_frame(lf, x, y, kind, int(number_param(params, "degree", 2, "The degree", whole=True, at_least=1)), group)
    pred_name = params.get("predicted_column") or f"{y}_fitted"
    resid_name = f"{y}_residual"
    for name in (pred_name, resid_name):
        if name in schema:
            raise ValueError(f"There is already a column called {name!r}. Choose another name for the fitted column")
    pred = predict_by_group(fits, x, group, bool(group) and schema[group].is_numeric())
    out = lf.with_columns([pred.alias(pred_name), (pl.col(y).cast(pl.Float64) - pred).alias(resid_name)])
    report: dict[str, Any] = {"kind": kind, "x": x, "y": y, "group": group, "fits": [f.to_dict() for f in fits]}
    msgs = []
    for f in fits:
        head = f"{f.group}: " if group else ""
        r2 = f" · R² {f.r2:.4f}" if f.r2 is not None else ""
        passes = f" · {f.iterations} passes" if f.iterations else ""
        msgs.append(f"{head}{f.equation}{r2} · typical error {f.rmse:.4g} · {f.n:,} points{passes}")
        if f.r2 is not None and f.r2 < 0:
            msgs.append(f"{head}this shape fits worse than a flat line through the average (R² below 0). Try another kind of curve")
        elif not f.converged:
            msgs.append(f"{head}the fit stopped after {f.iterations} passes without settling. Treat the equation as approximate")
    if not group:
        f = fits[0]
        report.update({"equation": f.equation, "r_squared": f.r2, "rmse": f.rmse, "points": f.n, "parameters": f.params})
        how = f"R² {f.r2:.3f}" if f.r2 is not None else "fit quality unknown"
        report["finding"] = finding("fit", f"{column_title(ctx, y)} tracks {column_title(ctx, x)}: {f.equation} "
                                            f"({how}, typical error {fmt_number(f.rmse)}, {f.n:,} points)",
                                    magnitude=f.r2, direction=("up" if f.params[0] >= 0 else "down"), exact=True)
    else:
        known = [f for f in fits if f.r2 is not None]
        best = max(known, key=lambda f: f.r2) if known else fits[0]
        report["finding"] = finding("fit", f"{column_title(ctx, y)} vs {column_title(ctx, x)}, fitted separately per "
                                            f"{group}: strongest for {best.group} ({best.equation}"
                                            + (f", R² {best.r2:.3f}" if best.r2 is not None else "") + ")",
                                    magnitude=best.r2, direction="flat", exact=True)
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="fit_curve", label="Fit a curve", category="Analyse & model", icon="≈",
    description="Find the relationship between two columns (straight line, levelling-off curve, exponential…). "
                "Adds fitted and residual columns and reports the equation and how well it fits.",
    apply=_fit,
    summary=lambda p: f"{p.get('y') or '?'} against {p.get('x') or '?'} · {dict(KINDS).get(p.get('kind') or 'linear', '')}" + (f" per {p['group']}" if p.get("group") else ""),
    params=[
        Param("x", "X (the cause, along the bottom)", "column", column_group="numeric", required=True),
        Param("y", "Y (the effect, up the side)", "column", column_group="numeric", required=True),
        Param("kind", "Shape of the relationship", "choice", default="linear", choices=KINDS),
        Param("degree", "Curve bends", "int", default=2, min=1, max=6, visible_when={"kind": "polynomial"}, help="2 = one bend, 3 = two bends"),
        Param("group", "Fit separately for each", "column", help="A category column; blank = one fit for everything"),
        Param("predicted_column", "Name for the fitted column", "text", default="", advanced=True),
    ],
))


# --------------------------------------------------------------- predict
def _has_fit(meta: dict | None) -> bool:
    return bool(((meta or {}).get("report") or {}).get("fits"))


def _predict(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    # the fit is whichever connected step carries one, whatever port it was dropped on
    metas = ctx.upstream_meta or {}
    frames = {port: (inputs.get(port) or [None])[0] for port in ("data", "model")}
    fit_port = next((port for port in ("model", "data") if any(_has_fit(m) for m in (metas.get(port) or []))), None)
    if fit_port is None:
        connected = [port for port in ("data", "model") if frames[port] is not None]
        if len(connected) == 2 and any((metas.get(port) or [{}])[0].get("node_type") == "fit_curve" for port in connected):
            raise ValueError("Run the 'Fit a curve' step first")
        raise ValueError("Connect a 'Fit a curve' step to this step (and the table with the values to predict for)")
    data_port = "data" if fit_port == "model" else "model"
    if frames[data_port] is None:
        raise ValueError("Connect the table with the values to predict for")
    data = [frames[data_port]]
    rep = next(m for m in metas[fit_port] if _has_fit(m))["report"]
    fits = [Fit.from_dict(f) for f in rep["fits"]]
    lf = data[0]
    schema = schema_of(lf)
    x = require_column(schema, params.get("x") or rep.get("x"), "x column", NUM)
    out_name = params.get("output") or f"predicted_{rep.get('y', 'y')}"
    group = None
    if rep.get("group") or len(fits) > 1:
        group = params.get("group") or rep.get("group") or None
        if not group or group not in schema:
            raise ValueError("The fit was made per group. Choose the matching group column here")
    if out_name in schema:
        raise ValueError(f"There is already a column called {out_name!r}. Choose another name for the predicted column")
    numeric = bool(group) and schema[group].is_numeric()
    out = lf.with_columns(predict_by_group(fits, x, group, numeric).alias(out_name))
    if group:
        msgs = [f"Predicted {out_name} from {x} per {group} using {len(fits)} fits"] + [f"{f.group}: {f.equation}" for f in fits]
        report = {"equations": {f.group: f.equation for f in fits}}
    else:
        msgs = [f"Predicted {out_name} from {x} using: {fits[0].equation}"]
        report = {"equation": fits[0].equation}
    beyond = int(out.select(outside_range_by_group(fits, x, group, numeric).fill_null(False).sum()).collect(engine="streaming")[0, 0])
    if beyond:
        rng = f" ({fits[0].x_min:.4g} to {fits[0].x_max:.4g})" if not group else ""
        msgs.append(f"Caution: {beyond:,} rows are outside the range the fit was made on{rng}")
    return NodeResult(out, messages=msgs, report=report)


registry.register(NodeType(
    key="predict", label="Predict from a fit", category="Analyse & model", icon="→",
    description="Use a fitted curve to predict values for new x values. Connect the table with the x values and the Fit step.",
    apply=_predict,
    inputs=[InputSpec("data", "Values to predict for"), InputSpec("model", "The fit")],
    route=lambda src_type, taken: ("model" if src_type == "fit_curve" else "data") if not taken.get("model" if src_type == "fit_curve" else "data") else None,
    summary=lambda p: f"predict {p.get('output') or 'y'}" + (f" from {p['x']}" if p.get("x") else ""),
    params=[
        Param("x", "X column in this table", "column", column_group="numeric", port="data", help="Blank = the same column name the fit used"),
        Param("output", "Name for the predicted column", "text", default=""),
        Param("group", "Group column (if the fit was per group)", "column", port="data"),
    ],
))


# --------------------------------------------------------------- check limits
def _limits(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    lf = first_input(inputs)
    schema = schema_of(lf)
    col = require_column(schema, params.get("column"), "column", NUM)
    lo = resolve_number(params.get("min"), ctx.inputs, "Minimum")
    hi = resolve_number(params.get("max"), ctx.inputs, "Maximum")
    if lo is None and hi is None:
        return NodeResult(lf, messages=["No limit yet. Enter a minimum, a maximum or both, as a number or the name of an input"])
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"The minimum ({lo:g}) is above the maximum ({hi:g}), so no value could pass")
    v = pl.col(col).cast(pl.Float64)
    has = v.is_not_null() & v.is_not_nan()          # a blank (or NaN) is not a measurement: it is not checked
    inside = pl.lit(True)
    if lo is not None:
        inside = inside & (v >= lo)
    if hi is not None:
        inside = inside & (v <= hi)
    ok = pl.when(has).then(inside)                  # true, false, or blank when there is no value
    flag = (params.get("flag_column") or "").strip() or f"{col}_ok"
    action = params.get("action") or "flag"
    if action == "flag" and flag in schema:
        raise ValueError(f"There is already a column called {flag!r}. Choose another name for the true/false column")
    counts = lf.select([pl.len().alias("n"), has.sum().alias("checked"), ok.fill_null(False).sum().alias("ok")]).collect(engine="streaming").row(0, named=True)
    n, checked, n_ok = int(counts["n"]), int(counts["checked"] or 0), int(counts["ok"] or 0)
    bad, blank = checked - n_ok, n - checked
    limit_txt = " and ".join(t for t in [f"at least {lo:g}" if lo is not None else "", f"at most {hi:g}" if hi is not None else ""] if t)
    # nothing to check (every value blank) is not a pass: there is no evidence either way
    verdict = "NOTHING CHECKED" if checked == 0 else ("PASS" if bad == 0 else "FAIL")
    report = {"limit": limit_txt, "rows": n, "checked": checked, "within": n_ok, "outside": bad, "blank": blank,
              "outside_percent": (100.0 * bad / checked) if checked else 0.0, "verdict": verdict}
    if checked:
        msgs = [f"{verdict}: {bad:,} of {checked:,} values ({100.0 * bad / checked:.2f}%) have {col} outside {limit_txt}"]
    else:
        msgs = [f"NOTHING CHECKED: {col} has no values{' (the table is empty)' if n == 0 else ''}, so nothing was compared with {limit_txt}"]
    if blank:
        msgs.append(f"{blank:,} rows have no {col} value and were not checked")
    if action == "remove":
        out = lf.filter(ok.fill_null(False))
    elif action == "keep_failing":
        out = lf.filter((~ok).fill_null(False))
    else:
        out = lf.with_columns(ok.alias(flag))
    if checked == 0:
        say = f"Nothing was checked: {column_title(ctx, col)} has no values to compare with {limit_txt}"
    elif bad == 0:
        say = f"All {checked:,} {column_title(ctx, col)} values are within {limit_txt}"
    else:
        say = f"{bad:,} of {checked:,} {column_title(ctx, col)} values ({fmt_pct(report['outside_percent'])}%) are outside {limit_txt}"
    report["finding"] = finding("limit", say, magnitude=report["outside_percent"], direction="up", exact=True)
    return NodeResult(out, report=report, messages=msgs)


registry.register(NodeType(
    key="check_limits", label="Check against limits", category="Analyse & model", icon="✓",
    description="Mark each row as within or outside a limit and report how many fail. Limits can be numbers or the names of inputs.",
    apply=_limits,
    summary=lambda p: f"{p.get('column') or '?'} " + " and ".join(t for t in [f"≥ {p['min']}" if p.get("min") not in (None, "") else "", f"≤ {p['max']}" if p.get("max") not in (None, "") else ""] if t),
    params=[
        Param("column", "Column to check", "column", column_group="numeric", required=True),
        Param("min", "Must be at least", "text", default="", placeholder="a number or an input name"),
        Param("max", "Must be at most", "text", default="", placeholder="a number or an input name"),
        Param("action", "Then", "choice", default="flag", choices=[
            ("flag", "Add a true/false column"), ("remove", "Keep only rows within the limits"), ("keep_failing", "Keep only rows outside the limits")]),
        Param("flag_column", "Name of the true/false column", "text", default="", advanced=True, visible_when={"action": "flag"}),
    ],
))
