"""Analyse & model: fit a curve, predict from it, check against limits."""
from __future__ import annotations

from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..fits import KINDS, fit_frame, predict_by_group, outside_range_by_group, Fit
from ._common import first_input, schema_of, require_column
from ..expr import NUM
from ..dtypes import number_from_text


def _resolve_number(text: Any, ctx: Ctx, what: str) -> float | None:
    """A number, or the name of an input that holds one; blank -> None."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    try:
        return number_from_text(text, what)
    except ValueError:
        pass
    s = str(text).strip()
    for k, v in (ctx.inputs or {}).items():
        if k.lower() == s.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    raise ValueError(f"{what}: {text!r} is not a number or the name of an input")


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
    fits = fit_frame(lf, x, y, kind, int(params.get("degree") or 2), group)
    pred_name = params.get("predicted_column") or f"{y}_fitted"
    resid_name = f"{y}_residual"
    for name in (pred_name, resid_name):
        if name in schema:
            raise ValueError(f"There is already a column called {name!r}; choose another name for the fitted column")
    pred = predict_by_group(fits, x, group)
    out = lf.with_columns([pred.alias(pred_name), (pl.col(y).cast(pl.Float64) - pred).alias(resid_name)])
    report: dict[str, Any] = {"kind": kind, "x": x, "y": y, "group": group, "fits": [f.to_dict() for f in fits]}
    msgs = []
    for f in fits:
        head = f"{f.group}: " if group else ""
        r2 = f" · R² {f.r2:.4f}" if f.r2 is not None else ""
        passes = f" · {f.iterations} passes" if f.iterations else ""
        msgs.append(f"{head}{f.equation}{r2} · typical error {f.rmse:.4g} · {f.n:,} points{passes}")
        if not f.converged:
            msgs.append(f"{head}the fit stopped after {f.iterations} passes without settling; treat the equation as approximate")
    if not group:
        f = fits[0]
        report.update({"equation": f.equation, "r_squared": f.r2, "rmse": f.rmse, "points": f.n, "parameters": f.params})
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
            raise ValueError("The fit was made per group; choose the matching group column here")
    if out_name in schema:
        raise ValueError(f"There is already a column called {out_name!r}; choose another name for the predicted column")
    out = lf.with_columns(predict_by_group(fits, x, group).alias(out_name))
    if group:
        msgs = [f"Predicted {out_name} from {x} per {group} using {len(fits)} fits"] + [f"{f.group}: {f.equation}" for f in fits]
        report = {"equations": {f.group: f.equation for f in fits}}
    else:
        msgs = [f"Predicted {out_name} from {x} using: {fits[0].equation}"]
        report = {"equation": fits[0].equation}
    beyond = int(out.select(outside_range_by_group(fits, x, group).fill_null(False).sum()).collect(engine="streaming")[0, 0])
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
    lo = _resolve_number(params.get("min"), ctx, "Minimum")
    hi = _resolve_number(params.get("max"), ctx, "Maximum")
    if lo is None and hi is None:
        return NodeResult(lf, messages=["No limit yet: enter a minimum, a maximum, or both (a number or the name of an input)"])
    v = pl.col(col).cast(pl.Float64)
    ok = pl.lit(True)
    if lo is not None:
        ok = ok & (v >= lo)
    if hi is not None:
        ok = ok & (v <= hi)
    ok = ok.fill_null(False)
    flag = (params.get("flag_column") or "").strip() or f"{col}_ok"
    counts = lf.select([pl.len().alias("n"), ok.sum().alias("ok")]).collect(engine="streaming").row(0, named=True)
    n, n_ok = int(counts["n"]), int(counts["ok"] or 0)
    bad = n - n_ok
    limit_txt = " and ".join(t for t in [f"at least {lo:g}" if lo is not None else "", f"at most {hi:g}" if hi is not None else ""] if t)
    verdict = "PASS" if bad == 0 else "FAIL"
    report = {"limit": limit_txt, "rows": n, "within": n_ok, "outside": bad,
              "outside_percent": (100.0 * bad / n) if n else 0.0, "verdict": verdict}
    msgs = [f"{verdict}: {bad:,} of {n:,} rows ({(100.0 * bad / n) if n else 0:.2f}%) have {col} outside {limit_txt}"]
    action = params.get("action") or "flag"
    if action == "remove":
        out = lf.filter(ok)
    elif action == "keep_failing":
        out = lf.filter(~ok)
    else:
        out = lf.with_columns(ok.alias(flag))
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
