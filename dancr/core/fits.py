"""Curve fitting shared by the Fit step, the chart overlay and Predict.

Every fit uses every row of the prepared data (finite x and y; x > 0 where the shape needs it, which is
reported). The closed-form shapes come from running sums over the data in streaming passes; the three
nonlinear shapes start from a closed-form estimate and are refined with Gauss-Newton passes, each a
streaming aggregation. R² and RMSE are then computed exactly over all rows.

Kinds (chosen by name, each a familiar shape):
  linear      y = a·x + b
  saturating  y = a·x / (b + x)          (levels off)
  exponential y = a·exp(b·x)
  power       y = a·x^b                   (straight on log-log axes)
  logarithmic y = a·ln(x) + b
  polynomial  y = c0 + c1·z + … + cn·zⁿ  with z = (x − centre) / scale, so the numbers stay well
              conditioned whatever the size of x (timestamps, serial numbers). params = [centre, scale, c0…cn].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import polars as pl

KINDS = [
    ("linear", "Straight line"), ("saturating", "Levels off (saturating)"), ("exponential", "Exponential"),
    ("power", "Power law (straight on log-log)"), ("logarithmic", "Logarithmic"), ("polynomial", "Curve (polynomial)"),
]
STREAM = "streaming"
GN_ITERATIONS = 25
POSITIVE_X = ("power", "logarithmic")


@dataclass
class Fit:
    kind: str
    params: list[float]
    r2: float | None
    rmse: float | None
    n: int
    x: str
    y: str
    group: str | None = None        # group value as text (the group column is compared as text too)
    equation: str = ""
    x_min: float | None = None
    x_max: float | None = None
    iterations: int = 0             # Gauss-Newton passes used (0 for closed-form shapes)
    converged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fit":
        fields = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in fields if k in d})


# ---------------------------------------------------------------- model expressions
def predict_expr(kind: str, params: list[float], x: pl.Expr) -> pl.Expr:
    """Predicted y for a column of x. Shapes that need x > 0 give null elsewhere."""
    x = x.cast(pl.Float64)
    if kind == "linear":
        return params[0] * x + params[1]
    if kind == "polynomial":
        centre, scale, coef = params[0], params[1], params[2:]
        z = (x - centre) / scale
        e: pl.Expr = pl.lit(coef[-1])
        for c in reversed(coef[:-1]):
            e = e * z + c
        return e
    if kind == "logarithmic":
        return pl.when(x > 0).then(params[0] * x.log() + params[1])
    if kind == "power":
        return pl.when(x > 0).then(params[0] * x.pow(params[1]))
    if kind == "exponential":
        return params[0] * (params[1] * x).exp()
    if kind == "saturating":
        return params[0] * x / (params[1] + x)
    raise ValueError(f"Unknown fit kind {kind!r}")


def predict_arrays(kind: str, params: list[float], xs: np.ndarray) -> np.ndarray:
    """Same models on numpy arrays (chart overlays). Shapes that need x > 0 give NaN elsewhere."""
    xs = np.asarray(xs, dtype=float)
    if kind == "linear":
        return params[0] * xs + params[1]
    if kind == "polynomial":
        z = (xs - params[0]) / params[1]
        return np.polyval(list(reversed(params[2:])), z)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "logarithmic":
            return np.where(xs > 0, params[0] * np.log(np.where(xs > 0, xs, 1.0)) + params[1], np.nan)
        if kind == "power":
            return np.where(xs > 0, params[0] * np.power(np.where(xs > 0, xs, 1.0), params[1]), np.nan)
    if kind == "exponential":
        return params[0] * np.exp(params[1] * xs)
    if kind == "saturating":
        return params[0] * xs / (params[1] + xs)
    raise ValueError(f"Unknown fit kind {kind!r}")


def equation(kind: str, params: list[float], x: str = "x", y: str = "y") -> str:
    g = lambda v: f"{v:.4g}"
    p = params
    if kind == "linear":
        return f"{y} = {g(p[0])}·{x} {'+' if p[1] >= 0 else '−'} {g(abs(p[1]))}"
    if kind == "saturating":
        return f"{y} = {g(p[0])}·{x} / ({g(p[1])} + {x})"
    if kind == "exponential":
        return f"{y} = {g(p[0])}·e^({g(p[1])}·{x})"
    if kind == "power":
        return f"{y} = {g(p[0])}·{x}^{g(p[1])}"
    if kind == "logarithmic":
        return f"{y} = {g(p[0])}·ln({x}) {'+' if p[1] >= 0 else '−'} {g(abs(p[1]))}"
    if kind == "polynomial":
        centre, scale, coef = p[0], p[1], p[2:]
        terms = [g(coef[0])] + [f"{g(c)}·z" if i == 1 else f"{g(c)}·z^{i}" for i, c in enumerate(coef) if i >= 1]
        return f"{y} = " + " + ".join(terms) + f"  where z = ({x} {'−' if centre >= 0 else '+'} {g(abs(centre))}) / {g(scale)}"
    return kind


def predict_by_group(fits: list[Fit], x: str, group: str | None) -> pl.Expr:
    """One prediction expression for a table: per group when the fits were made per group."""
    if not group:
        return predict_expr(fits[0].kind, fits[0].params, pl.col(x))
    e: pl.Expr = pl.lit(None).cast(pl.Float64)
    key = pl.col(group).cast(pl.Utf8)
    for f in fits:
        e = pl.when(key == pl.lit(f.group)).then(predict_expr(f.kind, f.params, pl.col(x))).otherwise(e)
    return e


def outside_range_by_group(fits: list[Fit], x: str, group: str | None) -> pl.Expr:
    """True where x is outside the range its fit was made on (per group when fitted per group)."""
    xe = pl.col(x).cast(pl.Float64)
    if not group:
        f = fits[0]
        return (xe < f.x_min) | (xe > f.x_max)
    e: pl.Expr = pl.lit(False)
    key = pl.col(group).cast(pl.Utf8)
    for f in fits:
        e = pl.when(key == pl.lit(f.group)).then((xe < f.x_min) | (xe > f.x_max)).otherwise(e)
    return e


# ---------------------------------------------------------------- streaming passes
X, Y = "__x", "__y"


def _row(lf: pl.LazyFrame, exprs: list[pl.Expr]) -> dict[str, Any]:
    return lf.select(exprs).collect(engine=STREAM).row(0, named=True)


def _linear_sums(lf: pl.LazyFrame, xe: pl.Expr, ye: pl.Expr) -> tuple[int, float, float, float, float, float]:
    """n, x̄, ȳ, Sxx, Sxy, Syy of the given x/y expressions (centred on the means: two streaming passes)."""
    m = _row(lf, [pl.len().alias("n"), xe.mean().alias("mx"), ye.mean().alias("my")])
    n, mx, my = int(m["n"]), m["mx"], m["my"]
    if n < 2 or mx is None or my is None:
        raise ValueError("Need at least two points to fit")
    dx, dy = xe - mx, ye - my
    s = _row(lf, [(dx * dx).sum().alias("sxx"), (dx * dy).sum().alias("sxy"), (dy * dy).sum().alias("syy")])
    return n, float(mx), float(my), float(s["sxx"]), float(s["sxy"]), float(s["syy"])


def _fit_line(lf: pl.LazyFrame, xe: pl.Expr, ye: pl.Expr) -> tuple[float, float]:
    """Least-squares slope and intercept of ye against xe over every row."""
    n, mx, my, sxx, sxy, _ = _linear_sums(lf, xe, ye)
    if sxx <= 0:
        raise ValueError("All x values are the same, so no line can be fitted")
    a = sxy / sxx
    return a, my - a * mx


def _fit_polynomial(lf: pl.LazyFrame, degree: int) -> list[float]:
    """[centre, scale, c0..cn] with the curve expressed in z = (x - centre) / scale."""
    deg = max(1, min(int(degree), 6))
    m = _row(lf, [pl.len().alias("n"), pl.col(X).n_unique().alias("distinct"), pl.col(X).mean().alias("mx"), pl.col(X).std(ddof=0).alias("sx")])
    n, distinct, mx, sx = int(m["n"]), int(m["distinct"]), m["mx"], m["sx"]
    if n < 2:
        raise ValueError("Need at least two points to fit")
    if not sx:
        raise ValueError("All x values are the same, so no curve can be fitted")
    if distinct <= deg:
        raise ValueError(f"A degree-{deg} curve needs more than {deg} distinct x values (this data has {distinct})")
    z = (pl.col(X) - mx) / sx
    exprs = [z.pow(k).sum().alias(f"z{k}") for k in range(2 * deg + 1)] + [(z.pow(k) * pl.col(Y)).sum().alias(f"zy{k}") for k in range(deg + 1)]
    s = _row(lf, exprs)
    A = np.array([[s[f"z{i + j}"] for j in range(deg + 1)] for i in range(deg + 1)], dtype=float)
    b = np.array([s[f"zy{i}"] for i in range(deg + 1)], dtype=float)
    try:
        cz = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError(f"A degree-{deg} curve cannot be fitted to this data (the x values do not spread enough). Try fewer bends.") from None
    return [float(mx), float(sx), *(float(c) for c in cz)]


def _gauss_newton(lf: pl.LazyFrame, kind: str, params: list[float]) -> tuple[list[float], int, bool]:
    """Refine a two-parameter nonlinear fit on every row (Levenberg damping, streaming sums each pass).
    Returns (params, passes used, converged)."""
    a, b = params
    lam = 1e-3
    x, y = pl.col(X), pl.col(Y)

    def parts(a: float, b: float) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """model, d/da, d/db"""
        if kind == "exponential":
            g = (b * x).exp()
            return a * g, g, a * x * g
        if kind == "power":
            g = x.pow(b)
            return a * g, g, a * g * x.log()
        return a * x / (b + x), x / (b + x), -a * x / (b + x).pow(2)

    def sums(a: float, b: float) -> dict[str, float]:
        f, j1, j2 = parts(a, b)
        r = y - f
        return _row(lf, [(r * r).sum().alias("ss"), (j1 * j1).sum().alias("s11"), (j1 * j2).sum().alias("s12"),
                         (j2 * j2).sum().alias("s22"), (j1 * r).sum().alias("g1"), (j2 * r).sum().alias("g2")])

    cur = sums(a, b)
    if cur["ss"] is None or not math.isfinite(cur["ss"]):
        raise ValueError("Could not fit this shape: the values overflow (try a straight line or a power law)")
    converged = False
    passes = 0
    for passes in range(1, GN_ITERATIONS + 1):
        s11, s12, s22, g1, g2 = cur["s11"], cur["s12"], cur["s22"], cur["g1"], cur["g2"]
        try:
            da, db = np.linalg.solve(np.array([[s11 * (1 + lam), s12], [s12, s22 * (1 + lam)]]), np.array([g1, g2]))
        except np.linalg.LinAlgError:
            break
        na, nb = a + float(da), b + float(db)
        if kind == "saturating":
            na, nb = max(na, 0.0), max(nb, 1e-12)
        nxt = sums(na, nb)
        if nxt["ss"] is not None and math.isfinite(nxt["ss"]) and nxt["ss"] < cur["ss"]:
            negligible = (cur["ss"] - nxt["ss"]) <= 1e-12 * max(cur["ss"], 1e-300)
            a, b, cur, lam = na, nb, nxt, max(lam / 3, 1e-9)
            if negligible:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e8:
                converged = True        # no step improves the fit any more: we are at the minimum
                break
    return [a, b], passes, converged


def _score(lf: pl.LazyFrame, kind: str, params: list[float]) -> tuple[float | None, float, int]:
    """Exact R² and RMSE over every row."""
    pred = predict_expr(kind, params, pl.col(X))
    r = pl.col(Y) - pred
    s = _row(lf, [pl.len().alias("n"), (r * r).sum().alias("ss_res"), ((pl.col(Y) - pl.col(Y).mean()).pow(2)).sum().alias("ss_tot")])
    n, ss_res, ss_tot = int(s["n"]), float(s["ss_res"] or 0.0), float(s["ss_tot"] or 0.0)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return r2, math.sqrt(ss_res / n) if n else 0.0, n


def fit_lazy(lf: pl.LazyFrame, kind: str, degree: int = 2) -> tuple[list[float], int, bool]:
    """Parameters of `kind` fitted to columns __x and __y of `lf`, using every row. Returns (params, passes, converged)."""
    x, y = pl.col(X), pl.col(Y)
    if kind == "linear":
        return list(_fit_line(lf, x, y)), 0, True
    if kind == "polynomial":
        return _fit_polynomial(lf, degree), 0, True
    if kind == "logarithmic":
        return list(_fit_line(lf, x.log(), y)), 0, True
    if kind == "power":
        pos = lf.filter(y > 0)
        try:
            b, la = _fit_line(pos, x.log(), y.log())
            start = [math.exp(la), b]
        except ValueError:                       # too few positive y for a log-log start
            start = [float(_row(lf, [y.mean().alias("m")])["m"] or 1.0), 1.0]
        return _gauss_newton(lf, kind, start)
    if kind == "exponential":
        pos = lf.filter(y > 0)
        try:
            b, la = _fit_line(pos, x, y.log())
            start = [math.exp(la), b]
        except ValueError:
            start = [float(_row(lf, [y.mean().alias("m")])["m"] or 1.0), 0.0]
        return _gauss_newton(lf, kind, start)
    if kind == "saturating":
        m = _row(lf, [y.max().alias("ymax"), x.mean().alias("xmean"), pl.len().alias("n")])
        if int(m["n"]) < 2:
            raise ValueError("Need at least two points to fit")
        return _gauss_newton(lf, kind, [float(m["ymax"] or 1.0) * 1.2, max(float(m["xmean"] or 1.0), 1e-9)])
    raise ValueError(f"Unknown fit kind {kind!r}")


def _prepared(lf: pl.LazyFrame, x: str, y: str, group: str | None) -> pl.LazyFrame:
    cols = [pl.col(x).cast(pl.Float64).alias(X), pl.col(y).cast(pl.Float64).alias(Y)] + ([pl.col(group).cast(pl.Utf8).alias(group)] if group else [])
    return lf.select(cols).filter(pl.col(X).is_finite() & pl.col(Y).is_finite())


def _require_positive_x(lf: pl.LazyFrame, kind: str) -> None:
    if kind in POSITIVE_X:
        k = int(_row(lf, [(pl.col(X) <= 0).sum().alias("k")])["k"])
        if k:
            shape = "logarithmic" if kind == "logarithmic" else "power-law"
            raise ValueError(f"A {shape} fit needs all x values above zero ({k:,} rows are 0 or below). Filter them out first.")


def fit_frame(lf: pl.LazyFrame, x: str, y: str, kind: str = "linear", degree: int = 2, group: str | None = None) -> list[Fit]:
    """Fit on every row; one Fit per group (or one overall). Group values are handled as text."""
    base = _prepared(lf, x, y, group)
    if group:
        keys = base.select(pl.col(group).drop_nulls().unique().sort()).collect(engine=STREAM)[group].to_list()
        parts = [(k, base.filter(pl.col(group) == k)) for k in keys]
    else:
        parts = [(None, base)]
    fits: list[Fit] = []
    errors: list[str] = []
    for key, part in parts:
        try:
            _require_positive_x(part, kind)
            params, passes, converged = fit_lazy(part, kind, degree)
        except ValueError as e:
            if group:
                errors.append(f"{key}: {e}")
                continue
            raise
        r2, rmse, n = _score(part, kind, params)
        ext = _row(part, [pl.col(X).min().alias("lo"), pl.col(X).max().alias("hi")])
        fits.append(Fit(kind, [float(p) for p in params], r2, rmse, n, x, y, key, equation(kind, params, x, y),
                        float(ext["lo"]), float(ext["hi"]), passes, converged))
    if not fits:
        raise ValueError("Nothing could be fitted (not enough valid points)" + (": " + "; ".join(errors[:3]) if errors else ""))
    return fits


def fit_arrays(kind: str, xs: np.ndarray, ys: np.ndarray, degree: int = 2) -> tuple[list[float], np.ndarray]:
    """Fit in-memory arrays (same engine); returns (params, predicted ys)."""
    lf = pl.DataFrame({X: np.asarray(xs, dtype=float), Y: np.asarray(ys, dtype=float)}).lazy()
    lf = lf.filter(pl.col(X).is_finite() & pl.col(Y).is_finite())
    _require_positive_x(lf, kind)
    params, _, _ = fit_lazy(lf, kind, degree)
    return params, predict_arrays(kind, params, np.asarray(xs, dtype=float))


def curve_points(fit: Fit, n: int = 200, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    lo = fit.x_min if lo is None else lo
    hi = fit.x_max if hi is None else hi
    if lo is None or hi is None or hi <= lo:
        return np.empty(0), np.empty(0)
    if fit.kind in POSITIVE_X and lo <= 0:
        lo = max(lo, 1e-9)
    xs = np.linspace(lo, hi, n)
    return xs, predict_arrays(fit.kind, fit.params, xs)
