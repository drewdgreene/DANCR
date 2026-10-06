"""DANCR formula language: Excel-flavoured expressions compiled to Polars.

    Pressure_B - Pressure_A
    IF(Temp > 20, "warm", "cold")
    ROUND(SQRT(ABS(x)), 2)
    [Probe A] * 1.5 + `Probe B`
    ROLLING_MEAN(Pressure, 20)

Column names may be written bare (``Pressure``), in square brackets
(``[Pressure (psi)]``) or in backticks. Function names are case-insensitive.
``and`` / ``or`` / ``not`` work as keywords. ``^`` is power, ``&`` joins text.
No Python is ever evaluated: the formula is parsed into a tree and turned into
a :class:`polars.Expr`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from ..params import find_input


from ._parse import (  # noqa: E402 - the compiler dispatches on these node types
    Binary,
    Call,
    Col,
    Const,
    FormulaError,
    Num,
    Parser,
    Str,
    Unary,
)


# ----------------------------------------------------------------- compiler
NUM, STR, BOOL, TIME, DUR, ANY = "number", "text", "true/false", "date/time", "duration", "any"


def kind_of_dtype(dt: pl.DataType) -> str:
    if dt.is_numeric():
        return NUM
    if dt == pl.Boolean:
        return BOOL
    if dt in (pl.Utf8, pl.String, pl.Categorical) or isinstance(dt, (pl.Categorical, pl.Enum)):
        return STR
    if dt in (pl.Datetime, pl.Date) or isinstance(dt, (pl.Datetime, pl.Date)):
        return TIME
    if dt == pl.Duration or isinstance(dt, pl.Duration):
        return DUR
    return ANY


@dataclass
class Typed:
    expr: pl.Expr
    kind: str
    literal: Any = None     # python value when this is a plain literal
    dtype: Any = None       # polars dtype when known (columns)


class Compiler:
    def __init__(self, schema: dict[str, pl.DataType], inputs: dict[str, Any] | None = None,
                 *, strict_columns: bool = False, notes: list[str] | None = None) -> None:
        self.schema = schema
        self.lower = {k.lower(): k for k in schema}
        self.columns_used: set[str] = set()
        self.inputs = inputs or {}
        self.inputs_used: set[str] = set()
        self.strict_columns = strict_columns
        self.notes = notes if notes is not None else []     # things worth telling the person (a forgiving match)

    # -- columns ----------------------------------------------------------
    def resolve_column(self, name: str, pos: int) -> str:
        if name in self.schema:
            return name
        if name.lower() in self.lower:
            return self.lower[name.lower()]
        # forgiving: strip spaces/underscores. Announced in `notes`, or refused outright in strict mode, so a typo
        # never silently binds to the wrong (but plausible) column.
        squash = lambda s: re.sub(r"[\s_]+", "", s.lower())
        squashed = [c for c in self.schema if squash(c) == squash(name)]
        if len(squashed) == 1:
            real = squashed[0]
            if self.strict_columns:
                raise FormulaError(f"{name!r} is not the exact column name {real!r}. Write it exactly, or turn off "
                                   "'Column names must match exactly'", pos)
            self.notes.append(f"matched “{name}” to column “{real}” (spaces and underscores are ignored)")
            return real
        if len(squashed) > 1:
            raise FormulaError(f"{name!r} could mean {squashed[0]!r} or {squashed[1]!r}. Write the exact name in [brackets]", pos)
        close = [c for c in self.schema if name.lower()[:3] and name.lower()[:3] in c.lower()]
        hint = f" Did you mean {close[0]!r}?" if close else ""
        cols = ", ".join(list(self.schema)[:12]) + (" ..." if len(self.schema) > 12 else "")
        extra = f" Inputs: {', '.join(self.inputs)}." if self.inputs else ""
        raise FormulaError(f"There is no column or input called {name!r}.{hint} Columns: {cols}.{extra}", pos)

    # -- entry ------------------------------------------------------------
    def compile(self, node: Any) -> Typed:
        m = getattr(self, f"c_{type(node).__name__}")
        return m(node)

    def c_Num(self, n: Num) -> Typed:
        v = n.value if isinstance(n.value, int) else (int(n.value) if n.value.is_integer() and abs(n.value) < 2**53 else n.value)
        return Typed(pl.lit(v), NUM, v)

    def c_Str(self, n: Str) -> Typed:
        return Typed(pl.lit(n.value), STR, n.value)

    def c_Const(self, n: Const) -> Typed:
        if n.name == "true":
            return Typed(pl.lit(True), BOOL, True)
        if n.name == "false":
            return Typed(pl.lit(False), BOOL, False)
        return Typed(pl.lit(None), ANY, None)

    def c_Col(self, n: Col) -> Typed:
        found = None if n.name in self.schema or n.name.lower() in self.lower else find_input(self.inputs, n.name)
        if found is not None:                   # a column always wins over an input of the same name
            real, v = found
            self.inputs_used.add(real)
            if isinstance(v, bool):
                return Typed(pl.lit(v), BOOL, v)
            if isinstance(v, (int, float)):
                return Typed(pl.lit(v), NUM, v)
            return Typed(pl.lit(str(v)), STR, str(v))
        name = self.resolve_column(n.name, n.pos)
        self.columns_used.add(name)
        return Typed(pl.col(name), kind_of_dtype(self.schema[name]), dtype=self.schema[name])

    def c_Unary(self, n: Unary) -> Typed:
        v = self.compile(n.operand)
        if n.op == "-":
            if v.kind in (STR, TIME, BOOL):
                raise FormulaError(f"Cannot negate a {v.kind} value")
            if v.kind == NUM and isinstance(v.literal, (int, float)) and not isinstance(v.literal, bool):
                return Typed(pl.lit(-v.literal), NUM, -v.literal)    # still a plain number: ROUND(x, -2), LAG(x, -1)
            # 0 - x rather than -x: Polars cannot negate Int128 (a widened UInt64)
            return Typed(pl.lit(0, pl.Int64) - _wide(v) if v.kind == NUM else -v.expr, v.kind if v.kind == DUR else NUM)
        if n.op == "not":
            return Typed(~self.as_bool(v), BOOL)
        raise FormulaError(f"Unknown operator {n.op}")

    def as_bool(self, v: Typed) -> pl.Expr:
        if v.kind == BOOL or v.kind == ANY:
            return v.expr
        if v.kind == NUM:
            return v.expr != 0
        raise FormulaError(f"Expected a true/false value but got {v.kind}")

    def coerce_time_literal(self, a: Typed, b: Typed) -> tuple[Typed, Typed]:
        """If one side is date/time and the other a text literal, parse the literal to match the column."""
        from ..dtypes import datetime_literal, align_time_column
        if a.kind == TIME and b.kind == STR and isinstance(b.literal, str):
            b = Typed(datetime_literal(b.literal, a.dtype or pl.Datetime("us"), "date"), TIME, dtype=a.dtype)
            if a.dtype is not None and isinstance(a.dtype, pl.Date):
                a = Typed(a.expr.cast(pl.Datetime("us")), TIME, dtype=pl.Datetime("us"))
        elif b.kind == TIME and a.kind == STR and isinstance(a.literal, str):
            a = Typed(datetime_literal(a.literal, b.dtype or pl.Datetime("us"), "date"), TIME, dtype=b.dtype)
            if b.dtype is not None and isinstance(b.dtype, pl.Date):
                b = Typed(b.expr.cast(pl.Datetime("us")), TIME, dtype=pl.Datetime("us"))
        elif a.kind == TIME and b.kind == TIME and a.dtype is not None and b.dtype is not None and a.dtype != b.dtype:
            b = Typed(align_time_column(b.expr, b.dtype, a.dtype), TIME, dtype=a.dtype)
        if (a.kind == TIME and b.kind == NUM) or (a.kind == NUM and b.kind == TIME):
            raise FormulaError("Cannot compare a date/time with a plain number. Write the date as text, e.g. \"2024-06-01\"")
        if (a.kind == TIME and b.kind == BOOL) or (a.kind == BOOL and b.kind == TIME):
            raise FormulaError("Cannot compare a date/time with a true/false value")
        return a, b

    def coerce_duration_compare(self, a: Typed, b: Typed) -> tuple[Typed, Typed]:
        """A duration compares with another duration, with a text span ("1d"), or with a blank; anything else
        (a plain number, a date, true/false) is refused rather than failing later inside Polars. Both sides
        are taken to microseconds, so a span read here and a duration a column holds keep the same unit."""
        from ..timeutil import parse_duration
        other = b if a.kind == DUR else a
        if other.kind == ANY:
            return a, b                                 # a blank compares blank: leave the duration alone
        if other.kind == STR and isinstance(other.literal, str):
            try:
                _text, secs = parse_duration(other.literal)
            except ValueError as e:
                raise FormulaError(f"Cannot read {other.literal!r} as a time span: {e}") from None
            span = Typed(pl.lit(round(secs * 1e6), pl.Int64), NUM)
            if a.kind == DUR:
                b = span
            else:
                a = span
        elif other.kind != DUR:
            raise FormulaError("Cannot compare a duration with a plain number. Write the span as text, e.g. \"1d\"")
        if a.kind == DUR:
            a = Typed(a.expr.dt.total_microseconds().cast(pl.Float64), NUM)
        if b.kind == DUR:
            b = Typed(b.expr.dt.total_microseconds().cast(pl.Float64), NUM)
        return a, b

    def c_Binary(self, n: Binary) -> Typed:
        op = n.op
        if op in ("and", "or"):
            a, b = self.compile(n.left), self.compile(n.right)
            e = (self.as_bool(a) & self.as_bool(b)) if op == "and" else (self.as_bool(a) | self.as_bool(b))
            return Typed(e, BOOL)
        a, b = self.compile(n.left), self.compile(n.right)
        if op in ("=", "!=", "<", ">", "<=", ">="):
            a, b = self.coerce_time_literal(a, b)
            if DUR in (a.kind, b.kind):
                a, b = self.coerce_duration_compare(a, b)
            if a.kind == STR and b.kind == NUM or a.kind == NUM and b.kind == STR:
                # numbers kept as text (a column never converted): compare them as numbers, so "10" > 5 and
                # "1" = 1.0; text that is not a number gives a blank answer, which filters treat as false
                from ..dtypes import text_to_number_expr
                if a.kind == STR:
                    a = Typed(text_to_number_expr(a.expr), NUM)
                else:
                    b = Typed(text_to_number_expr(b.expr), NUM)
            kind = STR if STR in (a.kind, b.kind) else (NUM if NUM in (a.kind, b.kind) else a.kind)
            ea, eb = a.expr, b.expr
            if kind == NUM:                            # [flag] = 1: TRUE is 1, as in Excel's arithmetic
                ea = ea.cast(pl.Int8) if a.kind == BOOL else ea
                eb = eb.cast(pl.Int8) if b.kind == BOOL else eb
            return Typed(excel_compare(op, ea, eb, kind), BOOL)
        if op == "&":
            return Typed(pl.concat_str([_as_text(a), _as_text(b)]), STR)
        if op == "+":
            if a.kind == STR and b.kind == STR:
                return Typed(pl.concat_str([_as_text(a), _as_text(b)]), STR)
            if a.kind == TIME and b.kind == DUR or a.kind == DUR and b.kind == TIME:
                return Typed(a.expr + b.expr, TIME)
            return Typed(_wide(a) + _wide(b), self._numkind(a, b))
        if op == "-":
            if a.kind == TIME and b.kind == TIME:
                a, b = self.coerce_time_literal(a, b)
                return Typed(a.expr - b.expr, DUR)
            if a.kind == TIME and b.kind == DUR:
                return Typed(a.expr - b.expr, TIME)
            return Typed(_wide(a) - _wide(b), self._numkind(a, b))
        if op == "*":
            return Typed(_wide(a) * _wide(b), self._numkind(a, b))
        if op == "/":
            if a.kind == DUR and b.kind == DUR:          # a duration divided by a duration is a plain ratio
                den = b.expr.cast(pl.Float64)
                return Typed(pl.when(den == 0).then(None).otherwise(a.expr.cast(pl.Float64) / den), NUM)
            if a.kind == DUR and b.kind in (NUM, ANY):   # a duration scaled by a number stays a duration
                den = b.expr.cast(pl.Float64)
                return Typed(pl.when(den == 0).then(None).otherwise(a.expr / b.expr), DUR)
            if b.kind == DUR:
                raise FormulaError("Cannot divide a number by a duration")
            self._numkind(a, b)
            den = b.expr.cast(pl.Float64)
            # dividing by zero has no answer (Excel shows #DIV/0!): a blank, not an infinity that breaks charts and totals
            return Typed(pl.when(den == 0).then(None).otherwise(a.expr.cast(pl.Float64) / den), NUM)
        if op == "%":
            return _mod(a, b)
        if op == "^":
            self._numkind(a, b)
            return Typed(_finite(a.expr.cast(pl.Float64).pow(b.expr.cast(pl.Float64))), NUM)
        raise FormulaError(f"Unknown operator {op}")

    def _numkind(self, a: Typed, b: Typed) -> str:
        for t in (a, b):
            if t.kind in (STR, BOOL, TIME):
                raise FormulaError(f"Cannot do arithmetic on a {t.kind} value")
        return NUM if (a.kind in (NUM, ANY) and b.kind in (NUM, ANY)) else (DUR if DUR in (a.kind, b.kind) else NUM)

    def c_Call(self, n: Call) -> Typed:
        name = n.name.upper()
        spec = FUNCTIONS.get(name)
        if spec is None:
            close = [f for f in FUNCTIONS if f.startswith(name[:2])]
            hint = f" Similar: {', '.join(sorted(close)[:6])}." if close else ""
            raise FormulaError(f"Unknown function {name}.{hint}", n.pos)
        lo, hi, fn, _doc = spec
        if len(n.args) < lo or (hi is not None and len(n.args) > hi):
            want = f"{lo}" if hi == lo else (f"at least {lo}" if hi is None else f"{lo} to {hi}")
            raise FormulaError(f"{name} takes {want} argument(s), got {len(n.args)}", n.pos)
        args = [self.compile(a) for a in n.args]
        try:
            return fn(self, args)
        except FormulaError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise FormulaError(f"{name}: {e}", n.pos) from e


def excel_compare(op: str, a: pl.Expr, b: pl.Expr, kind: str, case_sensitive: bool = False) -> pl.Expr:
    """``a op b`` as Excel compares, one rule for formulas and filter rules. Text is compared ignoring case
    ("abc" = "ABC"), and for = and <> a blank cell is empty text, so a blank equals "" and differs from "a".
    A blank number or date differs from every value (<> is true) and matches no other comparison; NaN
    counts as blank. ``op`` is one of = != < > <= >=."""
    if kind == STR:
        a, b = a.cast(pl.Utf8), b.cast(pl.Utf8)
        if op in ("=", "!="):
            a, b = a.fill_null(""), b.fill_null("")
        if not case_sensitive:
            a, b = a.str.to_lowercase(), b.str.to_lowercase()
    elif kind == NUM:
        a, b = a.fill_nan(None), b.fill_nan(None)      # NaN is a blank: never "above" anything
    if op == "!=":
        return a.ne_missing(b)
    fn = {"=": pl.Expr.eq, "<": pl.Expr.lt, ">": pl.Expr.gt, "<=": pl.Expr.le, ">=": pl.Expr.ge}[op]
    return fn(a, b)


# ----------------------------------------------------------------- functions
def number_text(e: pl.Expr, integer: bool = False) -> pl.Expr:
    """A number as Excel shows it in text: 15 significant digits (0.1 + 0.2 is "0.3") and no ".0" on a
    whole number. Whole-number columns (``integer``) keep every digit. A blank stays blank."""
    if integer:
        return e.cast(pl.Utf8)
    x = e.cast(pl.Float64)
    x = pl.when(x.is_finite()).then(x)                     # NaN and infinity are Excel errors: blank here
    # 15 significant digits only where they are ordinary decimals: rounding to significant digits is itself
    # inexact at extreme sizes (1e300 would gain digits), and those print shortest-exact already
    f = pl.when((x.abs() >= 1e-15) & (x.abs() < 1e15)).then(x.round_sig_figs(15)).otherwise(x)
    whole = (f == f.round(0)) & (f.abs() < 1e15)
    return pl.when(whole).then(f.cast(pl.Int64, strict=False).cast(pl.Utf8)).otherwise(f.cast(pl.Utf8).str.strip_suffix(".0"))


def _to_text(t: Typed) -> pl.Expr:
    if t.kind == NUM:
        return number_text(t.expr, t.dtype is not None and t.dtype.is_integer())
    return t.expr.cast(pl.Utf8)


def _as_text(t: Typed) -> pl.Expr:
    """A value as text for joining, as Excel's & and CONCAT do: a blank is "" (not a blank result)."""
    return _to_text(t).fill_null("")


def _mod(a: Typed, b: Typed) -> Typed:
    """The remainder with the sign of the divisor, as Excel's MOD; MOD(x, 0) is blank (Excel: #DIV/0!)."""
    return Typed(pl.when(_num(b) == 0).then(None).otherwise(_num(a) % _num(b)), NUM)


def _wide(t: Typed) -> pl.Expr:
    """A number for + - * and negation, in at least 64 bits. Small and unsigned integers (Parquet files,
    MONTH(), LEN()) would otherwise wrap around: UInt32 1 - 3 is 4294967294, Int8 100 + 100 is -56.
    UInt64 becomes Int128, which holds every value exactly (ids and hashes stay exact). Int64, Int128,
    floats and literals are unchanged."""
    if t.kind != NUM or t.literal is not None:
        return t.expr
    dt = t.dtype
    if dt is None:                                  # a computed value: the Int64 supertype widens only what needs it
        return t.expr + pl.lit(0, pl.Int64)
    if dt == pl.UInt64:
        return t.expr.cast(pl.Int128)
    if dt.is_integer() and dt not in (pl.Int64, pl.Int128):
        return t.expr.cast(pl.Int64)
    return t.expr


def _nn(e: pl.Expr) -> pl.Expr:
    """NaN as a blank, whatever the number type (a whole-number column has none): the rule everywhere in DANCR,
    so one 0/0 upstream does not turn a column's total or average into NaN."""
    return pl.when(e.cast(pl.Float64).is_nan()).then(None).otherwise(e)


def _finite(e: pl.Expr) -> pl.Expr:
    """A result with no answer (SQRT(-1), LN(0), 0^-1) is a blank, as Excel's #NUM! is not a number either."""
    x = e.cast(pl.Float64)
    return pl.when(x.is_finite()).then(x)


def _num(t: Typed) -> pl.Expr:
    if t.kind in (STR, TIME):
        raise FormulaError(f"Expected a number but got a {t.kind} value")
    return t.expr.cast(pl.Float64) if t.kind != NUM else t.expr


def _int_lit(t: Typed, what: str) -> int:
    if t.literal is None or isinstance(t.literal, bool) or not isinstance(t.literal, (int, float)):
        raise FormulaError(f"{what} must be a plain number")
    if float(t.literal) != int(t.literal):
        raise FormulaError(f"{what} must be a whole number, not {t.literal}")
    return int(t.literal)


def _same_kind(args: list[Typed], what: str) -> str:
    """The one kind shared by the given values (blanks match anything); mixed kinds are an error."""
    kinds = {a.kind for a in args if a.kind != ANY}
    if len(kinds) > 1:
        names = " and ".join(sorted(kinds))
        raise FormulaError(f"{what} must give the same kind of value on every branch, not {names}")
    return next(iter(kinds), ANY)


def _horizontal_or_column(hfn: Callable, cfn: Callable) -> Callable[[Compiler, list[Typed]], Typed]:
    """With one argument: aggregate down the column (broadcast). With several: row-wise."""
    def f(c: Compiler, args: list[Typed]) -> Typed:
        if len(args) == 1:
            return Typed(cfn(_nn(_num(args[0]))), NUM)
        return Typed(hfn([_nn(_num(a)) for a in args]), NUM)
    return f


def _simple(fn: Callable[[pl.Expr], pl.Expr], kind: str = NUM, cast: bool = True) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        a = args[0]
        return Typed(fn(_nn(_wide(a) if a.kind == NUM else _num(a)) if cast else a.expr), kind)   # ABS(Int8 -128) = 128
    return f


def _f_log(c: Compiler, args: list[Typed]) -> Typed:
    if len(args) == 2:
        base = args[1].literal
        if base is None:
            raise FormulaError("LOG base must be a plain number")
        return Typed(_finite(_num(args[0]).log(float(base))), NUM)
    return Typed(_finite(_num(args[0]).log10()), NUM)


def excel_round(e: pl.Expr, digits: int = 0) -> pl.Expr:
    """Round as Excel does: halves away from zero (2.5 -> 3, -2.5 -> -3), and a value that is a half in
    decimal but not quite in binary counts as a half (1.005 -> 1.01). ``digits`` may be negative (-2 rounds
    to hundreds)."""
    try:
        scale = 10.0 ** digits
    except OverflowError:
        scale = float("inf")
    x = e.cast(pl.Float64)
    # digits so large that no fractional place is left: the value is unchanged, as Excel leaves it. digits so
    # negative that the scale underflows to zero: every value rounds to zero (never a NaN from 0 * inf).
    if scale == float("inf"):
        return x
    if scale == 0.0:
        return pl.lit(0.0)
    # 15 significant digits first, as Excel keeps: 1.005 * 100 is 100.49999999999 in binary, which is the
    # 100.5 the person typed, while 2.4999999999 stays below the half
    scaled = x * scale
    # at 1e15 and beyond a double has no fractional digits left to round: the value stays as it is
    small = scaled.abs() < 1e15
    r = pl.when(small).then(scaled.round_sig_figs(15).round(0, mode="half_away_from_zero") / scale).otherwise(x)
    return pl.when(r == 0).then(0.0).otherwise(r)          # no "-0"


def _f_round(c: Compiler, args: list[Typed]) -> Typed:
    d = _int_lit(args[1], "ROUND digits") if len(args) == 2 else 0
    return Typed(excel_round(_num(args[0]), d), NUM)


def _f_if(c: Compiler, args: list[Typed]) -> Typed:
    cond = c.as_bool(args[0])
    a, b = args[1], args[2]
    return Typed(pl.when(cond).then(a.expr).otherwise(b.expr), _same_kind([a, b], "IF"))


def _f_coalesce(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.coalesce([a.expr for a in args]), _same_kind(args, "COALESCE / IFNULL"))


def _empty(e: pl.Expr, kind: str) -> pl.Expr:
    """A true/false expression that is true where the value counts as empty: a blank, a NaN number, or text
    that is only spaces. One rule, shared by ISBLANK/ISNULL and COUNT so they never disagree."""
    if kind == STR:
        return e.is_null() | (e.cast(pl.Utf8).str.strip_chars() == "")
    if kind == NUM:
        return e.is_null() | e.cast(pl.Float64).is_nan()
    return e.is_null()


def _f_isnull(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(_empty(args[0].expr, args[0].kind), BOOL)


def _f_concat(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.concat_str([_as_text(a) for a in args]), STR)


def _count_lit(t: Typed, what: str) -> int:
    n = _int_lit(t, what)
    if n < 0:
        raise FormulaError(f"{what} cannot be negative")
    return n


def _f_left(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(_to_text(args[0]).str.slice(0, _count_lit(args[1], "LEFT length")), STR)


def _f_right(c: Compiler, args: list[Typed]) -> Typed:
    n = _count_lit(args[1], "RIGHT length")
    if n == 0:
        return Typed(pl.when(args[0].expr.is_null()).then(None).otherwise(pl.lit("")), STR)
    return Typed(_to_text(args[0]).str.slice(-n, n), STR)


def _f_mid(c: Compiler, args: list[Typed]) -> Typed:
    start = _int_lit(args[1], "MID start") - 1
    if start < 0:
        raise FormulaError("MID start must be 1 or more")
    n = _count_lit(args[2], "MID length")
    return Typed(_to_text(args[0]).str.slice(start, n), STR)


def _str_fn(method: str, kind: str = STR) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        e = _to_text(args[0])
        rest = [a.expr for a in args[1:]]
        return Typed(getattr(e.str, method)(*rest), kind)
    return f


def _f_replace(c: Compiler, args: list[Typed]) -> Typed:
    """Excel's REPLACE(text, start, num_chars, new): num_chars characters from position start become new."""
    for a, what in ((args[1], "REPLACE start"), (args[2], "REPLACE number of characters")):
        if a.kind not in (NUM, ANY):
            raise FormulaError(f"{what} must be a number")
    if args[1].literal is not None and _int_lit(args[1], "REPLACE start") < 1:
        raise FormulaError("REPLACE start must be 1 or more")
    if args[2].literal is not None:
        _count_lit(args[2], "REPLACE number of characters")
    s = _to_text(args[0])
    start = args[1].expr.cast(pl.Float64).floor().cast(pl.Int64)
    count = args[2].expr.cast(pl.Float64).floor().cast(pl.Int64)
    # from a column, a start below 1 or a negative count is Excel's #VALUE!: a blank cell here, never a
    # slice counted from the end
    ok = (start >= 1) & (count >= 0)
    first = pl.when(ok).then(start - 1).otherwise(0).cast(pl.UInt64)
    rest = pl.when(ok).then(start - 1 + count).otherwise(0).cast(pl.UInt64)
    out = pl.concat_str([s.str.slice(0, first), _as_text(args[3]), s.str.slice(rest)])
    return Typed(pl.when(ok).then(out), STR)


def _f_substitute(c: Compiler, args: list[Typed]) -> Typed:
    """Excel's SUBSTITUTE(text, old, new[, instance]): every old becomes new, or only the instance-th one."""
    s, old, new = _to_text(args[0]), _as_text(args[1]), _as_text(args[2])
    if len(args) == 3:
        out = s.str.replace_all(old, new, literal=True)
    else:
        n = _int_lit(args[3], "SUBSTITUTE instance")
        if n < 1:
            raise FormulaError("SUBSTITUTE instance must be 1 or more")
        parts = s.str.split(old)
        out = (pl.when(parts.list.len() > n)
               .then(pl.concat_str([parts.list.slice(0, n).list.join(old), new, parts.list.slice(n).list.join(old)]))
               .otherwise(s))
    return Typed(pl.when(old == "").then(s).otherwise(out), STR)        # nothing to find: the text is unchanged


def _f_text(c: Compiler, args: list[Typed]) -> Typed:
    if len(args) == 2:
        if args[0].kind != TIME:
            raise FormulaError("TEXT with a format needs a date/time value, e.g. TEXT(time, \"%Y-%m-%d\")")
        if not isinstance(args[1].literal, str):
            raise FormulaError("TEXT format must be plain text in quotes, e.g. \"%Y-%m-%d\"")
        return Typed(args[0].expr.dt.strftime(args[1].literal), STR)
    return Typed(_to_text(args[0]), STR)


def _f_value(c: Compiler, args: list[Typed]) -> Typed:
    from ..dtypes import text_to_number_expr
    e = text_to_number_expr(args[0].expr) if args[0].kind == STR else args[0].expr.cast(pl.Float64, strict=False)
    return Typed(e, NUM)


def _f_date(c: Compiler, args: list[Typed]) -> Typed:
    e = args[0].expr
    if args[0].kind == TIME:
        if len(args) == 2:
            raise FormulaError("DATE is already given a date/time here. A format is only needed when parsing text, e.g. DATE(\"01/02/2024\", \"%d/%m/%Y\")")
        return Typed(e, TIME)
    if len(args) == 2:
        if not isinstance(args[1].literal, str):
            raise FormulaError("DATE format must be plain text in quotes, e.g. \"%d/%m/%Y\"")
        return Typed(e.cast(pl.Utf8).str.to_datetime(args[1].literal, strict=False), TIME)
    if args[0].kind == NUM:
        raise FormulaError("DATE of a number needs a unit. Use DATE(TEXT(x)) or convert the column type first")
    # the formats DANCR reads everywhere, month first when a date reads both ways (01/02/2024 is 2 January),
    # tried in order for each value; times with a UTC offset need their format given
    from ..timeutil import DATE_FORMATS, DAY_FIRST, has_offset
    text = e.cast(pl.Utf8).str.strip_chars()
    fmts = []
    for f in DATE_FORMATS:
        if has_offset(f):
            continue
        if f in DAY_FIRST:                         # %d/%m/…: its month-first twin goes first
            fmts.append(DAY_FIRST[f])
        if f not in fmts:
            fmts.append(f)
    def plausible(f: str) -> pl.Expr:                # "01/02/24" read with %Y is the year 24: not a date anyone meant
        d = text.str.to_datetime(f, strict=False, time_unit="us")
        return pl.when(d.dt.year().is_between(1000, 9999)).then(d)
    return Typed(pl.coalesce([plausible(f) for f in fmts]), TIME)


def _dt_part(attr: str) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        if args[0].kind != TIME:
            raise FormulaError("This function needs a date/time column")
        return Typed(getattr(args[0].expr.dt, attr)(), NUM)
    return f


def _f_weekday(c: Compiler, args: list[Typed]) -> Typed:
    """Excel's WEEKDAY: return type 1 (default) Sunday = 1 … Saturday = 7, 2 Monday = 1 … Sunday = 7,
    3 Monday = 0 … Sunday = 6, 11-17 the week starting Monday … Sunday = 1."""
    if args[0].kind != TIME:
        raise FormulaError("WEEKDAY needs a date/time column")
    rt = _int_lit(args[1], "WEEKDAY return type") if len(args) == 2 else 1
    iso = args[0].expr.dt.weekday().cast(pl.Int64)          # Monday = 1 … Sunday = 7
    if rt == 3:
        return Typed(iso - 1, NUM)
    first = {1: 7, 2: 1}.get(rt, rt - 10 if 11 <= rt <= 17 else None)     # the day counted as 1
    if first is None:
        raise FormulaError(f"WEEKDAY return type must be 1, 2, 3 or 11 to 17, not {rt}")
    return Typed((iso - first) % 7 + 1, NUM)


def _f_elapsed(c: Compiler, args: list[Typed]) -> Typed:
    if args[0].kind != TIME:
        raise FormulaError("ELAPSED needs a date/time column")
    unit = str(args[1].literal).lower() if len(args) == 2 else "s"
    div = {"ms": 1e3, "s": 1e6, "sec": 1e6, "seconds": 1e6, "m": 60e6, "min": 60e6, "minutes": 60e6,
           "h": 3600e6, "hours": 3600e6, "d": 86400e6, "days": 86400e6}.get(unit)
    if div is None:
        raise FormulaError("ELAPSED unit must be one of ms, s, min, h, d")
    e = args[0].expr
    first = e.drop_nulls().first()                  # the first row with a time, as documented (not the earliest)
    return Typed((e - first).dt.total_microseconds().cast(pl.Float64) / div, NUM)


def _f_seconds_between(c: Compiler, args: list[Typed]) -> Typed:
    a, b = args
    if a.kind != TIME or b.kind != TIME:
        raise FormulaError("SECONDS_BETWEEN needs two date/time values")
    a, b = c.coerce_time_literal(a, b)
    return Typed((b.expr - a.expr).dt.total_microseconds().cast(pl.Float64) / 1e6, NUM)


def _f_row(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.int_range(1, pl.len() + 1), NUM)


def _shift(sign: int) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        n = _int_lit(args[1], "shift amount") if len(args) == 2 else 1
        return Typed(args[0].expr.shift(sign * n), args[0].kind)
    return f


def _f_diff(c: Compiler, args: list[Typed]) -> Typed:
    n = _int_lit(args[1], "DIFF lag") if len(args) == 2 else 1
    e = args[0].expr
    if args[0].kind == TIME:
        return Typed((e - e.shift(n)).dt.total_microseconds().cast(pl.Float64) / 1e6, NUM)
    return Typed(_num(args[0]).diff(n), NUM)


def _rolling(method: str) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        n = _int_lit(args[1], "window size")
        if n < 1:
            raise FormulaError("window size must be at least 1")
        center = True                         # centred, as the Smooth step and spike removal are by default
        if len(args) == 3:
            if not isinstance(args[2].literal, bool):
                raise FormulaError("The third argument says whether to centre the window: TRUE or FALSE")
            center = args[2].literal
        e = _nn(_wide(args[0]) if args[0].kind == NUM else _num(args[0]))
        return Typed(getattr(e, f"rolling_{method}")(window_size=n, min_samples=1, center=center), NUM)
    return f


def _f_zscore(c: Compiler, args: list[Typed]) -> Typed:
    e = _nn(_num(args[0]))
    sd = e.std()
    return Typed(pl.when(sd == 0).then(None).otherwise((e - e.mean()) / sd), NUM)


def _f_percentile(c: Compiler, args: list[Typed]) -> Typed:
    lit = args[1].literal
    if lit is None or isinstance(lit, bool) or not isinstance(lit, (int, float)):
        raise FormulaError("PERCENTILE needs a plain number between 0 and 1 (or 0 and 100)")
    q = float(lit)
    if q > 1:
        q = q / 100.0
    if not 0 <= q <= 1:
        raise FormulaError(f"PERCENTILE must be between 0 and 1 (or 0 and 100), not {lit}")
    return Typed(_num(args[0]).fill_nan(None).quantile(q, interpolation="linear"), NUM)     # as Excel's PERCENTILE.INC


def _f_clip(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(_num(args[0]).clip(_num(args[1]), _num(args[2])), NUM)


def _f_pct_change(c: Compiler, args: list[Typed]) -> Typed:
    n = _int_lit(args[1], "lag") if len(args) == 2 else 1
    return Typed(_finite(_nn(_num(args[0])).pct_change(n) * 100.0), NUM)


def _f_and(c: Compiler, args: list[Typed]) -> Typed:
    e = c.as_bool(args[0])
    for a in args[1:]:
        e = e & c.as_bool(a)
    return Typed(e, BOOL)


def _f_or(c: Compiler, args: list[Typed]) -> Typed:
    e = c.as_bool(args[0])
    for a in args[1:]:
        e = e | c.as_bool(a)
    return Typed(e, BOOL)


def _f_fill_forward(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(args[0].expr.fill_null(strategy="forward"), args[0].kind)


def _f_interpolate(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(_num(args[0]).interpolate(), NUM)


def _f_rank(c: Compiler, args: list[Typed]) -> Typed:
    """As Excel's RANK: the largest value is 1, unless the order is 1 (then the smallest is 1). Ties share a rank."""
    order = _int_lit(args[1], "RANK order") if len(args) > 1 else 0
    e = _nn(args[0].expr) if args[0].kind == NUM else args[0].expr     # NaN is a blank: unranked, not the largest
    return Typed(e.rank(method="min", descending=order == 0), NUM)


def _f_count(c: Compiler, args: list[Typed]) -> Typed:
    """Number of non-empty values: a blank, a NaN or whitespace-only text does not count, matching ISBLANK."""
    return Typed((~_empty(args[0].expr, args[0].kind)).sum(), NUM)


def _f_pow(c: Compiler, args: list[Typed]) -> Typed:
    """POW(x, y): exactly what x^y does, so an answer with no value (a negative base to a fractional power)
    is a blank, never a NaN."""
    c._numkind(args[0], args[1])
    return Typed(_finite(_num(args[0]).cast(pl.Float64).pow(_num(args[1]).cast(pl.Float64))), NUM)


def _f_pi(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.lit(math.pi), NUM, math.pi)


# name -> (min_args, max_args or None, builder, doc)
FUNCTIONS: dict[str, tuple[int, int | None, Callable[[Compiler, list[Typed]], Typed], str]] = {
    # math
    "ABS": (1, 1, _simple(lambda e: e.abs()), "Absolute value"),
    "SQRT": (1, 1, _simple(lambda e: _finite(e.sqrt())), "Square root"),
    "EXP": (1, 1, _simple(lambda e: _finite(e.exp())), "e to the power x"),
    "LN": (1, 1, _simple(lambda e: _finite(e.log())), "Natural log"),
    "LOG": (1, 2, _f_log, "LOG(x) base 10, or LOG(x, base)"),
    "LOG2": (1, 1, _simple(lambda e: e.log(2)), "Log base 2"),
    "POW": (2, 2, _f_pow, "POW(x, y) = x^y"),
    "ROUND": (1, 2, _f_round, "ROUND(x, digits)"),
    "FLOOR": (1, 1, _simple(lambda e: e.floor()), "Round down"),
    "CEIL": (1, 1, _simple(lambda e: e.ceil()), "Round up"),
    "CEILING": (1, 1, _simple(lambda e: e.ceil()), "Round up"),
    "SIGN": (1, 1, _simple(lambda e: e.sign()), "-1, 0 or 1"),
    "MOD": (2, 2, lambda c, a: _mod(a[0], a[1]), "MOD(a, b) remainder, with the sign of b; blank when b is 0"),
    "SIN": (1, 1, _simple(lambda e: e.sin()), "Sine (radians)"),
    "COS": (1, 1, _simple(lambda e: e.cos()), "Cosine (radians)"),
    "TAN": (1, 1, _simple(lambda e: e.tan()), "Tangent (radians)"),
    "ASIN": (1, 1, _simple(lambda e: e.arcsin()), "Inverse sine"),
    "ACOS": (1, 1, _simple(lambda e: e.arccos()), "Inverse cosine"),
    "ATAN": (1, 1, _simple(lambda e: e.arctan()), "Inverse tangent"),
    "ATAN2": (2, 2, lambda c, a: Typed(pl.arctan2(_num(a[0]), _num(a[1])), NUM), "ATAN2(y, x)"),
    "PI": (0, 0, _f_pi, "3.14159..."),
    "CLIP": (3, 3, _f_clip, "CLIP(x, low, high) limits x to a range"),
    # aggregates / row-wise
    "SUM": (1, None, _horizontal_or_column(pl.sum_horizontal, lambda e: e.sum()), "SUM(col) column total; SUM(a, b, ...) row-wise"),
    "MIN": (1, None, _horizontal_or_column(pl.min_horizontal, lambda e: e.min()), "MIN(col) or MIN(a, b, ...)"),
    "MAX": (1, None, _horizontal_or_column(pl.max_horizontal, lambda e: e.max()), "MAX(col) or MAX(a, b, ...)"),
    "AVERAGE": (1, None, _horizontal_or_column(pl.mean_horizontal, lambda e: e.mean()), "AVERAGE(col) or AVERAGE(a, b, ...)"),
    "MEAN": (1, None, _horizontal_or_column(pl.mean_horizontal, lambda e: e.mean()), "Same as AVERAGE"),
    "MEDIAN": (1, 1, _simple(lambda e: e.fill_nan(None).median()), "Column median"),
    "STDEV": (1, 1, _simple(lambda e: e.std()), "Column standard deviation"),
    "STD": (1, 1, _simple(lambda e: e.std()), "Column standard deviation"),
    "VAR": (1, 1, _simple(lambda e: e.var()), "Column variance"),
    "COUNT": (1, 1, _f_count, "Number of non-empty values"),
    "PERCENTILE": (2, 2, _f_percentile, "PERCENTILE(col, 0.95)"),
    "ZSCORE": (1, 1, _f_zscore, "(x - mean) / std"),
    "RANK": (1, 2, _f_rank, "Rank of each value, largest first; RANK(x, 1) smallest first"),
    "CUMSUM": (1, 1, _simple(lambda e: e.cum_sum()), "Running total"),
    "CUMMAX": (1, 1, _simple(lambda e: e.cum_max()), "Running maximum"),
    "CUMMIN": (1, 1, _simple(lambda e: e.cum_min()), "Running minimum"),
    # sequence
    "ROW": (0, 0, _f_row, "Row number starting at 1"),
    "LAG": (1, 2, _shift(1), "LAG(col, n) value n rows earlier"),
    "LEAD": (1, 2, _shift(-1), "LEAD(col, n) value n rows later"),
    "DIFF": (1, 2, _f_diff, "Change from previous row (seconds for date/time)"),
    "PCT_CHANGE": (1, 2, _f_pct_change, "Percent change from previous row"),
    "ROLLING_MEAN": (2, 3, _rolling("mean"), "ROLLING_MEAN(col, n) over n rows centred on each row; ROLLING_MEAN(col, n, FALSE) the n rows ending at it"),
    "ROLLING_MEDIAN": (2, 3, _rolling("median"), "ROLLING_MEDIAN(col, n[, centred])"),
    "ROLLING_STD": (2, 3, _rolling("std"), "ROLLING_STD(col, n[, centred])"),
    "ROLLING_MIN": (2, 3, _rolling("min"), "ROLLING_MIN(col, n[, centred])"),
    "ROLLING_MAX": (2, 3, _rolling("max"), "ROLLING_MAX(col, n[, centred])"),
    "ROLLING_SUM": (2, 3, _rolling("sum"), "ROLLING_SUM(col, n[, centred])"),
    "FILL_FORWARD": (1, 1, _f_fill_forward, "Replace blanks with the previous value"),
    "INTERPOLATE": (1, 1, _f_interpolate, "Fill blanks by straight-line interpolation"),
    # logic
    "IF": (3, 3, _f_if, "IF(test, value_if_true, value_if_false)"),
    "AND": (1, None, _f_and, "AND(a, b, ...)"),
    "OR": (1, None, _f_or, "OR(a, b, ...)"),
    "NOT": (1, 1, lambda c, a: Typed(~c.as_bool(a[0]), BOOL), "NOT(a)"),
    "ISBLANK": (1, 1, _f_isnull, "True when the value is empty"),
    "ISNULL": (1, 1, _f_isnull, "True when the value is empty"),
    "COALESCE": (1, None, _f_coalesce, "First non-empty value"),
    "IFNULL": (2, 2, _f_coalesce, "IFNULL(x, fallback)"),
    # text
    "LEN": (1, 1, _str_fn("len_chars", NUM), "Text length"),
    "UPPER": (1, 1, _str_fn("to_uppercase"), "UPPER CASE"),
    "LOWER": (1, 1, _str_fn("to_lowercase"), "lower case"),
    "TRIM": (1, 1, _str_fn("strip_chars"), "Remove surrounding spaces"),
    "LEFT": (2, 2, _f_left, "LEFT(text, n)"),
    "RIGHT": (2, 2, _f_right, "RIGHT(text, n)"),
    "MID": (3, 3, _f_mid, "MID(text, start, n)"),
    "CONTAINS": (2, 2, lambda c, a: Typed(_to_text(a[0]).str.contains(_to_text(a[1]), literal=True), BOOL), "CONTAINS(text, part)"),
    "STARTSWITH": (2, 2, _str_fn("starts_with", BOOL), "STARTSWITH(text, prefix)"),
    "ENDSWITH": (2, 2, _str_fn("ends_with", BOOL), "ENDSWITH(text, suffix)"),
    "REPLACE": (4, 4, _f_replace, "REPLACE(text, start, n, new) puts new in place of n characters from position start"),
    "SUBSTITUTE": (3, 4, _f_substitute, "SUBSTITUTE(text, old, new) replaces every old; SUBSTITUTE(text, old, new, k) only the k-th"),
    "CONCAT": (1, None, _f_concat, "Join text pieces (a blank joins as nothing)"),
    "TEXT": (1, 2, _f_text, "Convert to text (15 significant digits, as Excel); TEXT(date, \"%Y-%m-%d\")"),
    "VALUE": (1, 1, _f_value, "Convert text to a number"),
    "NUMBER": (1, 1, _f_value, "Convert text to a number"),
    # dates
    "DATE": (1, 2, _f_date, "Parse text as date/time; DATE(text, \"%d/%m/%Y\")"),
    "YEAR": (1, 1, _dt_part("year"), "Year"),
    "MONTH": (1, 1, _dt_part("month"), "Month 1-12"),
    "DAY": (1, 1, _dt_part("day"), "Day of month"),
    "HOUR": (1, 1, _dt_part("hour"), "Hour"),
    "MINUTE": (1, 1, _dt_part("minute"), "Minute"),
    "SECOND": (1, 1, _dt_part("second"), "Second"),
    "WEEKDAY": (1, 2, _f_weekday, "1 = Sunday ... 7 = Saturday; WEEKDAY(d, 2): 1 = Monday ... 7 = Sunday; WEEKDAY(d, 3): 0 = Monday"),
    "DAYOFYEAR": (1, 1, _dt_part("ordinal_day"), "Day of year"),
    "ELAPSED": (1, 2, _f_elapsed, "Time since the first row: ELAPSED(time, \"min\")"),
    "SECONDS_BETWEEN": (2, 2, _f_seconds_between, "SECONDS_BETWEEN(start, end)"),
}


def function_docs() -> list[tuple[str, str]]:
    return [(name, spec[3]) for name, spec in FUNCTIONS.items()]


# ----------------------------------------------------------------- public API
def compile_formula(src: str, schema: dict[str, pl.DataType], inputs: dict[str, Any] | None = None,
                    *, strict_columns: bool = False, notes: list[str] | None = None) -> tuple[pl.Expr, str, set[str]]:
    """Parse and compile. Returns (expr, result_kind, columns_used). `inputs` are named project values.

    A forgiving column match (spaces/underscores ignored) is written into ``notes`` so a caller can tell the
    person; with ``strict_columns`` it is refused instead."""
    if not src or not src.strip():
        raise FormulaError("The formula is empty")
    try:
        tree = Parser(src).parse()
        c = Compiler(dict(schema), inputs, strict_columns=strict_columns, notes=notes)
        t = c.compile(tree)
    except RecursionError:
        raise FormulaError("The formula is too long or too deeply nested") from None
    return t.expr, t.kind, c.columns_used


def check_formula(src: str, schema: dict[str, pl.DataType], inputs: dict[str, Any] | None = None) -> str | None:
    """Return an error message or None."""
    try:
        compile_formula(src, schema, inputs)
        return None
    except FormulaError as e:
        return str(e)
