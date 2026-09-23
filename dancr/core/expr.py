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


class FormulaError(ValueError):
    def __init__(self, message: str, pos: int | None = None) -> None:
        super().__init__(message)
        self.pos = pos


# ----------------------------------------------------------------- tokenizer
_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<bracket>\[[^\]]+\])
  | (?P<backtick>`[^`]+`)
  | (?P<ident>[^\W\d][\w.]*)
  | (?P<op><=|>=|<>|!=|==|[-+*/^%&<>=(),])
""", re.VERBOSE)


@dataclass
class Tok:
    kind: str
    text: str
    pos: int


def tokenize(src: str) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise FormulaError(f"Unexpected character {src[i]!r}", i)
        kind = m.lastgroup or ""
        if kind != "ws":
            out.append(Tok(kind, m.group(0), i))
        i = m.end()
    out.append(Tok("eof", "", len(src)))
    return out


# ----------------------------------------------------------------- AST
@dataclass
class Num:
    value: float
@dataclass
class Str:
    value: str
@dataclass
class Col:
    name: str
    pos: int
@dataclass
class Const:
    name: str          # TRUE FALSE NULL
@dataclass
class Unary:
    op: str
    operand: Any
@dataclass
class Binary:
    op: str
    left: Any
    right: Any
@dataclass
class Call:
    name: str
    args: list[Any]
    pos: int


class Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def take(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Tok:
        t = self.take()
        if t.text != text:
            raise FormulaError(f"Expected {text!r} but found {t.text!r}" if t.text else f"Expected {text!r} before the end", t.pos)
        return t

    def parse(self) -> Any:
        node = self.parse_or()
        t = self.peek()
        if t.kind != "eof":
            raise FormulaError(f"Unexpected {t.text!r}", t.pos)
        return node

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.peek().text.lower() == "or":
            self.take()
            left = Binary("or", left, self.parse_and())
        return left

    def parse_and(self) -> Any:
        left = self.parse_not()
        while self.peek().text.lower() == "and":
            self.take()
            left = Binary("and", left, self.parse_not())
        return left

    def parse_not(self) -> Any:
        if self.peek().text.lower() == "not":
            self.take()
            return Unary("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self) -> Any:
        left = self.parse_concat()
        t = self.peek()
        if t.text in ("=", "==", "!=", "<>", "<", ">", "<=", ">="):
            self.take()
            right = self.parse_concat()
            op = {"==": "=", "<>": "!="}.get(t.text, t.text)
            return Binary(op, left, right)
        return left

    def parse_concat(self) -> Any:
        left = self.parse_add()
        while self.peek().text == "&":
            self.take()
            left = Binary("&", left, self.parse_add())
        return left

    def parse_add(self) -> Any:
        left = self.parse_mul()
        while self.peek().text in ("+", "-"):
            op = self.take().text
            left = Binary(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> Any:
        left = self.parse_pow()
        while self.peek().text in ("*", "/", "%"):
            op = self.take().text
            left = Binary(op, left, self.parse_pow())
        return left

    def parse_pow(self) -> Any:
        """Excel precedence: a sign on the base binds tighter than ^ (-2^2 = 4); ^ is right-associative."""
        base = self.parse_signed()
        if self.peek().text == "^":
            self.take()
            return Binary("^", base, self.parse_exponent())
        return base

    def parse_signed(self) -> Any:
        if self.peek().text == "-":
            self.take()
            return Unary("-", self.parse_signed())
        if self.peek().text == "+":
            self.take()
            return self.parse_signed()
        return self.parse_atom()

    def parse_exponent(self) -> Any:
        if self.peek().text == "-":
            self.take()
            return Unary("-", self.parse_exponent())
        if self.peek().text == "+":
            self.take()
            return self.parse_exponent()
        return self.parse_pow()

    def parse_atom(self) -> Any:
        t = self.take()
        if t.kind == "number":
            return Num(float(t.text))
        if t.kind == "string":
            body = t.text[1:-1]
            if "\\" in body:
                try:
                    body = body.encode("latin-1", "backslashreplace").decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError) as e:
                    raise FormulaError(f"Bad escape in text: {e}", t.pos) from e
            return Str(body)
        if t.kind == "bracket":
            return Col(t.text[1:-1].strip(), t.pos)
        if t.kind == "backtick":
            return Col(t.text[1:-1].strip(), t.pos)
        if t.text == "(":
            node = self.parse_or()
            self.expect(")")
            return node
        if t.kind == "ident":
            low = t.text.lower()
            if low in ("true", "false", "null"):
                return Const(low)
            if self.peek().text == "(":
                self.take()
                args: list[Any] = []
                if self.peek().text != ")":
                    args.append(self.parse_or())
                    while self.peek().text == ",":
                        self.take()
                        args.append(self.parse_or())
                self.expect(")")
                return Call(t.text.upper(), args, t.pos)
            return Col(t.text, t.pos)
        if t.kind == "eof":
            raise FormulaError("The formula ends too early", t.pos)
        raise FormulaError(f"Unexpected {t.text!r}", t.pos)


# ----------------------------------------------------------------- compiler
NUM, STR, BOOL, TIME, DUR, ANY = "number", "text", "true/false", "date/time", "duration", "any"


def _kind_of_dtype(dt: pl.DataType) -> str:
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
    def __init__(self, schema: dict[str, pl.DataType], inputs: dict[str, Any] | None = None) -> None:
        self.schema = schema
        self.lower = {k.lower(): k for k in schema}
        self.columns_used: set[str] = set()
        self.inputs = inputs or {}
        self.inputs_lower = {k.lower(): k for k in self.inputs}
        self.inputs_used: set[str] = set()

    # -- columns ----------------------------------------------------------
    def resolve_column(self, name: str, pos: int) -> str:
        if name in self.schema:
            return name
        if name.lower() in self.lower:
            return self.lower[name.lower()]
        # forgiving: strip spaces/underscores
        squash = lambda s: re.sub(r"[\s_]+", "", s.lower())
        squashed = [c for c in self.schema if squash(c) == squash(name)]
        if len(squashed) == 1:
            return squashed[0]
        if len(squashed) > 1:
            raise FormulaError(f"{name!r} could mean {squashed[0]!r} or {squashed[1]!r}; write the exact name in [brackets]", pos)
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
        v = int(n.value) if n.value.is_integer() and abs(n.value) < 2**53 else n.value
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
        key = n.name.lower()
        if n.name not in self.schema and key not in self.lower and key in self.inputs_lower:
            real = self.inputs_lower[key]
            self.inputs_used.add(real)
            v = self.inputs[real]
            if isinstance(v, bool):
                return Typed(pl.lit(v), BOOL, v)
            if isinstance(v, (int, float)):
                return Typed(pl.lit(v), NUM, v)
            return Typed(pl.lit(str(v)), STR, str(v))
        name = self.resolve_column(n.name, n.pos)
        self.columns_used.add(name)
        return Typed(pl.col(name), _kind_of_dtype(self.schema[name]), dtype=self.schema[name])

    def c_Unary(self, n: Unary) -> Typed:
        v = self.compile(n.operand)
        if n.op == "-":
            if v.kind in (STR, TIME, BOOL):
                raise FormulaError(f"Cannot negate a {v.kind} value")
            return Typed(-v.expr, NUM)
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
        from .dtypes import datetime_literal, align_time_column
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
            if a.kind == STR and b.kind == NUM or a.kind == NUM and b.kind == STR:
                # compare text column to number: cast number to text
                if a.kind == NUM:
                    a = Typed(a.expr.cast(pl.Utf8), STR)
                else:
                    b = Typed(b.expr.cast(pl.Utf8), STR)
            fn = {"=": pl.Expr.eq, "!=": pl.Expr.ne, "<": pl.Expr.lt, ">": pl.Expr.gt,
                  "<=": pl.Expr.le, ">=": pl.Expr.ge}[op]
            return Typed(fn(a.expr, b.expr), BOOL)
        if op == "&":
            return Typed(pl.concat_str([a.expr.cast(pl.Utf8), b.expr.cast(pl.Utf8)]), STR)
        if op == "+":
            if a.kind == STR and b.kind == STR:
                return Typed(pl.concat_str([a.expr, b.expr]), STR)
            if a.kind == TIME and b.kind == DUR or a.kind == DUR and b.kind == TIME:
                return Typed(a.expr + b.expr, TIME)
            return Typed(a.expr + b.expr, self._numkind(a, b))
        if op == "-":
            if a.kind == TIME and b.kind == TIME:
                a, b = self.coerce_time_literal(a, b)
                return Typed(a.expr - b.expr, DUR)
            if a.kind == TIME and b.kind == DUR:
                return Typed(a.expr - b.expr, TIME)
            return Typed(a.expr - b.expr, self._numkind(a, b))
        if op == "*":
            return Typed(a.expr * b.expr, self._numkind(a, b))
        if op == "/":
            return Typed(a.expr.cast(pl.Float64) / b.expr.cast(pl.Float64), NUM)
        if op == "%":
            return Typed(_num(a) % _num(b), NUM)
        if op == "^":
            return Typed(a.expr.cast(pl.Float64).pow(b.expr.cast(pl.Float64)), NUM)
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


# ----------------------------------------------------------------- functions
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
            return Typed(cfn(_num(args[0])), NUM)
        return Typed(hfn([_num(a) for a in args]), NUM)
    return f


def _simple(fn: Callable[[pl.Expr], pl.Expr], kind: str = NUM, cast: bool = True) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        return Typed(fn(_num(args[0]) if cast else args[0].expr), kind)
    return f


def _f_log(c: Compiler, args: list[Typed]) -> Typed:
    if len(args) == 2:
        base = args[1].literal
        if base is None:
            raise FormulaError("LOG base must be a plain number")
        return Typed(_num(args[0]).log(float(base)), NUM)
    return Typed(_num(args[0]).log10(), NUM)


def _f_round(c: Compiler, args: list[Typed]) -> Typed:
    d = _int_lit(args[1], "ROUND digits") if len(args) == 2 else 0
    return Typed(_num(args[0]).round(d), NUM)


def _f_if(c: Compiler, args: list[Typed]) -> Typed:
    cond = c.as_bool(args[0])
    a, b = args[1], args[2]
    return Typed(pl.when(cond).then(a.expr).otherwise(b.expr), _same_kind([a, b], "IF"))


def _f_coalesce(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.coalesce([a.expr for a in args]), _same_kind(args, "COALESCE / IFNULL"))


def _f_isnull(c: Compiler, args: list[Typed]) -> Typed:
    e = args[0].expr
    if args[0].kind == STR:
        return Typed(e.is_null() | (e.cast(pl.Utf8).str.strip_chars() == ""), BOOL)
    if args[0].kind == NUM:
        return Typed(e.is_null() | e.cast(pl.Float64).is_nan(), BOOL)
    return Typed(e.is_null(), BOOL)


def _f_concat(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.concat_str([a.expr.cast(pl.Utf8) for a in args]), STR)


def _f_left(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(args[0].expr.cast(pl.Utf8).str.slice(0, _int_lit(args[1], "LEFT length")), STR)


def _f_right(c: Compiler, args: list[Typed]) -> Typed:
    n = _int_lit(args[1], "RIGHT length")
    return Typed(args[0].expr.cast(pl.Utf8).str.slice(-n, n), STR)


def _f_mid(c: Compiler, args: list[Typed]) -> Typed:
    start = _int_lit(args[1], "MID start") - 1
    if start < 0:
        raise FormulaError("MID start must be 1 or more")
    n = _int_lit(args[2], "MID length")
    return Typed(args[0].expr.cast(pl.Utf8).str.slice(start, n), STR)


def _str_fn(method: str, kind: str = STR) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        e = args[0].expr.cast(pl.Utf8)
        rest = [a.expr for a in args[1:]]
        return Typed(getattr(e.str, method)(*rest), kind)
    return f


def _f_replace(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(args[0].expr.cast(pl.Utf8).str.replace_all(args[1].expr, args[2].expr, literal=True), STR)


def _f_text(c: Compiler, args: list[Typed]) -> Typed:
    if len(args) == 2:
        if args[0].kind != TIME:
            raise FormulaError("TEXT with a format needs a date/time value, e.g. TEXT(time, \"%Y-%m-%d\")")
        if not isinstance(args[1].literal, str):
            raise FormulaError("TEXT format must be plain text in quotes, e.g. \"%Y-%m-%d\"")
        return Typed(args[0].expr.dt.strftime(args[1].literal), STR)
    return Typed(args[0].expr.cast(pl.Utf8), STR)


def _f_value(c: Compiler, args: list[Typed]) -> Typed:
    from .dtypes import text_to_number_expr
    e = text_to_number_expr(args[0].expr) if args[0].kind == STR else args[0].expr.cast(pl.Float64, strict=False)
    return Typed(e, NUM)


def _f_date(c: Compiler, args: list[Typed]) -> Typed:
    e = args[0].expr
    if args[0].kind == TIME:
        return Typed(e, TIME)
    if len(args) == 2:
        if not isinstance(args[1].literal, str):
            raise FormulaError("DATE format must be plain text in quotes, e.g. \"%d/%m/%Y\"")
        return Typed(e.cast(pl.Utf8).str.to_datetime(args[1].literal, strict=False), TIME)
    if args[0].kind == NUM:
        raise FormulaError("DATE of a number needs a unit; use DATE(TEXT(x)) or convert the column type first")
    return Typed(e.cast(pl.Utf8).str.to_datetime(strict=False), TIME)


def _dt_part(attr: str) -> Callable:
    def f(c: Compiler, args: list[Typed]) -> Typed:
        if args[0].kind != TIME:
            raise FormulaError("This function needs a date/time column")
        return Typed(getattr(args[0].expr.dt, attr)(), NUM)
    return f


def _f_elapsed(c: Compiler, args: list[Typed]) -> Typed:
    if args[0].kind != TIME:
        raise FormulaError("ELAPSED needs a date/time column")
    unit = str(args[1].literal).lower() if len(args) == 2 else "s"
    div = {"ms": 1e3, "s": 1e6, "sec": 1e6, "seconds": 1e6, "m": 60e6, "min": 60e6, "minutes": 60e6,
           "h": 3600e6, "hours": 3600e6, "d": 86400e6, "days": 86400e6}.get(unit)
    if div is None:
        raise FormulaError("ELAPSED unit must be one of ms, s, min, h, d")
    e = args[0].expr
    return Typed((e - e.min()).dt.total_microseconds().cast(pl.Float64) / div, NUM)


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
        center = bool(args[2].literal) if len(args) == 3 else False
        e = _num(args[0])
        return Typed(getattr(e, f"rolling_{method}")(window_size=n, min_samples=1, center=center), NUM)
    return f


def _f_zscore(c: Compiler, args: list[Typed]) -> Typed:
    e = _num(args[0])
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
    return Typed(_num(args[0]).quantile(q), NUM)


def _f_clip(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(_num(args[0]).clip(_num(args[1]), _num(args[2])), NUM)


def _f_pct_change(c: Compiler, args: list[Typed]) -> Typed:
    n = _int_lit(args[1], "lag") if len(args) == 2 else 1
    return Typed(_num(args[0]).pct_change(n) * 100.0, NUM)


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
    return Typed(args[0].expr.rank(method="min"), NUM)


def _f_count(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(args[0].expr.count(), NUM)


def _f_pi(c: Compiler, args: list[Typed]) -> Typed:
    return Typed(pl.lit(math.pi), NUM, math.pi)


# name -> (min_args, max_args or None, builder, doc)
FUNCTIONS: dict[str, tuple[int, int | None, Callable[[Compiler, list[Typed]], Typed], str]] = {
    # math
    "ABS": (1, 1, _simple(lambda e: e.abs()), "Absolute value"),
    "SQRT": (1, 1, _simple(lambda e: e.sqrt()), "Square root"),
    "EXP": (1, 1, _simple(lambda e: e.exp()), "e to the power x"),
    "LN": (1, 1, _simple(lambda e: e.log()), "Natural log"),
    "LOG": (1, 2, _f_log, "LOG(x) base 10, or LOG(x, base)"),
    "LOG2": (1, 1, _simple(lambda e: e.log(2)), "Log base 2"),
    "POW": (2, 2, lambda c, a: Typed(_num(a[0]).pow(_num(a[1])), NUM), "POW(x, y) = x^y"),
    "ROUND": (1, 2, _f_round, "ROUND(x, digits)"),
    "FLOOR": (1, 1, _simple(lambda e: e.floor()), "Round down"),
    "CEIL": (1, 1, _simple(lambda e: e.ceil()), "Round up"),
    "CEILING": (1, 1, _simple(lambda e: e.ceil()), "Round up"),
    "SIGN": (1, 1, _simple(lambda e: e.sign()), "-1, 0 or 1"),
    "MOD": (2, 2, lambda c, a: Typed(_num(a[0]) % _num(a[1]), NUM), "Remainder"),
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
    "MEDIAN": (1, 1, _simple(lambda e: e.median()), "Column median"),
    "STDEV": (1, 1, _simple(lambda e: e.std()), "Column standard deviation"),
    "STD": (1, 1, _simple(lambda e: e.std()), "Column standard deviation"),
    "VAR": (1, 1, _simple(lambda e: e.var()), "Column variance"),
    "COUNT": (1, 1, _f_count, "Number of non-empty values"),
    "PERCENTILE": (2, 2, _f_percentile, "PERCENTILE(col, 0.95)"),
    "ZSCORE": (1, 1, _f_zscore, "(x - mean) / std"),
    "RANK": (1, 1, _f_rank, "Rank of each value"),
    "CUMSUM": (1, 1, _simple(lambda e: e.cum_sum()), "Running total"),
    "CUMMAX": (1, 1, _simple(lambda e: e.cum_max()), "Running maximum"),
    "CUMMIN": (1, 1, _simple(lambda e: e.cum_min()), "Running minimum"),
    # sequence
    "ROW": (0, 0, _f_row, "Row number starting at 1"),
    "LAG": (1, 2, _shift(1), "LAG(col, n) value n rows earlier"),
    "LEAD": (1, 2, _shift(-1), "LEAD(col, n) value n rows later"),
    "DIFF": (1, 2, _f_diff, "Change from previous row (seconds for date/time)"),
    "PCT_CHANGE": (1, 2, _f_pct_change, "Percent change from previous row"),
    "ROLLING_MEAN": (2, 3, _rolling("mean"), "ROLLING_MEAN(col, n[, centered])"),
    "ROLLING_MEDIAN": (2, 3, _rolling("median"), "ROLLING_MEDIAN(col, n)"),
    "ROLLING_STD": (2, 3, _rolling("std"), "ROLLING_STD(col, n)"),
    "ROLLING_MIN": (2, 3, _rolling("min"), "ROLLING_MIN(col, n)"),
    "ROLLING_MAX": (2, 3, _rolling("max"), "ROLLING_MAX(col, n)"),
    "ROLLING_SUM": (2, 3, _rolling("sum"), "ROLLING_SUM(col, n)"),
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
    "CONTAINS": (2, 2, lambda c, a: Typed(a[0].expr.cast(pl.Utf8).str.contains(a[1].expr, literal=True), BOOL), "CONTAINS(text, part)"),
    "STARTSWITH": (2, 2, _str_fn("starts_with", BOOL), "STARTSWITH(text, prefix)"),
    "ENDSWITH": (2, 2, _str_fn("ends_with", BOOL), "ENDSWITH(text, suffix)"),
    "REPLACE": (3, 3, _f_replace, "REPLACE(text, old, new)"),
    "CONCAT": (1, None, _f_concat, "Join text pieces"),
    "TEXT": (1, 2, _f_text, "Convert to text; TEXT(date, \"%Y-%m-%d\")"),
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
    "WEEKDAY": (1, 1, _dt_part("weekday"), "1 = Monday ... 7 = Sunday"),
    "DAYOFYEAR": (1, 1, _dt_part("ordinal_day"), "Day of year"),
    "ELAPSED": (1, 2, _f_elapsed, "Time since the first row: ELAPSED(time, \"min\")"),
    "SECONDS_BETWEEN": (2, 2, _f_seconds_between, "SECONDS_BETWEEN(start, end)"),
}


def function_docs() -> list[tuple[str, str]]:
    return [(name, spec[3]) for name, spec in FUNCTIONS.items()]


# ----------------------------------------------------------------- public API
def compile_formula(src: str, schema: dict[str, pl.DataType], inputs: dict[str, Any] | None = None) -> tuple[pl.Expr, str, set[str]]:
    """Parse and compile. Returns (expr, result_kind, columns_used). `inputs` are named project values."""
    if not src or not src.strip():
        raise FormulaError("The formula is empty")
    try:
        tree = Parser(src).parse()
        c = Compiler(dict(schema), inputs)
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
