"""Tokenizer, AST nodes and parser for the DANCR formula language."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class FormulaError(ValueError):
    def __init__(self, message: str, pos: int | None = None) -> None:
        super().__init__(message)
        self.pos = pos


# ----------------------------------------------------------------- tokenizer
_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"]|"")*"|'(?:[^']|'')*')
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
        """Excel precedence: a sign binds tighter than ^ (-2^2 = 4), and ^ works left to right (2^3^2 = 64)."""
        left = self.parse_signed()
        while self.peek().text == "^":
            self.take()
            left = Binary("^", left, self.parse_signed())
        return left

    def parse_signed(self) -> Any:
        if self.peek().text == "-":
            self.take()
            return Unary("-", self.parse_signed())
        if self.peek().text == "+":
            self.take()
            return self.parse_signed()
        return self.parse_atom()

    def parse_atom(self) -> Any:
        t = self.take()
        if t.kind == "number":
            text = t.text
            # a whole number stays whole, however long: 1234567890123456789 as a float is its neighbour too
            return Num(int(text) if text.isdigit() else float(text))
        if t.kind == "string":
            # as in Excel: a backslash is just a character ("C:\\new"), a quote inside text is doubled ("say ""hi""")
            q = t.text[0]
            return Str(t.text[1:-1].replace(q + q, q))
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
