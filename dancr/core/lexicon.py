"""Deterministic spelling repair for typed questions.

People mistype: ``saels``, ``presure``, ``regoin``. The grammar refuses a question it cannot read, honestly,
and this module gives it one more chance before it does: a word it does not know is matched to the closest
word the *project* actually uses (a column, a table, a category value, or a fixed grammar word), and the
question is read again with the correction. It is a fix, not a guess dressed up as one — every substitution
is recorded so the person can see it, and a word with no close match is left alone so the refusal stands.

Pure standard library, no model, no randomness: the same words always repair the same way.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from .ask import STOP, _number, _DATE


CUTOFF = 0.82        # how close a word must be to a known one before it is replaced
MIN_LEN = 4          # a word this short is not corrected (too little to go on)


def known_words(vocab: dict[tuple[str, ...], Any]) -> list[str]:
    """The words a typed question may use on their own: single-token phrases of the project's vocabulary."""
    return sorted({k[0] for k in vocab if len(k) == 1 and k[0] not in STOP})


def _known(tok: str, words: set[str]) -> bool:
    return tok in words


def _transposition(tok: str, words: set[str]) -> str | None:
    """A known word one adjacent swap away ('regoin' -> 'region', 'saels' -> 'sales'), the commonest typo."""
    for i in range(len(tok) - 1):
        if tok[i] == tok[i + 1]:
            continue
        cand = tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
        if cand in words:
            return cand
    return None


def correct(tokens: list[str], vocab: dict[tuple[str, ...], Any],
            aliases: dict[str, str] | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """``(tokens, fixes)``: every token the project knows, and any unknown one replaced by its closest known
    word (or a word this project has learned to read). A token that is a stop word, a number or a date is
    never touched."""
    words = known_words(vocab)
    wordset = set(words)
    # a near miss is only ever one of the project's own names (a column, a table, a value): "season" must not become
    # the grammar word "reason" and change what is asked. A swapped pair of letters is a typo of any word
    own = [w for w in words if any(getattr(m, "kind", "") in ("col", "table", "value") for m in vocab.get((w,), []))]
    # a word that is part of one of the project's own names ('area' in 'inner area') is a real word here: it is
    # never "corrected" into something else ('are'); the refusal names the phrases it belongs to instead
    inside = {w for k in vocab if len(k) > 1 for w in k}
    learned = {str(k).strip().lower(): str(v).strip() for k, v in (aliases or {}).items() if k and v}
    out: list[str] = []
    fixes: list[tuple[str, str]] = []
    for t in tokens:
        if _known(t, wordset) or t in STOP or _number(t) is not None or re.fullmatch(_DATE, t):
            out.append(t)
            continue
        if t in learned:
            out.append(learned[t])
            fixes.append((t, learned[t]))
            continue
        if len(t) < MIN_LEN or t in inside or any(ch.isdigit() for ch in t):
            out.append(t)                            # a word with digits in it (q3, top5) is never "corrected"
            continue
        swap = _transposition(t, wordset)
        close = [swap] if swap else difflib.get_close_matches(t, own, n=1, cutoff=CUTOFF)
        if close and close[0] != t:
            out.append(close[0])
            fixes.append((t, close[0]))
        else:
            out.append(t)
    return out, fixes
