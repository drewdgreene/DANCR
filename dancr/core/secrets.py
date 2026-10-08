"""Secrets in connector settings: expand ``${ENV_VAR}`` references and redact credentials from everything shown.

A database connection string or URL can carry a password. Storing it in a plain project file is unsafe, so the
convention is to write ``${PG_DSN}`` and keep the value in the environment; ``expand_env`` resolves it at use
time. ``redact``/``redact_params`` blank a password wherever a step's settings are printed (the window, the CLI,
``describe_pipeline``), so a secret never lands in a log, a report or an agent's context. Pure standard library.
"""
from __future__ import annotations

import os
import re
from typing import Any

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
# a password inside scheme://user:password@host. The password part is matched greedily up to the *last* "@" in
# the token, so a password that itself contains "@" or "/" is still swallowed whole rather than leaking its tail.
_CRED = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/@\s]+):(?P<pw>\S+)@")
_PW_PARAM = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)=([^&;\s]+)")
# an Authorization header written as text ("Authorization: Bearer xyz", "authorization=xyz")
_AUTH = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\"']+")
# the key of a header/credential mapping whose value should always be blanked
_SENSITIVE_KEY = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[-_]?key|authorization|proxy-authorization|"
                            r"auth|cookie|credential|bearer)")


def expand_env(text: Any) -> str:
    """``${VAR}`` and ``$VAR`` replaced by the environment's value; a variable that is not set is left as written,
    so the failure names it rather than becoming an empty password. Setting ``DANCR_ALLOW_ENV`` to a comma list of
    names restricts expansion to those, so an untrusted project cannot read an unrelated secret out of the
    environment and send it to a connector."""
    s = "" if text is None else str(text)
    allowed_raw = os.environ.get("DANCR_ALLOW_ENV")
    allow = {n.strip() for n in allowed_raw.split(",") if n.strip()} if allowed_raw else None

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if allow is not None and name not in allow:
            return m.group(0)                       # not allowlisted: leave it as written, never expand
        return os.environ.get(name, m.group(0))
    return _ENV.sub(sub, s)


def redact(text: Any) -> Any:
    """A connection string, URL or header text with its secret blanked (``user:****@host``, ``password=****``,
    ``Authorization: ****``)."""
    if not isinstance(text, str):
        return text
    s = _CRED.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:****@", text)
    s = _PW_PARAM.sub(lambda m: f"{m.group(1)}=****", s)
    return _AUTH.sub(lambda m: f"{m.group(1)}****", s)


def _redact_value(v: Any) -> Any:
    """A setting value safe to show: strings are redacted, and mappings (headers) blank a sensitive key's value."""
    if isinstance(v, dict):
        out: dict[Any, Any] = {}
        for k, val in v.items():
            if isinstance(k, str) and _SENSITIVE_KEY.search(k) and val not in (None, ""):
                out[k] = "****"
            else:
                out[k] = _redact_value(val)
        return out
    if isinstance(v, (list, tuple)):
        return [_redact_value(x) for x in v]
    if isinstance(v, str):
        return redact(v)
    return v


def redact_params(node_type: Any, params: dict[str, Any]) -> dict[str, Any]:
    """A step's settings safe to show: a param marked ``secret`` becomes ``****`` (when set), and any other value
    has a password, header token or other credential in it redacted (including inside header mappings)."""
    out: dict[str, Any] = {}
    for p in node_type.params:
        v = params.get(p.name)
        if p.secret and v:
            out[p.name] = "****"
        else:
            out[p.name] = _redact_value(v)
    return out
