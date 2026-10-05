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
# a password inside scheme://user:password@host
_CRED = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@")
_PW_PARAM = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)=([^&;\s]+)")


def expand_env(text: Any) -> str:
    """``${VAR}`` and ``$VAR`` replaced by the environment's value; a variable that is not set is left as written,
    so the failure names it rather than becoming an empty password."""
    s = "" if text is None else str(text)
    return _ENV.sub(lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)), s)


def redact(text: Any) -> Any:
    """A connection string or URL with its password blanked (``user:****@host``, ``password=****``)."""
    if not isinstance(text, str):
        return text
    s = _CRED.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:****@", text)
    return _PW_PARAM.sub(lambda m: f"{m.group(1)}=****", s)


def redact_params(node_type: Any, params: dict[str, Any]) -> dict[str, Any]:
    """A step's settings safe to show: a param marked ``secret`` becomes ``****`` (when set), and any other text
    value has a password in it redacted."""
    out: dict[str, Any] = {}
    for p in node_type.params:
        v = params.get(p.name)
        if p.secret and v:
            out[p.name] = "****"
        elif isinstance(v, str):
            out[p.name] = redact(v)
        else:
            out[p.name] = v
    return out
