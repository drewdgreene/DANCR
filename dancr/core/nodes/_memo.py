"""A small bounded time-to-live memo shared by the connector and document steps.

Computing a step's plan hash must not reach the network, a database or a subprocess on every GUI poll, so a
short-lived probe (a HEAD request, a ``max()`` check, a MinerU version query) is remembered for a few seconds.
Each caller keeps its own cache, so one kind of source's traffic never evicts another's.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable


def make_memo(max_entries: int = 256) -> Callable[[str, float, Callable[[], Any]], Any]:
    """A memo keyed by string: ``memo(key, ttl_seconds, probe)`` returns the remembered value or runs ``probe``."""
    entries: dict[str, tuple[float, Any]] = {}
    lock = threading.Lock()

    def memo(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
        with lock:
            hit = entries.get(key)
            if hit is not None and time.time() - hit[0] < ttl:
                return hit[1]
        value = fn()
        with lock:
            if len(entries) > max_entries:      # bounded: never grow without limit
                entries.clear()
            entries[key] = (time.time(), value)  # stamped after the probe: a slow probe must not shorten the TTL
        return value

    def clear() -> None:                        # tests (and a forced refresh) can empty the cache
        with lock:
            entries.clear()

    memo.clear = clear                          # type: ignore[attr-defined]
    return memo
