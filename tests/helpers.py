"""Helpers shared by the window tests: drive the Qt event loop until something is true."""
import time


def pump(app, ms: int = 50) -> None:
    """Process events for ``ms`` milliseconds."""
    t = time.time()
    while time.time() - t < ms / 1000:
        app.processEvents()


def settle(app, until, secs: float = 10) -> None:
    """Process events until ``until()`` is true; fail if it is not within ``secs``."""
    t = time.time()
    while not until() and time.time() - t < secs:
        app.processEvents()
    assert until()


def wait_run(window, app, secs: float = 30) -> None:
    """Wait for a pending auto-run to start and every run to settle."""
    settle(app, lambda: not (window.doc._auto_pending or window.doc.running), secs)
