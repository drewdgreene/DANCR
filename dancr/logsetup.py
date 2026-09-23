"""One log file per machine user, plus a fault handler so native crashes leave a trace.

Set DANCR_HOME to move every per-user folder (logs, untitled caches) on any platform.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_LIMIT = 3_000_000       # bytes: faults.log and gui-stdio.log are rotated past this


def log_dir() -> Path:
    home = os.environ.get("DANCR_HOME")
    if home:
        return Path(home).expanduser() / "logs"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "DANCR" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "DANCR"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dancr"


def log_path() -> Path:
    return log_dir() / "dancr.log"


def untitled_cache_root() -> Path:
    """Where results of projects that have not been saved yet are kept: a real disk, never the temp folder."""
    home = os.environ.get("DANCR_HOME")
    if home:
        return Path(home).expanduser() / "untitled"
    return log_dir().parent / "untitled" if sys.platform == "win32" else log_dir() / "untitled"


def rotate_if_large(path: Path, limit: int = LOG_LIMIT) -> None:
    """Move `x.log` aside to `x.log.1` (replacing the previous one) once it passes `limit` bytes."""
    try:
        if path.exists() and path.stat().st_size > limit:
            path.replace(path.with_name(path.name + ".1"))
    except OSError:
        pass


_configured = False
_fault_file = None


def configure(level: int = logging.INFO, stderr_level: int = logging.INFO) -> Path:
    """Send logging to the rotating file and to stderr (never stdout: the CLI and MCP server own it);
    enable faulthandler on faults.log. The window uses the defaults; the CLI and MCP pass stderr_level=WARNING."""
    global _configured, _fault_file
    path = log_path()
    if _configured:
        return path
    _configured = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
        faults = path.parent / "faults.log"
        rotate_if_large(faults)
        _fault_file = open(faults, "a", buffering=1)          # stays open for the life of the process
        faulthandler.enable(file=_fault_file, all_threads=True)
    except OSError as e:
        print(f"dancr: cannot write log files in {path.parent} ({e}); logging to stderr only", file=sys.stderr)
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(stderr_level)
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)

    def hook(exc_type, exc, tb):
        logging.getLogger("dancr").error("Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = hook
    logging.getLogger("dancr").info("started %s", " ".join(sys.argv))
    return path


# ---------------------------------------------------------------- GUI hang watchdog
_last_tick = 0.0
_watchdog_started = False


def gui_tick() -> None:
    """Call from a GUI-thread timer; the watchdog dumps stacks when ticks stop."""
    global _last_tick
    import time
    _last_tick = time.monotonic()


def start_watchdog(stall_seconds: float = 3.0) -> None:
    """Background thread: if the GUI thread has not ticked for `stall_seconds`, write every
    thread's stack to faults.log (once per stall) so the next freeze names its cause."""
    global _watchdog_started, _last_tick
    if _watchdog_started:
        return
    _watchdog_started = True
    import threading, time, sys, traceback
    _last_tick = time.monotonic()
    log = logging.getLogger("dancr.watchdog")

    def run() -> None:
        reported = False
        while True:
            time.sleep(0.5)
            stalled = time.monotonic() - _last_tick
            if stalled >= stall_seconds and not reported:
                reported = True
                log.error("GUI thread has not processed events for %.1f s; dumping stacks", stalled)
                try:
                    with open(log_path().parent / "faults.log", "a") as f:
                        f.write(f"\n=== GUI stalled {stalled:.1f}s at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        for tid, frame in sys._current_frames().items():
                            name = next((t.name for t in threading.enumerate() if t.ident == tid), "?")
                            f.write(f"--- thread {name} ({tid})\n")
                            f.write("".join(traceback.format_stack(frame)))
                except OSError:
                    pass
            elif stalled < stall_seconds and reported:
                log.warning("GUI thread responsive again")
                reported = False
    threading.Thread(target=run, name="dancr-watchdog", daemon=True).start()


def install_signal_logging(on_terminate=None) -> None:
    import signal

    def handler(signum, frame):
        logging.getLogger("dancr").warning("received signal %s; exiting", signal.Signals(signum).name)
        if on_terminate:
            on_terminate()
        else:
            raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP) if hasattr(signal, "SIGHUP") else (signal.SIGTERM,):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def breadcrumb(msg: str) -> None:
    logging.getLogger("dancr.ui").info(msg)
