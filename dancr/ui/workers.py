"""Background execution helpers."""
from __future__ import annotations

import logging
import os
import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot

log = logging.getLogger("dancr.ui")


class TaskSignals(QObject):
    done = Signal(object)          # result
    failed = Signal(str)           # short message (the full traceback is in the log)
    finished = Signal()


# Set while no pipeline run is executing. View queries that read results wait for it (held in the pool,
# not on a thread) so memory is never shared between a run and a query.
run_gate = threading.Event()
run_gate.set()


class Task(QRunnable):
    """Run fn(*args) in a pool; results come back on the GUI thread via signals.
    ``cancelled`` is honoured before the work starts and before any result is delivered."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = TaskSignals()
        self.cancelled = False
        self.waits_for_run = True
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            try:
                if self.cancelled:
                    return
                r = self.fn(*self.args, **self.kwargs)
            except Exception as e:  # noqa: BLE001
                log.error("Background task %s failed:\n%s", getattr(self.fn, "__qualname__", self.fn), traceback.format_exc())
                if not self.cancelled:
                    self.signals.failed.emit(str(e) or type(e).__name__)
            else:
                if not self.cancelled:
                    self.signals.done.emit(r)
            finally:
                self.signals.finished.emit()
        except RuntimeError:
            pass    # the receiving widget was destroyed (app shutting down)


class ViewPool(QThreadPool):
    """A small pool for previews, chart queries, page fetches and searches.

    Tasks with ``waits_for_run`` are held here while a run executes and started when it ends, so a
    waiting task never occupies a thread and page fetches or searches keep flowing during a run."""

    def __init__(self) -> None:
        super().__init__()
        self.setMaxThreadCount(min(4, os.cpu_count() or 1))
        self._held: list[Task] = []
        self._active: set[Task] = set()

    def start(self, task: Task) -> None:  # type: ignore[override]
        if task.waits_for_run and not run_gate.is_set():
            self._held.append(task)
            return
        self._active.add(task)
        task.signals.finished.connect(lambda t=task: self._active.discard(t))
        super().start(task)

    def take(self, task: Task) -> bool:
        """Remove a task that has not started; True if it was removed."""
        if task in self._held:
            self._held.remove(task)
            return True
        if self.tryTake(task):
            self._active.discard(task)
            return True
        return False

    def hold(self) -> None:
        run_gate.clear()

    def release(self) -> None:
        run_gate.set()
        held, self._held = self._held, []
        for t in held:
            if not t.cancelled:
                self.start(t)

    def shutdown(self, wait_ms: int = 5000) -> None:
        """Cancel everything: held tasks are dropped, queued ones removed, running ones told to discard their result."""
        for t in self._held:
            t.cancelled = True
        self._held = []
        for t in list(self._active):
            t.cancelled = True
        self.clear()
        self.waitForDone(wait_ms)
        run_gate.set()


_VIEW_POOL: ViewPool | None = None


def view_pool() -> ViewPool:
    global _VIEW_POOL
    if _VIEW_POOL is None:
        _VIEW_POOL = ViewPool()
    return _VIEW_POOL


class Serial:
    """Keeps only the latest task of a kind: earlier results are ignored. Tasks that have not started
    (queued or held for a run) are dropped when a newer one arrives."""

    def __init__(self) -> None:
        self._current: Task | None = None
        self._inflight: set[Task] = set()
        self.pool = view_pool()

    def submit(self, fn: Callable[..., Any], on_done: Callable[[Any], None], on_fail: Callable[[str], None] | None = None, *args: Any, **kwargs: Any) -> Task:
        self.cancel()
        t = Task(fn, *args, **kwargs)
        self._current = t
        self._inflight.add(t)
        t.signals.done.connect(on_done)
        if on_fail:
            t.signals.failed.connect(on_fail)
        t.signals.finished.connect(lambda t=t: self._inflight.discard(t))
        self.pool.start(t)
        return t

    def cancel(self) -> None:
        if self._current is not None:
            self._current.cancelled = True
            if self.pool.take(self._current):
                self._inflight.discard(self._current)


class RunThread(QThread):
    """Runs the executor off the GUI thread; progress arrives as signals on the GUI thread."""
    event = Signal(dict)
    failed = Signal(str)

    def __init__(self, executor, targets: list[str] | None, force: bool = False) -> None:
        super().__init__()
        self.executor = executor
        self.targets = targets
        self.force = force
        self.cancel = threading.Event()
        self.results: dict = {}

    def run(self) -> None:
        from ..core.executor import ExecutionCancelled
        try:
            self.results = self.executor.run(targets=self.targets, on_event=self._emit, cancel=self.cancel, force=self.force)
        except ExecutionCancelled:
            self.event.emit({"type": "run_cancelled"})
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

    def _emit(self, e: dict) -> None:
        if e.get("type") == "run_cancelled":
            self.results = dict(e.get("states") or {})
        self.event.emit(e)
