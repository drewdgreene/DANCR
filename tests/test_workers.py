"""Background work: tasks that wait for a run, and results that arrive too late."""
import threading

from helpers import pump


def test_waiting_task_does_not_block_a_page_fetch():
    import os, time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from dancr.ui.workers import Task, view_pool, run_gate
    pool = view_pool()
    pool.hold()
    got = []
    try:
        waiting = Task(lambda: "later"); waiting.signals.done.connect(got.append)
        quick = Task(lambda: "now"); quick.waits_for_run = False; quick.signals.done.connect(got.append)
        pool.start(waiting); pool.start(quick)
        t = time.time()
        while "now" not in got and time.time() - t < 5:
            app.processEvents()
        assert got == ["now"] and not run_gate.is_set()     # the held task never ran while the gate was closed
    finally:
        pool.release()                                      # never leave the gate closed for later tests
    t = time.time()
    while "later" not in got and time.time() - t < 5:
        app.processEvents()
    assert got == ["now", "later"]


def test_page_fetches_are_not_starved_by_long_queries(app):
    # every pool thread busy with a query that cannot be interrupted (a superseded Describe of a huge table)
    import time
    from dancr.ui.workers import Task, view_pool
    pool = view_pool()
    release = threading.Event()
    slow = []
    for _ in range(pool.maxThreadCount() + 2):
        t = Task(release.wait, 10); t.waits_for_run = False
        slow.append(t); pool.start(t)
    got = []
    page = Task(lambda: "page"); page.waits_for_run = False; page.quick = True
    page.signals.done.connect(got.append)
    try:
        pool.start(page)
        t0 = time.time()
        while not got and time.time() - t0 < 5:
            app.processEvents()
        assert got == ["page"]
    finally:
        release.set()
        pool.waitForDone(5000)
    pump(app, 50)


def test_an_older_result_already_queued_is_not_delivered(app):
    from dancr.ui.workers import Serial, view_pool
    got = []
    s = Serial(waits_for_run=False)
    s.submit(lambda: "old", got.append)
    view_pool().waitForDone(5000)                       # "old" is done and its signal is queued
    release = threading.Event()
    s.submit(lambda: (release.wait(5), "new")[1], got.append)
    release.set()
    view_pool().waitForDone(5000)
    pump(app, 100)
    assert got == ["new"]


def test_garbage_holding_a_widget_is_collected_on_the_gui_thread(app):
    """A widget in a reference cycle must never be freed by a worker thread that happens to trigger a collection
    (Qt deadlocks or crashes); the GUI thread collects it instead."""
    import gc
    import time
    from PySide6.QtWidgets import QWidget
    from dancr.ui.app import GuiThreadGC
    guard = app.findChild(GuiThreadGC)
    assert guard is not None and not gc.isenabled()
    freed_on: list[int] = []

    class Cyclic(QWidget):
        def __del__(self):
            freed_on.append(threading.get_ident())

    w = Cyclic(); w.me = w
    del w

    def allocate() -> None:                     # holds far more than a collection's worth at once
        held = [[] for _ in range(5 * gc.get_threshold()[0])]
        assert len(held)
    t = threading.Thread(target=allocate)
    t.start(); t.join()
    assert freed_on == []
    guard._ticks = guard.FULL_EVERY - 1         # the next tick collects everything
    end = time.time() + 5
    while not freed_on and time.time() < end:
        pump(app, 50)
    assert freed_on == [threading.get_ident()]
