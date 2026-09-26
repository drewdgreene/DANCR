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
