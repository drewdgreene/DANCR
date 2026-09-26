"""The log file and the watchdog that reports a stalled window."""


def test_log_setup_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_HOME", str(tmp_path))
    import importlib, dancr.logsetup as ls
    importlib.reload(ls)
    p = ls.configure()
    import logging
    logging.getLogger("dancr").warning("hello from test")
    for h in logging.getLogger().handlers:
        h.flush()
    assert p.exists() and "hello from test" in p.read_text()


def test_watchdog_dumps_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCR_HOME", str(tmp_path))
    import importlib, time, dancr.logsetup as ls
    importlib.reload(ls)
    ls.configure()
    ls.start_watchdog(0.6)
    ls.gui_tick()
    faults_log = tmp_path / "logs" / "faults.log"
    try:
        t = time.time()                          # no ticks -> a stall is reported within about a second
        while not (faults_log.exists() and "GUI stalled" in faults_log.read_text()) and time.time() - t < 10:
            time.sleep(0.1)
        faults = faults_log.read_text()
        assert "GUI stalled" in faults and "thread MainThread" in faults
    finally:
        monkeypatch.undo()
        importlib.reload(ls)                     # later tests get the module configured for the test home again
