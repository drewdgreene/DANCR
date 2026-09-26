import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# keep the tests' log, watchdog and untitled-cache files away from the person's real folders on every platform, and their windows off the screen
os.environ["DANCR_HOME"] = tempfile.mkdtemp(prefix="dancr-test-home-")
atexit.register(shutil.rmtree, os.environ["DANCR_HOME"], True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# and their window settings (recent projects, theme, panel sizes) away from the person's own DANCR settings
from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, os.environ["DANCR_HOME"])
QCoreApplication.setOrganizationName("DANCR-test")
QCoreApplication.setApplicationName("DANCR-test")

import polars as pl
import pytest

from dancr.synth import write_dataset
from dancr.core import Pipeline
from dancr.core.executor import Executor


@pytest.fixture(scope="session")
def probe_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("probes")
    write_dataset(d, hours=0.5, rate=20.0, seed=3)
    return d


@pytest.fixture(scope="session")
def truth(probe_dir) -> dict:
    return json.loads((probe_dir / "truth.json").read_text())


@pytest.fixture
def pipe(probe_dir, tmp_path) -> Pipeline:
    p = Pipeline("t")
    p.add_node("load_file", "A", {"path": str(probe_dir / "probe_A.csv")}, id="a")
    p.add_node("load_file", "B", {"path": str(probe_dir / "probe_B.csv")}, id="b")
    p.path = tmp_path / "p.json"
    return p


@pytest.fixture(scope="session")
def app():
    """The one QApplication, styled as the app styles it."""
    from PySide6.QtWidgets import QApplication
    from dancr.ui.theme import apply_app_style
    a = QApplication.instance() or QApplication([])
    apply_app_style(a)
    return a


@pytest.fixture
def window(app):
    """A new, empty main window on screen, with the rail listing steps by flow."""
    from PySide6.QtCore import QSettings
    from dancr.ui.mainwindow import MainWindow
    QSettings().setValue("rail_mode", "flow")
    w = MainWindow()
    w.show(); app.processEvents()
    yield w
    w.doc.stop(wait=True)                     # a run still going would ask "Stop the run and quit?"
    w.doc.undo.setClean()
    w.close()


@pytest.fixture
def opened_window(app, tmp_path, probe_dir):
    """A main window that opened a saved project: probe A loaded ("a") feeding a one-minute time-buckets step ("tb")."""
    from dancr.ui.mainwindow import MainWindow
    p = Pipeline("gui")
    p.add_node("load_file", "A", {"path": str(probe_dir / "probe_A.csv")}, id="a", x=0, y=0)
    p.add_node("time_buckets", params={"every": "1m"}, id="tb", x=300, y=0)
    p.connect("a", "tb")
    path = p.save(tmp_path / "gui.json")
    w = MainWindow()
    w.open_path(path)
    w.show()
    app.processEvents()
    yield w
    w.doc.undo.setClean()
    w.close()


@pytest.fixture
def sample(tmp_path) -> Path:
    """The built-in sample data (6000 rows) written to the test's folder."""
    from dancr.core.samples import write_sample
    return write_sample(tmp_path, rows=6000)


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """A message box nobody patched would wait for a click forever (a failing test's "Save changes?" at
    teardown looked like a hang). Answer it the harmless way instead: Discard, No or Cancel, else OK.
    A test that cares patches QMessageBox itself, which takes precedence."""
    if "PySide6.QtWidgets" not in sys.modules:
        yield
        return
    from PySide6.QtWidgets import QMessageBox as M

    def answer(*args, **kwargs):
        buttons = next((a for a in args if isinstance(a, M.StandardButton)), kwargs.get("buttons", M.Ok))
        for b in (M.Discard, M.No, M.Cancel, M.Ok):
            if buttons & b:
                return b
        return M.Ok
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(M, name, staticmethod(answer))
    yield


@pytest.fixture(autouse=True)
def _dispose_windows():
    """Delete every window a test leaves behind. Closed windows are otherwise only hidden: hundreds stay
    alive, keep their timers and signal connections, and slow down and entangle every later test."""
    yield
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return
    mw = sys.modules.get("dancr.ui.mainwindow")
    if mw is None:
        return
    # only DANCR's own windows: other parentless widgets (pyqtgraph's menus) are still referenced from Python
    for w in [w for w in app.topLevelWidgets() if isinstance(w, mw.MainWindow)]:
        if w.isVisible():
            w.doc.stop(wait=True)
            w.doc.undo.setClean()
            w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def mcp_root(tmp_path, monkeypatch) -> Path:
    """The MCP server only creates pipelines under its root folder; tests use their temporary folder."""
    import dancr.mcp_server as server
    monkeypatch.setattr(server, "ROOT", tmp_path.resolve())
    return tmp_path


@pytest.fixture
def ex(pipe) -> Executor:
    return Executor(pipe)


def run_one(pipe: Pipeline, node_id: str) -> pl.DataFrame:
    ex = Executor(pipe)
    res = ex.run(targets=[node_id])
    st = res[node_id]
    assert st.status == "done", st.error
    return pl.read_parquet(st.output)


@pytest.fixture
def small_df() -> pl.DataFrame:
    return pl.DataFrame({
        "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 0, 9), "1s", eager=True),
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2, 18.0, 100.0],
        "name": ["a", "b", "a", "b", "a", "b", "a", "b", "a", None],
    })


@pytest.fixture
def small_csv(small_df, tmp_path) -> Path:
    p = tmp_path / "small.csv"
    small_df.with_columns(pl.col("t").dt.strftime("%Y-%m-%d %H:%M:%S")).write_csv(p)
    return p
