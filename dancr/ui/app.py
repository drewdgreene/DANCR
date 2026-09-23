"""python -m dancr.ui.app [pipeline.json]"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QEvent
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QComboBox, QAbstractSpinBox


class WheelGuard(QObject):
    """The mouse wheel never changes a combo box or spin box; it scrolls the page they sit on.
    Values change by opening the list or typing, so scrolling through Settings cannot alter a step by accident."""

    def eventFilter(self, obj, e) -> bool:
        if e.type() == QEvent.Wheel and isinstance(obj, (QComboBox, QAbstractSpinBox)):
            w = obj.parentWidget()
            while w is not None:                 # hand the wheel to the nearest ancestor that scrolls
                e.accept()
                QApplication.sendEvent(w, e)
                if e.isAccepted():
                    break
                w = w.parentWidget()
            return True
        return False


def install_wheel_guard(app: QApplication) -> WheelGuard:
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard


def _rebuild_for_theme(app: QApplication, win_ref: list) -> None:
    """Rebuild the window with the new palette. Widgets capture theme colours when they are built, so a
    live theme change recreates the chrome; the Document (project, undo stack, cache) is carried over."""
    old = win_ref[0] if win_ref else None
    if old is None:
        return
    state = old.capture_ui_state()
    from .mainwindow import MainWindow
    new = MainWindow(doc=old.doc)
    win_ref[0] = new
    new.restore_ui_state(state)
    new.show()
    old.dispose_for_theme()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    from ..logsetup import configure
    configure(logging.INFO)
    QCoreApplication.setOrganizationName("DANCR")
    QCoreApplication.setApplicationName("DANCR")
    app = QApplication(argv)
    from .theme import apply_app_style, theme_manager
    apply_app_style(app)
    theme = theme_manager(app)
    install_wheel_guard(app)
    icon = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    from ..logsetup import start_watchdog, gui_tick, install_signal_logging, breadcrumb
    from PySide6.QtCore import QTimer
    import threading
    tick = QTimer(); tick.setInterval(250); tick.timeout.connect(gui_tick); tick.start()
    start_watchdog(3.0)
    win_ref: list = []
    if theme is not None:
        theme.changed.connect(lambda: _rebuild_for_theme(app, win_ref))
    install_signal_logging(lambda: QTimer.singleShot(0, win_ref[0].terminate) if win_ref else app.quit())
    # matplotlib's first import can build a font cache for many seconds; do it off the GUI thread now
    threading.Thread(target=lambda: __import__("matplotlib.pyplot"), name="mpl-warmup", daemon=True).start()
    from ..core.executor import sweep_untitled_caches
    threading.Thread(target=sweep_untitled_caches, name="cache-sweep", daemon=True).start()
    from .mainwindow import MainWindow
    win = MainWindow()
    win_ref.append(win)
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:
        win.open_path(files[0])
    win.show()
    code = app.exec()
    breadcrumb(f"exiting normally with code {code}")
    logging.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
