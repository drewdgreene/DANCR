"""MainWindow help methods."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QPoint, QSettings, QTimer, QSize, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (QMainWindow, QMenu, QFileDialog, QMessageBox, QSplitter, QToolBar, QStatusBar, QInputDialog, QLabel,
                               QProgressDialog, QApplication, QToolButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFrame, QProgressBar, QDialog, QSizePolicy, QCheckBox)

from ..core import registry, PipelineError
from ..core.model import Pipeline
from ..core.samples import write_sample, build_template
from .document import Document
from .common import listen
from .dialogs import Toast, VersionsDialog
from .stepfactory import StepFactory
from .workers import view_pool
from .canvas import CanvasScene, CanvasView, NODE_W, NODE_H
from .inspector import InspectorPanel
from .steppicker import StepPicker
from .rail import Rail, VIEW_TYPES, REPORT_TYPES
from .tableview import TableView
from .chartview import ChartView
from .mapview import MapView
from .reportview import ReportView
from .inputsview import InputsView
from .enterdata import EnterDataView
from .answering import Understanding, AskBar, AnswerPanel
from .assistant import AssistantPanel, SideDock
from .startpage import StartPage
from .insight import InsightBar
from .sources import SourcesTray
from .theme import T
from .icons import icon

from .window_common import DATA_FILTER, FILE_FILTER, log  # noqa: F401


class WindowHelp:
    # ------------------------------------------------------------ help
    def show_help(self, section: str = "") -> None:
        from .helpdialog import HelpDialog
        HelpDialog(self, section).show()

    def show_onboarding(self) -> None:
        """The getting-started tour. Modeless, so it never blocks the window."""
        existing = getattr(self, "_onboarding", None)
        if existing is not None and existing.isVisible():
            existing.raise_(); existing.activateWindow(); return
        from .onboarding import OnboardingDialog
        dlg = OnboardingDialog(self, on_assistant=self.setup_assistant, on_agents=self.setup_agents)
        dlg.destroyed.connect(lambda: self.start.set_first_run(False))
        self._onboarding = dlg
        dlg.show()

    def _maybe_first_run(self) -> None:
        """On the very first launch, show the banner and open the tour once. Never on a project already open."""
        if self._disposed or not self._on_start_page():
            return
        first = not bool(self.settings.value("onboarding/seen", False, type=bool))
        self.start.set_first_run(first)
        if first:
            self.show_onboarding()

    def setup_assistant(self) -> None:
        from .setup import AssistantSetupDialog
        AssistantSetupDialog(self, on_saved=lambda: self.assistant._refresh_dot()).exec()

    def setup_agents(self) -> None:
        from .setup import AgentSetupDialog
        AgentSetupDialog(self).exec()

    def check_build(self) -> None:
        from .setup import CapabilitiesDialog
        CapabilitiesDialog(self).exec()

    def show_shortcuts(self) -> None:
        from .shortcuts import ShortcutDialog
        ShortcutDialog(self).exec()

    def show_log(self) -> None:
        from ..logsetup import log_path
        lp = log_path()
        if lp.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(lp)))
        self.status.showMessage(f"Log file: {lp}", 10000)

    def about(self) -> None:
        from .. import __version__
        QMessageBox.about(self, "About DANCR",
                          f"<b>DANCR {__version__}</b><br>Data Analysis Node-based Canvas for Research<br><br>"
                          "Designed and developed by Drew Greene. Commissioned by Jens Dancer.<br>"
                          "Engine: Polars. UI: Qt / PySide6 / pyqtgraph. Icons: Phosphor (MIT).<br>"
                          "Document reading: MinerU (Apache-2.0).")
