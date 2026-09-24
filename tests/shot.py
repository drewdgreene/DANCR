"""Headless screenshot helper: python tests/shot.py pipeline.json out.png [node_id]"""
import sys, os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QCoreApplication, QSettings
from PySide6.QtGui import QFont
from dancr.ui.theme import apply_app_style
QCoreApplication.setOrganizationName("DANCR"); QCoreApplication.setApplicationName("DANCR")   # as ui/app.py does, so this clears the app's own settings
app = QApplication([]); app.setFont(QFont("Adwaita Mono", 11)); apply_app_style(app)
QSettings().clear()
from dancr.ui.mainwindow import MainWindow
w = MainWindow(); w.resize(int(os.environ.get('SHOT_W', '1250')), int(os.environ.get('SHOT_H', '900')))
if len(sys.argv) > 1 and sys.argv[1] != "-":
    w.open_path(sys.argv[1])
w.show()
def go():
    w.view.fit_all()
    if len(sys.argv) > 3:
        w.show_node(sys.argv[3])
    if len(sys.argv) > 4:
        w.table.tabs.setCurrentIndex(int(sys.argv[4]))
    QTimer.singleShot(int(os.environ.get('SHOT_DELAY', '1200')), lambda: (w.grab().save(sys.argv[2]), app.quit()))
QTimer.singleShot(300, go)
app.exec()
