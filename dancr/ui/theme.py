"""Colors, fonts and the application style sheet. One accent, neutral greys, no decoration."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QObject, Signal, QSettings
from PySide6.QtGui import QColor, QPalette, QGuiApplication, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

# Category tints: used only for the step icon, never for large surfaces.
CATEGORY_COLORS = {
    "Get data": "#2563eb", "Filter & sort": "#7c3aed", "Calculate": "#9333ea", "Clean up": "#d97706", "Combine": "#db2777",
    "Time": "#059669", "Analyse & model": "#0891b2", "Share": "#475569",
}
STATUS_COLORS = {"idle": "#8b8b92", "stale": "#d97706", "running": "#2563eb", "done": "#16a34a", "failed": "#dc2626", "preview": "#7c3aed"}
SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#78716c"]

UI_FONTS = ["Adwaita Sans", "Inter", "Noto Sans", "Cantarell", "Segoe UI", "SF Pro Text", "Helvetica Neue", "DejaVu Sans", "Liberation Sans", "Arial"]
MONO_FONTS = ["Adwaita Mono", "JetBrains Mono", "Source Code Pro", "Noto Sans Mono", "DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]


THEME_SETTING = "theme"
THEME_CHOICES = ("system", "light", "dark")


def preference() -> str:
    """The chosen appearance: 'system' (follow the OS), 'light' or 'dark'.
    The DANCR_THEME environment variable is a hard override, handy for screenshots and CI."""
    forced = os.environ.get("DANCR_THEME", "").strip().lower()
    if forced in ("dark", "light"):
        return forced
    value = str(QSettings().value(THEME_SETTING, "system") or "system").strip().lower()
    return value if value in THEME_CHOICES else "system"


def set_preference(value: str) -> None:
    value = str(value).strip().lower()
    QSettings().setValue(THEME_SETTING, value if value in THEME_CHOICES else "system")


def _system_is_dark() -> bool:
    app = QGuiApplication.instance()
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        pal = app.palette() if app else QPalette()
        return pal.color(QPalette.Window).lightness() < 128


def is_dark() -> bool:
    pref = preference()
    if pref in ("dark", "light"):
        return pref == "dark"
    return _system_is_dark()


class T:
    dark = False
    bg = "#f2f2f4"; canvas = "#f7f7f8"; grid = "#cfcfd6"; dot_major = "#a8a8b2"; panel = "#ffffff"; border = "#d6d6db"; border_soft = "#e6e6ea"
    text = "#1c1c1e"; muted = "#6b6b73"; faint = "#a0a0a8"; node = "#ffffff"; node_border = "#c9c9cf"; select = "#2563eb"
    edge = "#9a9aa2"; edge_hover = "#2563eb"; accent = "#2563eb"; note = "#fff8dc"; note_text = "#5c4b00"
    hover = "#ebebef"; danger = "#dc2626"; ok = "#16a34a"; warn = "#d97706"
    ui_font = "sans-serif"; mono_font = "monospace"

    @classmethod
    def resolve(cls) -> None:
        cls.dark = is_dark()
        if cls.dark:
            cls.bg = "#1c1c1f"; cls.canvas = "#202024"; cls.grid = "#35353e"; cls.dot_major = "#4e4e5a"; cls.panel = "#26262b"; cls.border = "#3a3a42"; cls.border_soft = "#30303a"
            cls.text = "#ececef"; cls.muted = "#a1a1aa"; cls.faint = "#6f6f78"; cls.node = "#2a2a30"; cls.node_border = "#46464f"; cls.select = "#60a5fa"
            cls.edge = "#6b6b75"; cls.edge_hover = "#93c5fd"; cls.accent = "#3b82f6"; cls.note = "#3a341f"; cls.note_text = "#f3e2a4"
            cls.hover = "#31313a"; cls.danger = "#f87171"; cls.ok = "#4ade80"; cls.warn = "#fbbf24"
        else:
            cls.bg = "#f2f2f4"; cls.canvas = "#f7f7f8"; cls.grid = "#cfcfd6"; cls.dot_major = "#a8a8b2"; cls.panel = "#ffffff"; cls.border = "#d6d6db"; cls.border_soft = "#e6e6ea"
            cls.text = "#1c1c1e"; cls.muted = "#6b6b73"; cls.faint = "#a0a0a8"; cls.node = "#ffffff"; cls.node_border = "#c9c9cf"; cls.select = "#2563eb"
            cls.edge = "#9a9aa2"; cls.edge_hover = "#2563eb"; cls.accent = "#2563eb"; cls.note = "#fff8dc"; cls.note_text = "#5c4b00"
            cls.hover = "#ebebef"; cls.danger = "#dc2626"; cls.ok = "#16a34a"; cls.warn = "#d97706"
        families = set(QFontDatabase.families())
        cls.ui_font = next((f for f in UI_FONTS if f in families), "sans-serif")
        cls.mono_font = next((f for f in MONO_FONTS if f in families), "monospace")


def apply_app_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    T.resolve()
    f = QFont(T.ui_font, 10)
    f.setStyleHint(QFont.SansSerif)
    app.setFont(f)
    pal = QPalette()
    roles = [
        (QPalette.Window, T.bg), (QPalette.WindowText, T.text), (QPalette.Base, T.panel), (QPalette.AlternateBase, T.bg),
        (QPalette.Text, T.text), (QPalette.Button, T.panel), (QPalette.ButtonText, T.text), (QPalette.Highlight, T.accent),
        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, T.panel), (QPalette.ToolTipText, T.text),
        (QPalette.PlaceholderText, T.faint), (QPalette.Mid, T.border), (QPalette.Dark, T.border), (QPalette.Light, T.panel),
    ]
    for role, color in roles:
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(T.faint))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(T.faint))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(T.faint))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QWidget {{ font-family: "{T.ui_font}"; }}
        QToolTip {{ color: {T.text}; background: {T.panel}; border: 1px solid {T.border}; padding: 5px 7px; font-family: "{T.ui_font}"; font-size: 10pt; }}
        QMenuBar {{ background: {T.bg}; border-bottom: 1px solid {T.border}; }}
        QMenuBar::item:selected {{ background: {T.hover}; }}
        QMenu {{ background: {T.panel}; border: 1px solid {T.border}; padding: 4px; }}
        QMenu::item {{ padding: 5px 24px 5px 12px; }}
        QMenu::item:selected {{ background: {T.hover}; }}
        QMenu::separator {{ height: 1px; background: {T.border_soft}; margin: 4px 6px; }}
        QToolBar {{ background: {T.bg}; border: none; border-bottom: 1px solid {T.border}; padding: 3px 6px; spacing: 2px; }}
        QToolBar QToolButton {{ padding: 4px 8px; border: 1px solid transparent; border-radius: 4px; }}
        QToolBar QToolButton:hover {{ background: {T.hover}; }}
        QToolBar QToolButton:pressed {{ background: {T.border}; }}
        QToolBar QToolButton:disabled {{ color: {T.faint}; }}
        QToolBar::separator {{ width: 1px; background: {T.border}; margin: 4px 6px; }}
        QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
        QDockWidget::title {{ padding: 0; margin: 0; background: {T.bg}; text-align: left; }}
        QTabWidget::pane {{ border: none; border-top: 1px solid {T.border}; }}
        QTabBar::tab {{ padding: 6px 14px; border: none; color: {T.muted}; }}
        QTabBar::tab:selected {{ color: {T.text}; border-bottom: 2px solid {T.accent}; }}
        QTabBar::tab:hover {{ color: {T.text}; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
            border: 1px solid {T.border}; border-radius: 4px; padding: 4px 6px; background: {T.panel}; selection-background-color: {T.accent}; }}
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {T.accent}; }}
        QLineEdit:disabled, QComboBox:disabled {{ color: {T.faint}; background: {T.bg}; }}
        QComboBox {{ combobox-popup: 0; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{ background: {T.panel}; border: 1px solid {T.border}; selection-background-color: {T.hover}; selection-color: {T.text}; }}
        QPushButton {{ padding: 5px 12px; border: 1px solid {T.border}; border-radius: 4px; background: {T.panel}; }}
        QPushButton:hover {{ background: {T.hover}; }}
        QPushButton:pressed {{ background: {T.border}; }}
        QPushButton:disabled {{ color: {T.faint}; }}
        QPushButton#primary {{ background: {T.accent}; color: white; border: 1px solid {T.accent}; }}
        QPushButton#primary:hover {{ background: {'#2f6fe0' if not T.dark else '#4f8ff7'}; }}
        QPushButton#primary:disabled {{ background: {T.border}; color: {T.faint}; border-color: {T.border}; }}
        QPushButton#quiet, QToolButton#quiet {{ border: none; background: transparent; padding: 3px 6px; color: {T.muted}; }}
        QPushButton#quiet:hover, QToolButton#quiet:hover {{ background: {T.hover}; color: {T.text}; }}
        QHeaderView::section {{ padding: 4px 6px; border: none; border-right: 1px solid {T.border_soft}; border-bottom: 1px solid {T.border}; background: {T.bg}; color: {T.muted}; }}
        QTableView {{ gridline-color: {T.border_soft}; border: none; selection-background-color: {'#dbeafe' if not T.dark else '#1e3a5f'}; selection-color: {T.text}; }}
        QTableCornerButton::section {{ background: {T.bg}; border: none; }}
        QLabel#muted {{ color: {T.muted}; }}
        QLabel#faint {{ color: {T.faint}; }}
        QLabel#error {{ color: {T.danger}; }}
        QLabel#heading {{ font-weight: 600; font-size: 12pt; }}
        QLabel#section {{ color: {T.muted}; font-size: 8.5pt; font-weight: 600; letter-spacing: 0.4px; }}
        QFrame#panel {{ background: {T.panel}; border: none; }}
        QFrame#hline {{ background: {T.border}; max-height: 1px; min-height: 1px; border: none; }}
        QListWidget {{ border: none; background: {T.panel}; outline: none; }}
        QListWidget::item {{ padding: 5px 8px; }}
        QListWidget::item:selected {{ background: {T.hover}; color: {T.text}; }}
        QListWidget::item:hover {{ background: {T.hover}; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {T.border}; border-radius: 5px; min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {T.faint}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: {T.border}; border-radius: 5px; min-width: 24px; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        QSplitter::handle {{ background: {T.border}; }}
        QSplitter::handle:horizontal {{ width: 1px; }} QSplitter::handle:vertical {{ height: 1px; }}
        QStatusBar {{ border-top: 1px solid {T.border}; color: {T.muted}; }}
        QStatusBar::item {{ border: none; }}
        QProgressBar {{ border: none; background: {T.border_soft}; height: 4px; }}
        QProgressBar::chunk {{ background: {T.accent}; }}
        QCheckBox {{ spacing: 6px; }}
        QRadioButton {{ spacing: 6px; }}
        QGroupBox {{ border: 1px solid {T.border}; border-radius: 4px; margin-top: 8px; }}
    """)


def category_color(category: str) -> QColor:
    return QColor(CATEGORY_COLORS.get(category, "#64748b"))


class ThemeManager(QObject):
    """Keeps the palette in step with the OS. ``changed`` is emitted after a new palette is applied,
    so the window can rebuild its chrome (widgets capture T colours when they are built)."""

    changed = Signal()

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self.app = app
        try:
            app.styleHints().colorSchemeChanged.connect(self._on_system_changed)
        except Exception:
            pass

    def _on_system_changed(self, *_: object) -> None:
        if preference() == "system":
            self.apply()

    def apply(self) -> None:
        apply_app_style(self.app)
        self.changed.emit()

    def set_preference(self, value: str) -> None:
        set_preference(value)
        self.apply()


_manager: ThemeManager | None = None


def theme_manager(app: QApplication | None = None) -> ThemeManager | None:
    """The one ThemeManager for this process (created on first call that passes an app)."""
    global _manager
    if _manager is None and app is not None:
        _manager = ThemeManager(app)
    return _manager
