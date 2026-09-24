"""A searchable popup for adding a step. Opened from the toolbar, the canvas, the + key,
or the + button on a selected step's output."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QPoint, QEvent, QSize, QRect
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QLabel, QApplication, QStyledItemDelegate, QStyle)


from ..core import registry
from .theme import T, category_color
from .icons import icon, node_icon_name

CATEGORY_ORDER = ["Get data", "Clean up", "Filter & sort", "Calculate", "Combine", "Time", "Analyse & model", "Share"]


class StepDelegate(QStyledItemDelegate):
    """Icon, bold title, muted wrapped description; category headers as small caps."""

    def sizeHint(self, option, index) -> QSize:
        if not index.data(Qt.UserRole):
            return QSize(option.rect.width(), 26)
        fm = QFontMetrics(option.font)
        width = max(120, option.rect.width() - 52)
        desc = index.data(Qt.UserRole + 1) or ""
        h = fm.boundingRect(QRect(0, 0, width, 1000), Qt.TextWordWrap, desc).height()
        return QSize(option.rect.width(), 26 + h + 8)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        r = option.rect
        key = index.data(Qt.UserRole)
        if not key:
            painter.setPen(QColor(T.muted))
            f = QFont(option.font); f.setBold(True); f.setPointSizeF(f.pointSizeF() - 1); painter.setFont(f)
            painter.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignLeft | Qt.AlignBottom, str(index.data(Qt.DisplayRole)).upper())
            painter.restore(); return
        if option.state & QStyle.State_Selected or option.state & QStyle.State_MouseOver:
            painter.fillRect(r, QColor(T.hover))
        icon_ = index.data(Qt.DecorationRole)
        if icon_ is not None:
            icon_.paint(painter, QRect(r.left() + 12, r.top() + 8, 20, 20))
        f = QFont(option.font); f.setBold(True); painter.setFont(f); painter.setPen(QColor(T.text))
        painter.drawText(QRect(r.left() + 44, r.top() + 6, r.width() - 52, 20), Qt.AlignLeft | Qt.AlignVCenter, str(index.data(Qt.DisplayRole)))
        painter.setFont(option.font); painter.setPen(QColor(T.muted))
        painter.drawText(QRect(r.left() + 44, r.top() + 26, r.width() - 52, r.height() - 30), Qt.TextWordWrap | Qt.AlignTop, index.data(Qt.UserRole + 1) or "")
        painter.restore()


class StepPicker(QFrame):
    chosen = Signal(str)          # node type key

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("picker")
        self.setStyleSheet(f"QFrame#picker {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 6px; }}"
                           f"QListWidget {{ background: {T.panel}; }}")
        self.setFixedSize(380, 460)
        lay = QVBoxLayout(self); lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)
        self.search = QLineEdit(); self.search.setPlaceholderText("Search steps — e.g. filter, average, chart")
        self.search.setClearButtonEnabled(True)
        self.list = QListWidget(); self.list.setUniformItemSizes(False)
        self.list.setSpacing(0); self.list.setItemDelegate(StepDelegate(self.list)); self.list.setMouseTracking(True)
        self.list.setStyleSheet(f"QListWidget {{ background: {T.panel}; }} QListWidget::item {{ padding: 0; }} QListWidget::item:selected {{ background: transparent; }} QListWidget::item:hover {{ background: transparent; }}")
        self.hint = QLabel("Enter to add · Esc to close"); self.hint.setObjectName("faint")
        lay.addWidget(self.search); lay.addWidget(self.list, 1); lay.addWidget(self.hint)
        self.search.textChanged.connect(self.refill)
        self.search.installEventFilter(self)
        self.list.itemActivated.connect(self._pick)
        self.list.itemClicked.connect(self._pick)
        self.refill()

    def open_at(self, global_pos: QPoint) -> None:
        screen = QApplication.screenAt(global_pos) or QApplication.primaryScreen()
        g = screen.availableGeometry()
        x = min(max(global_pos.x(), g.left()), g.right() - self.width())
        y = min(max(global_pos.y(), g.top()), g.bottom() - self.height())
        self.move(x, y)
        self.search.clear()
        self.show()
        self.search.setFocus()

    def refill(self) -> None:
        q = self.search.text().strip().lower()
        self.list.clear()
        cats = registry.by_category()
        first: QListWidgetItem | None = None
        for cat in CATEGORY_ORDER + [c for c in cats if c not in CATEGORY_ORDER]:
            types = [t for t in cats.get(cat, []) if not q or q in t.label.lower() or q in t.description.lower() or q in t.key]
            if not types:
                continue
            head = QListWidgetItem(cat)
            head.setFlags(Qt.NoItemFlags)
            self.list.addItem(head)
            for t in types:
                it = QListWidgetItem(icon(node_icon_name(t.key), category_color(t.category).name(), 20), t.label)
                it.setData(Qt.UserRole, t.key)
                it.setData(Qt.UserRole + 1, t.description)
                self.list.addItem(it)
                if first is None:
                    first = it
        if first is not None:
            self.list.setCurrentItem(first)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.search and ev.type() == QEvent.KeyPress:
            key = ev.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                row = self.list.currentRow()
                step = 1 if key == Qt.Key_Down else -1
                n = self.list.count()
                if row < 0:                     # no current row: Down starts at the top, Up at the bottom
                    row = -1 if step > 0 else 0
                for _ in range(n):
                    row = (row + step) % n
                    if self.list.item(row).flags() & Qt.ItemIsSelectable:
                        self.list.setCurrentRow(row); break
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                it = self.list.currentItem()
                if it and it.data(Qt.UserRole):
                    self._pick(it)
                return True
        return super().eventFilter(obj, ev)

    def _pick(self, it: QListWidgetItem) -> None:
        k = it.data(Qt.UserRole)
        if k:
            self.hide()
            self.chosen.emit(k)
