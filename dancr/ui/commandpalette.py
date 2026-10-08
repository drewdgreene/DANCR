"""A searchable palette for everything the window can do.

Opened with Ctrl+Shift+P from anywhere. It lists the menu actions, then the project's own steps, answers,
recent files, templates and examples, filtered as you type. Enter runs the highlighted one. It is the keyboard
answer to a mouse-heavy canvas: no menu hunting, no remembering shortcuts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, QEvent, QSize, QRect
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (QFrame, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyledItemDelegate, QStyle,
                               QVBoxLayout, QWidget)

from .theme import T


@dataclass
class Command:
    group: str
    label: str
    shortcut: str
    run: Callable[[], None]


class _Delegate(QStyledItemDelegate):
    """Small-caps group header; then a bold label with its shortcut right-aligned."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 26 if not index.data(Qt.UserRole) else 30)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        r = option.rect
        if not index.data(Qt.UserRole):
            painter.setPen(QColor(T.muted))
            f = QFont(option.font); f.setBold(True); f.setPointSizeF(f.pointSizeF() - 1); painter.setFont(f)
            painter.drawText(r.adjusted(10, 0, -10, 0), Qt.AlignLeft | Qt.AlignBottom, str(index.data(Qt.DisplayRole)).upper())
            painter.restore(); return
        if option.state & (QStyle.State_Selected | QStyle.State_MouseOver):
            painter.fillRect(r, QColor(T.hover))
        painter.setFont(option.font); painter.setPen(QColor(T.text))
        painter.drawText(QRect(r.left() + 12, r.top(), r.width() - 140, r.height()), Qt.AlignLeft | Qt.AlignVCenter,
                         str(index.data(Qt.DisplayRole)))
        shortcut = index.data(Qt.UserRole + 1) or ""
        if shortcut:
            painter.setPen(QColor(T.faint)); painter.drawText(r.adjusted(12, 0, -12, 0), Qt.AlignRight | Qt.AlignVCenter, shortcut)
        painter.restore()


class CommandPalette(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("palette")
        self.setStyleSheet(f"QFrame#palette {{ background: {T.panel}; border: 1px solid {T.border}; border-radius: 8px; }}"
                           f"QListWidget {{ background: {T.panel}; }}")
        self.setFixedSize(520, 440)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 10, 10, 8); lay.setSpacing(6)
        self.search = QLineEdit(); self.search.setPlaceholderText("Type a command, a step, an answer or a file…")
        self.search.setClearButtonEnabled(True)
        self.list = QListWidget(); self.list.setUniformItemSizes(False); self.list.setItemDelegate(_Delegate(self.list))
        self.list.setMouseTracking(True)
        self.list.setStyleSheet(f"QListWidget {{ background: {T.panel}; }} QListWidget::item {{ padding: 0; }} "
                                f"QListWidget::item:selected, QListWidget::item:hover {{ background: transparent; }}")
        hint = QLabel("Enter to run · Esc to close"); hint.setObjectName("faint")
        lay.addWidget(self.search); lay.addWidget(self.list, 1); lay.addWidget(hint)
        self._commands: list[Command] = []
        self.search.textChanged.connect(self._refill)
        self.search.installEventFilter(self)
        self.list.itemActivated.connect(self._run)
        self.list.itemClicked.connect(self._run)

    def open_center(self, commands: list[Command]) -> None:
        self._commands = commands
        self.search.clear()
        self._refill("")
        parent = self.parentWidget()
        if parent is not None:
            g = parent.window().geometry()
            self.move(max(0, g.center().x() - self.width() // 2), max(0, g.top() + 90))
        self.show()
        self.search.setFocus()

    def _refill(self, query: str) -> None:
        q = query.strip().lower()
        self.list.clear()
        first: QListWidgetItem | None = None
        last_group = None
        for i, c in enumerate(self._commands):
            if q and q not in c.label.lower() and q not in c.group.lower():
                continue
            if c.group != last_group:
                head = QListWidgetItem(c.group); head.setFlags(Qt.NoItemFlags)
                self.list.addItem(head); last_group = c.group
            it = QListWidgetItem(c.label)
            it.setData(Qt.UserRole, True); it.setData(Qt.UserRole + 1, c.shortcut); it.setData(Qt.UserRole + 2, i)
            self.list.addItem(it)
            if first is None:
                first = it
        if first is not None:
            self.list.setCurrentItem(first)

    def _run(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.UserRole + 2)
        if index is None:
            return
        self.hide()
        self._commands[index].run()

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.search and ev.type() == QEvent.KeyPress:
            key = ev.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                row = self.list.currentRow(); step = 1 if key == Qt.Key_Down else -1
                n = self.list.count()
                for _ in range(n):
                    row = (row + step) % n
                    if self.list.item(row).data(Qt.UserRole):
                        self.list.setCurrentRow(row); break
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                it = self.list.currentItem()
                if it and it.data(Qt.UserRole):
                    self._run(it)
                return True
            if key == Qt.Key_Escape:
                self.hide(); return True
        return super().eventFilter(obj, ev)
