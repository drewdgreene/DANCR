"""A click anywhere on an editable combo opens its list (typing still filters it)."""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtWidgets import QComboBox


class OpenOnClick(QObject):
    """The list opens on the press: Wayland compositors only allow a popup to grab input with the serial of a
    button press, so opening on release makes the list flash and vanish. The release that follows would
    normally close a list-style popup again, so the one that lands right after our open is swallowed."""

    HOLD_MS = 600

    def __init__(self, combo: QComboBox) -> None:
        super().__init__(combo)
        self.combo = combo
        self._opened_at = 0.0
        self._container = None

    def _watch_container(self) -> None:
        cont = self.combo.view().parentWidget()
        if cont is not None and cont is not self._container:
            self._container = cont
            cont.installEventFilter(self)
            self.combo.view().installEventFilter(self)

    def eventFilter(self, obj, e) -> bool:
        t = e.type()
        if t not in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease) or e.button() != Qt.LeftButton:
            return False
        try:
            combo = self.combo
            view = combo.view()
            if obj is combo.lineEdit():
                if t == QEvent.MouseButtonPress and not view.isVisible():
                    obj.setFocus()
                    combo.showPopup()
                    self._opened_at = time.monotonic()
                    self._watch_container()
                    return True
                if t == QEvent.MouseButtonRelease:
                    self._opened_at = 0.0         # a release here must not swallow a later one
                return False
            if t == QEvent.MouseButtonRelease and time.monotonic() - self._opened_at < self.HOLD_MS / 1000:
                self._opened_at = 0.0
                if not view.rect().contains(view.mapFromGlobal(e.globalPosition().toPoint())):
                    return True                  # the release of the click that opened the list: keep it open
        except RuntimeError:                     # the combo is being destroyed
            return False
        return False


def open_list_on_click(combo: QComboBox) -> None:
    if combo.lineEdit() is not None:
        combo.lineEdit().installEventFilter(OpenOnClick(combo))
