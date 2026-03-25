"""VTKWidget: placeholder 3-D view widget (VTK removed).

Shows a static label in place of a 3-D render window.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class VTKWidget(QWidget):
    """Placeholder widget shown where the VTK render window would appear."""

    ready = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._renderer = None
        self._render_window_interactor = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("3D View\n(VTK unavailable)", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #555; background: #111; font-size: 14px;")
        layout.addWidget(lbl)

    def initialize(self): pass
    def get_renderer(self): return None
    def get_render_window(self): return None
    def get_interactor(self): return None
    def render(self): pass
    def reset_camera(self): pass
    def set_background(self, r: float, g: float, b: float): pass
    def add_actor(self, actor): pass
    def remove_actor(self, actor): pass

    def showEvent(self, event):
        super().showEvent(event)

    def closeEvent(self, event):
        super().closeEvent(event)
