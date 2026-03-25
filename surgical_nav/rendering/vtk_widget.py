"""VTKWidget: thin wrapper around QVTKRenderWindowInteractor.

Falls back to a plain placeholder label when VTK is unavailable
(set SURGICAL_NAV_NO_VTK=1 to force the stub path).
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

try:
    import vtkmodules.all as vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    _VTK = True
except ImportError:
    _VTK = False


class VTKWidget(QWidget):
    """Embeds a VTK render window inside a Qt widget.

    Falls back to a plain placeholder when VTK is unavailable.
    """

    ready = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._initialized = False
        self._renderer = None
        self._render_window_interactor = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _VTK:
            self._render_window_interactor = QVTKRenderWindowInteractor(self)
            self._renderer = vtk.vtkRenderer()
            self._renderer.SetBackground(0.1, 0.1, 0.1)
            self._render_window_interactor.GetRenderWindow().AddRenderer(self._renderer)
            layout.addWidget(self._render_window_interactor)
        else:
            lbl = QLabel("3D View\n(VTK unavailable)", self)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #555; background: #111; font-size: 14px;")
            layout.addWidget(lbl)

    def initialize(self):
        if not _VTK or self._initialized:
            return
        self._render_window_interactor.Initialize()
        self._initialized = True
        self.ready.emit()

    def get_renderer(self):
        return self._renderer

    def get_render_window(self):
        if not _VTK:
            return None
        return self._render_window_interactor.GetRenderWindow()

    def get_interactor(self):
        return self._render_window_interactor

    def render(self):
        if _VTK and self._initialized:
            self._render_window_interactor.GetRenderWindow().Render()

    def reset_camera(self):
        if _VTK and self._renderer:
            self._renderer.ResetCamera()
            self.render()

    def set_background(self, r: float, g: float, b: float):
        if _VTK and self._renderer:
            self._renderer.SetBackground(r, g, b)
            self.render()

    def add_actor(self, actor):
        if _VTK and self._renderer:
            self._renderer.AddActor(actor)

    def remove_actor(self, actor):
        if _VTK and self._renderer:
            self._renderer.RemoveActor(actor)

    def showEvent(self, event):
        super().showEvent(event)
        self.initialize()

    def closeEvent(self, event):
        if _VTK and self._render_window_interactor:
            self._render_window_interactor.Finalize()
        super().closeEvent(event)
