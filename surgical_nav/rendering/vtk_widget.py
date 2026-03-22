"""VTKWidget: thin wrapper around QVTKRenderWindowInteractor.

Windows quirk: Initialize() must be called *after* the widget is shown.
Callers should connect to the ``ready`` signal or call ``initialize()``
inside the parent's ``showEvent()``.

Usage::

    widget = VTKWidget(parent)
    layout.addWidget(widget)
    # ... later, after show() ...
    widget.initialize()
    renderer = widget.get_renderer()
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VTKWidget(QWidget):
    """Embeds a VTK render window inside a Qt widget.

    Attributes
    ----------
    ready : Signal
        Emitted once after ``initialize()`` succeeds.
    """

    ready = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._initialized = False

        # Core VTK objects
        self._render_window_interactor = QVTKRenderWindowInteractor(self)
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.1)

        render_window = self._render_window_interactor.GetRenderWindow()
        render_window.AddRenderer(self._renderer)

        # Layout — fill the widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._render_window_interactor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Initialize the render window interactor.

        Must be called after the widget has been shown (Windows requirement).
        Safe to call multiple times.
        """
        if self._initialized:
            return
        self._render_window_interactor.Initialize()
        self._initialized = True
        self.ready.emit()

    def get_renderer(self) -> vtk.vtkRenderer:
        return self._renderer

    def get_render_window(self) -> vtk.vtkRenderWindow:
        return self._render_window_interactor.GetRenderWindow()

    def get_interactor(self) -> QVTKRenderWindowInteractor:
        return self._render_window_interactor

    def render(self):
        """Trigger a render. Safe to call before initialize(); no-ops then."""
        if self._initialized:
            self._render_window_interactor.GetRenderWindow().Render()

    def reset_camera(self):
        self._renderer.ResetCamera()
        self.render()

    def set_background(self, r: float, g: float, b: float):
        self._renderer.SetBackground(r, g, b)
        self.render()

    def add_actor(self, actor: vtk.vtkProp):
        self._renderer.AddActor(actor)

    def remove_actor(self, actor: vtk.vtkProp):
        self._renderer.RemoveActor(actor)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.initialize()

    def closeEvent(self, event):
        self._render_window_interactor.Finalize()
        super().closeEvent(event)
