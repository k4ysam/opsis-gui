"""VolumeViewer: 3-D volume renderer stub.

Displays a placeholder until a volume is loaded. VTK is loaded lazily only
when set_volume() is called — so the app launches without any VTK
initialization at startup.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PySide6.QtCore import Qt


class VolumeViewer(QWidget):
    """3-D volume renderer with surface overlay and pointer actor.

    VTK is only initialised when actual volume data is provided via
    set_volume().  Until then a placeholder label is shown so the app
    starts without any VTK/OpenGL work.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._stack = QStackedWidget(self)

        # Placeholder (shown until a volume is loaded)
        self._placeholder = QLabel("No volume loaded", self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #555; font-size: 14px;")
        self._stack.addWidget(self._placeholder)   # index 0

        # VTK viewer slot (added lazily)
        self._vtk_widget   = None
        self._vtk_viewer   = None    # the vtkRenderer-level viewer
        self._vtk_ready    = False

        # Pointer / trajectory state (buffered until VTK is ready)
        self._buffered_pointer_matrix: Optional[np.ndarray] = None
        self._pointer_status: str = "NEVER_SEEN"

        # Model actors deferred until VTK is ready
        self._pending_surfaces: list = []
        self._model_actors: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # Public API — volume
    # ------------------------------------------------------------------

    def set_volume(self, vtk_image_data):
        """Load a vtkImageData into the volume mapper and reset camera."""
        if vtk_image_data is None:
            return
        self._ensure_vtk()
        self._vtk_viewer.set_volume(vtk_image_data)
        self._stack.setCurrentIndex(1)

        # Flush deferred operations
        for poly_data, color, opacity in self._pending_surfaces:
            self._vtk_viewer.add_surface(poly_data, color, opacity)
        self._pending_surfaces.clear()

        if self._buffered_pointer_matrix is not None:
            self._vtk_viewer.set_pointer_transform(self._buffered_pointer_matrix)
            self._buffered_pointer_matrix = None

    def clear_volume(self):
        if self._vtk_viewer is not None:
            self._vtk_viewer.clear_volume()
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Public API — surface models
    # ------------------------------------------------------------------

    def add_surface(self, poly_data, color=(0.9, 0.75, 0.65), opacity: float = 0.6):
        if self._vtk_ready:
            self._vtk_viewer.add_surface(poly_data, color, opacity)
        else:
            self._pending_surfaces.append((poly_data, color, opacity))

    def add_model(self, model_node):
        if self._vtk_ready:
            self._vtk_viewer.add_model(model_node)

    def remove_model(self, node_id: str):
        if self._vtk_ready:
            self._vtk_viewer.remove_model(node_id)

    def update_model_visibility(self, node_id: str, visible: bool):
        if self._vtk_ready:
            self._vtk_viewer.update_model_visibility(node_id, visible)

    # ------------------------------------------------------------------
    # Public API — pointer / needle
    # ------------------------------------------------------------------

    def set_pointer_transform(self, matrix_4x4: np.ndarray):
        if self._vtk_ready:
            self._vtk_viewer.set_pointer_transform(matrix_4x4)
        else:
            self._buffered_pointer_matrix = matrix_4x4

    def set_pointer_status(self, status: str):
        self._pointer_status = status
        if self._vtk_ready:
            self._vtk_viewer.set_pointer_status(status)

    def hide_pointer(self):
        if self._vtk_ready:
            self._vtk_viewer.hide_pointer()

    def set_trajectory(self, entry: np.ndarray, target: np.ndarray):
        if self._vtk_ready:
            self._vtk_viewer.set_trajectory(entry, target)

    # ------------------------------------------------------------------
    # Public API — camera / misc
    # ------------------------------------------------------------------

    def reset_camera(self):
        if self._vtk_ready:
            self._vtk_viewer.reset_camera()

    def render(self):
        if self._vtk_ready:
            self._vtk_viewer.render()

    def initialize(self):
        # Called by main_window._initialize_vtk_children — no-op until VTK loaded
        pass

    def get_renderer(self):
        if self._vtk_ready:
            return self._vtk_viewer.get_renderer()
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_vtk(self):
        """Lazily create the real VTK viewer on first use."""
        if self._vtk_ready:
            return
        try:
            from surgical_nav.rendering._vtk_volume_viewer import _VTKVolumeViewer
            self._vtk_viewer = _VTKVolumeViewer()
            self._stack.addWidget(self._vtk_viewer)   # index 1
            self._vtk_ready = True
        except Exception as exc:
            print(f"[VolumeViewer] VTK unavailable: {exc}")
