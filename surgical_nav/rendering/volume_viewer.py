"""VolumeViewer: 3-D viewer placeholder (VTK removed).

All rendering methods are no-ops; the widget shows the VTKWidget placeholder.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

from surgical_nav.rendering.vtk_widget import VTKWidget
from surgical_nav.app.scene_graph import ModelNode


class VolumeViewer(QWidget):
    """3-D volume viewer stub — renders nothing without VTK."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._vtk_widget = VTKWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)
        self._model_actors: Dict = {}

    def set_volume(self, vtk_image_data): pass
    def clear_volume(self): pass
    def add_model(self, model_node: ModelNode): pass
    def remove_model(self, node_id: str): pass
    def update_model_visibility(self, node_id: str, visible: bool): pass
    def add_surface(self, poly_data, color=(0.9, 0.75, 0.65), opacity: float = 0.6): pass
    def set_pointer_transform(self, matrix_4x4: np.ndarray): pass
    def set_pointer_status(self, status: str): pass
    def hide_pointer(self): pass
    def set_trajectory(self, entry: np.ndarray, target: np.ndarray): pass
    def reset_camera(self): pass
    def render(self): pass
    def initialize(self): pass
    def get_renderer(self): return None

    def showEvent(self, event):
        super().showEvent(event)

    def closeEvent(self, event):
        super().closeEvent(event)
