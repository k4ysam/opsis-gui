"""SliceViewer: single-plane MPR view placeholder (VTK removed).

Shows a labelled placeholder for each anatomical plane.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

# Plane definitions: (normal, view-up, label)
_PLANE_CONFIGS = {
    "axial":    {"normal": (0, 0, 1), "view_up": (0, 1, 0), "label": "Axial"},
    "coronal":  {"normal": (0, 1, 0), "view_up": (0, 0, 1), "label": "Coronal"},
    "sagittal": {"normal": (1, 0, 0), "view_up": (0, 0, 1), "label": "Sagittal"},
}


class SliceViewer(QWidget):
    """2-D slice viewer placeholder for one anatomical plane."""

    def __init__(self, plane: str = "axial", parent: Optional[QWidget] = None):
        super().__init__(parent)
        if plane not in _PLANE_CONFIGS:
            raise ValueError(f"plane must be one of {list(_PLANE_CONFIGS)}")

        cfg = _PLANE_CONFIGS[plane]
        self._plane = plane
        self._normal: Tuple[float, float, float] = cfg["normal"]
        self._view_up: Tuple[float, float, float] = cfg["view_up"]
        self._initialized = False
        self._vtk_image = None
        self._renderer = None
        self._interactor = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel(f"{cfg['label']}\n(VTK unavailable)", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #555; background: #000; font-size: 13px;")
        layout.addWidget(lbl)

    def initialize(self): pass
    def set_volume(self, vtk_image_data): pass
    def set_window_level(self, window: float, level: float): pass
    def set_slice_position(self, world_x: float, world_y: float, world_z: float): pass
    def set_crosshair(self, world_x: float, world_y: float, world_z: float): pass
    def render(self): pass
    def get_renderer(self): return None
    def get_interactor(self): return None

    def showEvent(self, event):
        super().showEvent(event)
        self.initialize()

    def closeEvent(self, event):
        super().closeEvent(event)
