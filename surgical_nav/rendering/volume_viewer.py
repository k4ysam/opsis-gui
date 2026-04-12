"""VolumeViewer: lightweight placeholder — no VTK required.

The 3-D volume renderer has been replaced with a simple dark panel so the
application starts quickly on any machine. All public API methods are no-ops
so the rest of the app wires up identically.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class VolumeViewer(QWidget):
    """Placeholder 3-D viewer — accepts all VolumeViewer API calls as no-ops."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("3D View")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "background: #111; color: #333; font-size: 14px; font-style: italic;"
        )
        layout.addWidget(lbl)

    # ------------------------------------------------------------------
    # Public API — all no-ops so callers need no changes
    # ------------------------------------------------------------------

    def set_volume(self, vtk_image_data) -> None:
        pass

    def clear_volume(self) -> None:
        pass

    def add_model(self, model_node) -> None:
        pass

    def remove_model(self, node_id: str) -> None:
        pass

    def update_model_visibility(self, node_id: str, visible: bool) -> None:
        pass

    def add_surface(self, poly_data, color=(0.9, 0.75, 0.65), opacity: float = 0.6) -> None:
        pass

    def set_pointer_transform(self, matrix_4x4: np.ndarray) -> None:
        pass

    def set_pointer_status(self, status: str) -> None:
        pass

    def hide_pointer(self) -> None:
        pass

    def set_trajectory(self, entry: np.ndarray, target: np.ndarray) -> None:
        pass

    def reset_camera(self) -> None:
        pass

    def render(self) -> None:
        pass

    def initialize(self) -> None:
        pass

    def get_renderer(self):
        return None
