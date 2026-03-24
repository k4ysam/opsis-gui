"""LayoutManager: switches the central rendering area between 2-up and 6-up layouts.

Layouts
-------
2-up  : 3D volume viewer (left, 60%) + one slice viewer (right, 40%)
6-up  : 3D viewer (top-left large) + axial + coronal + sagittal in a 2×2 grid
single: only the 3D viewer (used during planning / segmentation steps)

The manager owns the container QWidget and reparents the viewer widgets on
each layout switch without recreating them.

Usage::

    lm = LayoutManager(container_widget)
    lm.set_viewers(volume_viewer, axial, coronal, sagittal)
    lm.set_layout("2up")     # or "6up" / "single"
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QSplitter, QGridLayout, QStackedWidget, QSizePolicy,
)
from PySide6.QtCore import Qt


class LayoutManager:
    """Manages switching between viewer layouts inside a container widget."""

    LAYOUTS = ("single", "2up", "6up")

    def __init__(self, container: QWidget):
        self._container = container
        self._volume_viewer: Optional[QWidget] = None
        self._axial:    Optional[QWidget] = None
        self._coronal:  Optional[QWidget] = None
        self._sagittal: Optional[QWidget] = None
        self._current_layout: Optional[str] = None

        # The container needs a layout to hold whatever we put in it
        from PySide6.QtWidgets import QVBoxLayout
        if container.layout() is None:
            outer = QVBoxLayout(container)
            outer.setContentsMargins(0, 0, 0, 0)
        self._outer_layout = container.layout()

        # We swap a single child widget inside the outer layout
        self._current_widget: Optional[QWidget] = None

    # ------------------------------------------------------------------
    # Viewer registration
    # ------------------------------------------------------------------

    def set_viewers(
        self,
        volume_viewer: QWidget,
        axial: QWidget,
        coronal: QWidget,
        sagittal: QWidget,
    ):
        self._volume_viewer = volume_viewer
        self._axial    = axial
        self._coronal  = coronal
        self._sagittal = sagittal

    # ------------------------------------------------------------------
    # Layout switching
    # ------------------------------------------------------------------

    def set_layout(self, layout: str):
        if layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}")
        if layout == self._current_layout:
            return

        # Remove current child from container
        if self._current_widget is not None:
            self._outer_layout.removeWidget(self._current_widget)
            self._current_widget.setParent(None)   # detach without deleting

        builder = {
            "single": self._build_single,
            "2up":    self._build_2up,
            "6up":    self._build_6up,
        }[layout]

        new_widget = builder()
        self._outer_layout.addWidget(new_widget)
        new_widget.show()
        self._current_widget = new_widget
        self._current_layout = layout

    def current_layout(self) -> Optional[str]:
        return self._current_layout

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_single(self) -> QWidget:
        """One full-size viewer.

        Prefer a 2-D slice view for single-panel layouts because the 3-D
        volume renderer is substantially heavier on some macOS/VTK setups.
        """
        wrapper = QWidget()
        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        single_view = self._axial if self._axial is not None else self._volume_viewer
        layout.addWidget(single_view)
        single_view.show()
        return wrapper

    def _build_2up(self) -> QSplitter:
        """3D viewer (left 60%) | one slice viewer (right 40%)."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._volume_viewer)
        splitter.addWidget(self._axial)
        self._volume_viewer.show()
        self._axial.show()
        # Initial size ratio ~60/40
        splitter.setSizes([600, 400])
        return splitter

    def _build_6up(self) -> QWidget:
        """2×2 grid: 3D (top-left), axial (top-right), coronal (btm-left), sagittal (btm-right)."""
        wrapper = QWidget()
        grid = QGridLayout(wrapper)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        grid.addWidget(self._volume_viewer, 0, 0)
        grid.addWidget(self._axial,         0, 1)
        grid.addWidget(self._coronal,       1, 0)
        grid.addWidget(self._sagittal,      1, 1)

        # Equal columns and rows
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        for v in (self._volume_viewer, self._axial, self._coronal, self._sagittal):
            if v is not None:
                v.show()
        return wrapper
