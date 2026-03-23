"""NavigationPage: Stage 4 — real-time guided needle navigation.

Displays the 6-up viewer layout (3D + 3 MPR slices) and drives slice
positions from live tracker data via ResliceDriver.

Features
--------
- Distance-to-target label (updates at 10 Hz)
- Freeze toggle (pauses slice scrolling without dropping tracker data)
- Trajectory line in 3D view (Entry → Target from TRAJECTORY_POINTS)
- Pointer/needle actor coloured by tool status
- Pointer-tip depth gauge along trajectory axis
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.app.scene_graph import SceneGraph, FiducialSetNode
from surgical_nav.rendering.reslice_driver import ResliceDriver
from surgical_nav.rendering.slice_viewer import SliceViewer
from surgical_nav.rendering.volume_viewer import VolumeViewer


class NavigationPage(WorkflowPage):
    """Stage 4: real-time navigation with MPR slice updates."""

    def __init__(
        self,
        slice_viewers: List[SliceViewer],
        volume_viewer: VolumeViewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("Navigation", parent)
        self._slice_viewers  = slice_viewers
        self._volume_viewer  = volume_viewer

        # ResliceDriver drives all slice viewers
        self._reslice_driver = ResliceDriver(
            SceneGraph.instance(), slice_viewers, render_hz=20.0
        )

        # Distance-to-target polling timer
        self._dist_timer = QTimer(self)
        self._dist_timer.setInterval(100)   # 10 Hz
        self._dist_timer.timeout.connect(self._update_distance)

        self._build_ui()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        self._dist_timer.start()
        self._refresh_trajectory()

    def on_leave(self):
        self._dist_timer.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_transform(self, name: str, matrix: np.ndarray):
        """Forward tracking transform to the ResliceDriver."""
        self._reslice_driver.on_transform(name, matrix)

    def set_pointer_status(self, status: str):
        """Update 3D needle actor colour: 'SEEN'|'NOT_SEEN'|'NEVER_SEEN'."""
        self._volume_viewer.set_pointer_status(status)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout

        # Status panel
        status_box = QGroupBox("Navigation Status")
        status_layout = QHBoxLayout(status_box)

        self._dist_lbl = QLabel("Distance to target: — mm")
        self._dist_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self._dist_lbl)

        status_layout.addStretch()

        self._freeze_btn = QPushButton("Freeze")
        self._freeze_btn.setCheckable(True)
        self._freeze_btn.setFixedWidth(80)
        self._freeze_btn.toggled.connect(self._on_freeze_toggled)
        status_layout.addWidget(self._freeze_btn)

        layout.addWidget(status_box)

        # Depth gauge
        depth_box = QGroupBox("Trajectory Depth")
        depth_layout = QHBoxLayout(depth_box)
        self._depth_lbl = QLabel("Depth along trajectory: — mm")
        depth_layout.addWidget(self._depth_lbl)
        layout.addWidget(depth_box)

        layout.addStretch()

    def _on_freeze_toggled(self, frozen: bool):
        self._reslice_driver.set_frozen(frozen)
        self._freeze_btn.setText("Unfreeze" if frozen else "Freeze")

    # ------------------------------------------------------------------
    # Distance and depth updates
    # ------------------------------------------------------------------

    def _update_distance(self):
        tip = self._reslice_driver.tip_in_image()
        if tip is None:
            self._dist_lbl.setText("Distance to target: — mm")
            self._depth_lbl.setText("Depth along trajectory: — mm")
            return

        target_pt = self._get_target_position()
        if target_pt is not None:
            dist = float(np.linalg.norm(tip - target_pt))
            self._dist_lbl.setText(f"Distance to target: {dist:.1f} mm")

            # Depth along trajectory axis
            entry_pt = self._get_entry_position()
            if entry_pt is not None:
                axis = target_pt - entry_pt
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    axis_hat = axis / axis_len
                    depth = float(np.dot(tip - entry_pt, axis_hat))
                    self._depth_lbl.setText(
                        f"Depth along trajectory: {depth:.1f} mm "
                        f"/ {axis_len:.1f} mm total"
                    )
        else:
            self._dist_lbl.setText("Distance to target: — mm (no target planned)")

    def _get_target_position(self) -> Optional[np.ndarray]:
        node = SceneGraph.instance().get_node("TRAJECTORY_POINTS")
        if not isinstance(node, FiducialSetNode):
            return None
        for p in node.points:
            if p.get("label") == "Target":
                return np.asarray(p["position"], dtype=np.float64)
        return None

    def _get_entry_position(self) -> Optional[np.ndarray]:
        node = SceneGraph.instance().get_node("TRAJECTORY_POINTS")
        if not isinstance(node, FiducialSetNode):
            return None
        for p in node.points:
            if p.get("label") == "Entry":
                return np.asarray(p["position"], dtype=np.float64)
        return None

    def _refresh_trajectory(self):
        """Push trajectory line to VolumeViewer if entry + target exist."""
        entry  = self._get_entry_position()
        target = self._get_target_position()
        if entry is not None and target is not None:
            self._volume_viewer.set_trajectory(entry, target)
