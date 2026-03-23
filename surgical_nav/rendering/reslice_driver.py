"""ResliceDriver: drives MPR slice viewers from real-time tracker data.

Listens to tracking transforms, computes the pointer-tip position in image
space by walking the SceneGraph transform chain, and updates all three
SliceViewers at up to 20 Hz (one render every 50 ms via QTimer).

Transform chain (same as SlicerOpenNav)::

    tip_in_image = IMG_REG_INV @ POINTER_CAL @ PointerToTracker @ [0,0,0,1]

Optionally includes the ICP refinement::

    tip_in_image = IMG_REG_REFINEMENT_INV @ IMG_REG_INV @ POINTER_CAL @ PointerToTracker @ [0,0,0,1]

Usage::

    driver = ResliceDriver(scene_graph, axial, coronal, sagittal)
    tracker.transform_received.connect(driver.on_transform)
    # driver starts its own 50 ms QTimer on construction
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PySide6.QtCore import QTimer, QObject

from surgical_nav.app.scene_graph import SceneGraph
from surgical_nav.utils.math_utils import invert_transform, compose
from surgical_nav.rendering.slice_viewer import SliceViewer


class ResliceDriver(QObject):
    """Converts live tracker transforms into slice-viewer position updates.

    Parameters
    ----------
    scene_graph : SceneGraph
        Shared scene graph (reads POINTER_CALIBRATION, IMAGE_REGISTRATION,
        IMAGE_REGISTRATION_REFINEMENT transform nodes).
    slice_viewers : list[SliceViewer]
        All viewers whose slice position and crosshair should be updated.
    render_hz : float
        Maximum render rate (default 20 Hz = 50 ms timer).
    """

    def __init__(
        self,
        scene_graph: SceneGraph,
        slice_viewers: List[SliceViewer],
        render_hz: float = 20.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._sg      = scene_graph
        self._viewers = list(slice_viewers)
        self._frozen  = False

        self._latest_ptr: Optional[np.ndarray] = None   # PointerToTracker
        self._dirty = False

        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(1000 / render_hz)))
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_transform(self, name: str, matrix: np.ndarray) -> None:
        """Slot: receive a new tracking transform."""
        if name == "PointerToTracker":
            self._latest_ptr = np.asarray(matrix, dtype=np.float64)
            self._dirty = True

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze slice updates (pointer still tracked internally)."""
        self._frozen = frozen

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def tip_in_image(self) -> Optional[np.ndarray]:
        """Return the current pointer-tip position in image space, or None."""
        if self._latest_ptr is None:
            return None
        return self._compute_tip(self._latest_ptr)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Called by QTimer at render_hz; pushes latest tip to viewers."""
        if not self._dirty or self._frozen or self._latest_ptr is None:
            return
        self._dirty = False
        tip = self._compute_tip(self._latest_ptr)
        if tip is None:
            return
        x, y, z = float(tip[0]), float(tip[1]), float(tip[2])
        for viewer in self._viewers:
            viewer.set_slice_position(x, y, z)
            viewer.set_crosshair(x, y, z)

    def _compute_tip(self, ptr_to_tracker: np.ndarray) -> Optional[np.ndarray]:
        """Compute tip position in image coordinates.

        Returns (3,) world position or None if required transforms are missing.
        """
        # Step 1: apply pivot calibration to get tip in tracker space
        cal_node = self._sg.get_node("POINTER_CALIBRATION")
        if cal_node is None:
            # No calibration yet — use raw tracker origin as tip
            tip_tracker = ptr_to_tracker[:3, 3].copy()
        else:
            cal = cal_node.matrix
            tip_tracker = (ptr_to_tracker @ cal)[:3, 3]

        # Step 2: apply image registration (patient → image)
        reg_node = self._sg.get_node("IMAGE_REGISTRATION")
        if reg_node is None:
            return None   # registration required for navigation

        reg_inv = invert_transform(reg_node.matrix)

        # Step 3: optionally apply ICP refinement
        refine_node = self._sg.get_node("IMAGE_REGISTRATION_REFINEMENT")
        if refine_node is not None:
            refine_inv = invert_transform(refine_node.matrix)
            T = compose(refine_inv, reg_inv)
        else:
            T = reg_inv

        # Apply to tip position (homogeneous)
        tip_h = np.array([*tip_tracker, 1.0])
        result = T @ tip_h
        return result[:3]
