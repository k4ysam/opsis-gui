"""PlanningPage: Stage 2 — segmentation, trajectory, and landmark planning.

Four sequential steps:
  Step 1 — Skin segmentation (HU threshold → surface mesh)
  Step 2 — Target segmentation (paint brush + connected threshold seed)
  Step 3 — Trajectory definition (entry + target fiducial points)
  Step 4 — Anatomical landmark placement (≥3 points required)

All results are pushed to the shared SceneGraph so downstream stages
(Registration, Navigation) can access them by node ID.
"""

from __future__ import annotations

from typing import Optional, List
import time

import numpy as np
import SimpleITK as sitk

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QTimer

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.app.scene_graph import (
    SceneGraph, ModelNode, FiducialSetNode, VolumeNode
)
from surgical_nav.segmentation.threshold_segmenter import ThresholdSegmenter
from surgical_nav.segmentation.surface_extractor import SurfaceExtractor


# ---------------------------------------------------------------------------
# Background worker for segmentation (keeps UI responsive)
# ---------------------------------------------------------------------------

class _SegWorker(QObject):
    finished = Signal(object, object)   # (sitk.Image label, vtkPolyData mesh)
    error    = Signal(str)

    def __init__(self, sitk_image, lower_hu, upper_hu, closing_radius):
        super().__init__()
        self._img   = sitk_image
        self._lower = lower_hu
        self._upper = upper_hu
        self._rad   = closing_radius

    def run(self):
        try:
            t0 = time.perf_counter()
            seg  = ThresholdSegmenter()
            label = seg.segment_skin(self._img, self._lower, self._upper,
                                     self._rad, shrink_factor=2)
            t1 = time.perf_counter()
            mesh  = SurfaceExtractor(smooth_iterations=8).extract(label)
            t2 = time.perf_counter()
            print(
                f"[timing] planning skin pipeline: segmentation={t1 - t0:.2f}s "
                f"mesh={t2 - t1:.2f}s total={t2 - t0:.2f}s",
                flush=True,
            )
            self.finished.emit(label, mesh)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# PlanningPage
# ---------------------------------------------------------------------------

class PlanningPage(WorkflowPage):
    """Stage 2: segmentation, trajectory, and landmark planning."""

    # Emitted when skin mesh or target label changes (for VolumeViewer)
    skin_mesh_ready   = Signal(object)   # vtkPolyData
    target_mesh_ready = Signal(object)   # vtkPolyData
    target_label_updated = Signal(object)    # sitk.Image
    seed_label_updated = Signal(object)      # np.ndarray labels: 1 inside, 2 outside
    target_preview_updated = Signal(object)  # np.ndarray
    trajectory_updated = Signal(object, object)  # entry_ras, target_ras
    landmarks_updated = Signal(object)      # list[np.ndarray]
    target_outline_updated = Signal(object, object)  # plane, outline points
    interaction_mode_changed = Signal(str)

    # Emitted when the user activates/deactivates a point-placement mode.
    # Value: "entry" | "target" | "landmark" | "" (none)
    interaction_mode_changed = Signal(str)

    # Emitted after entry/target points change so slice viewers can redraw.
    # Arguments: entry_xyz (np.ndarray or None), target_xyz (np.ndarray or None)
    trajectory_points_updated = Signal(object, object)

    # Emitted whenever the landmark list changes (list of np.ndarray world coords)
    landmarks_updated = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Planning", parent, show_back=True)
        self._step = 1          # current active step (1–4)
        self._seg_thread: Optional[QThread] = None

        # Trajectory point cache (world coords)
        self._entry_xyz:  Optional[np.ndarray] = None
        self._target_xyz: Optional[np.ndarray] = None

        # Cached data
        self._sitk_image:   Optional[sitk.Image]      = None
        self._vtk_image:    Optional[vtk.vtkImageData] = None
        self._skin_label:   Optional[sitk.Image]      = None
        self._target_label: Optional[sitk.Image]      = None
        self._target_array: Optional[np.ndarray]      = None
        self._seed_array: Optional[np.ndarray]        = None
        self._target_preview_array: Optional[np.ndarray] = None
        self._target_mesh_timer = QTimer(self)
        self._target_mesh_timer.setSingleShot(True)
        self._target_mesh_timer.setInterval(120)
        self._target_mesh_timer.timeout.connect(self._rebuild_target_mesh)
        self._target_overlay_timer = QTimer(self)
        self._target_overlay_timer.setSingleShot(True)
        self._target_overlay_timer.setInterval(33)
        self._target_overlay_timer.timeout.connect(self._emit_step2_overlays)

        self._build_ui()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        """Refresh volume reference from SceneGraph when page becomes active."""
        node = self._scene_graph.get_node("ACTIVE_VOLUME")
        if isinstance(node, VolumeNode):
            self._sitk_image = node.sitk_image
            self._vtk_image  = node.vtk_image_data
        self._last_paint_ijk = None
        if self._sitk_image is not None and self._target_label is None:
            self._target_array = None
        self._seed_array = None
        self._target_preview_array = None
        self._update_step_visibility()
        self._emit_interaction_mode()

    # ------------------------------------------------------------------
    # Step 1 — Skin Segmentation
    # ------------------------------------------------------------------

    def _build_step1(self) -> QGroupBox:
        box = QGroupBox("Step 1 — Skin Segmentation")
        layout = QVBoxLayout(box)

        hu_row = QHBoxLayout()
        hu_row.addWidget(QLabel("Lower HU:"))
        self._lower_spin = QSpinBox()
        self._lower_spin.setRange(-2000, 2000)
        self._lower_spin.setValue(-200)
        hu_row.addWidget(self._lower_spin)

        hu_row.addWidget(QLabel("Upper HU:"))
        self._upper_spin = QSpinBox()
        self._upper_spin.setRange(-2000, 4000)
        self._upper_spin.setValue(500)
        hu_row.addWidget(self._upper_spin)

        hu_row.addWidget(QLabel("Closing radius:"))
        self._closing_spin = QSpinBox()
        self._closing_spin.setRange(0, 10)
        self._closing_spin.setValue(3)
        hu_row.addWidget(self._closing_spin)
        layout.addLayout(hu_row)

        self._seg_btn = QPushButton("Segment Skin")
        self._seg_btn.clicked.connect(self._run_skin_segmentation)
        layout.addWidget(self._seg_btn)

        self._seg_status = QLabel("")
        self._seg_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._seg_status)

        nav_row = QHBoxLayout()
        nav_row.addStretch()
        next_btn = QPushButton("Next → Target Segmentation")
        next_btn.clicked.connect(lambda: self._go_to_step(2))
        nav_row.addWidget(next_btn)
        layout.addLayout(nav_row)
        return box

    def _run_skin_segmentation(self):
        if self._sitk_image is None:
            QMessageBox.warning(self, "No Volume", "Load a DICOM volume first.")
            return

        self._seg_btn.setEnabled(False)
        self._seg_status.setText("Segmenting…")

        worker = _SegWorker(
            self._sitk_image,
            self._lower_spin.value(),
            self._upper_spin.value(),
            self._closing_spin.value(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self._on_skin_done)
        worker.error.connect(self._on_skin_error)
        thread.started.connect(worker.run)
        self._seg_thread = thread
        self._seg_worker = worker
        thread.start()

    def _on_skin_done(self, label: sitk.Image, mesh):
        if self._seg_thread:
            self._seg_thread.quit()
        self._seg_btn.setEnabled(True)
        self._skin_label = label

        # Push to SceneGraph
        model_node = ModelNode(
            node_id="SKIN_MODEL",
            vtk_poly_data=mesh,
            color=(0.9, 0.75, 0.65),
            opacity=0.6,
        )
        self._scene_graph.add_node(model_node)
        self._seg_status.setText("Skin segmented")
        self.skin_mesh_ready.emit(mesh)

    def _on_skin_error(self, msg: str):
        if self._seg_thread:
            self._seg_thread.quit()
        self._seg_btn.setEnabled(True)
        self._seg_status.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Step 2 — Target Segmentation
    # ------------------------------------------------------------------

    def _build_step2(self) -> QGroupBox:
        box = QGroupBox("Step 2 — Target Segmentation")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel(
            "Paint inside the target and outside the target on a few slices, "
            "then preview the interpolated segmentation. This follows the "
            "SlicerOpenNav inside/outside seed workflow."
        ))

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush radius (voxels):"))
        self._brush_radius = QSpinBox(); self._brush_radius.setRange(1, 20)
        self._brush_radius.setValue(3)
        brush_row.addWidget(self._brush_radius)
        self._paint_inside_btn = QPushButton("Paint Inside")
        self._paint_inside_btn.setCheckable(True)
        brush_row.addWidget(self._paint_inside_btn)
        self._paint_outside_btn = QPushButton("Paint Outside")
        self._paint_outside_btn.setCheckable(True)
        brush_row.addWidget(self._paint_outside_btn)
        layout.addLayout(brush_row)

        action_row = QHBoxLayout()
        self._preview_target_btn = QPushButton("Preview Target")
        self._preview_target_btn.setEnabled(False)
        self._preview_target_btn.clicked.connect(self._preview_target_from_seeds)
        self._apply_target_btn = QPushButton("Apply Target")
        self._apply_target_btn.setEnabled(False)
        self._apply_target_btn.clicked.connect(self._apply_target_preview)
        self._clear_seeds_btn = QPushButton("Clear Seeds")
        self._clear_seeds_btn.setEnabled(False)
        self._clear_seeds_btn.clicked.connect(self._clear_seed_paint)
        self._reset_target_btn = QPushButton("Reset Target")
        self._reset_target_btn.setEnabled(False)
        self._reset_target_btn.clicked.connect(self._reset_target_segmentation)
        self._update_target_mesh_btn = QPushButton("Update Target Mesh")
        self._update_target_mesh_btn.setEnabled(False)
        self._update_target_mesh_btn.clicked.connect(self._rebuild_target_mesh)
        action_row.addWidget(self._preview_target_btn)
        action_row.addWidget(self._apply_target_btn)
        action_row.addWidget(self._clear_seeds_btn)
        action_row.addWidget(self._reset_target_btn)
        action_row.addWidget(self._update_target_mesh_btn)
        layout.addLayout(action_row)
        self._paint_inside_btn.toggled.connect(self._on_inside_mode_toggled)
        self._paint_outside_btn.toggled.connect(self._on_outside_mode_toggled)

        self._target_status = QLabel("No target defined")
        self._target_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._target_status)

        nav_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(lambda: self._go_to_step(1))
        nav_row.addWidget(back_btn)
        nav_row.addStretch()
        next_btn = QPushButton("Next → Trajectory")
        next_btn.clicked.connect(lambda: self._go_to_step(3))
        nav_row.addWidget(next_btn)
        layout.addLayout(nav_row)
        return box

    def paint_seed_at_ijk(self, ijk: tuple, plane: str | None = None):
        """Paint inside/outside seed voxels around an index using the brush radius."""
        if self._sitk_image is None:
            return
        seed_value = self._active_seed_value()
        if seed_value == 0:
            return
        radius = int(self._brush_radius.value())
        if self._seed_array is None:
            size = self._sitk_image.GetSize()  # x, y, z
            self._seed_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        current = np.asarray(ijk, dtype=int)
        previous = getattr(self, "_last_paint_ijk", None)
        points = [current] if previous is None else self._interpolate_ijk_points(previous, current)
        for point in points:
            self._paint_target_brush(
                self._seed_array, tuple(int(v) for v in point), radius, plane, value=seed_value
            )
        self._last_paint_ijk = current
        inside = int(np.count_nonzero(self._seed_array == 1))
        outside = int(np.count_nonzero(self._seed_array == 2))
        self._target_status.setText(
            f"Seeds: inside {inside:,}, outside {outside:,}. "
            "Click 'Preview Target' to interpolate."
        )
        self._target_overlay_timer.start()
        self._preview_target_btn.setEnabled(inside > 0)
        self._clear_seeds_btn.setEnabled(inside > 0 or outside > 0)

    def _emit_step2_overlays(self):
        self.seed_label_updated.emit(self._seed_array)
        self.target_preview_updated.emit(self._target_preview_array)
        self.target_label_updated.emit(self._target_array)

    def _preview_target_from_seeds(self):
        if self._sitk_image is None or self._seed_array is None:
            return
        inside = (self._seed_array == 1).astype(np.uint8, copy=False)
        outside = (self._seed_array == 2).astype(np.uint8, copy=False)
        seg = ThresholdSegmenter()
        preview = seg.segment_from_seeds(self._sitk_image, inside, outside)
        self._target_preview_array = sitk.GetArrayFromImage(preview).astype(np.uint8, copy=False)
        voxels = int(self._target_preview_array.sum())
        self._target_status.setText(
            f"Preview target: {voxels:,} voxels. Click 'Apply Target' to confirm."
        )
        self._apply_target_btn.setEnabled(voxels > 0)
        self._reset_target_btn.setEnabled(True)
        self._emit_step2_overlays()

    def _apply_target_preview(self):
        if self._target_preview_array is None:
            return
        self._target_array = self._target_preview_array.copy()
        self._target_preview_array = None
        self._update_target_mesh_btn.setEnabled(True)
        n = int(self._target_array.sum())
        self._target_status.setText(
            f"Target: {n:,} voxels applied. Click 'Update Target Mesh' to rebuild."
        )
        self._apply_target_btn.setEnabled(False)
        self._emit_step2_overlays()

    def _clear_seed_paint(self):
        self._seed_array = None
        self._last_paint_ijk = None
        self._preview_target_btn.setEnabled(False)
        self._clear_seeds_btn.setEnabled(False)
        self._emit_step2_overlays()

    def _reset_target_segmentation(self):
        self._seed_array = None
        self._target_preview_array = None
        self._target_array = None
        self._target_label = None
        self._preview_target_btn.setEnabled(False)
        self._apply_target_btn.setEnabled(False)
        self._clear_seeds_btn.setEnabled(False)
        self._reset_target_btn.setEnabled(False)
        self._update_target_mesh_btn.setEnabled(False)
        self._target_status.setText("No target defined")
        self._emit_step2_overlays()

    def _emit_interaction_mode(self):
        if self._step != 2:
            mode = "navigate"
        elif self._paint_inside_btn.isChecked() or self._paint_outside_btn.isChecked():
            mode = "paint"
        else:
            mode = "navigate"
        self.interaction_mode_changed.emit(mode)

    def _on_inside_mode_toggled(self, checked: bool):
        if checked and self._paint_outside_btn.isChecked():
            self._paint_outside_btn.setChecked(False)
        self._last_paint_ijk = None
        self._emit_interaction_mode()

    def _on_outside_mode_toggled(self, checked: bool):
        if checked and self._paint_inside_btn.isChecked():
            self._paint_inside_btn.setChecked(False)
        self._last_paint_ijk = None
        self._emit_interaction_mode()

    def _active_seed_value(self) -> int:
        if self._paint_inside_btn.isChecked():
            return 1
        if self._paint_outside_btn.isChecked():
            return 2
        return 0

    @staticmethod
    def _interpolate_ijk_points(start: np.ndarray, end: np.ndarray):
        steps = int(max(np.abs(end - start).max(), 1))
        return [
            np.round(start + (end - start) * t).astype(int)
            for t in np.linspace(0.0, 1.0, steps + 1)
        ]

    @staticmethod
    def _paint_target_brush(
        arr: np.ndarray,
        ijk: tuple[int, int, int],
        radius: int,
        plane: str | None,
        value: int = 1,
    ):
        ix, iy, iz = ijk
        nz, ny, nx = arr.shape
        r2 = radius * radius
        dz_range = range(-radius, radius + 1) if plane not in ("axial", "coronal", "sagittal") else range(0, 1)
        dy_range = range(-radius, radius + 1)
        dx_range = range(-radius, radius + 1)

        for dz in dz_range:
            for dy in dy_range:
                for dx in dx_range:
                    if plane == "axial":
                        dist2 = dx * dx + dy * dy
                    elif plane == "coronal":
                        dist2 = dx * dx + dz * dz
                    elif plane == "sagittal":
                        dist2 = dy * dy + dz * dz
                    else:
                        dist2 = dx * dx + dy * dy + dz * dz
                    if dist2 > r2:
                        continue
                    x = ix + dx
                    y = iy + dy
                    z = iz + dz
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        arr[z, y, x] = value

    # ------------------------------------------------------------------
    # Step 3 — Trajectory (entry + target points)
    # ------------------------------------------------------------------

    def _build_step3(self) -> QGroupBox:
        box = QGroupBox("Step 3 — Trajectory")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel(
            "Place an Entry Point and a Target Point on the slice views."
        ))

        self._entry_btn  = QPushButton("Set Entry Point")
        self._target_btn = QPushButton("Set Target Point")
        self._entry_btn.setCheckable(True)
        self._target_btn.setCheckable(True)
        self._entry_btn.toggled.connect(self._on_entry_btn_toggled)
        self._target_btn.toggled.connect(self._on_target_btn_toggled)
        layout.addWidget(self._entry_btn)
        layout.addWidget(self._target_btn)

        self._traj_status = QLabel("Entry: —   Target: —")
        self._traj_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._traj_status)

        nav_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(lambda: self._go_to_step(2))
        nav_row.addWidget(back_btn)
        nav_row.addStretch()
        next_btn = QPushButton("Next → Landmarks")
        next_btn.clicked.connect(lambda: self._go_to_step(4))
        nav_row.addWidget(next_btn)
        layout.addLayout(nav_row)
        return box

    def _on_entry_btn_toggled(self, checked: bool):
        if checked:
            self._target_btn.setChecked(False)
            self.interaction_mode_changed.emit("entry")
        else:
            self.interaction_mode_changed.emit("")

    def _on_target_btn_toggled(self, checked: bool):
        if checked:
            self._entry_btn.setChecked(False)
            self.interaction_mode_changed.emit("target")
        else:
            self.interaction_mode_changed.emit("")

    def place_trajectory_point(self, point_type: str, world_xyz: np.ndarray):
        """Called externally when user clicks on a slice view in trajectory mode.

        Parameters
        ----------
        point_type : str
            'entry' or 'target'
        world_xyz : np.ndarray
            (3,) RAS world coordinates.
        """
        node_id = "TRAJECTORY_POINTS"
        existing = self._scene_graph.get_node(node_id)
        if existing is None:
            self._scene_graph.add_node(FiducialSetNode(node_id=node_id))

        label = "Entry" if point_type == "entry" else "Target"
        # Remove existing point of this type before replacing
        node = self._scene_graph.get_node(node_id)
        node.points = [p for p in node.points if p["label"] != label]
        self._scene_graph.add_fiducial(node_id, label, world_xyz)

        # Cache and broadcast so slice viewers can redraw markers
        if point_type == "entry":
            self._entry_xyz = world_xyz.copy()
        else:
            self._target_xyz = world_xyz.copy()
        self.trajectory_points_updated.emit(self._entry_xyz, self._target_xyz)

        self._refresh_traj_status()
        self.trajectory_updated.emit(
            self._get_trajectory_position("Entry"),
            self._get_trajectory_position("Target"),
        )

    def _refresh_traj_status(self):
        node = self._scene_graph.get_node("TRAJECTORY_POINTS")
        if node is None:
            return
        labels = {p["label"] for p in node.points}
        entry  = "✓" if "Entry"  in labels else "—"
        target = "✓" if "Target" in labels else "—"
        self._traj_status.setText(f"Entry: {entry}   Target: {target}")

    def _get_trajectory_position(self, label: str):
        node = self._scene_graph.get_node("TRAJECTORY_POINTS")
        if node is None:
            return None
        for point in node.points:
            if point["label"] == label:
                return np.asarray(point["position"], dtype=np.float64)
        return None

    # ------------------------------------------------------------------
    # Step 4 — Anatomical Landmarks
    # ------------------------------------------------------------------

    def _build_step4(self) -> QGroupBox:
        box = QGroupBox("Step 4 — Anatomical Landmarks (≥3 required)")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel(
            "Click on slice views to place landmarks used for patient registration."
        ))

        self._place_lm_btn = QPushButton("Place Landmark")
        self._place_lm_btn.setCheckable(True)
        self._place_lm_btn.toggled.connect(
            lambda checked: self.interaction_mode_changed.emit("landmark" if checked else "")
        )
        layout.addWidget(self._place_lm_btn)

        self._lm_table = QTableWidget(0, 2)
        self._lm_table.setHorizontalHeaderLabels(["Name", "Position (RAS)"])
        self._lm_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._lm_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        # Allow double-click editing; Position column items will be flagged non-editable
        self._lm_table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self._lm_table.itemChanged.connect(self._on_landmark_name_changed)
        layout.addWidget(self._lm_table)

        row = QHBoxLayout()
        self._del_lm_btn = QPushButton("Delete Selected")
        self._del_lm_btn.clicked.connect(self._delete_landmark)
        row.addWidget(self._del_lm_btn)
        row.addStretch()
        layout.addLayout(row)

        self._lm_status = QLabel("0 landmarks placed (need ≥3)")
        self._lm_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lm_status)

        nav_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(lambda: self._go_to_step(3))
        nav_row.addWidget(back_btn)
        nav_row.addStretch()
        done_btn = QPushButton("Complete Planning")
        done_btn.clicked.connect(self._complete_planning)
        nav_row.addWidget(done_btn)
        layout.addLayout(nav_row)
        return box

    def place_landmark(self, world_xyz: np.ndarray):
        """Called externally when user clicks on a slice view in landmark mode."""
        n = len(self._scene_graph.get_node("PLANNING_LANDMARKS").points
                if self._scene_graph.get_node("PLANNING_LANDMARKS") else [])
        label = f"LM{n + 1}"
        self._scene_graph.add_fiducial("PLANNING_LANDMARKS", label, world_xyz)
        self._refresh_landmark_table()
        self.landmarks_updated.emit(self._current_landmark_positions())

    def _delete_landmark(self):
        row = self._lm_table.currentRow()
        if row < 0:
            return
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        if node and row < len(node.points):
            node.points.pop(row)
            self._scene_graph._fire("node_modified", node)
        self._refresh_landmark_table()
        self.landmarks_updated.emit(self._current_landmark_positions())

    def _refresh_landmark_table(self):
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        points = node.points if node else []
        self._lm_table.blockSignals(True)
        self._lm_table.setRowCount(len(points))
        for i, p in enumerate(points):
            pos = p["position"]

            name_item = QTableWidgetItem(p["label"])
            self._lm_table.setItem(i, 0, name_item)

            pos_item = QTableWidgetItem(f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            pos_item.setFlags(pos_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._lm_table.setItem(i, 1, pos_item)
        self._lm_table.blockSignals(False)

        n = len(points)
        self._lm_status.setText(
            f"{n} landmark{'s' if n != 1 else ''} placed"
            + (" — ready" if n >= 3 else f" (need ≥3)")
        )
        self.landmarks_updated.emit([p["position"] for p in points])

    def _on_landmark_name_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        new_name = item.text().strip()
        if not new_name:
            return
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        row = item.row()
        if node and row < len(node.points):
            node.points[row]["label"] = new_name
            self._scene_graph._fire("node_modified", node)

    def _current_landmark_positions(self):
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        if node is None:
            return []
        return [np.asarray(p["position"], dtype=np.float64) for p in node.points]

    def _complete_planning(self):
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        n_landmarks = len(node.points) if node else 0
        if n_landmarks < 3:
            QMessageBox.warning(
                self, "Insufficient Landmarks",
                f"Place at least 3 landmarks (currently {n_landmarks})."
            )
            return
        self.emit_complete()

    def handle_preview_click(self, plane: str, ijk: tuple, ras_xyz: tuple[float, float, float]):
        """Handle a click from the Qt preview panes."""
        if self._step == 2:
            if self._paint_inside_btn.isChecked() or self._paint_outside_btn.isChecked():
                self.paint_seed_at_ijk(ijk, plane=plane)
            return
        if self._step == 3:
            point = np.asarray(ras_xyz, dtype=np.float64)
            if self._entry_btn.isChecked():
                self.place_trajectory_point("entry", point)
            elif self._target_btn.isChecked():
                self.place_trajectory_point("target", point)
            return
        if self._step == 4 and self._place_lm_btn.isChecked():
            self.place_landmark(np.asarray(ras_xyz, dtype=np.float64))

    def handle_preview_drag(self, plane: str, ijk: tuple, ras_xyz: tuple[float, float, float]):
        """Handle a drag from the Qt preview panes."""
        if self._step == 2 and (self._paint_inside_btn.isChecked() or self._paint_outside_btn.isChecked()):
            self.paint_seed_at_ijk(ijk, plane=plane)
        else:
            self._last_paint_ijk = None

    def _rebuild_target_mesh(self):
        """Rebuild the target surface after painting settles."""
        if self._sitk_image is None or self._target_array is None:
            return
        t0 = time.perf_counter()
        smoothed = sitk.GetImageFromArray(self._target_array.astype(np.uint8))
        smoothed.CopyInformation(self._sitk_image)
        smoothed = sitk.BinaryMorphologicalClosing(smoothed, [1, 1, 1], sitk.sitkBall)
        self._target_label = smoothed
        mesh = SurfaceExtractor(smooth_iterations=4).extract(self._target_label)
        model_node = ModelNode(
            node_id="TARGET_MODEL",
            vtk_poly_data=mesh,
            color=(0.2, 0.6, 1.0),
            opacity=0.8,
        )
        self._scene_graph.add_node(model_node)
        self._target_array = sitk.GetArrayFromImage(self._target_label).astype(np.uint8, copy=False)
        self._target_preview_array = None
        self._emit_step2_overlays()
        n = int(self._target_array.sum())
        self._target_status.setText(
            f"Target: {n:,} voxels — mesh updated in {time.perf_counter() - t0:.2f}s"
        )
        self._update_target_mesh_btn.setEnabled(False)
        self._apply_target_btn.setEnabled(False)
        self.target_mesh_ready.emit(mesh)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def _go_to_step(self, step: int):
        self._step = step
        self._update_step_visibility()
        self._emit_interaction_mode()

    def _update_step_visibility(self):
        widgets = [self._step1_box, self._step2_box,
                   self._step3_box, self._step4_box]
        for i, w in enumerate(widgets):
            w.setVisible(i + 1 == self._step)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout

        self._step1_box = self._build_step1()
        self._step2_box = self._build_step2()
        self._step3_box = self._build_step3()
        self._step4_box = self._build_step4()

        layout.addWidget(self._step1_box)
        layout.addWidget(self._step2_box)
        layout.addWidget(self._step3_box)
        layout.addWidget(self._step4_box)
        layout.addStretch()

        self._update_step_visibility()
