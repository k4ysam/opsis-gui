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

import numpy as np
import SimpleITK as sitk

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject

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
            seg  = ThresholdSegmenter()
            label = seg.segment_skin(self._img, self._lower, self._upper,
                                     self._rad)
            mesh  = SurfaceExtractor(smooth_iterations=30).extract(label)
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

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Planning", parent)
        self._step = 1          # current active step (1–4)
        self._seg_thread: Optional[QThread] = None

        # Cached data
        self._sitk_image:   Optional[sitk.Image] = None
        self._vtk_image:    Optional[object]    = None
        self._skin_label:   Optional[sitk.Image] = None
        self._target_label: Optional[sitk.Image] = None

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
        self._update_step_visibility()

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

        next_btn = QPushButton("Next → Target Segmentation")
        next_btn.clicked.connect(lambda: self._go_to_step(2))
        layout.addWidget(next_btn)
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
            "Click a voxel on a slice view to seed the target region, "
            "or use the paint brush to define it manually."
        ))

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed HU range:"))
        self._seed_lower = QSpinBox(); self._seed_lower.setRange(-2000, 4000)
        self._seed_lower.setValue(50)
        self._seed_upper = QSpinBox(); self._seed_upper.setRange(-2000, 4000)
        self._seed_upper.setValue(300)
        seed_row.addWidget(self._seed_lower)
        seed_row.addWidget(QLabel("–"))
        seed_row.addWidget(self._seed_upper)
        layout.addLayout(seed_row)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush radius (voxels):"))
        self._brush_radius = QSpinBox(); self._brush_radius.setRange(1, 20)
        self._brush_radius.setValue(3)
        brush_row.addWidget(self._brush_radius)
        self._paint_btn = QPushButton("Enable Paint Brush")
        self._paint_btn.setCheckable(True)
        brush_row.addWidget(self._paint_btn)
        layout.addLayout(brush_row)

        self._target_status = QLabel("No target defined")
        self._target_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._target_status)

        next_btn = QPushButton("Next → Trajectory")
        next_btn.clicked.connect(lambda: self._go_to_step(3))
        layout.addWidget(next_btn)
        return box

    def seed_target_at_ijk(self, ijk: tuple):
        """Called by SliceViewer when user clicks in seed mode."""
        if self._sitk_image is None:
            return
        seg = ThresholdSegmenter()
        self._target_label = seg.segment_target(
            self._sitk_image, ijk,
            self._seed_lower.value(), self._seed_upper.value()
        )
        mesh = SurfaceExtractor(smooth_iterations=15).extract(self._target_label)
        model_node = ModelNode(
            node_id="TARGET_MODEL",
            vtk_poly_data=mesh,
            color=(0.2, 0.6, 1.0),
            opacity=0.8,
        )
        self._scene_graph.add_node(model_node)
        n = int(sitk.GetArrayFromImage(self._target_label).sum())
        self._target_status.setText(f"Target: {n:,} voxels")
        self.target_mesh_ready.emit(mesh)

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
        self._entry_btn.clicked.connect(
            lambda: self._target_btn.setChecked(False)
        )
        self._target_btn.clicked.connect(
            lambda: self._entry_btn.setChecked(False)
        )
        layout.addWidget(self._entry_btn)
        layout.addWidget(self._target_btn)

        self._traj_status = QLabel("Entry: —   Target: —")
        self._traj_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._traj_status)

        next_btn = QPushButton("Next → Landmarks")
        next_btn.clicked.connect(lambda: self._go_to_step(4))
        layout.addWidget(next_btn)
        return box

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

        self._refresh_traj_status()

    def _refresh_traj_status(self):
        node = self._scene_graph.get_node("TRAJECTORY_POINTS")
        if node is None:
            return
        labels = {p["label"] for p in node.points}
        entry  = "✓" if "Entry"  in labels else "—"
        target = "✓" if "Target" in labels else "—"
        self._traj_status.setText(f"Entry: {entry}   Target: {target}")

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
        layout.addWidget(self._place_lm_btn)

        self._lm_table = QTableWidget(0, 2)
        self._lm_table.setHorizontalHeaderLabels(["#", "Position (RAS)"])
        self._lm_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._lm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
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

        done_btn = QPushButton("Complete Planning")
        done_btn.clicked.connect(self._complete_planning)
        layout.addWidget(done_btn)
        return box

    def place_landmark(self, world_xyz: np.ndarray):
        """Called externally when user clicks on a slice view in landmark mode."""
        n = len(self._scene_graph.get_node("PLANNING_LANDMARKS").points
                if self._scene_graph.get_node("PLANNING_LANDMARKS") else [])
        label = f"LM{n + 1}"
        self._scene_graph.add_fiducial("PLANNING_LANDMARKS", label, world_xyz)
        self._refresh_landmark_table()

    def _delete_landmark(self):
        row = self._lm_table.currentRow()
        if row < 0:
            return
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        if node and row < len(node.points):
            node.points.pop(row)
            self._scene_graph._fire("node_modified", node)
        self._refresh_landmark_table()

    def _refresh_landmark_table(self):
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        points = node.points if node else []
        self._lm_table.setRowCount(len(points))
        for i, p in enumerate(points):
            pos = p["position"]
            self._lm_table.setItem(i, 0, QTableWidgetItem(p["label"]))
            self._lm_table.setItem(
                i, 1,
                QTableWidgetItem(f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            )
        n = len(points)
        self._lm_status.setText(
            f"{n} landmark{'s' if n != 1 else ''} placed"
            + (" — ready" if n >= 3 else f" (need ≥3)")
        )

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

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def _go_to_step(self, step: int):
        self._step = step
        self._update_step_visibility()

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
