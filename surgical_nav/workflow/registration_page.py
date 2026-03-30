"""RegistrationPage: Stage 3 — calibration + patient registration wizard.

Nine sequential steps:
  Step 1 — Connect to PLUS server
  Step 2 — Verify tool visibility
  Step 3 — Pivot calibration (pointer tip)
  Step 4 — Spin calibration (shaft axis)
  Step 5 — Review calibration results
  Step 6 — Landmark collection (touch patient landmarks)
  Step 7 — Landmark registration
  Step 8 — Surface trace collection
  Step 9 — ICP surface refinement + verification

All results are pushed to the shared SceneGraph as TransformNodes:
  POINTER_CALIBRATION          → pivot result (tool-tip offset)
  IMAGE_REGISTRATION           → landmark registration result
  IMAGE_REGISTRATION_REFINEMENT → ICP refinement result
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.app.scene_graph import SceneGraph, TransformNode, FiducialSetNode
from surgical_nav.calibration.pivot_calibrator import PivotCalibrator
from surgical_nav.calibration.spin_calibrator import SpinCalibrator
from surgical_nav.registration.landmark_registrar import LandmarkRegistrar
from surgical_nav.registration.surface_registrar import SurfaceRegistrar


_COLLECT_TARGET = 150   # samples for pivot/spin
_MIN_LANDMARKS  = 3


class RegistrationPage(WorkflowPage):
    """Stage 3: instrument calibration + patient-to-image registration."""

    # Emitted with the final IMAGE_REGISTRATION 4×4 matrix
    registration_complete = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Registration", parent, show_back=True)
        self._step = 1

        # Calibration state
        self._pivot_cal  = PivotCalibrator(min_samples=50, max_rmse_mm=0.8)
        self._spin_cal   = SpinCalibrator(min_samples=50, min_linearity=0.90)
        self._collecting_pivot = False
        self._collecting_spin  = False

        # Registration state
        self._landmark_reg = LandmarkRegistrar(min_pairs=_MIN_LANDMARKS, max_rmse_mm=3.0)
        self._image_landmarks: List[np.ndarray] = []   # from SceneGraph PLANNING_LANDMARKS
        self._surface_points:  List[np.ndarray] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        self._load_image_landmarks()
        self._update_step_visibility()

    # ------------------------------------------------------------------
    # Public API (called by main.py when tracker emits transforms)
    # ------------------------------------------------------------------

    def receive_transform(self, name: str, matrix: np.ndarray):
        """Feed a new tracking transform into the active collection."""
        if name != "PointerToTracker":
            return
        if self._collecting_pivot:
            self._pivot_cal.add_sample(matrix)
            self._pivot_progress.setValue(
                min(self._pivot_cal.sample_count, _COLLECT_TARGET)
            )
            if self._pivot_cal.sample_count >= _COLLECT_TARGET:
                self._stop_pivot_collection()
        elif self._collecting_spin:
            self._spin_cal.add_sample(matrix)
            self._spin_progress.setValue(
                min(self._spin_cal.sample_count, _COLLECT_TARGET)
            )
            if self._spin_cal.sample_count >= _COLLECT_TARGET:
                self._stop_spin_collection()

    def add_patient_landmark(self, world_xyz: np.ndarray):
        """Called when user touches a patient landmark point with the pointer."""
        n = len(self._image_landmarks)
        if n == 0:
            QMessageBox.warning(self, "No Image Landmarks",
                                "Load a planning result with landmarks first.")
            return
        idx = self._landmark_reg.pair_count
        if idx >= n:
            return
        self._landmark_reg.add_pair(world_xyz, self._image_landmarks[idx])
        self._refresh_landmark_table()

    def add_surface_point(self, world_xyz: np.ndarray):
        """Called when user traces the skin surface with the pointer."""
        self._surface_points.append(world_xyz.copy())
        self._surface_count_lbl.setText(
            f"{len(self._surface_points)} surface points collected"
        )

    # ------------------------------------------------------------------
    # Step 3 — Pivot calibration
    # ------------------------------------------------------------------

    def _build_step3(self) -> QGroupBox:
        box = QGroupBox("Step 3 — Pivot Calibration")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Hold the pointer tip stationary and rotate the handle.\n"
            f"Collecting {_COLLECT_TARGET} samples automatically."
        ))
        self._pivot_progress = QProgressBar()
        self._pivot_progress.setRange(0, _COLLECT_TARGET)
        layout.addWidget(self._pivot_progress)

        row = QHBoxLayout()
        self._pivot_start_btn = QPushButton("Start Collection")
        self._pivot_start_btn.clicked.connect(self._start_pivot_collection)
        self._pivot_calibrate_btn = QPushButton("Calibrate")
        self._pivot_calibrate_btn.setEnabled(False)
        self._pivot_calibrate_btn.clicked.connect(self._run_pivot_calibration)
        row.addWidget(self._pivot_start_btn)
        row.addWidget(self._pivot_calibrate_btn)
        layout.addLayout(row)

        self._pivot_result_lbl = QLabel("")
        self._pivot_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._pivot_result_lbl)

        next_btn = QPushButton("Next → Spin Calibration")
        next_btn.clicked.connect(lambda: self._go_to_step(4))
        layout.addWidget(next_btn)
        return box

    def _start_pivot_collection(self):
        self._pivot_cal.clear()
        self._pivot_progress.setValue(0)
        self._collecting_pivot = True
        self._pivot_start_btn.setEnabled(False)
        self._pivot_calibrate_btn.setEnabled(False)
        self._pivot_result_lbl.setText("Collecting…")

    def _stop_pivot_collection(self):
        self._collecting_pivot = False
        self._pivot_calibrate_btn.setEnabled(True)
        self._pivot_result_lbl.setText(
            f"Collection complete — {self._pivot_cal.sample_count} samples"
        )

    def _run_pivot_calibration(self):
        result = self._pivot_cal.calibrate()
        if not result.success:
            QMessageBox.warning(self, "Calibration Failed", result.message)
            self._pivot_result_lbl.setText(f"Failed: {result.message}")
            return
        # Store in SceneGraph
        self._scene_graph.add_node(TransformNode(
            node_id="POINTER_CALIBRATION",
            matrix=result.as_transform(),
        ))
        self._pivot_result_lbl.setText(
            f"RMSE: {result.rmse_mm:.3f} mm  "
            f"Tip: ({result.p_tool[0]:.1f}, {result.p_tool[1]:.1f}, {result.p_tool[2]:.1f})"
        )
        self._pivot_start_btn.setEnabled(True)
        self._pivot_calibrate_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Step 4 — Spin calibration
    # ------------------------------------------------------------------

    def _build_step4(self) -> QGroupBox:
        box = QGroupBox("Step 4 — Spin Calibration (Shaft Axis)")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Spin the pointer around its own shaft axis.\n"
            f"Collecting {_COLLECT_TARGET} samples automatically."
        ))
        self._spin_progress = QProgressBar()
        self._spin_progress.setRange(0, _COLLECT_TARGET)
        layout.addWidget(self._spin_progress)

        row = QHBoxLayout()
        self._spin_start_btn = QPushButton("Start Collection")
        self._spin_start_btn.clicked.connect(self._start_spin_collection)
        self._spin_calibrate_btn = QPushButton("Calibrate")
        self._spin_calibrate_btn.setEnabled(False)
        self._spin_calibrate_btn.clicked.connect(self._run_spin_calibration)
        row.addWidget(self._spin_start_btn)
        row.addWidget(self._spin_calibrate_btn)
        layout.addLayout(row)

        self._spin_result_lbl = QLabel("")
        self._spin_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._spin_result_lbl)

        next_btn = QPushButton("Next → Review Calibration")
        next_btn.clicked.connect(lambda: self._go_to_step(5))
        layout.addWidget(next_btn)
        return box

    def _start_spin_collection(self):
        self._spin_cal.clear()
        self._spin_progress.setValue(0)
        self._collecting_spin = True
        self._spin_start_btn.setEnabled(False)
        self._spin_calibrate_btn.setEnabled(False)
        self._spin_result_lbl.setText("Collecting…")

    def _stop_spin_collection(self):
        self._collecting_spin = False
        self._spin_calibrate_btn.setEnabled(True)
        self._spin_result_lbl.setText(
            f"Collection complete — {self._spin_cal.sample_count} samples"
        )

    def _run_spin_calibration(self):
        result = self._spin_cal.calibrate()
        if not result.success:
            QMessageBox.warning(self, "Spin Calibration Failed", result.message)
            self._spin_result_lbl.setText(f"Failed: {result.message}")
            return
        self._spin_result_lbl.setText(
            f"Planarity: {result.linearity:.3f}  "
            f"Shaft: ({result.shaft_in_tool[0]:.2f}, "
            f"{result.shaft_in_tool[1]:.2f}, {result.shaft_in_tool[2]:.2f})"
        )
        self._spin_start_btn.setEnabled(True)
        self._spin_calibrate_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Step 5 — Review
    # ------------------------------------------------------------------

    def _build_step5(self) -> QGroupBox:
        box = QGroupBox("Step 5 — Review Calibration")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Verify the calibration results above before proceeding.\n"
            "Return to Step 3 or 4 if RMSE / planarity are unsatisfactory."
        ))
        next_btn = QPushButton("Next → Landmark Collection")
        next_btn.clicked.connect(lambda: self._go_to_step(6))
        layout.addWidget(next_btn)
        return box

    # ------------------------------------------------------------------
    # Step 6 — Landmark collection
    # ------------------------------------------------------------------

    def _build_step6(self) -> QGroupBox:
        box = QGroupBox("Step 6 — Landmark Collection")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Touch each anatomical landmark (same order as Planning).\n"
            "The pointer tip position is recorded when you press 'Record'."
        ))

        self._lm_collect_table = QTableWidget(0, 3)
        self._lm_collect_table.setHorizontalHeaderLabels(["#", "Image", "Physical"])
        self._lm_collect_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._lm_collect_table)

        self._lm_collect_status = QLabel("0 / 0 pairs collected")
        self._lm_collect_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lm_collect_status)

        next_btn = QPushButton("Next → Landmark Registration")
        next_btn.clicked.connect(lambda: self._go_to_step(7))
        layout.addWidget(next_btn)
        return box

    def _load_image_landmarks(self):
        node = self._scene_graph.get_node("PLANNING_LANDMARKS")
        if node is None:
            return
        self._image_landmarks = [p["position"].copy() for p in node.points]
        self._lm_collect_table.setRowCount(len(self._image_landmarks))
        for i, pos in enumerate(self._image_landmarks):
            self._lm_collect_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._lm_collect_table.setItem(
                i, 1,
                QTableWidgetItem(f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            )

    def _refresh_landmark_table(self):
        for i in range(self._landmark_reg.pair_count):
            p = self._landmark_reg._p[i]
            self._lm_collect_table.setItem(
                i, 2,
                QTableWidgetItem(f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
            )
        n = self._landmark_reg.pair_count
        total = len(self._image_landmarks)
        self._lm_collect_status.setText(f"{n} / {total} pairs collected")

    # ------------------------------------------------------------------
    # Step 7 — Landmark registration
    # ------------------------------------------------------------------

    def _build_step7(self) -> QGroupBox:
        box = QGroupBox("Step 7 — Landmark Registration")
        layout = QVBoxLayout(box)

        self._reg_result_lbl = QLabel("Not yet computed")
        self._reg_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._reg_result_lbl)

        reg_btn = QPushButton("Compute Landmark Registration")
        reg_btn.clicked.connect(self._run_landmark_registration)
        layout.addWidget(reg_btn)

        next_btn = QPushButton("Next → Surface Trace")
        next_btn.clicked.connect(lambda: self._go_to_step(8))
        layout.addWidget(next_btn)
        return box

    def _run_landmark_registration(self):
        result = self._landmark_reg.register()
        if not result.success:
            QMessageBox.warning(self, "Registration Failed", result.message)
            self._reg_result_lbl.setText(f"Failed: {result.message}")
            return
        self._scene_graph.add_node(TransformNode(
            node_id="IMAGE_REGISTRATION",
            matrix=result.transform,
        ))
        self._reg_result_lbl.setText(
            f"RMSE: {result.rmse_mm:.3f} mm  ({result.n_pairs} pairs)"
        )

    # ------------------------------------------------------------------
    # Step 8 — Surface trace
    # ------------------------------------------------------------------

    def _build_step8(self) -> QGroupBox:
        box = QGroupBox("Step 8 — Surface Trace Collection")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Glide the pointer tip across the patient's skin surface.\n"
            "Points are recorded automatically while tracing is active."
        ))

        self._trace_btn = QPushButton("Start Tracing")
        self._trace_btn.setCheckable(True)
        layout.addWidget(self._trace_btn)

        self._surface_count_lbl = QLabel("0 surface points collected")
        self._surface_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._surface_count_lbl)

        next_btn = QPushButton("Next → ICP Refinement")
        next_btn.clicked.connect(lambda: self._go_to_step(9))
        layout.addWidget(next_btn)
        return box

    # ------------------------------------------------------------------
    # Step 9 — ICP refinement
    # ------------------------------------------------------------------

    def _build_step9(self) -> QGroupBox:
        box = QGroupBox("Step 9 — ICP Surface Refinement")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Refines the landmark registration against the skin surface mesh.\n"
            "Requires a skin segmentation from the Planning stage."
        ))

        icp_btn = QPushButton("Run ICP Refinement")
        icp_btn.clicked.connect(self._run_icp_refinement)
        layout.addWidget(icp_btn)

        self._icp_result_lbl = QLabel("Not yet computed")
        self._icp_result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icp_result_lbl)

        done_btn = QPushButton("Accept Registration & Proceed")
        done_btn.clicked.connect(self._accept_registration)
        layout.addWidget(done_btn)
        return box

    def _run_icp_refinement(self):
        from surgical_nav.app.scene_graph import ModelNode
        skin_node = self._scene_graph.get_node("SKIN_MODEL")
        if skin_node is None or not isinstance(skin_node, ModelNode):
            QMessageBox.warning(self, "No Skin Model",
                                "Run skin segmentation in Planning first.")
            return
        if len(self._surface_points) < 10:
            QMessageBox.warning(self, "Too Few Points",
                                "Collect at least 10 surface points.")
            return

        init_node = self._scene_graph.get_node("IMAGE_REGISTRATION")
        init_T = init_node.matrix if init_node is not None else None

        reg = SurfaceRegistrar(max_mean_distance_mm=3.0)
        result = reg.register(None, skin_node.vtk_poly_data, init_T)

        if not result.success:
            self._icp_result_lbl.setText(f"Failed: {result.message}")
            QMessageBox.warning(self, "ICP Failed", result.message)
            return

        self._scene_graph.add_node(TransformNode(
            node_id="IMAGE_REGISTRATION_REFINEMENT",
            matrix=result.transform,
        ))
        self._icp_result_lbl.setText(
            f"Mean distance: {result.mean_distance:.3f} mm"
        )

    def _accept_registration(self):
        reg_node = self._scene_graph.get_node("IMAGE_REGISTRATION")
        if reg_node is None:
            QMessageBox.warning(self, "No Registration",
                                "Complete landmark registration first.")
            return
        self.registration_complete.emit(reg_node.matrix)
        self.emit_complete()

    # ------------------------------------------------------------------
    # Steps 1 & 2 — PLUS connection and tool check (stubs)
    # ------------------------------------------------------------------

    def _build_step1(self) -> QGroupBox:
        box = QGroupBox("Step 1 — Connect to PLUS Server")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Ensure PLUS Server is running and the tracker is connected.\n"
            "Check that the status lights in the toolbar show green."
        ))
        next_btn = QPushButton("Next → Verify Tools")
        next_btn.clicked.connect(lambda: self._go_to_step(2))
        layout.addWidget(next_btn)
        return box

    def _build_step2(self) -> QGroupBox:
        box = QGroupBox("Step 2 — Verify Tool Visibility")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Confirm all required tools are tracked (green status lights).\n"
            "Wave the pointer in front of the camera."
        ))
        next_btn = QPushButton("Next → Pivot Calibration")
        next_btn.clicked.connect(lambda: self._go_to_step(3))
        layout.addWidget(next_btn)
        return box

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def _go_to_step(self, step: int):
        self._step = step
        self._update_step_visibility()

    def _update_step_visibility(self):
        boxes = [
            self._step1_box, self._step2_box, self._step3_box,
            self._step4_box, self._step5_box, self._step6_box,
            self._step7_box, self._step8_box, self._step9_box,
        ]
        for i, box in enumerate(boxes):
            box.setVisible(i + 1 == self._step)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout
        self._step1_box = self._build_step1()
        self._step2_box = self._build_step2()
        self._step3_box = self._build_step3()
        self._step4_box = self._build_step4()
        self._step5_box = self._build_step5()
        self._step6_box = self._build_step6()
        self._step7_box = self._build_step7()
        self._step8_box = self._build_step8()
        self._step9_box = self._build_step9()
        for box in (
            self._step1_box, self._step2_box, self._step3_box,
            self._step4_box, self._step5_box, self._step6_box,
            self._step7_box, self._step8_box, self._step9_box,
        ):
            layout.addWidget(box)
        layout.addStretch()
        self._update_step_visibility()
