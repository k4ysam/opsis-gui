"""LarkCapturePage: capture physical landmarks with LARK optical tracker.

Stage 5 of the surgical navigation workflow.  Accessible immediately after
a patient (DICOM) is loaded — does not require Planning / Registration.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from surgical_nav.app.scene_graph import SceneGraph
from surgical_nav.registration.validation_engine import ValidationEngine
from surgical_nav.tracking.transform_store import TransformStore
from surgical_nav.workflow.base_page import WorkflowPage

log = logging.getLogger(__name__)

_GREEN  = QColor("#c6efce")
_YELLOW = QColor("#ffeb9c")
_RED    = QColor("#ffc7ce")


class LarkCapturePage(WorkflowPage):
    """Stage 5: capture physical landmarks via LARK and validate against GT."""

    def __init__(
        self,
        tracker_store: TransformStore,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("LARK Capture", parent, show_back=False)
        self._store = tracker_store
        self._pointer_status: str = "NEVER_SEEN"
        self._gt_path: Optional[Path] = None
        self._captured: list[np.ndarray] = []          # list of (3,) arrays
        self._validation_result = None

        # Throttle UI updates to 10 Hz
        self._pending_transform: Optional[np.ndarray] = None
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(100)                # 100 ms = 10 Hz
        self._ui_timer.timeout.connect(self._flush_transform)
        self._ui_timer.start()

        self._build_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        self.setFocus()

    # ------------------------------------------------------------------
    # External slots
    # ------------------------------------------------------------------

    def on_transform(self, name: str, matrix: np.ndarray) -> None:
        """Receive tracking transform — called at tracker rate (~10-100 Hz)."""
        if name != "PointerToTracker":
            return
        self._pending_transform = matrix

    def on_tool_status(self, name: str, status: str) -> None:
        """Update pointer status for capture gating."""
        if "Pointer" not in name:
            return
        self._pointer_status = status
        color = {"SEEN": "#44cc44", "NOT_SEEN": "#cccc00", "NEVER_SEEN": "#cc4444"}.get(
            status, "#888888"
        )
        self._lbl_status.setText(f"Status: {status}")
        self._lbl_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._update_capture_enabled()

    # ------------------------------------------------------------------
    # Key handler
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._do_capture()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        lay = self._root_layout

        # --- Connection status ---
        self._lbl_status = QLabel("Status: NEVER_SEEN")
        self._lbl_status.setStyleSheet("color: #cc4444; font-weight: bold;")
        lay.addWidget(self._lbl_status)

        # --- Live position ---
        self._lbl_position = QLabel("Live Position: —")
        self._lbl_position.setStyleSheet("font-family: monospace;")
        lay.addWidget(self._lbl_position)

        # --- Ground truth ---
        gt_row = QHBoxLayout()
        self._lbl_gt = QLabel("Ground Truth: (none loaded)")
        gt_row.addWidget(self._lbl_gt, stretch=1)
        load_gt_btn = QPushButton("Load .tag File…")
        load_gt_btn.clicked.connect(self._load_ground_truth)
        gt_row.addWidget(load_gt_btn)
        lay.addLayout(gt_row)

        # --- Captured table ---
        lay.addWidget(QLabel("Captured Landmarks:"))
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Label", "X (mm)", "Y (mm)", "Z (mm)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay.addWidget(self._table)

        # --- Capture buttons ---
        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton("Capture  [Space]")
        self._btn_capture.clicked.connect(self._do_capture)
        self._btn_capture.setEnabled(False)
        btn_row.addWidget(self._btn_capture)

        undo_btn = QPushButton("Undo Last")
        undo_btn.clicked.connect(self._undo_last)
        btn_row.addWidget(undo_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

        # --- Validation ---
        lay.addWidget(_separator())

        validate_btn = QPushButton("Run FRE/TRE Validation")
        validate_btn.clicked.connect(self._run_validation)
        lay.addWidget(validate_btn)

        results_row = QHBoxLayout()
        self._lbl_fre = QLabel("FRE RMSE: —")
        self._lbl_tre = QLabel("TRE RMSE: —")
        results_row.addWidget(self._lbl_fre)
        results_row.addWidget(self._lbl_tre)
        results_row.addStretch()
        lay.addLayout(results_row)

        lay.addWidget(QLabel("Per-point errors:"))
        self._results_table = QTableWidget(0, 4)
        self._results_table.setHorizontalHeaderLabels(["#", "Label", "FRE (mm)", "TRE (mm)"])
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._results_table)

        # --- Bottom actions ---
        bottom_row = QHBoxLayout()
        export_btn = QPushButton("Export Results CSV…")
        export_btn.clicked.connect(self._export_csv)
        bottom_row.addWidget(export_btn)

        accept_btn = QPushButton("Accept & Complete")
        accept_btn.clicked.connect(self.emit_complete)
        bottom_row.addWidget(accept_btn)

        bottom_row.addStretch()
        lay.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_transform(self) -> None:
        """Called by QTimer at 10 Hz to update the UI with latest transform."""
        m = self._pending_transform
        if m is None:
            return
        self._pending_transform = None
        xyz = m[:3, 3]
        self._lbl_position.setText(
            f"Live Position:  X={xyz[0]:+.2f}  Y={xyz[1]:+.2f}  Z={xyz[2]:+.2f}  mm"
        )

    def _update_capture_enabled(self) -> None:
        ok = self._pointer_status == "SEEN"
        self._btn_capture.setEnabled(ok)

    def _next_label(self) -> str:
        return f"LM{len(self._captured) + 1}"

    def _do_capture(self) -> None:
        if self._pointer_status != "SEEN":
            return
        m = self._store.get("PointerToTracker")
        if m is None:
            self.status_message.emit("No transform in store — tracker not ready")
            return
        xyz = np.array(m[:3, 3], dtype=np.float64)
        label = self._next_label()
        self._captured.append(xyz)

        # Persist to SceneGraph
        SceneGraph.instance().add_fiducial("LARK_PHYSICAL_LANDMARKS", label, xyz)

        # Update table
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self._table.setItem(row, 1, QTableWidgetItem(label))
        for col, v in enumerate(xyz):
            item = QTableWidgetItem(f"{v:.3f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, col + 2, item)

        self.status_message.emit(f"Captured {label}: ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm")

    def _undo_last(self) -> None:
        if not self._captured:
            return
        self._captured.pop()
        self._table.removeRow(self._table.rowCount() - 1)
        # Remove from SceneGraph
        sg = SceneGraph.instance()
        from surgical_nav.app.scene_graph import FiducialSetNode
        node = sg.get_node("LARK_PHYSICAL_LANDMARKS")
        if isinstance(node, FiducialSetNode) and node.points:
            node.points.pop()

    def _clear_all(self) -> None:
        self._captured.clear()
        self._table.setRowCount(0)
        sg = SceneGraph.instance()
        from surgical_nav.app.scene_graph import FiducialSetNode
        node = sg.get_node("LARK_PHYSICAL_LANDMARKS")
        if isinstance(node, FiducialSetNode):
            node.points.clear()

    def _load_ground_truth(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth", str(Path.home()), "TAG files (*.tag);;All files (*)"
        )
        if not path:
            return
        self._gt_path = Path(path)
        self._lbl_gt.setText(f"Ground Truth: {self._gt_path.name}")
        self.status_message.emit(f"Loaded GT: {self._gt_path.name}")

    def _run_validation(self) -> None:
        if len(self._captured) < 3:
            QMessageBox.warning(self, "Not enough points", "Capture at least 3 landmarks first.")
            return
        if self._gt_path is None:
            QMessageBox.warning(self, "No ground truth", "Load a .tag ground truth file first.")
            return

        pts = np.stack(self._captured)
        engine = ValidationEngine()
        result = engine.validate(pts, self._gt_path)
        self._validation_result = result

        if not result.success:
            QMessageBox.critical(self, "Validation failed", result.message)
            return

        self._lbl_fre.setText(f"FRE RMSE: {result.fre_rmse:.3f} mm")
        self._lbl_tre.setText(f"TRE RMSE: {result.tre_rmse:.3f} mm")

        # Per-point table
        n = len(result.fre_per_point)
        self._results_table.setRowCount(n)
        for i in range(n):
            fre = float(result.fre_per_point[i])
            tre = float(result.tre_per_point[i]) if i < len(result.tre_per_point) else float("nan")
            label = f"LM{i + 1}"

            self._results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._results_table.setItem(i, 1, QTableWidgetItem(label))

            fre_item = QTableWidgetItem(f"{fre:.3f}")
            fre_item.setBackground(_error_color(fre))
            self._results_table.setItem(i, 2, fre_item)

            if not np.isnan(tre):
                tre_item = QTableWidgetItem(f"{tre:.3f}")
                tre_item.setBackground(_error_color(tre))
                self._results_table.setItem(i, 3, tre_item)

        self.status_message.emit(
            f"Validation: FRE={result.fre_rmse:.3f}mm  TRE={result.tre_rmse:.3f}mm"
        )

    def _export_csv(self) -> None:
        if self._validation_result is None or not self._validation_result.success:
            QMessageBox.information(self, "No results", "Run validation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Results", str(Path.home() / "lark_validation.csv"),
            "CSV files (*.csv)"
        )
        if not path:
            return

        result = self._validation_result
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label", "x_mm", "y_mm", "z_mm", "fre_mm", "tre_mm"])
            n = len(result.fre_per_point)
            for i in range(n):
                xyz = self._captured[i] if i < len(self._captured) else [0, 0, 0]
                fre = float(result.fre_per_point[i])
                tre = (
                    float(result.tre_per_point[i])
                    if i < len(result.tre_per_point)
                    else ""
                )
                writer.writerow([f"LM{i+1}", *xyz.tolist(), f"{fre:.3f}", f"{tre:.3f}" if tre != "" else ""])

        self.status_message.emit(f"Exported validation results → {path}")


def _separator() -> QWidget:
    sep = QWidget()
    sep.setFixedHeight(1)
    sep.setStyleSheet("background: #3a5a8a;")
    return sep


def _error_color(err_mm: float) -> QColor:
    if err_mm < 2.0:
        return _GREEN
    if err_mm < 3.0:
        return _YELLOW
    return _RED
