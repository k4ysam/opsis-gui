"""LandmarkManagerPage: view and manage all fiducial sets in the scene.

Displays every FiducialSetNode currently in the SceneGraph in a tabbed
QTableWidget.  Supports:
  - Refreshing the table from the SceneGraph
  - Deleting individual rows
  - Exporting any fiducial set to CSV
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, List

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QFileDialog, QLabel, QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import Qt

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.app.scene_graph import SceneGraph, FiducialSetNode


class LandmarkManagerPage(WorkflowPage):
    """Stage 4 (optional): inspect and export all landmark/trajectory sets."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Landmark Manager", parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-populate tabs from the current SceneGraph state."""
        sg = SceneGraph.instance()
        fiducial_nodes: List[FiducialSetNode] = [
            n for n in sg.nodes_of_type(FiducialSetNode)
        ]

        # Remove all existing tabs
        while self._tabs.count():
            self._tabs.removeTab(0)

        if not fiducial_nodes:
            self._status_lbl.setText("No landmark sets in scene.")
            return

        self._status_lbl.setText(
            f"{len(fiducial_nodes)} set(s) loaded — "
            f"{sum(len(n.points) for n in fiducial_nodes)} total points"
        )

        for node in fiducial_nodes:
            table = self._make_table(node)
            self._tabs.addTab(table, node.node_id)

    def current_node_id(self) -> Optional[str]:
        idx = self._tabs.currentIndex()
        if idx < 0:
            return None
        return self._tabs.tabText(idx)

    def export_current(self, path: Optional[str] = None):
        """Export the active tab's fiducial set to CSV.

        If *path* is None a file-dialog is shown.
        Returns the path written, or None if cancelled/empty.
        """
        node_id = self.current_node_id()
        if node_id is None:
            return None
        node = SceneGraph.instance().get_node(node_id)
        if not isinstance(node, FiducialSetNode) or not node.points:
            return None

        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Landmarks",
                str(Path.home() / f"{node_id}.csv"),
                "CSV files (*.csv)",
            )
        if not path:
            return None

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label", "x_mm", "y_mm", "z_mm"])
            for p in node.points:
                pos = np.asarray(p["position"])
                writer.writerow([p.get("label", ""), *pos.tolist()])

        self.status_message.emit(f"Exported {node_id} → {path}")
        return path

    def delete_selected_row(self):
        """Delete the selected row from the active tab and update SceneGraph."""
        idx = self._tabs.currentIndex()
        if idx < 0:
            return
        table: QTableWidget = self._tabs.widget(idx)
        row = table.currentRow()
        if row < 0:
            return

        node_id = self._tabs.tabText(idx)
        node = SceneGraph.instance().get_node(node_id)
        if isinstance(node, FiducialSetNode) and row < len(node.points):
            del node.points[row]

        table.removeRow(row)
        self._status_lbl.setText(
            f"{self._tabs.count()} set(s) — "
            f"{sum(self._tabs.widget(i).rowCount() for i in range(self._tabs.count()))} total points"
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout

        self._status_lbl = QLabel("No landmark sets loaded.")
        layout.addWidget(self._status_lbl)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_row)
        btn_row.addWidget(delete_btn)

        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(lambda: self.export_current())
        btn_row.addWidget(export_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _make_table(self, node: FiducialSetNode) -> QTableWidget:
        table = QTableWidget(len(node.points), 4)
        table.setHorizontalHeaderLabels(["Label", "X (mm)", "Y (mm)", "Z (mm)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        for row, p in enumerate(node.points):
            pos = np.asarray(p["position"], dtype=float)
            table.setItem(row, 0, QTableWidgetItem(p.get("label", "")))
            for col, v in enumerate(pos):
                item = QTableWidgetItem(f"{v:.3f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, col + 1, item)

        return table
