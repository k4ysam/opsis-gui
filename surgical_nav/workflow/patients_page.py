"""PatientsPage: Stage 1 — DICOM loading and case creation.

Workflow:
  1. User picks a directory → DICOMIndexer scans it → series table populates
  2. User selects a series + enters a case name → "Load" button
  3. DICOMLoader loads the series → VolumeNode pushed to SceneGraph
  4. stage_complete emitted → Planning stage unlocked

The page owns no viewers itself; it emits ``volume_loaded`` so the
MainWindow can push the vtkImageData to all SliceViewers and the VolumeViewer.
"""

from __future__ import annotations

import os
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog,
    QHeaderView, QProgressBar, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.dicom.dicom_indexer import DICOMIndexer, SeriesEntry
from surgical_nav.dicom.dicom_loader import DICOMLoader
from surgical_nav.app.scene_graph import VolumeNode


# ---------------------------------------------------------------------------
# Background worker: scan DICOM directory without blocking UI
# ---------------------------------------------------------------------------

class _ScanWorker(QObject):
    finished = Signal(list)   # List[SeriesEntry]
    progress = Signal(int, int)
    error    = Signal(str)

    def __init__(self, directory: str):
        super().__init__()
        self._directory = directory

    def run(self):
        try:
            indexer = DICOMIndexer(progress_callback=self.progress.emit)
            series  = indexer.scan(self._directory)
            self.finished.emit(series)
        except Exception as exc:
            self.error.emit(str(exc))


class _LoadWorker(QObject):
    finished = Signal(object, object)   # (vtkImageData, sitk.Image)
    error    = Signal(str)

    def __init__(self, file_paths: List[str]):
        super().__init__()
        self._file_paths = file_paths

    def run(self):
        try:
            vtk_img, sitk_img = DICOMLoader().load_series(self._file_paths)
            self.finished.emit(vtk_img, sitk_img)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# PatientsPage
# ---------------------------------------------------------------------------

class PatientsPage(WorkflowPage):
    """Stage 1: select a DICOM directory, pick a series, name the case."""

    volume_loaded = Signal(object, object, str)
    # ^ (vtkImageData, sitk.Image, case_name)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Patients", parent)
        self._series_list: List[SeriesEntry] = []
        self._selected_series: Optional[SeriesEntry] = None
        self._scan_thread: Optional[QThread] = None
        self._load_thread: Optional[QThread] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout

        # --- Directory picker ---
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("DICOM directory…")
        self._dir_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(QLabel("Directory:"))
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # --- Progress bar (hidden until scan) ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        # --- Series table ---
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Patient", "Description", "Modality", "Slices"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._table.itemSelectionChanged.connect(self._on_series_selected)
        layout.addWidget(self._table, stretch=1)

        # --- Case name + Load button ---
        case_row = QHBoxLayout()
        case_row.addWidget(QLabel("Case name:"))
        self._case_edit = QLineEdit()
        self._case_edit.setPlaceholderText("e.g. Patient_001")
        self._case_edit.textChanged.connect(self._update_load_btn)
        case_row.addWidget(self._case_edit, stretch=1)
        layout.addLayout(case_row)

        self._load_btn = QPushButton("Load Series")
        self._load_btn.setEnabled(False)
        self._load_btn.setFixedHeight(36)
        self._load_btn.clicked.connect(self._load_series)
        layout.addWidget(self._load_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select DICOM Directory", os.path.expanduser("~")
        )
        if not directory:
            return
        self._dir_edit.setText(directory)
        self._start_scan(directory)

    def _start_scan(self, directory: str):
        self._table.setRowCount(0)
        self._series_list.clear()
        self._selected_series = None
        self._update_load_btn()

        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status_lbl.setText("Scanning…")

        worker = _ScanWorker(directory)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self._on_scan_finished)
        worker.progress.connect(self._on_scan_progress)
        worker.error.connect(self._on_scan_error)
        thread.started.connect(worker.run)
        self._scan_thread = thread
        self._scan_worker = worker
        thread.start()

    def _on_scan_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_scan_finished(self, series: list):
        if self._scan_thread is not None:
            self._scan_thread.quit()
        self._progress.setVisible(False)
        self._series_list = series
        self._populate_table(series)
        n = len(series)
        self._status_lbl.setText(
            f"Found {n} series" if n else "No DICOM series found"
        )

    def _on_scan_error(self, msg: str):
        if self._scan_thread is not None:
            self._scan_thread.quit()
        self._progress.setVisible(False)
        self._status_lbl.setText(f"Scan error: {msg}")

    def _populate_table(self, series: List[SeriesEntry]):
        self._table.setRowCount(len(series))
        for row, s in enumerate(series):
            self._table.setItem(row, 0, QTableWidgetItem(s.patient_name))
            self._table.setItem(row, 1, QTableWidgetItem(s.series_description))
            self._table.setItem(row, 2, QTableWidgetItem(s.modality))
            self._table.setItem(row, 3, QTableWidgetItem(str(len(s.file_paths))))
        self._table.resizeColumnsToContents()

    def _on_series_selected(self):
        rows = self._table.selectedItems()
        if rows:
            row = self._table.currentRow()
            if 0 <= row < len(self._series_list):
                self._selected_series = self._series_list[row]
        else:
            self._selected_series = None
        self._update_load_btn()

    def _update_load_btn(self):
        has_series = self._selected_series is not None
        has_name   = bool(self._case_edit.text().strip())
        self._load_btn.setEnabled(has_series and has_name)

    def _load_series(self):
        if self._selected_series is None:
            return
        case_name = self._case_edit.text().strip()
        if not case_name:
            return

        self._load_btn.setEnabled(False)
        self._status_lbl.setText("Loading volume…")
        self._progress.setMaximum(0)   # indeterminate / marquee
        self._progress.setVisible(True)

        worker = _LoadWorker(self._selected_series.file_paths)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(
            lambda vtk_img, sitk_img: self._on_load_finished(vtk_img, sitk_img, case_name)
        )
        worker.error.connect(self._on_load_error)
        thread.started.connect(worker.run)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    def _on_load_finished(self, vtk_image, sitk_image, case_name: str):
        if self._load_thread is not None:
            self._load_thread.quit()
        self._progress.setVisible(False)
        self._load_btn.setEnabled(True)

        # Push to scene graph
        node = VolumeNode(
            node_id="ACTIVE_VOLUME",
            name=case_name,
            vtk_image_data=vtk_image,
            sitk_image=sitk_image,
        )
        self._scene_graph.add_node(node)

        self._status_lbl.setText(f"Loaded: {case_name}")
        self.volume_loaded.emit(vtk_image, sitk_image, case_name)
        self.emit_complete()

    def _on_load_error(self, msg: str):
        if self._load_thread is not None:
            self._load_thread.quit()
        self._progress.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_lbl.setText(f"Load error: {msg}")
        QMessageBox.critical(self, "Load Error", msg)
