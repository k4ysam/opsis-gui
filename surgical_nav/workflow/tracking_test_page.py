"""TrackingTestPage: test the FALCON multi-camera tracker with video files.

Lets the user select 5 video files, start/stop the tracker, view the live
6D pose, and see a 3D trajectory in the right-panel TrackingViewer.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Signal

from surgical_nav.workflow.base_page import WorkflowPage
from surgical_nav.tracking.falcon_tracker import FalconTracker

# Default video paths
_VIDEO_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'falcon', 'Tracking', 'videos',
    'drive-download-20260402T223017Z-1-001',
))
_DEFAULT_VIDEOS = [
    os.path.join(_VIDEO_DIR, f"{i}.mp4") for i in range(1, 6)
]


class TrackingTestPage(WorkflowPage):
    """Stage 5 (always accessible): test FALCON tracker with video files."""

    tracker_started  = Signal(object)   # emits FalconTracker instance
    tracker_stopped  = Signal()
    transform_received = Signal(str, object)   # forwarded for TrackingViewer

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Tracking Test", parent, show_back=False)
        self._tracker: Optional[FalconTracker] = None
        self._video_edits: list[QLineEdit] = []
        self._pose_labels: dict[str, QLabel] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_video_paths(self) -> list[str]:
        return [e.text() for e in self._video_edits]

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = self._root_layout

        # --- Video source selection ---
        video_box = QGroupBox("Video Sources")
        video_layout = QVBoxLayout(video_box)
        video_layout.setSpacing(4)

        for i in range(1, 6):
            row = QHBoxLayout()
            row.setSpacing(4)

            lbl = QLabel(f"Cam {i}:")
            lbl.setFixedWidth(42)
            row.addWidget(lbl)

            edit = QLineEdit()
            default = _DEFAULT_VIDEOS[i - 1]
            if os.path.isfile(default):
                edit.setText(default)
            else:
                edit.setPlaceholderText("Select .mp4 file…")
            edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row.addWidget(edit)
            self._video_edits.append(edit)

            browse = QPushButton("…")
            browse.setFixedWidth(28)
            browse.clicked.connect(lambda _, idx=i: self._browse(idx))
            row.addWidget(browse)

            video_layout.addLayout(row)

        layout.addWidget(video_box)

        # --- Start / Stop ---
        self._start_btn = QPushButton("Start Tracking")
        self._start_btn.setCheckable(True)
        self._start_btn.setStyleSheet(
            "QPushButton { background: #2d6a2d; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton:checked { background: #8b2020; }"
        )
        self._start_btn.toggled.connect(self._on_toggled)
        layout.addWidget(self._start_btn)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_lbl)

        # --- Current pose display ---
        pose_box = QGroupBox("Current Pose")
        pose_layout = QVBoxLayout(pose_box)
        pose_layout.setSpacing(2)

        for field, unit in [("X", "mm"), ("Y", "mm"), ("Z", "mm"),
                             ("Yaw", "°"), ("Pitch", "°"), ("Roll", "°")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{field}:"))
            val = QLabel("—")
            val.setStyleSheet("font-weight: bold; color: #4a9eff;")
            row.addWidget(val)
            row.addWidget(QLabel(unit))
            row.addStretch()
            pose_layout.addLayout(row)
            self._pose_labels[field.lower()] = val

        layout.addWidget(pose_box)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self, index: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select video for Camera {index}",
            _VIDEO_DIR, "Video files (*.mp4 *.avi *.mov);;All files (*)"
        )
        if path:
            self._video_edits[index - 1].setText(path)

    def _on_toggled(self, checked: bool):
        if checked:
            self._start_tracking()
        else:
            self._stop_tracking()

    def _start_tracking(self):
        paths = self.get_video_paths()
        valid = [p for p in paths if p and os.path.isfile(p)]
        if not valid:
            self._status_lbl.setText("No valid video files selected")
            self._start_btn.setChecked(False)
            return

        self._tracker = FalconTracker(video_paths=valid, fps=30)
        self._tracker.transform_received.connect(self._on_transform)
        self._tracker.disconnected.connect(self._on_tracker_disconnected)
        self._tracker.start()

        self._start_btn.setText("Stop Tracking")
        self._status_lbl.setText(f"Tracking {len(valid)} camera(s)…")
        self.tracker_started.emit(self._tracker)

    def _stop_tracking(self):
        if self._tracker:
            self._tracker.stop()
            self._tracker = None
        self._start_btn.setText("Start Tracking")
        self._status_lbl.setText("Stopped")
        self._reset_pose_labels()
        self.tracker_stopped.emit()

    def _on_tracker_disconnected(self):
        # Called if tracker stops on its own
        if self._start_btn.isChecked():
            self._start_btn.setChecked(False)

    def _on_transform(self, name: str, matrix: np.ndarray):
        x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        self._pose_labels["x"].setText(f"{x:.2f}")
        self._pose_labels["y"].setText(f"{y:.2f}")
        self._pose_labels["z"].setText(f"{z:.2f}")

        try:
            from scipy.spatial.transform import Rotation as R_scipy
            rot = R_scipy.from_matrix(matrix[:3, :3].T)
            yaw, pitch, roll = rot.as_euler('ZYX', degrees=True)
            self._pose_labels["yaw"].setText(f"{yaw:.1f}")
            self._pose_labels["pitch"].setText(f"{pitch:.1f}")
            self._pose_labels["roll"].setText(f"{roll:.1f}")
        except Exception:
            pass

        self.transform_received.emit(name, matrix)

    def _reset_pose_labels(self):
        for lbl in self._pose_labels.values():
            lbl.setText("—")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_leave(self):
        if self._start_btn.isChecked():
            self._start_btn.setChecked(False)
