"""MainWindow: top-level application window.

Hosts:
  - Toolbar with workflow stage buttons (Patients / Planning / Registration /
    Navigation / Landmarks)
  - QStackedWidget for page content (populated by callers)
  - Status bar showing PLUS connection, pointer tool, head-frame tool state
  - Rendering container managed by LayoutManager

The window does NOT own the page objects — callers create pages and register
them via ``add_page()``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QLabel, QStackedWidget, QSizePolicy, QSplitter,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QColor

from surgical_nav.rendering.layout_manager import LayoutManager


# Workflow stage definitions: (button label, page index, required previous stage)
_STAGES = [
    ("Patients",      0, -1),
    ("Planning",      1,  0),
    ("Registration",  2,  1),
    ("Navigation",    3,  2),
    ("Landmarks",     4, -1),   # accessible any time
    ("Tracking Test", 5, -1),   # accessible any time
]


class MainWindow(QMainWindow):
    """Top-level window for the surgical navigation application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgical Navigation")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # Track which stages have been completed
        self._completed_stages: set[int] = set()

        # --- Central area ---------------------------------------------------
        # QSplitter: left = workflow controls stack, right = viewer panel
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self._splitter)

        # Left panel: workflow page controls
        self._stack = QStackedWidget()
        self._stack.setMinimumWidth(260)
        self._stack.setMaximumWidth(420)
        self._splitter.addWidget(self._stack)

        # Right panel: placeholder until set_viewer_panel() is called
        self._viewer_panel = QLabel(
            "No viewer", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self._splitter.addWidget(self._viewer_panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        # --- Toolbar --------------------------------------------------------
        self._toolbar = QToolBar("Workflow")
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        _tooltips = [
            "Load a DICOM volume to begin",
            "Segment anatomy and plan trajectory (requires loaded volume)",
            "Calibrate instruments and register patient (requires planning)",
            "Real-time navigation (requires registration)",
            "Manage anatomical landmarks",
            "Test FALCON tracker with video files",
        ]

        self._stage_actions: list[QAction] = []
        for (label, idx, _), tip in zip(_STAGES, _tooltips):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(idx)
            action.setToolTip(tip)
            action.triggered.connect(self._on_stage_action)
            self._toolbar.addAction(action)
            self._stage_actions.append(action)

        # Disable stages that require a prerequisite; keep -1 stages always enabled
        for action in self._stage_actions:
            _, _, req = _STAGES[action.data()]
            action.setEnabled(req == -1)

        # --- Status bar -----------------------------------------------------
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._lbl_case = QLabel("No case")
        self._lbl_plus = _StatusLight("PLUS")
        self._lbl_pointer = _StatusLight("Pointer")
        self._lbl_head_frame = _StatusLight("Head Frame")

        self._status_bar.addPermanentWidget(self._lbl_case)
        self._status_bar.addPermanentWidget(self._lbl_plus)
        self._status_bar.addPermanentWidget(self._lbl_pointer)
        self._status_bar.addPermanentWidget(self._lbl_head_frame)

    # ------------------------------------------------------------------
    # Page management
    # ------------------------------------------------------------------

    def add_page(self, widget: QWidget) -> int:
        """Append a page to the stack. Returns the page index."""
        return self._stack.addWidget(widget)

    def set_page(self, index: int):
        """Switch to the given page index (no precondition check)."""
        if 0 <= index < self._stack.count():
            self._stack.setCurrentIndex(index)
            for action in self._stage_actions:
                action.setChecked(action.data() == index)

    def current_page(self) -> int:
        return self._stack.currentIndex()

    def set_viewer_panel(self, widget: QWidget):
        """Replace the right-hand viewer panel with *widget*."""
        old = self._splitter.widget(1)
        self._splitter.replaceWidget(1, widget)
        self._splitter.setStretchFactor(1, 1)
        if old is not None and old is not widget:
            old.setParent(None)

    # ------------------------------------------------------------------
    # Stage gate
    # ------------------------------------------------------------------

    def mark_stage_complete(self, stage_index: int):
        """Unlock the next stage once the current one is done."""
        self._completed_stages.add(stage_index)
        next_stage = stage_index + 1
        if next_stage < len(self._stage_actions):
            self._stage_actions[next_stage].setEnabled(True)

    def _on_stage_action(self):
        action: QAction = self.sender()
        idx = action.data()
        self.set_page(idx)

    # ------------------------------------------------------------------
    # Status bar helpers
    # ------------------------------------------------------------------

    def set_case_name(self, name: str):
        self._lbl_case.setText(f"Case: {name}")

    def set_plus_status(self, connected: bool):
        self._lbl_plus.set_state("green" if connected else "red")

    def set_tool_status(self, tool: str, status: str):
        """status: 'SEEN' | 'NOT_SEEN' | 'NEVER_SEEN'"""
        color = {"SEEN": "green", "NOT_SEEN": "yellow", "NEVER_SEEN": "red"}.get(status, "gray")
        if tool == "Pointer":
            self._lbl_pointer.set_state(color)
        elif tool == "HeadFrame":
            self._lbl_head_frame.set_state(color)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        # Initialise VTK widgets in the workflow stack pages
        for i in range(self._stack.count()):
            _initialize_vtk_children(self._stack.widget(i))
        # Initialise VTK widgets in the right-hand viewer panel
        viewer_panel = self._splitter.widget(1)
        if viewer_panel is not None:
            _initialize_vtk_children(viewer_panel)


def _initialize_vtk_children(widget: QWidget):
    """Recursively call initialize() on any VTKWidget descendants."""
    from surgical_nav.rendering.vtk_widget import VTKWidget
    from surgical_nav.rendering.slice_viewer import SliceViewer
    for child in widget.findChildren(QWidget):
        if isinstance(child, (VTKWidget, SliceViewer)):
            child.initialize()


class _StatusLight(QWidget):
    """A small coloured circle + label for the status bar."""

    _COLORS = {
        "green":  "#44cc44",
        "yellow": "#cccc00",
        "red":    "#cc4444",
        "gray":   "#888888",
    }

    def __init__(self, label: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 8, 0)
        layout.setSpacing(4)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        layout.addWidget(self._dot)
        layout.addWidget(QLabel(label))
        self.set_state("gray")

    def set_state(self, color: str):
        hex_color = self._COLORS.get(color, "#888888")
        self._dot.setStyleSheet(f"color: {hex_color}; font-size: 14px;")
