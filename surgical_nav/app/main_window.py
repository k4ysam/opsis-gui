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
    QToolBar, QStatusBar, QLabel, QStackedWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QColor

from surgical_nav.rendering.layout_manager import LayoutManager


# Workflow stage definitions: (button label, page index, required previous stage)
_STAGES = [
    ("Patients",     0, -1),
    ("Planning",     1,  0),
    ("Registration", 2,  1),
    ("Navigation",   3,  2),
    ("Landmarks",    4, -1),   # accessible any time after planning
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
        central = QWidget()
        self.setCentralWidget(central)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Page stack
        self._stack = QStackedWidget()
        central_layout.addWidget(self._stack)

        # --- Toolbar --------------------------------------------------------
        self._toolbar = QToolBar("Workflow")
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        self._stage_actions: list[QAction] = []
        for label, idx, _ in _STAGES:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(idx)
            action.triggered.connect(self._on_stage_action)
            self._toolbar.addAction(action)
            self._stage_actions.append(action)

        # Disable all stages except Patients initially
        for action in self._stage_actions[1:]:
            action.setEnabled(False)

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
        # Initialise all VTKWidget instances inside the stack
        for i in range(self._stack.count()):
            page = self._stack.widget(i)
            _initialize_vtk_children(page)


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
