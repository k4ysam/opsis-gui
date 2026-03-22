"""WorkflowPage: abstract base class for all workflow stage pages.

Each stage (Patients, Planning, Registration, Navigation, Landmarks) subclasses
this and implements ``on_enter`` / ``on_leave`` lifecycle hooks.

The base class holds a reference to the shared SceneGraph so subclasses
don't need to import it separately.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal

from surgical_nav.app.scene_graph import SceneGraph


class WorkflowPage(QWidget):
    """Abstract base for a workflow stage page.

    Signals
    -------
    stage_complete : Signal()
        Emit when the user has completed all required steps for this stage.
        The MainWindow connects this to ``mark_stage_complete()``.
    status_message : Signal(str)
        Emit a short string for the main window's status bar.
    """

    stage_complete = Signal()
    status_message = Signal(str)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = title
        self._scene_graph = SceneGraph.instance()

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(8, 8, 8, 8)
        self._root_layout.setSpacing(6)

        header = QLabel(title)
        header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        header.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 4px;"
            "background: #1e3a5f; color: white; border-radius: 4px;"
        )
        self._root_layout.addWidget(header)

    # ------------------------------------------------------------------
    # Lifecycle hooks — override in subclasses as needed
    # ------------------------------------------------------------------

    def on_enter(self):
        """Called when the page becomes visible (toolbar button pressed)."""

    def on_leave(self):
        """Called just before switching away from this page."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def scene_graph(self) -> SceneGraph:
        return self._scene_graph

    def emit_complete(self):
        """Convenience: emit stage_complete and a status message."""
        self.stage_complete.emit()
        self.status_message.emit(f"{self._title} complete")
