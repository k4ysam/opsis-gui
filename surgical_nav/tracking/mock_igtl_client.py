"""MockIGTLClient: synthetic circular-motion transforms for offline testing.

Emits the same Qt signals as IGTLClient so the rest of the application
can be developed and tested without a running PLUS server.

Circular path::

    PointerToTracker: tip traces a 50 mm radius circle in the XY-plane
    at 0.5 Hz.  HeadFrameToTracker: fixed identity.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal, QObject

from surgical_nav.tracking.transform_store import TransformStore


class MockIGTLClient(QObject):
    """Emits synthetic tracking transforms at a configurable rate.

    Signals
    -------
    transform_received(name, matrix)
        Emitted each time a transform is updated.  ``matrix`` is a
        (4, 4) float64 ndarray.
    tool_status_changed(name, status)
        Emitted when a tool transitions between
        ``"NEVER_SEEN" | "SEEN" | "NOT_SEEN"``.
    connected()
        Emitted when the mock "connection" starts.
    disconnected()
        Emitted when the mock stops.
    """

    transform_received = Signal(str, object)    # (name, ndarray 4×4)
    tool_status_changed = Signal(str, str)      # (name, status)
    connected          = Signal()
    disconnected       = Signal()

    # Tool names emitted by this mock
    TOOL_NAMES = ("PointerToTracker", "HeadFrameToTracker")

    def __init__(
        self,
        store: Optional[TransformStore] = None,
        hz: float = 10.0,
        radius_mm: float = 50.0,
        parent=None,
    ):
        super().__init__(parent)
        self._store     = store or TransformStore()
        self._period    = 1.0 / max(hz, 0.1)
        self._radius    = radius_mm
        self._running   = False
        self._thread: Optional[QThread] = None
        self._statuses: dict[str, str] = {n: "NEVER_SEEN" for n in self.TOOL_NAMES}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def store(self) -> TransformStore:
        return self._store

    def start(self) -> None:
        """Start emitting transforms in a background QThread."""
        if self._running:
            return
        self._running = True
        self._thread = QThread()
        # Move self to thread so run() executes there
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()
        self.connected.emit()

    def stop(self) -> None:
        """Stop emitting and clean up the thread."""
        self._running = False
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self.disconnected.emit()

    # ------------------------------------------------------------------
    # Internal loop (runs on background thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        t0 = time.monotonic()
        while self._running:
            t = time.monotonic() - t0
            self._emit_pointer(t)
            self._emit_headframe()
            time.sleep(self._period)

    def _emit_pointer(self, t: float) -> None:
        angle = 2.0 * math.pi * 0.5 * t   # 0.5 Hz
        tx = self._radius * math.cos(angle)
        ty = self._radius * math.sin(angle)
        tz = 0.0

        m = np.eye(4, dtype=np.float64)
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz

        name = "PointerToTracker"
        self._store.update(name, m)
        self.transform_received.emit(name, m.copy())
        self._update_status(name, "SEEN")

    def _emit_headframe(self) -> None:
        name = "HeadFrameToTracker"
        m = np.eye(4, dtype=np.float64)
        self._store.update(name, m)
        self.transform_received.emit(name, m.copy())
        self._update_status(name, "SEEN")

    def _update_status(self, name: str, new_status: str) -> None:
        if self._statuses.get(name) != new_status:
            self._statuses[name] = new_status
            self.tool_status_changed.emit(name, new_status)
