"""IGTLClient: OpenIGTLink client using pyigtl.

Polls for transforms from a PLUS server in a background QThread and emits
Qt signals to the main thread.  Falls back gracefully if pyigtl is not
installed (import-time check).

Signals are identical to MockIGTLClient so either class can be substituted
without changing the rest of the application.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal, QObject

from surgical_nav.tracking.transform_store import TransformStore

# pyigtl is optional; degrade gracefully if not installed
try:
    import pyigtl  # type: ignore
    _PYIGTL_AVAILABLE = True
except ImportError:
    _PYIGTL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Staleness thresholds
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_S = 1.0  # > 1 s without update → NOT_SEEN


# ---------------------------------------------------------------------------
# Internal worker (lives on background thread)
# ---------------------------------------------------------------------------

class _IGTLWorker(QObject):
    transform_ready = Signal(str, object)   # (name, ndarray 4×4)
    status_changed  = Signal(str, str)      # (name, status)
    error_occurred  = Signal(str)

    def __init__(
        self,
        host: str,
        port: int,
        store: TransformStore,
        poll_hz: float,
    ):
        super().__init__()
        self._host    = host
        self._port    = port
        self._store   = store
        self._period  = 1.0 / max(poll_hz, 1.0)
        self._running = False
        self._statuses: dict[str, str] = {}

    def run(self) -> None:
        if not _PYIGTL_AVAILABLE:
            self.error_occurred.emit(
                "pyigtl is not installed — install it with: pip install pyigtl"
            )
            return

        client = pyigtl.Client(host=self._host, port=self._port)
        self._running = True
        try:
            while self._running:
                messages = client.get_latest_messages()
                for msg in messages:
                    if isinstance(msg, pyigtl.TransformMessage):
                        name   = msg.device_name
                        matrix = np.array(msg.matrix, dtype=np.float64)
                        if matrix.shape == (4, 4):
                            self._store.update(name, matrix)
                            self.transform_ready.emit(name, matrix.copy())
                            self._maybe_status(name, "SEEN")

                # Detect stale tools
                for name in list(self._statuses):
                    if self._store.is_stale(name, _STALE_THRESHOLD_S):
                        self._maybe_status(name, "NOT_SEEN")

                time.sleep(self._period)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            try:
                client.disconnect()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False

    def _maybe_status(self, name: str, new_status: str) -> None:
        if self._statuses.get(name) != new_status:
            self._statuses[name] = new_status
            self.status_changed.emit(name, new_status)


# ---------------------------------------------------------------------------
# Public API — stays on main thread
# ---------------------------------------------------------------------------

class IGTLClient(QObject):
    """OpenIGTLink client that polls a PLUS server for tracking transforms.

    Parameters
    ----------
    host : str
        PLUS server hostname or IP (default "localhost").
    port : int
        PLUS server port (default 18944).
    store : TransformStore, optional
        Shared store; one is created internally if not provided.
    poll_hz : float
        How often to call ``get_latest_messages()`` (default 10 Hz).

    Signals
    -------
    transform_received(name, matrix)
        Emitted whenever a new transform is received.
    tool_status_changed(name, status)
        ``"NEVER_SEEN" | "SEEN" | "NOT_SEEN"``
    connected()
        Emitted once the client thread starts.
    disconnected()
        Emitted once the client thread stops.
    error(message)
        Emitted on connection or processing errors.
    """

    transform_received  = Signal(str, object)
    tool_status_changed = Signal(str, str)
    connected           = Signal()
    disconnected        = Signal()
    error               = Signal(str)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 18944,
        store: Optional[TransformStore] = None,
        poll_hz: float = 10.0,
        parent=None,
    ):
        super().__init__(parent)
        self._host    = host
        self._port    = port
        self._store   = store or TransformStore()
        self._poll_hz = poll_hz
        self._thread: Optional[QThread]     = None
        self._worker: Optional[_IGTLWorker] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def store(self) -> TransformStore:
        return self._store

    @property
    def pyigtl_available(self) -> bool:
        return _PYIGTL_AVAILABLE

    def start(self) -> None:
        """Connect to PLUS server and start polling in a background thread."""
        if self._thread is not None:
            return
        self._worker = _IGTLWorker(
            self._host, self._port, self._store, self._poll_hz
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._worker.transform_ready.connect(self.transform_received)
        self._worker.status_changed.connect(self.tool_status_changed)
        self._worker.error_occurred.connect(self.error)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self.connected.emit()

    def stop(self) -> None:
        """Stop polling and disconnect."""
        if self._worker:
            self._worker.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self.disconnected.emit()
