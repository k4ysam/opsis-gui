"""PlusLauncher: manages a PlusServer subprocess.

Launches ``PlusServer --config-file <path>`` and monitors its stdout for
the "Server started" string.  Emits Qt signals when the server is ready,
exits, or produces an error.

PlusServer is optional; if not found on PATH the class still creates but
``start()`` will emit ``error`` immediately.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal


_STARTED_MARKER = "Server started"


class PlusLauncher(QObject):
    """Manages a PlusServer subprocess.

    Signals
    -------
    server_started()
        Emitted once "Server started" appears in stdout.
    server_stopped()
        Emitted when the process exits (normally or by stop()).
    error(message)
        Emitted if PlusServer cannot be found or the process fails to start.
    output_line(line)
        Emitted for each line of PlusServer stdout/stderr (useful for logs).
    """

    server_started = Signal()
    server_stopped = Signal()
    error          = Signal(str)
    output_line    = Signal(str)

    def __init__(self, config_path: str = "", parent=None):
        super().__init__(parent)
        self._config_path = config_path
        self._process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._started_reported = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> str:
        return self._config_path

    @config_path.setter
    def config_path(self, path: str) -> None:
        self._config_path = path

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Launch PlusServer with the configured config file."""
        if self.is_running:
            return

        plus_exe = shutil.which("PlusServer") or shutil.which("PlusServer.exe")
        if plus_exe is None:
            self.error.emit(
                "PlusServer not found on PATH. "
                "Install PLUS Toolkit and ensure PlusServer.exe is on PATH."
            )
            return

        cmd = [plus_exe]
        if self._config_path:
            cmd += ["--config-file", self._config_path]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.error.emit(f"Failed to launch PlusServer: {exc}")
            return

        self._started_reported = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_stdout, daemon=True
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Terminate PlusServer if running."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitor_stdout(self) -> None:
        """Read PlusServer stdout in a daemon thread, emit signals."""
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                self.output_line.emit(line)
                if not self._started_reported and _STARTED_MARKER in line:
                    self._started_reported = True
                    self.server_started.emit()
        except Exception:
            pass
        finally:
            self.server_stopped.emit()
