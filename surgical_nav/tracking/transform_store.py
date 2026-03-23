"""TransformStore: thread-safe dict of latest 4×4 tracking transforms.

Each entry records the transform matrix and a timestamp so staleness can
be detected by callers (e.g. igtl_client).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np


class TransformStore:
    """Thread-safe store for the latest OpenIGTLink transforms.

    Keys are transform names (e.g. 'PointerToTracker', 'HeadFrameToTracker').
    Values are ``(4×4 float64 ndarray, timestamp_float)``.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Tuple[np.ndarray, float]] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, name: str, matrix: np.ndarray) -> None:
        """Store the latest matrix for *name* (thread-safe)."""
        mat = np.array(matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Expected (4,4) matrix, got {mat.shape}")
        with self._lock:
            self._data[name] = (mat, time.monotonic())

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[np.ndarray]:
        """Return latest matrix for *name*, or None if not seen."""
        with self._lock:
            entry = self._data.get(name)
        return entry[0].copy() if entry is not None else None

    def get_with_timestamp(self, name: str) -> Optional[Tuple[np.ndarray, float]]:
        """Return ``(matrix_copy, timestamp)`` or None."""
        with self._lock:
            entry = self._data.get(name)
        if entry is None:
            return None
        return entry[0].copy(), entry[1]

    def is_stale(self, name: str, max_age_s: float = 1.0) -> bool:
        """Return True if *name* has never been seen or hasn't updated within *max_age_s*."""
        with self._lock:
            entry = self._data.get(name)
        if entry is None:
            return True
        return (time.monotonic() - entry[1]) > max_age_s

    def names(self) -> list[str]:
        """Return a snapshot of all known transform names."""
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        """Remove all stored transforms."""
        with self._lock:
            self._data.clear()
