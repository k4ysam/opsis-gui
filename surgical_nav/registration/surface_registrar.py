"""SurfaceRegistrar: stub — VTK removed.

ICP surface-to-surface registration requires VTK.
Without VTK this module returns a failed ICPResult for all registration requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ICPResult:
    """Output of ICP surface registration."""

    transform:       np.ndarray    # (4,4) refinement transform
    mean_distance:   float         # mean closest-point distance (mm)
    n_source_points: int
    success:         bool = True
    message:         str  = "OK"


class SurfaceRegistrar:
    """Stub ICP registrar — returns failure result without VTK."""

    def __init__(
        self,
        max_iterations: int = 50,
        max_landmarks: int = 200,
        max_mean_distance_mm: float = 3.0,
        start_by_matching_centroids: bool = True,
    ):
        pass

    def register(self, source, target, initial_transform=None) -> ICPResult:
        """Return a failed result — VTK required for ICP registration."""
        return ICPResult(
            transform=np.eye(4),
            mean_distance=float("inf"),
            n_source_points=0,
            success=False,
            message="ICP registration unavailable: VTK not installed",
        )
