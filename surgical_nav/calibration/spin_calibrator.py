"""SpinCalibrator: shaft/axis-alignment calibration.

The spin calibration determines the tool shaft direction in tool
coordinates by fitting a line to the collection of tool-tip positions
as the tool is spun around its long axis.

When the tool is spun (rotated purely around its own shaft axis with the
tip stationary), the tracker origin traces a circle whose axis equals
the tool shaft direction.  We fit that direction via PCA on the centre-
subtracted tracker origin positions.

Result
------
The output is ``shaft_in_tool`` — a unit vector in tool (local)
coordinates pointing along the shaft.  Callers typically combine this
with the pivot result to get the full tool calibration matrix.

Quality check
-------------
The explained variance ratio of the first PCA component should be close
to 1 (ideal: only 1 degree of freedom in the motion).  If the ratio is
below ``min_linearity``, calibration is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpinResult:
    """Output of a spin calibration."""

    shaft_in_tool:  np.ndarray    # (3,) unit vector in tool coordinates
    shaft_in_world: np.ndarray    # (3,) unit vector in tracker coordinates
    linearity:      float         # fraction of variance on primary axis [0,1]
    n_samples:      int
    success:        bool = True
    message:        str  = "OK"


# ---------------------------------------------------------------------------
# SpinCalibrator
# ---------------------------------------------------------------------------

class SpinCalibrator:
    """Accumulates tracked transforms while the tool spins, then fits the axis.

    Parameters
    ----------
    min_samples : int
        Minimum number of transforms required (default 4).
    min_linearity : float
        Minimum fraction of variance that must lie on the primary PCA
        component (default 0.95 = 95 %).
    """

    def __init__(
        self,
        min_samples: int = 4,
        min_linearity: float = 0.95,
    ):
        self._min_samples   = min_samples
        self._min_linearity = min_linearity
        self._samples: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def add_sample(self, transform: np.ndarray) -> None:
        """Add a (4,4) float64 tracking transform."""
        m = np.asarray(transform, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Expected (4,4) matrix, got {m.shape}")
        self._samples.append(m.copy())

    def clear(self) -> None:
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> SpinResult:
        """Fit the tool shaft direction from the collected samples."""
        n = len(self._samples)

        if n < self._min_samples:
            return SpinResult(
                shaft_in_tool=np.array([0., 0., 1.]),
                shaft_in_world=np.array([0., 0., 1.]),
                linearity=0.0, n_samples=n, success=False,
                message=f"Need ≥{self._min_samples} samples (have {n})",
            )

        # Tracker origins (translation part of each transform)
        origins = np.stack([T[:3, 3] for T in self._samples])   # (N, 3)

        # PCA on centre-subtracted origins
        centre = origins.mean(axis=0)
        centred = origins - centre

        cov = centred.T @ centred / n
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending eigenvalues; largest is last
        primary_axis = eigenvectors[:, -1]
        total_var    = eigenvalues.sum()
        if total_var < 1e-12:
            return SpinResult(
                shaft_in_tool=np.array([0., 0., 1.]),
                shaft_in_world=np.array([0., 0., 1.]),
                linearity=0.0, n_samples=n, success=False,
                message="No motion detected — all tracker origins are identical",
            )

        linearity = float(eigenvalues[-1] / total_var)

        if linearity < self._min_linearity:
            return SpinResult(
                shaft_in_tool=np.array([0., 0., 1.]),
                shaft_in_world=primary_axis,
                linearity=linearity, n_samples=n, success=False,
                message=(
                    f"Linearity {linearity:.3f} below threshold "
                    f"{self._min_linearity:.3f}. Ensure pure spin motion."
                ),
            )

        # Express shaft in each tool's local frame and average
        # shaft_in_tool_i = R_i.T @ shaft_in_world
        shaft_local_sum = np.zeros(3)
        for T in self._samples:
            R = T[:3, :3]
            shaft_local_sum += R.T @ primary_axis
        shaft_in_tool = shaft_local_sum / n
        norm = np.linalg.norm(shaft_in_tool)
        if norm < 1e-9:
            shaft_in_tool = np.array([0., 0., 1.])
        else:
            shaft_in_tool /= norm

        return SpinResult(
            shaft_in_tool=shaft_in_tool,
            shaft_in_world=primary_axis,
            linearity=linearity,
            n_samples=n,
            success=True,
        )
