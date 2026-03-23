"""PivotCalibrator: sphere-fit least-squares pivot calibration.

Given N tracked transforms T_i = [R_i | t_i] with the tool tip held
stationary while the handle rotates, solves for the tool-tip offset in
tool coordinates (p_tool) and the pivot point in tracker coordinates
(p_ref).

Formulation
-----------
For each transform:
    R_i @ p_tool + t_i = p_ref

Rearranged into a linear system A x = b:
    [R_i | -I_3] [p_tool; p_ref] = [-t_i]

Stacked for all N transforms:
    A ∈ R^{3N × 6},  b ∈ R^{3N}

Solved via np.linalg.lstsq.

Quality checks
--------------
- Minimum 4 transforms required.
- Angular spread of rotation axes must exceed 30° (collinearity check).
- RMSE of fit must be < 0.8 mm (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PivotResult:
    """Output of a successful pivot calibration."""

    p_tool: np.ndarray          # (3,) tip offset in tool coordinates
    p_ref:  np.ndarray          # (3,) pivot point in tracker coordinates
    rmse_mm: float
    n_samples: int
    success: bool = True
    message: str  = "OK"

    def as_transform(self) -> np.ndarray:
        """Return a 4×4 matrix T such that T maps tool origin → tool tip.

        The translation column equals p_tool; rotation is identity
        (calibration only determines the tip offset, not orientation).
        """
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = self.p_tool
        return m


# ---------------------------------------------------------------------------
# PivotCalibrator
# ---------------------------------------------------------------------------

class PivotCalibrator:
    """Accumulates tracked transforms and performs pivot calibration.

    Usage::

        cal = PivotCalibrator()
        # collect while rotating the tool around a fixed tip:
        for T in tracked_transforms:
            cal.add_sample(T)
        result = cal.calibrate()
        if result.success:
            print(f"Tip offset: {result.p_tool}  RMSE: {result.rmse_mm:.3f} mm")

    Parameters
    ----------
    min_samples : int
        Minimum number of transforms required (default 4).
    min_angular_spread_deg : float
        Minimum angular spread of the rotation axes to reject collinear
        samples (default 30°).
    max_rmse_mm : float
        Maximum acceptable RMSE in mm (default 0.8).
    """

    def __init__(
        self,
        min_samples: int = 4,
        min_angular_spread_deg: float = 30.0,
        max_rmse_mm: float = 0.8,
    ):
        self._min_samples    = min_samples
        self._min_spread_deg = min_angular_spread_deg
        self._max_rmse       = max_rmse_mm
        self._samples: List[np.ndarray] = []   # list of (4,4) matrices

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
        """Discard all collected samples."""
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> PivotResult:
        """Run pivot calibration on the collected samples.

        Returns a PivotResult with ``success=False`` and a descriptive
        ``message`` if any quality check fails.
        """
        n = len(self._samples)

        if n < self._min_samples:
            return PivotResult(
                p_tool=np.zeros(3), p_ref=np.zeros(3), rmse_mm=float("inf"),
                n_samples=n, success=False,
                message=f"Need ≥{self._min_samples} samples (have {n})",
            )

        # Check angular spread
        spread_ok, spread_deg = self._check_angular_spread(self._samples)
        if not spread_ok:
            return PivotResult(
                p_tool=np.zeros(3), p_ref=np.zeros(3), rmse_mm=float("inf"),
                n_samples=n, success=False,
                message=(
                    f"Insufficient angular spread {spread_deg:.1f}° "
                    f"(need ≥{self._min_spread_deg}°). Rotate the tool more."
                ),
            )

        # Build linear system
        A = np.zeros((3 * n, 6), dtype=np.float64)
        b = np.zeros(3 * n, dtype=np.float64)
        for i, T in enumerate(self._samples):
            R = T[:3, :3]
            t = T[:3, 3]
            A[3*i:3*i+3, :3] = R
            A[3*i:3*i+3, 3:] = -np.eye(3)
            b[3*i:3*i+3]     = -t

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        p_tool = x[:3]
        p_ref  = x[3:]

        # Compute RMSE
        rmse = self._compute_rmse(self._samples, p_tool, p_ref)

        if rmse > self._max_rmse:
            return PivotResult(
                p_tool=p_tool, p_ref=p_ref, rmse_mm=rmse,
                n_samples=n, success=False,
                message=f"RMSE {rmse:.3f} mm exceeds threshold {self._max_rmse} mm",
            )

        return PivotResult(
            p_tool=p_tool, p_ref=p_ref, rmse_mm=rmse,
            n_samples=n, success=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rmse(
        samples: List[np.ndarray],
        p_tool: np.ndarray,
        p_ref:  np.ndarray,
    ) -> float:
        """RMSE of || R_i @ p_tool + t_i - p_ref || over all samples."""
        errors = []
        for T in samples:
            R, t = T[:3, :3], T[:3, 3]
            diff = R @ p_tool + t - p_ref
            errors.append(float(np.dot(diff, diff)))
        return float(np.sqrt(np.mean(errors)))

    def _check_angular_spread(
        self, samples: List[np.ndarray]
    ) -> tuple[bool, float]:
        """Return (passes, spread_deg).

        Computes the rotation axis of each transform via the axis-angle
        representation and checks the maximum pairwise angular spread.
        """
        axes = []
        for T in samples:
            axis = _rotation_to_axis(T[:3, :3])
            if axis is not None:
                axes.append(axis)

        if len(axes) < 2:
            return False, 0.0

        # Maximum pairwise angle between axes (treat as unsigned, so clamp dot
        # to [−1, 1] and use arccos)
        max_angle = 0.0
        ref = axes[0]
        for ax in axes[1:]:
            dot = float(np.clip(np.dot(ref, ax), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(abs(dot))))
            max_angle = max(max_angle, angle)

        return max_angle >= self._min_spread_deg, max_angle


def _rotation_to_axis(R: np.ndarray) -> Optional[np.ndarray]:
    """Extract the rotation axis from a 3×3 rotation matrix.

    Returns a unit vector, or None if the rotation is near-identity.
    """
    # Rodrigues: axis ∝ skew-symmetric part
    skew = R - R.T
    v = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    return v / norm
