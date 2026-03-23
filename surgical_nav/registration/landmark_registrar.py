"""LandmarkRegistrar: rigid-body landmark registration via Umeyama SVD.

Given corresponding point sets P (patient/physical space) and Q (image
space), finds the rigid transform T such that T @ P ≈ Q.

Algorithm (Umeyama 1991)
------------------------
    H = (P - P̄)ᵀ (Q - Q̄)
    U, _, Vᵀ = svd(H)
    d = sign(det(Vᵀᵀ Uᵀ))           # reflection fix
    R = Vᵀᵀ diag(1,1,d) Uᵀ
    t = Q̄ - R P̄

The 4×4 result matrix is stored as IMAGE_REGISTRATION in the SceneGraph.

Quality
-------
RMSE < 3.0 mm (configurable).  Minimum 3 non-collinear point pairs required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Output of a landmark registration."""

    transform: np.ndarray     # (4,4) rigid body — maps P → Q
    rmse_mm:   float
    n_pairs:   int
    success:   bool = True
    message:   str  = "OK"


# ---------------------------------------------------------------------------
# LandmarkRegistrar
# ---------------------------------------------------------------------------

class LandmarkRegistrar:
    """Computes a rigid registration from paired landmark sets.

    Parameters
    ----------
    min_pairs : int
        Minimum number of point pairs (default 3).
    max_rmse_mm : float
        Maximum acceptable RMSE in mm (default 3.0).
    """

    def __init__(self, min_pairs: int = 3, max_rmse_mm: float = 3.0):
        self._min_pairs  = min_pairs
        self._max_rmse   = max_rmse_mm
        self._p: List[np.ndarray] = []   # physical (patient) points
        self._q: List[np.ndarray] = []   # image points

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add_pair(self, p_physical: np.ndarray, q_image: np.ndarray) -> None:
        """Add a correspondence: p_physical ↔ q_image (both (3,) arrays)."""
        p = np.asarray(p_physical, dtype=np.float64).ravel()
        q = np.asarray(q_image,    dtype=np.float64).ravel()
        if p.shape != (3,) or q.shape != (3,):
            raise ValueError("Each point must be a (3,) array")
        self._p.append(p.copy())
        self._q.append(q.copy())

    def clear(self) -> None:
        self._p.clear()
        self._q.clear()

    @property
    def pair_count(self) -> int:
        return len(self._p)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> RegistrationResult:
        """Run Umeyama SVD rigid registration on the collected pairs."""
        n = len(self._p)
        if n < self._min_pairs:
            return RegistrationResult(
                transform=np.eye(4), rmse_mm=float("inf"), n_pairs=n,
                success=False,
                message=f"Need ≥{self._min_pairs} pairs (have {n})",
            )

        P = np.stack(self._p)   # (N,3) physical
        Q = np.stack(self._q)   # (N,3) image

        # Check for collinearity
        if _are_collinear(P):
            return RegistrationResult(
                transform=np.eye(4), rmse_mm=float("inf"), n_pairs=n,
                success=False,
                message="Physical points are collinear — spread them further apart",
            )

        R, t = _umeyama_svd(P, Q)

        # Compute RMSE
        P_aligned = (R @ P.T).T + t   # (N,3)
        rmse = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t

        if rmse > self._max_rmse:
            return RegistrationResult(
                transform=T, rmse_mm=rmse, n_pairs=n, success=False,
                message=f"RMSE {rmse:.3f} mm exceeds threshold {self._max_rmse} mm",
            )

        return RegistrationResult(transform=T, rmse_mm=rmse, n_pairs=n)

    # ------------------------------------------------------------------
    # Convenience: register from arrays directly (no state)
    # ------------------------------------------------------------------

    @classmethod
    def register_arrays(
        cls,
        P: np.ndarray,
        Q: np.ndarray,
        max_rmse_mm: float = 3.0,
    ) -> RegistrationResult:
        """Register two (N,3) arrays directly without accumulating state."""
        reg = cls(min_pairs=3, max_rmse_mm=max_rmse_mm)
        for p, q in zip(P, Q):
            reg.add_pair(p, q)
        return reg.register()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _umeyama_svd(P: np.ndarray, Q: np.ndarray):
    """Return (R, t) such that R @ P[i] + t ≈ Q[i].

    Parameters
    ----------
    P, Q : (N, 3) float64 arrays of corresponding points.
    """
    P_bar = P.mean(axis=0)
    Q_bar = Q.mean(axis=0)

    H = (P - P_bar).T @ (Q - Q_bar)
    U, _, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    t = Q_bar - R @ P_bar
    return R, t


def _are_collinear(pts: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if all rows of *pts* lie on a single line."""
    if len(pts) < 3:
        return True
    centred = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centred)
    # If second singular value is near zero, points are collinear
    return float(s[1]) < tol
