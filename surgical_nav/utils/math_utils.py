"""Core math utilities: homogeneous transforms, RMSE, coordinate conversions."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def identity() -> np.ndarray:
    """Return a 4×4 identity transform."""
    return np.eye(4, dtype=np.float64)


def compose(*matrices: np.ndarray) -> np.ndarray:
    """Left-multiply a sequence of 4×4 matrices: compose(A, B, C) = A @ B @ C."""
    result = np.eye(4, dtype=np.float64)
    for m in matrices:
        result = result @ m
    return result


def invert_transform(m: np.ndarray) -> np.ndarray:
    """Invert a 4×4 rigid-body transform efficiently (R^T, -R^T t)."""
    R = m[:3, :3]
    t = m[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 matrix from a 3×3 rotation and a 3-vector translation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def apply_transform(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to an (N, 3) array of points.

    Returns an (N, 3) array.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones])          # (N, 4)
    transformed = (m @ hom.T).T           # (N, 4)
    return transformed[:, :3]


def points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert (N, 3) to (N, 4) by appending a column of ones."""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    return np.hstack([pts, ones])


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error between two (N, 3) point sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = a - b
    return float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))


def mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between paired (N, 3) point sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------

def lps_to_ras_transform() -> np.ndarray:
    """4×4 matrix that flips LPS → RAS (negate X and Y)."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = -1.0   # L → R
    m[1, 1] = -1.0   # P → A
    return m


def lps_to_ras(points: np.ndarray) -> np.ndarray:
    """Convert (N, 3) LPS coordinates to RAS."""
    pts = np.asarray(points, dtype=np.float64).copy()
    pts[:, 0] *= -1
    pts[:, 1] *= -1
    return pts
