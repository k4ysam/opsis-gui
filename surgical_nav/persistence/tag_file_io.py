"""TagFileIO: read and write FALCON .tag and .xfm files.

.tag format (space-separated, one point-pair per line):
    x_fixed y_fixed z_fixed x_moving y_moving z_moving

.xfm format:
    Line 1: "Linear_Transform ="
    Lines 2-4: 3×4 matrix rows (rotation + translation in last column)
    Line 5: ";"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_tag_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a .tag file into (fixed_points, moving_points), each (N, 3).

    Skips blank lines and lines starting with '#'.
    Raises ValueError if any data line doesn't have exactly 6 floats.
    """
    fixed, moving = [], []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(
                f"{path}:{lineno}: expected 6 floats, got {len(parts)}: {raw!r}"
            )
        vals = [float(p) for p in parts]
        fixed.append(vals[:3])
        moving.append(vals[3:])
    if not fixed:
        raise ValueError(f"{path}: no valid point pairs found")
    return np.array(fixed, dtype=np.float64), np.array(moving, dtype=np.float64)


def write_tag_file(
    path: str | Path,
    fixed: np.ndarray,
    moving: np.ndarray,
) -> None:
    """Write paired point sets to .tag format."""
    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if fixed.shape != moving.shape or fixed.ndim != 2 or fixed.shape[1] != 3:
        raise ValueError("fixed and moving must both be (N, 3) arrays")
    lines = [
        f"{fx:.6f} {fy:.6f} {fz:.6f} {mx:.6f} {my:.6f} {mz:.6f}"
        for (fx, fy, fz), (mx, my, mz) in zip(fixed, moving)
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def read_xfm_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a .xfm file into (R_3x3, T_3).

    Expected format::
        Linear_Transform =
        r00 r01 r02 t0
        r10 r11 r12 t1
        r20 r21 r22 t2
        ;
    """
    rows = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line in ("Linear_Transform =", ";"):
            continue
        parts = line.split()
        if len(parts) == 4:
            rows.append([float(p) for p in parts])
    if len(rows) != 3:
        raise ValueError(f"{path}: expected 3 matrix rows, got {len(rows)}")
    mat = np.array(rows, dtype=np.float64)
    R = mat[:, :3]
    T = mat[:, 3]
    return R, T


def write_xfm_file(
    path: str | Path,
    R: np.ndarray,
    T: np.ndarray,
) -> None:
    """Write a rotation matrix and translation vector to .xfm format."""
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).ravel()
    if R.shape != (3, 3) or T.shape != (3,):
        raise ValueError("R must be (3,3) and T must be (3,)")
    lines = ["Linear_Transform ="]
    for i in range(3):
        lines.append(
            f"{R[i,0]:.6f} {R[i,1]:.6f} {R[i,2]:.6f} {T[i]:.6f}"
        )
    lines.append(";")
    Path(path).write_text("\n".join(lines) + "\n")
