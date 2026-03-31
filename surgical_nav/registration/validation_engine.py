"""ValidationEngine: FRE/TRE computation for LARK landmark validation.

Loads ground-truth .tag files, registers captured physical-space points
against image-space ground-truth using Procrustes (via LandmarkRegistrar),
and computes per-point FRE and TRE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from surgical_nav.persistence.tag_file_io import read_tag_file
from surgical_nav.registration.landmark_registrar import LandmarkRegistrar


@dataclass
class ValidationResult:
    """Output of a FRE/TRE validation run."""

    fre_per_point: np.ndarray     # (N,) mm — per-fiducial registration error
    fre_rmse: float               # mm
    tre_per_point: np.ndarray     # (M,) mm — per-target error
    tre_rmse: float               # mm
    registration_R: np.ndarray    # (3, 3)
    registration_T: np.ndarray    # (3,)
    n_fiducial_pairs: int
    n_target_pairs: int
    success: bool
    message: str


class ValidationEngine:
    """Compute FRE and TRE between captured LARK points and ground truth.

    Usage::

        engine = ValidationEngine()
        result = engine.validate(
            captured_points=np.array([[x1,y1,z1], ...]),  # physical-space from LARK
            ground_truth_tag="GT.tag",                     # .tag file path
        )
    """

    def validate(
        self,
        captured_points: np.ndarray,
        ground_truth_tag: str | Path,
        held_out_indices: list[int] | None = None,
    ) -> ValidationResult:
        """Register captured_points against ground truth and compute errors.

        Parameters
        ----------
        captured_points : (N, 3) array
            Physical-space points captured via LARK pointer.
        ground_truth_tag : path
            .tag file with (fixed=image-space, moving=physical-space) point pairs.
        held_out_indices : list of int, optional
            Indices to hold out as TRE targets. If None, uses leave-one-out.
        """
        captured = np.asarray(captured_points, dtype=np.float64)
        if captured.ndim != 2 or captured.shape[1] != 3:
            return ValidationResult(
                fre_per_point=np.array([]),
                fre_rmse=float("inf"),
                tre_per_point=np.array([]),
                tre_rmse=float("inf"),
                registration_R=np.eye(3),
                registration_T=np.zeros(3),
                n_fiducial_pairs=0,
                n_target_pairs=0,
                success=False,
                message="captured_points must be an (N, 3) array",
            )

        try:
            gt_image, _ = read_tag_file(ground_truth_tag)
        except Exception as exc:
            return ValidationResult(
                fre_per_point=np.array([]),
                fre_rmse=float("inf"),
                tre_per_point=np.array([]),
                tre_rmse=float("inf"),
                registration_R=np.eye(3),
                registration_T=np.zeros(3),
                n_fiducial_pairs=0,
                n_target_pairs=0,
                success=False,
                message=f"Failed to read ground truth: {exc}",
            )

        n = min(len(captured), len(gt_image))
        if n < 3:
            return ValidationResult(
                fre_per_point=np.array([]),
                fre_rmse=float("inf"),
                tre_per_point=np.array([]),
                tre_rmse=float("inf"),
                registration_R=np.eye(3),
                registration_T=np.zeros(3),
                n_fiducial_pairs=n,
                n_target_pairs=0,
                success=False,
                message=f"Need ≥3 point pairs for registration (have {n})",
            )

        P = captured[:n]       # physical (moving)
        Q = gt_image[:n]       # image (fixed)

        # Determine fiducial vs target split
        if held_out_indices:
            target_idx = [i for i in held_out_indices if 0 <= i < n]
            fid_idx = [i for i in range(n) if i not in target_idx]
        else:
            # Leave-one-out: all are fiducials, all are targets
            fid_idx = list(range(n))
            target_idx = list(range(n))

        P_fid = P[fid_idx]
        Q_fid = Q[fid_idx]

        result = LandmarkRegistrar.register_arrays(P_fid, Q_fid, max_rmse_mm=float("inf"))
        if not result.success and result.rmse_mm == float("inf"):
            return ValidationResult(
                fre_per_point=np.array([]),
                fre_rmse=float("inf"),
                tre_per_point=np.array([]),
                tre_rmse=float("inf"),
                registration_R=np.eye(3),
                registration_T=np.zeros(3),
                n_fiducial_pairs=len(fid_idx),
                n_target_pairs=len(target_idx),
                success=False,
                message=result.message,
            )

        R = result.transform[:3, :3]
        T = result.transform[:3, 3]

        # FRE: residual on fiducial points
        P_fid_aligned = (R @ P_fid.T).T + T
        fre_per_point = np.sqrt(np.sum((P_fid_aligned - Q_fid) ** 2, axis=1))
        fre_rmse = float(np.sqrt(np.mean(fre_per_point ** 2)))

        # TRE: apply same transform to target points
        P_tgt = P[target_idx]
        Q_tgt = Q[target_idx]
        P_tgt_aligned = (R @ P_tgt.T).T + T
        tre_per_point = np.sqrt(np.sum((P_tgt_aligned - Q_tgt) ** 2, axis=1))
        tre_rmse = float(np.sqrt(np.mean(tre_per_point ** 2))) if len(tre_per_point) > 0 else 0.0

        return ValidationResult(
            fre_per_point=fre_per_point,
            fre_rmse=fre_rmse,
            tre_per_point=tre_per_point,
            tre_rmse=tre_rmse,
            registration_R=R,
            registration_T=T,
            n_fiducial_pairs=len(fid_idx),
            n_target_pairs=len(target_idx),
            success=True,
            message=f"FRE={fre_rmse:.3f}mm, TRE={tre_rmse:.3f}mm",
        )
