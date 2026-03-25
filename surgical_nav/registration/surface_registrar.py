"""SurfaceRegistrar: ICP surface-to-surface registration using VTK.

Wraps ``vtkIterativeClosestPointTransform`` to refine an initial rigid
alignment (typically from landmark registration) against a surface mesh.

The result is stored as IMAGE_REGISTRATION_REFINEMENT in the SceneGraph.

Parameters (defaults match SlicerOpenNav)
-----------------------------------------
max_iterations : 50
max_landmarks  : 200
start_by_matching_centroids : True
mode : RigidBody

Quality
-------
Mean closest-point distance < 3.0 mm (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    import vtkmodules.all as vtk
    _VTK = True
except ImportError:
    vtk = None  # type: ignore
    _VTK = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ICPResult:
    """Output of ICP surface registration."""

    transform:      np.ndarray    # (4,4) refinement transform
    mean_distance:  float         # mean closest-point distance (mm)
    n_source_points: int
    success:        bool = True
    message:        str  = "OK"


# ---------------------------------------------------------------------------
# SurfaceRegistrar
# ---------------------------------------------------------------------------

class SurfaceRegistrar:
    """Refines a rigid alignment using ICP between two vtkPolyData surfaces.

    Parameters
    ----------
    max_iterations : int
        ICP iteration limit (default 50).
    max_landmarks : int
        Number of source points used per iteration (default 200).
    max_mean_distance_mm : float
        Acceptance threshold on mean closest-point distance (default 3.0).
    start_by_matching_centroids : bool
        Call ``StartByMatchingCentroidsOn()`` before running (default True).
    """

    def __init__(
        self,
        max_iterations: int = 50,
        max_landmarks: int = 200,
        max_mean_distance_mm: float = 3.0,
        start_by_matching_centroids: bool = True,
    ):
        self._max_iter      = max_iterations
        self._max_lm        = max_landmarks
        self._max_dist      = max_mean_distance_mm
        self._match_centroids = start_by_matching_centroids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        initial_transform: np.ndarray | None = None,
    ) -> ICPResult:
        """Run ICP aligning *source* onto *target*.

        Parameters
        ----------
        source : vtkPolyData
            The surface to be aligned (e.g. patient skin from tracker).
        target : vtkPolyData
            The reference surface (e.g. CT skin mesh).
        initial_transform : (4,4) ndarray, optional
            Pre-alignment from landmark registration.  Applied to *source*
            before ICP if provided.

        Returns
        -------
        ICPResult
        """
        if source.GetNumberOfPoints() == 0 or target.GetNumberOfPoints() == 0:
            return ICPResult(
                transform=np.eye(4), mean_distance=float("inf"),
                n_source_points=0, success=False,
                message="Source or target surface has no points",
            )

        # Apply initial transform to source if provided
        if initial_transform is not None:
            vtk_init = _numpy_to_vtk_transform(initial_transform)
            tf_filter = vtk.vtkTransformPolyDataFilter()
            tf_filter.SetInputData(source)
            tf_filter.SetTransform(vtk_init)
            tf_filter.Update()
            source_aligned = tf_filter.GetOutput()
        else:
            source_aligned = source

        # Build ICP
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source_aligned)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(self._max_iter)
        icp.SetMaximumNumberOfLandmarks(self._max_lm)
        if self._match_centroids:
            icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        # Extract 4×4 matrix
        vtk_mat = icp.GetMatrix()
        refinement = _vtk_matrix_to_numpy(vtk_mat)

        # Compute mean closest-point distance on the refined source
        combined = np.eye(4) if initial_transform is None else initial_transform
        full_transform = refinement @ combined if initial_transform is not None else refinement

        mean_dist = _mean_closest_point_distance(source, target, full_transform)

        if mean_dist > self._max_dist:
            return ICPResult(
                transform=refinement,
                mean_distance=mean_dist,
                n_source_points=source.GetNumberOfPoints(),
                success=False,
                message=(
                    f"Mean distance {mean_dist:.3f} mm exceeds "
                    f"threshold {self._max_dist} mm"
                ),
            )

        return ICPResult(
            transform=refinement,
            mean_distance=mean_dist,
            n_source_points=source.GetNumberOfPoints(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_vtk_transform(m: np.ndarray) -> vtk.vtkTransform:
    vtk_m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_m.SetElement(i, j, float(m[i, j]))
    t = vtk.vtkTransform()
    t.SetMatrix(vtk_m)
    return t


def _vtk_matrix_to_numpy(vtk_mat: vtk.vtkMatrix4x4) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            m[i, j] = vtk_mat.GetElement(i, j)
    return m


def _mean_closest_point_distance(
    source: vtk.vtkPolyData,
    target: vtk.vtkPolyData,
    transform: np.ndarray,
) -> float:
    """Apply *transform* to *source* points and compute mean distance to *target*."""
    # Build cell locator on target
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()

    pts = source.GetPoints()
    n = pts.GetNumberOfPoints()
    if n == 0:
        return float("inf")

    total = 0.0
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id  = vtk.reference(0)
    dist2   = vtk.reference(0.0)

    R = transform[:3, :3]
    t = transform[:3,  3]

    sample_step = max(1, n // 500)   # limit to ~500 samples for speed
    count = 0
    for i in range(0, n, sample_step):
        p = np.array(pts.GetPoint(i))
        p_t = R @ p + t
        locator.FindClosestPoint(p_t.tolist(), closest, cell_id, sub_id, dist2)
        total += float(dist2) ** 0.5
        count += 1

    return total / count if count > 0 else float("inf")
