"""SurfaceExtractor: stub — VTK removed.

Surface extraction from label maps requires VTK (marching cubes).
Without VTK this module returns None for all extraction requests.
"""

from __future__ import annotations

import SimpleITK as sitk


class SurfaceExtractor:
    """Stub surface extractor — returns None without VTK."""

    def __init__(
        self,
        smooth_iterations: int = 50,
        smooth_relaxation: float = 0.1,
        compute_normals: bool = True,
    ):
        self._smooth_iterations = smooth_iterations
        self._smooth_relaxation = smooth_relaxation
        self._compute_normals = compute_normals

    def extract(self, label_image: sitk.Image, isovalue: float = 0.5):
        """Return None — VTK required for marching-cubes surface extraction."""
        return None
