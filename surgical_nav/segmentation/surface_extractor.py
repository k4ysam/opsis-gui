"""SurfaceExtractor: converts a binary label map to a smooth vtkPolyData mesh.

Pipeline:
  1. sitk.Image (uint8 label) → vtkImageData via numpy bridge
  2. vtkMarchingCubes at isovalue 0.5
  3. vtkSmoothPolyDataFilter (Laplacian, 50 iterations)
  4. vtkPolyDataNormals for lighting

The resulting mesh is in the same coordinate space as the input image
(RAS when the caller has already converted from LPS).

Usage::

    extractor = SurfaceExtractor()
    skin_mesh = extractor.extract(skin_label_sitk, isovalue=0.5)
    # Returns vtkPolyData ready for VolumeViewer.add_model()
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    _VTK = True
except ImportError:
    vtk = None  # type: ignore
    numpy_to_vtk = None  # type: ignore
    _VTK = False


class SurfaceExtractor:
    """Extracts an isosurface mesh from a binary SimpleITK label image."""

    def __init__(
        self,
        smooth_iterations: int = 50,
        smooth_relaxation: float = 0.1,
        compute_normals: bool = True,
    ):
        self._smooth_iterations = smooth_iterations
        self._smooth_relaxation = smooth_relaxation
        self._compute_normals   = compute_normals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        label_image: sitk.Image,
        isovalue: float = 0.5,
    ) -> vtk.vtkPolyData:
        """Extract a surface mesh from a binary label image.

        Parameters
        ----------
        label_image : sitk.Image
            Binary uint8 label map (1 = foreground).
        isovalue : float
            Marching-cubes isovalue (default 0.5 — between 0 and 1).

        Returns
        -------
        vtkPolyData
            Smoothed surface mesh.  Empty polydata if the label is all zeros.
        """
        vtk_image = self._sitk_to_vtk(label_image)

        # --- Marching Cubes ---
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(vtk_image)
        mc.SetValue(0, isovalue)
        mc.ComputeNormalsOff()
        mc.Update()

        if mc.GetOutput().GetNumberOfPoints() == 0:
            return vtk.vtkPolyData()

        # --- Smoothing ---
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(self._smooth_iterations)
        smoother.SetRelaxationFactor(self._smooth_relaxation)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

        if not self._compute_normals:
            return smoother.GetOutput()

        # --- Normals ---
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.Update()

        return normals.GetOutput()

    # ------------------------------------------------------------------
    # Internal: SimpleITK → vtkImageData (float, preserves spacing/origin)
    # ------------------------------------------------------------------

    @staticmethod
    def _sitk_to_vtk(label_image: sitk.Image) -> vtk.vtkImageData:
        """Convert a SimpleITK image to vtkImageData (float32 scalars)."""
        arr = sitk.GetArrayFromImage(label_image).astype(np.float32)  # (z, y, x)
        nz, ny, nx = arr.shape

        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(*label_image.GetSpacing())
        vtk_img.SetOrigin(*label_image.GetOrigin())

        flat = arr.transpose(2, 1, 0).ravel(order="F")
        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName("labels")
        vtk_img.GetPointData().SetScalars(vtk_arr)

        return vtk_img
