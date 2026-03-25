"""DICOMLoader: loads a DICOM series into a vtkImageData (RAS coordinates).

SimpleITK reads the series and returns an LPS image.  We convert to RAS by
negating the X and Y components of origin and direction cosines before
constructing the vtkImageData.

Usage::

    loader = DICOMLoader()
    vtk_image, sitk_image = loader.load_series(file_paths)
    # vtk_image: vtkImageData in RAS, ready for rendering
    # sitk_image: original SimpleITK.Image (kept for segmentation)
"""

from __future__ import annotations

from typing import List, Tuple

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


class DICOMLoader:
    """Loads a sorted list of DICOM file paths into VTK + SimpleITK objects."""

    def load_series(
        self, file_paths: List[str]
    ) -> Tuple[object, sitk.Image]:
        """Load *file_paths* as one DICOM series.

        Parameters
        ----------
        file_paths : list of str
            Sorted (by InstanceNumber) paths to the DICOM slices.

        Returns
        -------
        vtk_image : vtkImageData
            Volume in RAS coordinate space, scalar type float32.
        sitk_image : SimpleITK.Image
            Original LPS image kept for downstream segmentation.
        """
        sitk_image = self._read_sitk(file_paths)
        vtk_image  = self.sitk_to_vtk(sitk_image)
        return vtk_image, sitk_image

    # ------------------------------------------------------------------
    # SimpleITK loading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_sitk(file_paths: List[str]) -> sitk.Image:
        """Read a series from an explicit sorted file list."""
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        return sitk.Cast(image, sitk.sitkFloat32)

    # ------------------------------------------------------------------
    # SimpleITK → vtkImageData (LPS → RAS conversion)
    # ------------------------------------------------------------------

    @classmethod
    def sitk_to_vtk(cls, sitk_image: sitk.Image):
        """Convert a SimpleITK image (LPS) to vtkImageData (RAS).

        The pixel array is the same; only the origin and direction cosines
        are flipped on X and Y (LPS → RAS).
        """
        if not _VTK:
            return None

        # --- Extract numpy array (z, y, x) → reorder to (x, y, z) for VTK ---
        arr = sitk.GetArrayFromImage(sitk_image)   # shape: (z, y, x)
        arr = arr.astype(np.float32)

        spacing   = sitk_image.GetSpacing()         # (sx, sy, sz)
        origin_lps = np.array(sitk_image.GetOrigin())  # (ox, oy, oz) in LPS
        direction  = np.array(sitk_image.GetDirection()).reshape(3, 3)  # LPS cosines

        # LPS → RAS: flip X and Y
        origin_ras = origin_lps.copy()
        origin_ras[0] *= -1
        origin_ras[1] *= -1

        direction_ras = direction.copy()
        direction_ras[0, :] *= -1   # row 0 = X cosines
        direction_ras[1, :] *= -1   # row 1 = Y cosines

        # Build vtkImageData
        vtk_image = vtk.vtkImageData()
        nz, ny, nx = arr.shape
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(*spacing)
        vtk_image.SetOrigin(*origin_ras)

        # Flatten in VTK (Fortran / x-fastest) order
        flat = arr.transpose(2, 1, 0).ravel(order="F")
        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName("scalars")
        vtk_image.GetPointData().SetScalars(vtk_arr)

        # Store direction cosines as field data for later retrieval
        cos_flat = direction_ras.ravel().astype(np.float64)
        cos_vtk = numpy_to_vtk(cos_flat, deep=True)
        cos_vtk.SetName("DirectionCosines")
        vtk_image.GetFieldData().AddArray(cos_vtk)

        return vtk_image

    # ------------------------------------------------------------------
    # Convenience: auto-detect series files from a directory
    # ------------------------------------------------------------------

    @staticmethod
    def get_series_file_names(directory: str, series_uid: str = "") -> List[str]:
        """Return sorted DICOM file paths for a series in *directory*.

        If *series_uid* is empty, returns files for the first (or only) series.
        """
        if series_uid:
            return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series_uid))
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
        if not uids:
            return []
        return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uids[0]))
