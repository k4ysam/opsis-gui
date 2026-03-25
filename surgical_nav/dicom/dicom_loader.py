"""DICOMLoader: loads a DICOM series into a SimpleITK image.

Without VTK, ``sitk_to_vtk`` always returns None. The SimpleITK image is
still loaded and returned for downstream segmentation use.

Usage::

    loader = DICOMLoader()
    vtk_image, sitk_image = loader.load_series(file_paths)
    # vtk_image: None (VTK not available)
    # sitk_image: SimpleITK.Image for segmentation
"""

from __future__ import annotations

from typing import List, Tuple

import SimpleITK as sitk


class DICOMLoader:
    """Loads a sorted list of DICOM file paths into a SimpleITK image."""

    def load_series(
        self, file_paths: List[str]
    ) -> Tuple[None, sitk.Image]:
        """Load *file_paths* as one DICOM series.

        Returns
        -------
        vtk_image : None
            VTK image data unavailable without VTK.
        sitk_image : SimpleITK.Image
            Original LPS image kept for downstream segmentation.
        """
        sitk_image = self._read_sitk(file_paths)
        return None, sitk_image

    @staticmethod
    def _read_sitk(file_paths: List[str]) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        return sitk.Cast(image, sitk.sitkFloat32)

    @classmethod
    def sitk_to_vtk(cls, sitk_image: sitk.Image):
        """Returns None — VTK not available."""
        return None

    @staticmethod
    def get_series_file_names(directory: str, series_uid: str = "") -> List[str]:
        """Return sorted DICOM file paths for a series in *directory*."""
        if series_uid:
            return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series_uid))
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
        if not uids:
            return []
        return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uids[0]))
