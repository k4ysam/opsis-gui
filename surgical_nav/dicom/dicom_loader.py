"""DICOMLoader: loads a DICOM series into a SimpleITK image.

Without VTK, ``sitk_to_vtk`` always returns None. The SimpleITK image is
still loaded and returned for downstream segmentation use.

Tries SimpleITK's ImageSeriesReader first; falls back to pydicom pixel-by-pixel
reading when SimpleITK cannot determine the ImageIO reader (e.g. compressed or
non-standard DICOM files).

Usage::

    loader = DICOMLoader()
    vtk_image, sitk_image = loader.load_series(file_paths)
    # vtk_image: None (VTK not available)
    # sitk_image: SimpleITK.Image for segmentation
"""

from __future__ import annotations

from typing import List, Tuple
import time

import numpy as np
import SimpleITK as sitk


class DICOMLoader:
    """Loads a sorted list of DICOM file paths into a SimpleITK image."""

    @staticmethod
    def _log(message: str):
        print(f"[timing] {message}", flush=True)

    def load_series(
        self, file_paths: List[str]
    ) -> Tuple[None, sitk.Image]:
        """Load *file_paths* as one DICOM series.

        Returns
        -------
        vtk_image : None
            VTK image data unavailable without VTK.
        sitk_image : SimpleITK.Image
            Original LPS image for downstream segmentation.
        """
        t0 = time.perf_counter()
        sitk_image = self._read_sitk(file_paths)
        t1 = time.perf_counter()
        vtk_image  = self.sitk_to_vtk(sitk_image)
        t2 = time.perf_counter()
        self._log(
            f"DICOM load: read={t1 - t0:.2f}s vtk_convert={t2 - t1:.2f}s total={t2 - t0:.2f}s"
        )
        return vtk_image, sitk_image

    # ------------------------------------------------------------------
    # SimpleITK loading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_sitk(file_paths: List[str]) -> sitk.Image:
        """Read a series from an explicit sorted file list.

        Tries SimpleITK's ImageSeriesReader first; falls back to pydicom when
        SimpleITK cannot determine the ImageIO reader.
        """
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_paths)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            image = reader.Execute()
            return sitk.Cast(image, sitk.sitkFloat32)
        except Exception as sitk_err:
            # Fallback: read pixel data with pydicom slice-by-slice
            try:
                return DICOMLoader._read_pydicom(file_paths)
            except Exception as pydicom_err:
                raise RuntimeError(
                    f"Could not load DICOM series.\n"
                    f"SimpleITK error: {sitk_err}\n"
                    f"pydicom fallback error: {pydicom_err}\n\n"
                    f"The files may be in an unsupported or compressed format."
                ) from sitk_err

    @staticmethod
    def _read_pydicom(file_paths: List[str]) -> sitk.Image:
        """Read pixel data slice-by-slice with pydicom and assemble into sitk.Image."""
        import pydicom

        slices = []
        for path in file_paths:
            try:
                ds = pydicom.dcmread(path, force=True)
            except Exception:
                continue
            if not hasattr(ds, "PixelData"):
                continue
            slices.append(ds)

        if not slices:
            raise ValueError(
                "No DICOM files with pixel data found. "
                "The selected series may contain metadata-only files "
                "(e.g. DICOMDIR records) or use an unsupported transfer syntax."
            )

        # Sort by InstanceNumber
        slices.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0) or 0))

        # Stack pixel arrays: result shape (nz, ny, nx)
        pixel_arrays = []
        for ds in slices:
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception:
                continue
            # Apply rescale slope/intercept if present (converts to HU)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            arr = arr * slope + intercept
            pixel_arrays.append(arr)

        if not pixel_arrays:
            raise ValueError(
                "Could not decode pixel data from any slice. "
                "The files may use a compressed transfer syntax that requires "
                "additional packages (e.g. pylibjpeg, pillow)."
            )
        volume = np.stack(pixel_arrays, axis=0).astype(np.float32)  # (nz, ny, nx)

        img = sitk.GetImageFromArray(volume)

        # Set geometry from the first slice
        ds0 = slices[0]
        if hasattr(ds0, "PixelSpacing"):
            row_sp, col_sp = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
        else:
            row_sp = col_sp = 1.0

        slice_thickness = float(getattr(ds0, "SliceThickness", 1.0) or 1.0)
        img.SetSpacing((col_sp, row_sp, slice_thickness))

        if hasattr(ds0, "ImagePositionPatient"):
            origin = [float(v) for v in ds0.ImagePositionPatient]
            img.SetOrigin(origin)

        if hasattr(ds0, "ImageOrientationPatient"):
            iop = [float(v) for v in ds0.ImageOrientationPatient]
            row_dir = iop[:3]
            col_dir = iop[3:]
            # Normal = cross product of row and col direction cosines
            normal = [
                row_dir[1] * col_dir[2] - row_dir[2] * col_dir[1],
                row_dir[2] * col_dir[0] - row_dir[0] * col_dir[2],
                row_dir[0] * col_dir[1] - row_dir[1] * col_dir[0],
            ]
            direction = row_dir + col_dir + normal
            img.SetDirection(direction)

        return img

    @classmethod
    def sitk_to_vtk(cls, sitk_image: sitk.Image) -> vtk.vtkImageData:
        """Convert a SimpleITK image (LPS) to vtkImageData (RAS).

        The pixel array is the same; only the origin and direction cosines
        are flipped on X and Y (LPS → RAS).
        """
        t0 = time.perf_counter()
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

        cls._log(
            f"sitk_to_vtk dims={nx}x{ny}x{nz} voxels={nx * ny * nz:,} took {time.perf_counter() - t0:.2f}s"
        )
        return vtk_image

    # ------------------------------------------------------------------
    # Convenience: auto-detect series files from a directory
    # ------------------------------------------------------------------

    @staticmethod
    def get_series_file_names(directory: str, series_uid: str = "") -> List[str]:
        """Return sorted DICOM file paths for a series in *directory*."""
        if series_uid:
            return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series_uid))
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
        if not uids:
            return []
        return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uids[0]))
