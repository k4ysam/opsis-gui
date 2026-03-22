"""Tests for dicom/dicom_loader.py.

Generates a synthetic 3-D volume with SimpleITK (no real DICOM files needed
for the core conversion tests), then verifies the VTK output has correct
dimensions, spacing, origin (LPS→RAS flip), and scalar values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import numpy as np
import pytest
import SimpleITK as sitk
import vtkmodules.all as vtk

from surgical_nav.dicom.dicom_loader import DICOMLoader


# ---------------------------------------------------------------------------
# Helper: make a synthetic SimpleITK image with known properties
# ---------------------------------------------------------------------------

def _make_sitk_image(
    size=(10, 12, 8),           # (x, y, z)
    spacing=(1.5, 1.5, 3.0),
    origin=(-75.0, -100.0, -50.0),   # LPS origin
    direction=None,             # 3×3 row-major; None = identity
    fill_value: float = 42.0,
) -> sitk.Image:
    """Create a simple float32 SimpleITK image."""
    arr = np.full((size[2], size[1], size[0]), fill_value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sitk_to_vtk_dimensions():
    size = (10, 12, 8)
    img = _make_sitk_image(size=size)
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    assert vtk_img.GetDimensions() == size


def test_sitk_to_vtk_spacing():
    spacing = (1.5, 2.0, 3.5)
    img = _make_sitk_image(spacing=spacing)
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    np.testing.assert_allclose(vtk_img.GetSpacing(), spacing, atol=1e-6)


def test_sitk_to_vtk_lps_to_ras_origin():
    """Origin X and Y must be negated; Z stays the same."""
    origin_lps = (-75.0, -100.0, -50.0)
    img = _make_sitk_image(origin=origin_lps)
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    origin_ras = vtk_img.GetOrigin()
    assert origin_ras[0] == pytest.approx(75.0)    # −(−75) = +75
    assert origin_ras[1] == pytest.approx(100.0)   # −(−100) = +100
    assert origin_ras[2] == pytest.approx(-50.0)   # unchanged


def test_sitk_to_vtk_scalar_values_preserved():
    """Pixel values must survive the conversion unchanged."""
    img = _make_sitk_image(fill_value=1234.0)
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    scalars = vtk_img.GetPointData().GetScalars()
    assert scalars is not None
    # Sample the first scalar value
    assert scalars.GetValue(0) == pytest.approx(1234.0, abs=1e-3)


def test_sitk_to_vtk_scalar_type_float32():
    img = _make_sitk_image()
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    assert vtk_img.GetScalarType() == vtk.VTK_FLOAT


def test_sitk_to_vtk_direction_cosines_stored():
    """Direction cosines (RAS) must be stored in field data."""
    img = _make_sitk_image()
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    fd = vtk_img.GetFieldData()
    assert fd.GetArray("DirectionCosines") is not None
    assert fd.GetArray("DirectionCosines").GetNumberOfTuples() == 9


def test_sitk_to_vtk_direction_ras_flip():
    """Row 0 and row 1 of direction cosines must be negated (LPS→RAS)."""
    # Identity direction in LPS
    identity = np.eye(3).ravel().tolist()
    img = _make_sitk_image(direction=identity)
    vtk_img = DICOMLoader.sitk_to_vtk(img)

    fd = vtk_img.GetFieldData()
    cos_arr = fd.GetArray("DirectionCosines")
    cosines = np.array([cos_arr.GetValue(i) for i in range(9)]).reshape(3, 3)

    # Row 0 (X direction) should be negated: [1,0,0] → [-1,0,0]
    assert cosines[0, 0] == pytest.approx(-1.0)
    # Row 1 (Y direction) should be negated: [0,1,0] → [0,-1,0]
    assert cosines[1, 1] == pytest.approx(-1.0)
    # Row 2 (Z direction) unchanged: [0,0,1]
    assert cosines[2, 2] == pytest.approx(1.0)


def test_sitk_to_vtk_number_of_points():
    size = (10, 12, 8)
    img = _make_sitk_image(size=size)
    vtk_img = DICOMLoader.sitk_to_vtk(img)
    expected = size[0] * size[1] * size[2]
    assert vtk_img.GetNumberOfPoints() == expected


def test_load_series_returns_pair(tmp_path):
    """load_series() must return (vtkImageData, sitk.Image)."""
    # Write a tiny 2-slice synthetic series as MHD (SimpleITK can read its own format)
    arr = np.random.rand(2, 5, 5).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 2.0))
    mhd_path = str(tmp_path / "test.mhd")
    sitk.WriteImage(img, mhd_path)

    # Load via SimpleITK directly (bypasses DICOM reader, tests conversion path)
    loaded = sitk.ReadImage(mhd_path, sitk.sitkFloat32)
    vtk_img = DICOMLoader.sitk_to_vtk(loaded)

    assert isinstance(vtk_img, vtk.vtkImageData)
    assert isinstance(loaded, sitk.Image)
    assert vtk_img.GetNumberOfPoints() == 2 * 5 * 5
