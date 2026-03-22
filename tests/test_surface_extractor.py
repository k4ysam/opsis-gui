"""Tests for segmentation/surface_extractor.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk
import vtkmodules.all as vtk

from surgical_nav.segmentation.surface_extractor import SurfaceExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_label(size=30, radius=10) -> sitk.Image:
    """Binary label image with a solid sphere at the centre."""
    arr = np.zeros((size, size, size), dtype=np.uint8)
    cx = cy = cz = size // 2
    for z in range(size):
        for y in range(size):
            for x in range(size):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                    arr[z, y, x] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _empty_label(size=10) -> sitk.Image:
    arr = np.zeros((size, size, size), dtype=np.uint8)
    return sitk.GetImageFromArray(arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_label_returns_empty_polydata():
    result = SurfaceExtractor().extract(_empty_label())
    assert result.GetNumberOfPoints() == 0


def test_sphere_produces_points():
    result = SurfaceExtractor(smooth_iterations=5).extract(_sphere_label())
    assert result.GetNumberOfPoints() > 0


def test_sphere_produces_cells():
    result = SurfaceExtractor(smooth_iterations=5).extract(_sphere_label())
    assert result.GetNumberOfCells() > 0


def test_output_is_vtkPolyData():
    result = SurfaceExtractor(smooth_iterations=5).extract(_sphere_label())
    assert isinstance(result, vtk.vtkPolyData)


def test_normals_computed_when_enabled():
    result = SurfaceExtractor(smooth_iterations=5, compute_normals=True).extract(
        _sphere_label()
    )
    assert result.GetPointData().GetNormals() is not None


def test_normals_absent_when_disabled():
    result = SurfaceExtractor(smooth_iterations=5, compute_normals=False).extract(
        _sphere_label()
    )
    # Marching cubes with ComputeNormalsOff + no normals filter → no point normals
    assert result.GetPointData().GetNormals() is None


def test_spacing_respected():
    """Anisotropic spacing must affect mesh bounds."""
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[5:15, 5:15, 5:15] = 1
    img_iso = sitk.GetImageFromArray(arr)
    img_iso.SetSpacing((1.0, 1.0, 1.0))

    img_aniso = sitk.GetImageFromArray(arr)
    img_aniso.SetSpacing((1.0, 1.0, 3.0))   # Z stretched ×3

    ext = SurfaceExtractor(smooth_iterations=2)
    mesh_iso   = ext.extract(img_iso)
    mesh_aniso = ext.extract(img_aniso)

    bounds_iso   = mesh_iso.GetBounds()
    bounds_aniso = mesh_aniso.GetBounds()

    z_extent_iso   = bounds_iso[5]   - bounds_iso[4]
    z_extent_aniso = bounds_aniso[5] - bounds_aniso[4]
    # Anisotropic Z extent should be roughly 3× larger
    assert z_extent_aniso > z_extent_iso * 2.0


def test_sitk_to_vtk_dimensions():
    """Internal converter preserves image dimensions."""
    arr = np.zeros((8, 10, 12), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    vtk_img = SurfaceExtractor._sitk_to_vtk(img)
    assert vtk_img.GetDimensions() == (12, 10, 8)


def test_sitk_to_vtk_spacing():
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 3.0, 4.0))
    vtk_img = SurfaceExtractor._sitk_to_vtk(img)
    np.testing.assert_allclose(vtk_img.GetSpacing(), (2.0, 3.0, 4.0))
