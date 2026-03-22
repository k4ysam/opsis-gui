"""Tests for rendering/paint_brush.py (no Qt/display needed)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk
import vtkmodules.all as vtk

from surgical_nav.rendering.paint_brush import PaintBrush


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(dims=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    label = vtk.vtkImageData()
    label.SetDimensions(*dims)
    label.SetSpacing(*spacing)
    label.SetOrigin(*origin)
    label.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    label.GetPointData().GetScalars().Fill(0)
    return label


def _read_voxel(label: vtk.vtkImageData, ix: int, iy: int, iz: int) -> int:
    dims = label.GetDimensions()
    flat = ix + iy * dims[0] + iz * dims[0] * dims[1]
    return int(label.GetPointData().GetScalars().GetValue(flat))


# ---------------------------------------------------------------------------
# create_label_volume
# ---------------------------------------------------------------------------

def test_create_label_volume_matches_reference():
    ref = _make_label(dims=(10, 12, 8), spacing=(1.5, 1.5, 3.0), origin=(5.0, -3.0, 0.0))
    label = PaintBrush.create_label_volume(ref)
    assert label.GetDimensions() == ref.GetDimensions()
    assert label.GetSpacing()    == pytest.approx(ref.GetSpacing())
    assert label.GetOrigin()     == pytest.approx(ref.GetOrigin())


def test_create_label_volume_all_zeros():
    from vtkmodules.util.numpy_support import vtk_to_numpy
    ref = _make_label()
    label = PaintBrush.create_label_volume(ref)
    arr = vtk_to_numpy(label.GetPointData().GetScalars())
    assert arr.sum() == 0


# ---------------------------------------------------------------------------
# paint_at_world
# ---------------------------------------------------------------------------

def test_paint_at_world_centre_voxel():
    label = _make_label(origin=(0.0, 0.0, 0.0))
    brush = PaintBrush(label)
    brush.paint_at_world(10.0, 10.0, 10.0, radius_voxels=0)
    assert _read_voxel(label, 10, 10, 10) == 1


def test_paint_radius_zero_single_voxel():
    label = _make_label()
    brush = PaintBrush(label)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    # Only (5,5,5) should be set
    assert _read_voxel(label, 5, 5, 5) == 1
    assert _read_voxel(label, 6, 5, 5) == 0


def test_paint_sphere_volume():
    """Radius-2 sphere should paint roughly the right number of voxels."""
    label = _make_label(dims=(30, 30, 30))
    brush = PaintBrush(label)
    brush.paint_at_world(15.0, 15.0, 15.0, radius_voxels=2)
    from vtkmodules.util.numpy_support import vtk_to_numpy
    arr = vtk_to_numpy(label.GetPointData().GetScalars())
    painted = arr.sum()
    # Sphere of radius 2 in discrete voxels: expected ~33 voxels
    assert 20 < painted < 50


def test_paint_outside_bounds_ignored():
    label = _make_label(dims=(10, 10, 10))
    brush = PaintBrush(label)
    brush.paint_at_world(100.0, 100.0, 100.0, radius_voxels=2)  # outside
    from vtkmodules.util.numpy_support import vtk_to_numpy
    arr = vtk_to_numpy(label.GetPointData().GetScalars())
    assert arr.sum() == 0


def test_custom_paint_value():
    label = _make_label()
    brush = PaintBrush(label)
    brush.set_paint_value(2)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    assert _read_voxel(label, 5, 5, 5) == 2


# ---------------------------------------------------------------------------
# erase_at_world
# ---------------------------------------------------------------------------

def test_erase_removes_painted_voxels():
    label = _make_label()
    brush = PaintBrush(label)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=1)
    brush.erase_at_world(5.0, 5.0, 5.0, radius_voxels=1)
    assert _read_voxel(label, 5, 5, 5) == 0


def test_erase_does_not_change_paint_value():
    label = _make_label()
    brush = PaintBrush(label)
    brush.set_paint_value(3)
    brush.erase_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    assert brush._paint_value == 3   # restored after erase


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_zeros_all_voxels():
    label = _make_label()
    brush = PaintBrush(label)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=3)
    brush.clear()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    arr = vtk_to_numpy(label.GetPointData().GetScalars())
    assert arr.sum() == 0


# ---------------------------------------------------------------------------
# get_label_sitk
# ---------------------------------------------------------------------------

def test_get_label_sitk_returns_sitk_image():
    label = _make_label()
    brush = PaintBrush(label)
    result = brush.get_label_sitk()
    assert isinstance(result, sitk.Image)


def test_get_label_sitk_painted_voxel_is_foreground():
    label = _make_label(dims=(10, 10, 10))
    brush = PaintBrush(label)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    sitk_img = brush.get_label_sitk()
    arr = sitk.GetArrayFromImage(sitk_img)
    # sitk array is (z, y, x); painted at ix=5, iy=5, iz=5
    assert arr[5, 5, 5] == 1


def test_get_label_sitk_preserves_spacing():
    label = _make_label(spacing=(2.0, 2.0, 4.0))
    brush = PaintBrush(label)
    sitk_img = brush.get_label_sitk()
    np.testing.assert_allclose(sitk_img.GetSpacing(), (2.0, 2.0, 4.0))
