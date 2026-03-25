"""Tests for rendering/paint_brush.py (numpy-based, no VTK)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk

from surgical_nav.rendering.paint_brush import PaintBrush


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brush(dims=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    arr = np.zeros(dims, dtype=np.uint8)
    return PaintBrush(arr, spacing, origin), arr


# ---------------------------------------------------------------------------
# create_label_volume
# ---------------------------------------------------------------------------

def test_create_label_volume_shape():
    dims = (10, 12, 8)
    arr = PaintBrush.create_label_volume(dims, spacing=(1.5, 1.5, 3.0), origin=(5.0, -3.0, 0.0))
    assert arr.shape == dims


def test_create_label_volume_all_zeros():
    arr = PaintBrush.create_label_volume((10, 10, 10))
    assert arr.sum() == 0


# ---------------------------------------------------------------------------
# paint_at_world
# ---------------------------------------------------------------------------

def test_paint_at_world_centre_voxel():
    brush, arr = _make_brush()
    brush.paint_at_world(10.0, 10.0, 10.0, radius_voxels=0)
    assert arr[10, 10, 10] == 1


def test_paint_radius_zero_single_voxel():
    brush, arr = _make_brush()
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    assert arr[5, 5, 5] == 1
    assert arr[6, 5, 5] == 0


def test_paint_sphere_volume():
    brush, arr = _make_brush(dims=(30, 30, 30))
    brush.paint_at_world(15.0, 15.0, 15.0, radius_voxels=2)
    painted = arr.sum()
    assert 20 < painted < 50


def test_paint_outside_bounds_ignored():
    brush, arr = _make_brush(dims=(10, 10, 10))
    brush.paint_at_world(100.0, 100.0, 100.0, radius_voxels=2)
    assert arr.sum() == 0


def test_custom_paint_value():
    brush, arr = _make_brush()
    brush.set_paint_value(2)
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    assert arr[5, 5, 5] == 2


# ---------------------------------------------------------------------------
# erase_at_world
# ---------------------------------------------------------------------------

def test_erase_removes_painted_voxels():
    brush, arr = _make_brush()
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=1)
    brush.erase_at_world(5.0, 5.0, 5.0, radius_voxels=1)
    assert arr[5, 5, 5] == 0


def test_erase_does_not_change_paint_value():
    brush, arr = _make_brush()
    brush.set_paint_value(3)
    brush.erase_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    assert brush._paint_value == 3


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_zeros_all_voxels():
    brush, arr = _make_brush()
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=3)
    brush.clear()
    assert arr.sum() == 0


# ---------------------------------------------------------------------------
# get_label_sitk
# ---------------------------------------------------------------------------

def test_get_label_sitk_returns_sitk_image():
    brush, _ = _make_brush()
    result = brush.get_label_sitk()
    assert isinstance(result, sitk.Image)


def test_get_label_sitk_painted_voxel_is_foreground():
    brush, _ = _make_brush(dims=(10, 10, 10))
    brush.paint_at_world(5.0, 5.0, 5.0, radius_voxels=0)
    sitk_img = brush.get_label_sitk()
    arr = sitk.GetArrayFromImage(sitk_img)
    # sitk array is (nz, ny, nx); painted at ix=5, iy=5, iz=5
    assert arr[5, 5, 5] == 1


def test_get_label_sitk_preserves_spacing():
    brush, _ = _make_brush(spacing=(2.0, 2.0, 4.0))
    sitk_img = brush.get_label_sitk()
    np.testing.assert_allclose(sitk_img.GetSpacing(), (2.0, 2.0, 4.0))
