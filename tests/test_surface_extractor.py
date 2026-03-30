"""Tests for segmentation/surface_extractor.py (VTK-free stub)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import SimpleITK as sitk

from surgical_nav.segmentation.surface_extractor import SurfaceExtractor


def _sphere_label(size=30, radius=10) -> sitk.Image:
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


def test_extract_returns_none_without_vtk():
    result = SurfaceExtractor().extract(_sphere_label())
    assert result is None


def test_extract_does_not_crash():
    SurfaceExtractor(smooth_iterations=5).extract(_sphere_label())


def test_extractor_creates():
    ext = SurfaceExtractor(smooth_iterations=10, smooth_relaxation=0.2, compute_normals=False)
    assert ext is not None
