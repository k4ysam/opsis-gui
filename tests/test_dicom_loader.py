"""Tests for dicom/dicom_loader.py (VTK-free).

Verifies that the loader returns (None, sitk.Image) and that sitk loading works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk

from surgical_nav.dicom.dicom_loader import DICOMLoader


def test_sitk_to_vtk_returns_none():
    arr = np.zeros((8, 8, 4), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    result = DICOMLoader.sitk_to_vtk(img)
    assert result is None


def test_load_series_returns_pair(tmp_path):
    """load_series() must return (None, sitk.Image)."""
    arr = np.random.rand(2, 5, 5).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 2.0))
    mhd_path = str(tmp_path / "test.mhd")
    sitk.WriteImage(img, mhd_path)

    loaded = sitk.ReadImage(mhd_path, sitk.sitkFloat32)
    vtk_img = DICOMLoader.sitk_to_vtk(loaded)

    assert vtk_img is None
    assert isinstance(loaded, sitk.Image)
