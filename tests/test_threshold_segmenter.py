"""Tests for segmentation/threshold_segmenter.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk

from surgical_nav.segmentation.threshold_segmenter import ThresholdSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Wrap a numpy (z, y, x) array as a float32 SimpleITK image."""
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing(spacing)
    return img


def _get_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


# ---------------------------------------------------------------------------
# segment_skin tests
# ---------------------------------------------------------------------------

def test_segment_skin_all_foreground():
    """Volume entirely within HU range → entire volume labelled."""
    arr = np.full((10, 10, 10), 100.0)
    img = _make_image(arr)
    seg = ThresholdSegmenter()
    result = seg.segment_skin(img, lower_hu=50, upper_hu=200)
    out = _get_array(result)
    assert out.max() == 1
    assert out.sum() == 10 * 10 * 10


def test_segment_skin_all_background():
    """Volume entirely outside HU range → all zeros."""
    arr = np.full((10, 10, 10), -500.0)
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_skin(img, lower_hu=0, upper_hu=100)
    assert _get_array(result).sum() == 0


def test_segment_skin_output_is_uint8():
    arr = np.full((5, 5, 5), 50.0)
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_skin(img)
    assert result.GetPixelID() == sitk.sitkUInt8


def test_segment_skin_only_values_0_and_1():
    arr = np.random.uniform(-500, 500, (8, 8, 8))
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_skin(img, lower_hu=-200, upper_hu=300)
    out = _get_array(result)
    assert set(out.ravel().tolist()).issubset({0, 1})


def test_segment_skin_largest_component_kept():
    """Two separate blobs — only the larger one should survive."""
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    # Large blob: 6×6×6 = 216 voxels
    arr[1:7, 1:7, 1:7] = 100.0
    # Small blob: 2×2×2 = 8 voxels
    arr[15:17, 15:17, 15:17] = 100.0
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_skin(img, lower_hu=50, upper_hu=200,
                                               closing_radius=0)
    out = _get_array(result)
    # Should have exactly 216 foreground voxels (the large blob)
    assert out.sum() == 216


def test_segment_skin_preserves_spacing():
    arr = np.full((5, 5, 5), 50.0)
    img = _make_image(arr, spacing=(1.5, 1.5, 3.0))
    result = ThresholdSegmenter().segment_skin(img)
    np.testing.assert_allclose(result.GetSpacing(), (1.5, 1.5, 3.0))


# ---------------------------------------------------------------------------
# segment_target tests
# ---------------------------------------------------------------------------

def test_segment_target_includes_seed():
    """The seed voxel must be foreground in the output."""
    arr = np.full((10, 10, 10), 100.0)
    img = _make_image(arr)
    seed = (5, 5, 5)
    result = ThresholdSegmenter().segment_target(img, seed_index=seed,
                                                  lower_hu=50, upper_hu=200)
    out = _get_array(result)
    # seed in (iz, iy, ix) order for numpy
    assert out[5, 5, 5] == 1


def test_segment_target_isolated_region():
    """Only voxels within the HU range connected to the seed are labelled."""
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    # A 3×3×3 cube of value 150 at the centre
    arr[3:6, 3:6, 3:6] = 150.0
    img = _make_image(arr)
    seed = (4, 4, 4)   # (ix, iy, iz) — centre of the cube
    result = ThresholdSegmenter().segment_target(img, seed_index=seed,
                                                  lower_hu=100, upper_hu=200)
    out = _get_array(result)
    assert out.sum() == 27   # 3×3×3


def test_segment_target_seed_outside_range_gives_empty():
    """Seed voxel HU outside [lower, upper] → empty result."""
    arr = np.full((5, 5, 5), -500.0)
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_target(img, seed_index=(2, 2, 2),
                                                  lower_hu=0, upper_hu=100)
    assert _get_array(result).sum() == 0


def test_segment_target_output_is_uint8():
    arr = np.full((5, 5, 5), 100.0)
    img = _make_image(arr)
    result = ThresholdSegmenter().segment_target(img, seed_index=(2, 2, 2))
    assert result.GetPixelID() == sitk.sitkUInt8
