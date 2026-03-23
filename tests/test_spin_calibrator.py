"""Tests for calibration/spin_calibrator.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from surgical_nav.calibration.spin_calibrator import SpinCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_around(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around *axis* by *angle_rad*."""
    ax = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def _make_spin_samples(
    shaft_world: np.ndarray,
    shaft_point: np.ndarray,
    tracker_offset_mm: float = 50.0,
    n: int = 60,
    noise_mm: float = 0.0,
    rng: np.random.Generator = None,
) -> list[np.ndarray]:
    """Generate transforms for a tool spinning around *shaft_world*.

    The tool spins around its shaft axis.  The tracker frame origin is
    offset from the shaft, so it traces a circle in the plane perpendicular
    to the shaft.

    Parameters
    ----------
    shaft_world : (3,) unit-ish vector — the shaft axis in world space
    shaft_point : (3,) a point on the shaft axis in world space
    tracker_offset_mm : radial distance from shaft to tracker origin (mm)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    shaft = shaft_world / np.linalg.norm(shaft_world)

    # Build a frame: shaft, perp1, perp2
    ref = np.array([1., 0., 0.]) if abs(shaft[2]) > 0.9 else np.array([0., 0., 1.])
    perp1 = np.cross(shaft, ref); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(shaft, perp1)

    # Tracker origin in the unrotated pose is offset along perp1
    initial_offset = tracker_offset_mm * perp1

    # Initial rotation: tool Z aligned with shaft
    R_base = np.stack([perp1, perp2, shaft], axis=1)   # columns = frame axes

    samples = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        R_spin = _rot_around(shaft, angle)
        R = R_spin @ R_base

        # Tracker origin orbits the shaft
        t = shaft_point + R_spin @ initial_offset
        if noise_mm > 0:
            t += rng.normal(scale=noise_mm, size=3)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        samples.append(T)
    return samples


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_initial_count():
    assert SpinCalibrator().sample_count == 0


def test_add_increments_count():
    cal = SpinCalibrator()
    cal.add_sample(np.eye(4))
    assert cal.sample_count == 1


def test_clear_resets():
    cal = SpinCalibrator()
    cal.add_sample(np.eye(4))
    cal.clear()
    assert cal.sample_count == 0


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        SpinCalibrator().add_sample(np.eye(3))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_too_few_fails():
    cal = SpinCalibrator(min_samples=4)
    cal.add_sample(np.eye(4))
    result = cal.calibrate()
    assert not result.success
    assert "samples" in result.message.lower()


def test_no_motion_fails():
    """All identical transforms → no variance → fail."""
    cal = SpinCalibrator()
    for _ in range(10):
        cal.add_sample(np.eye(4))
    result = cal.calibrate()
    assert not result.success


def test_non_linear_motion_fails():
    """Translations scattered in 3D (not a line) → linearity < threshold."""
    cal = SpinCalibrator(min_linearity=0.95)
    rng = np.random.default_rng(0)
    for _ in range(30):
        T = np.eye(4)
        T[:3, 3] = rng.normal(scale=50.0, size=3)   # random 3D scatter
        cal.add_sample(T)
    result = cal.calibrate()
    assert not result.success
    assert "planarity" in result.message.lower() or "linearity" in result.message.lower()


# ---------------------------------------------------------------------------
# Successful calibration
# ---------------------------------------------------------------------------

def test_noiseless_shaft_recovery_z():
    """Spinning around world Z should recover shaft ~ [0,0,1]."""
    shaft = np.array([0., 0., 1.])
    shaft_pt = np.array([0., 0., 50.])
    cal = SpinCalibrator()
    for T in _make_spin_samples(shaft, shaft_pt):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    dot = abs(float(np.dot(result.shaft_in_world, shaft)))
    assert dot > 0.999


def test_noiseless_shaft_recovery_y():
    """Spinning around world Y should recover shaft ~ [0,1,0]."""
    shaft = np.array([0., 1., 0.])
    shaft_pt = np.array([30., 50., 0.])
    cal = SpinCalibrator()
    for T in _make_spin_samples(shaft, shaft_pt):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    dot = abs(float(np.dot(result.shaft_in_world, shaft)))
    assert dot > 0.999


def test_noiseless_shaft_arbitrary_axis():
    """Spinning around an arbitrary axis should be recovered."""
    shaft = np.array([1., 1., 1.]) / np.sqrt(3)
    shaft_pt = np.array([10., 10., 10.])
    cal = SpinCalibrator()
    for T in _make_spin_samples(shaft, shaft_pt, n=72):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    dot = abs(float(np.dot(result.shaft_in_world, shaft)))
    assert dot > 0.999


def test_shaft_in_tool_is_unit_vector():
    shaft = np.array([0., 0., 1.])
    shaft_pt = np.array([0., 0., 50.])
    cal = SpinCalibrator()
    for T in _make_spin_samples(shaft, shaft_pt):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    np.testing.assert_allclose(np.linalg.norm(result.shaft_in_tool), 1.0, atol=1e-9)


def test_planarity_near_one_for_clean_spin():
    """Clean spin → planarity (linearity field) should be close to 1."""
    shaft = np.array([0., 0., 1.])
    shaft_pt = np.array([0., 0., 50.])
    cal = SpinCalibrator()
    for T in _make_spin_samples(shaft, shaft_pt):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    assert result.linearity > 0.99


def test_low_noise_still_passes():
    """σ=0.5 mm position noise should still pass linearity threshold."""
    shaft = np.array([0., 0., 1.])
    shaft_pt = np.array([0., 0., 50.])
    cal = SpinCalibrator(min_linearity=0.90)
    rng = np.random.default_rng(55)
    for T in _make_spin_samples(shaft, shaft_pt, n=120, noise_mm=0.5, rng=rng):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
