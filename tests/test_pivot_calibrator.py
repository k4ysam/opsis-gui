"""Tests for calibration/pivot_calibrator.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from surgical_nav.calibration.pivot_calibrator import PivotCalibrator, PivotResult


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def _make_pivot_samples(
    p_tool: np.ndarray,
    p_ref:  np.ndarray,
    n: int = 50,
    noise_mm: float = 0.0,
    rng: np.random.Generator = None,
) -> list[np.ndarray]:
    """Generate synthetic pivot transforms.

    For each sample, a random rotation R is drawn.  The tracker-space
    translation is computed as:
        t_i = p_ref - R_i @ p_tool
    so that R_i @ p_tool + t_i = p_ref (ideal pivot constraint).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    samples = []
    # Use rotations that span all axes to ensure angular spread
    for i in range(n):
        # Rotate around a varying axis to ensure spread
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        angle = rng.uniform(0.3, np.pi)
        # Rodrigues rotation
        K = np.array([
            [0, -ax[2], ax[1]],
            [ax[2], 0, -ax[0]],
            [-ax[1], ax[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        t = p_ref - R @ p_tool
        if noise_mm > 0:
            t += rng.normal(scale=noise_mm, size=3)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        samples.append(T)
    return samples


# ---------------------------------------------------------------------------
# Construction and accumulation
# ---------------------------------------------------------------------------

def test_initial_sample_count():
    cal = PivotCalibrator()
    assert cal.sample_count == 0


def test_add_sample_increments_count():
    cal = PivotCalibrator()
    cal.add_sample(np.eye(4))
    assert cal.sample_count == 1


def test_clear_resets_count():
    cal = PivotCalibrator()
    cal.add_sample(np.eye(4))
    cal.clear()
    assert cal.sample_count == 0


def test_wrong_shape_raises():
    cal = PivotCalibrator()
    with pytest.raises(ValueError):
        cal.add_sample(np.eye(3))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_too_few_samples_fails():
    cal = PivotCalibrator(min_samples=4)
    cal.add_sample(np.eye(4))
    result = cal.calibrate()
    assert not result.success
    assert "samples" in result.message.lower()


def test_collinear_samples_fails():
    """Rotating around a single axis gives zero spread → should fail."""
    cal = PivotCalibrator(min_angular_spread_deg=30.0)
    p_tool = np.array([0.0, 0.0, 100.0])
    p_ref  = np.array([10.0, 20.0, 30.0])
    # All rotations around the same Z axis
    for deg in range(0, 360, 10):
        R = _rot_z(deg)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = p_ref - R @ p_tool
        cal.add_sample(T)
    result = cal.calibrate()
    assert not result.success
    assert "spread" in result.message.lower() or "angular" in result.message.lower()


def test_high_noise_fails():
    """Noise σ=5 mm should push RMSE above 0.8 mm threshold."""
    cal = PivotCalibrator(max_rmse_mm=0.8)
    p_tool = np.array([0.0, 0.0, 100.0])
    p_ref  = np.array([10.0, 20.0, 30.0])
    rng = np.random.default_rng(0)
    for T in _make_pivot_samples(p_tool, p_ref, n=50, noise_mm=5.0, rng=rng):
        cal.add_sample(T)
    result = cal.calibrate()
    assert not result.success
    assert "RMSE" in result.message


# ---------------------------------------------------------------------------
# Successful calibration (noiseless)
# ---------------------------------------------------------------------------

def test_noiseless_recovers_p_tool():
    cal = PivotCalibrator()
    p_tool = np.array([5.0, -3.0, 120.0])
    p_ref  = np.array([10.0, 20.0, 30.0])
    for T in _make_pivot_samples(p_tool, p_ref):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    np.testing.assert_allclose(result.p_tool, p_tool, atol=1e-6)


def test_noiseless_recovers_p_ref():
    cal = PivotCalibrator()
    p_tool = np.array([5.0, -3.0, 120.0])
    p_ref  = np.array([10.0, 20.0, 30.0])
    for T in _make_pivot_samples(p_tool, p_ref):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    np.testing.assert_allclose(result.p_ref, p_ref, atol=1e-6)


def test_noiseless_rmse_near_zero():
    cal = PivotCalibrator()
    p_tool = np.array([0.0, 0.0, 100.0])
    p_ref  = np.array([0.0, 0.0, 0.0])
    for T in _make_pivot_samples(p_tool, p_ref):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.rmse_mm < 1e-6


# ---------------------------------------------------------------------------
# Low-noise calibration (σ = 0.3 mm → RMSE < 0.8 mm)
# ---------------------------------------------------------------------------

def test_low_noise_within_threshold():
    """σ=0.3 mm noise must still pass the 0.8 mm RMSE threshold."""
    cal = PivotCalibrator(max_rmse_mm=0.8)
    p_tool = np.array([0.0, 0.0, 100.0])
    p_ref  = np.array([50.0, 50.0, 0.0])
    rng = np.random.default_rng(7)
    for T in _make_pivot_samples(p_tool, p_ref, n=150, noise_mm=0.3, rng=rng):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    assert result.rmse_mm < 0.8


def test_low_noise_tip_within_half_mm():
    """With σ=0.3 mm noise, recovered tip should be within 0.5 mm."""
    cal = PivotCalibrator()
    p_tool = np.array([3.0, -2.0, 95.0])
    p_ref  = np.array([20.0, 30.0, 10.0])
    rng = np.random.default_rng(99)
    for T in _make_pivot_samples(p_tool, p_ref, n=150, noise_mm=0.3, rng=rng):
        cal.add_sample(T)
    result = cal.calibrate()
    assert result.success, result.message
    np.testing.assert_allclose(result.p_tool, p_tool, atol=0.5)


# ---------------------------------------------------------------------------
# as_transform
# ---------------------------------------------------------------------------

def test_as_transform_shape():
    cal = PivotCalibrator()
    p_tool = np.array([5.0, 0.0, 100.0])
    p_ref  = np.array([0.0, 0.0, 0.0])
    for T in _make_pivot_samples(p_tool, p_ref):
        cal.add_sample(T)
    result = cal.calibrate()
    m = result.as_transform()
    assert m.shape == (4, 4)


def test_as_transform_translation_equals_p_tool():
    cal = PivotCalibrator()
    p_tool = np.array([5.0, -3.0, 120.0])
    p_ref  = np.array([10.0, 20.0, 30.0])
    for T in _make_pivot_samples(p_tool, p_ref):
        cal.add_sample(T)
    result = cal.calibrate()
    m = result.as_transform()
    np.testing.assert_allclose(m[:3, 3], result.p_tool, atol=1e-9)
