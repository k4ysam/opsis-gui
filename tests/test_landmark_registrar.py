"""Tests for registration/landmark_registrar.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from surgical_nav.registration.landmark_registrar import (
    LandmarkRegistrar, RegistrationResult, _umeyama_svd, _are_collinear
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def _make_pairs(R: np.ndarray, t: np.ndarray, n: int = 6,
                noise_mm: float = 0.0, rng=None):
    """Generate n random physical points and their transformed image points."""
    if rng is None:
        rng = np.random.default_rng(42)
    P = rng.uniform(-50, 50, size=(n, 3))
    Q = (R @ P.T).T + t
    if noise_mm > 0:
        Q += rng.normal(scale=noise_mm, size=Q.shape)
    return P, Q


# ---------------------------------------------------------------------------
# Construction and accumulation
# ---------------------------------------------------------------------------

def test_initial_pair_count():
    assert LandmarkRegistrar().pair_count == 0


def test_add_pair_increments():
    reg = LandmarkRegistrar()
    reg.add_pair(np.zeros(3), np.ones(3))
    assert reg.pair_count == 1


def test_clear_resets():
    reg = LandmarkRegistrar()
    reg.add_pair(np.zeros(3), np.zeros(3))
    reg.clear()
    assert reg.pair_count == 0


def test_wrong_shape_raises():
    reg = LandmarkRegistrar()
    with pytest.raises(ValueError):
        reg.add_pair(np.zeros(4), np.zeros(3))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_too_few_pairs_fails():
    reg = LandmarkRegistrar(min_pairs=3)
    reg.add_pair(np.array([0.,0.,0.]), np.array([1.,0.,0.]))
    reg.add_pair(np.array([1.,0.,0.]), np.array([2.,0.,0.]))
    result = reg.register()
    assert not result.success
    assert "pairs" in result.message.lower()


def test_collinear_points_fails():
    reg = LandmarkRegistrar()
    for i in range(5):
        p = np.array([float(i)*10, 0., 0.])
        reg.add_pair(p, p + np.array([5., 0., 0.]))
    result = reg.register()
    assert not result.success
    assert "collinear" in result.message.lower()


def test_high_noise_fails():
    reg = LandmarkRegistrar(max_rmse_mm=3.0)
    rng = np.random.default_rng(0)
    R = _rot_z(30)
    t = np.array([10., 20., 5.])
    P, Q = _make_pairs(R, t, n=6, noise_mm=10.0, rng=rng)
    for p, q in zip(P, Q):
        reg.add_pair(p, q)
    result = reg.register()
    assert not result.success
    assert "RMSE" in result.message


# ---------------------------------------------------------------------------
# Successful registrations
# ---------------------------------------------------------------------------

def test_pure_translation_noiseless():
    t = np.array([30., -20., 10.])
    reg = LandmarkRegistrar()
    rng = np.random.default_rng(1)
    P, Q = _make_pairs(np.eye(3), t, rng=rng)
    for p, q in zip(P, Q):
        reg.add_pair(p, q)
    result = reg.register()
    assert result.success, result.message
    np.testing.assert_allclose(result.transform[:3, 3], t, atol=1e-9)
    assert result.rmse_mm < 1e-6


def test_pure_rotation_noiseless():
    R = _rot_z(45)
    t = np.zeros(3)
    reg = LandmarkRegistrar()
    rng = np.random.default_rng(2)
    P, Q = _make_pairs(R, t, rng=rng)
    for p, q in zip(P, Q):
        reg.add_pair(p, q)
    result = reg.register()
    assert result.success, result.message
    np.testing.assert_allclose(result.transform[:3, :3], R, atol=1e-9)
    assert result.rmse_mm < 1e-6


def test_rigid_transform_noiseless():
    R = _rot_z(30)
    t = np.array([5., -10., 15.])
    reg = LandmarkRegistrar()
    rng = np.random.default_rng(3)
    P, Q = _make_pairs(R, t, n=8, rng=rng)
    for p, q in zip(P, Q):
        reg.add_pair(p, q)
    result = reg.register()
    assert result.success, result.message
    assert result.rmse_mm < 1e-6


def test_low_noise_within_3mm():
    """σ=1 mm noise should still pass the 3 mm RMSE threshold."""
    R = _rot_z(20)
    t = np.array([10., 5., -3.])
    rng = np.random.default_rng(7)
    P, Q = _make_pairs(R, t, n=8, noise_mm=1.0, rng=rng)
    result = LandmarkRegistrar.register_arrays(P, Q, max_rmse_mm=3.0)
    assert result.success, result.message
    assert result.rmse_mm < 3.0


def test_result_transform_shape():
    R = _rot_z(15)
    t = np.array([1., 2., 3.])
    rng = np.random.default_rng(8)
    P, Q = _make_pairs(R, t, rng=rng)
    result = LandmarkRegistrar.register_arrays(P, Q)
    assert result.transform.shape == (4, 4)


def test_result_transform_is_rigid():
    """Bottom row must be [0,0,0,1] and rotation must be orthogonal."""
    R = _rot_z(60)
    t = np.array([3., -6., 9.])
    rng = np.random.default_rng(9)
    P, Q = _make_pairs(R, t, rng=rng)
    result = LandmarkRegistrar.register_arrays(P, Q)
    assert result.success
    T = result.transform
    np.testing.assert_allclose(T[3], [0., 0., 0., 1.], atol=1e-9)
    Rot = T[:3, :3]
    np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-9)


def test_register_arrays_convenience():
    R = _rot_z(10)
    t = np.zeros(3)
    rng = np.random.default_rng(5)
    P, Q = _make_pairs(R, t, rng=rng)
    result = LandmarkRegistrar.register_arrays(P, Q)
    assert result.success


# ---------------------------------------------------------------------------
# Helpers unit tests
# ---------------------------------------------------------------------------

def test_collinear_detector_true():
    pts = np.column_stack([np.linspace(0, 10, 5), np.zeros(5), np.zeros(5)])
    assert _are_collinear(pts)


def test_collinear_detector_false():
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(5, 3))
    assert not _are_collinear(pts)


def test_umeyama_identity():
    rng = np.random.default_rng(42)
    P = rng.uniform(-10, 10, size=(6, 3))
    R, t = _umeyama_svd(P, P)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-9)
