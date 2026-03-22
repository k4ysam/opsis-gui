"""Tests for utils/math_utils.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from surgical_nav.utils.math_utils import (
    identity, compose, invert_transform, make_transform,
    apply_transform, rmse, mean_distance, lps_to_ras_transform, lps_to_ras,
)


def test_identity():
    np.testing.assert_array_equal(identity(), np.eye(4))


def test_compose_two():
    A = np.eye(4); A[0, 3] = 1.0   # translate X by 1
    B = np.eye(4); B[1, 3] = 2.0   # translate Y by 2
    C = compose(A, B)
    assert C[0, 3] == pytest.approx(1.0)
    assert C[1, 3] == pytest.approx(2.0)


def test_compose_three():
    A = np.eye(4); A[0, 3] = 1.0
    B = np.eye(4); B[0, 3] = 2.0
    C = np.eye(4); C[0, 3] = 3.0
    result = compose(A, B, C)
    assert result[0, 3] == pytest.approx(6.0)


def test_invert_pure_translation():
    m = identity()
    m[:3, 3] = [5, -3, 7]
    inv = invert_transform(m)
    product = m @ inv
    np.testing.assert_allclose(product, np.eye(4), atol=1e-12)


def test_invert_rotation():
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    m = make_transform(R, np.array([1.0, 2.0, 3.0]))
    inv = invert_transform(m)
    np.testing.assert_allclose(m @ inv, np.eye(4), atol=1e-12)


def test_apply_transform_translation():
    m = identity()
    m[:3, 3] = [10, 0, 0]
    pts = np.array([[0, 0, 0], [1, 0, 0]])
    result = apply_transform(m, pts)
    np.testing.assert_allclose(result, [[10, 0, 0], [11, 0, 0]])


def test_apply_transform_single_point():
    m = identity()
    m[:3, 3] = [0, 5, 0]
    result = apply_transform(m, np.array([0, 0, 0]))
    np.testing.assert_allclose(result, [[0, 5, 0]])


def test_rmse_identical():
    pts = np.random.randn(10, 3)
    assert rmse(pts, pts) == pytest.approx(0.0)


def test_rmse_known():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    b = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
    assert rmse(a, b) == pytest.approx(1.0)


def test_mean_distance():
    a = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
    b = np.array([[3, 4, 0], [3, 4, 0]], dtype=float)   # distance = 5 each
    assert mean_distance(a, b) == pytest.approx(5.0)


def test_lps_to_ras_transform():
    m = lps_to_ras_transform()
    assert m[0, 0] == -1.0
    assert m[1, 1] == -1.0
    assert m[2, 2] == 1.0
    assert m[3, 3] == 1.0


def test_lps_to_ras():
    pts = np.array([[1.0, 2.0, 3.0]])
    converted = lps_to_ras(pts)
    np.testing.assert_allclose(converted, [[-1.0, -2.0, 3.0]])


def test_compose_with_inverse_is_identity():
    theta = np.pi / 6
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    T = make_transform(R, np.array([5.0, -2.0, 1.0]))
    np.testing.assert_allclose(compose(T, invert_transform(T)), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(compose(invert_transform(T), T), np.eye(4), atol=1e-12)
