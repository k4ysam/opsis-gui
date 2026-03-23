"""Tests for registration/surface_registrar.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import vtkmodules.all as vtk

from surgical_nav.registration.surface_registrar import (
    SurfaceRegistrar, ICPResult,
    _numpy_to_vtk_transform, _vtk_matrix_to_numpy, _mean_closest_point_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_polydata(radius: float = 20.0, center=(0., 0., 0.),
                     resolution: int = 16) -> vtk.vtkPolyData:
    """Create a vtkPolyData sphere."""
    src = vtk.vtkSphereSource()
    src.SetRadius(radius)
    src.SetCenter(*center)
    src.SetPhiResolution(resolution)
    src.SetThetaResolution(resolution)
    src.Update()
    return src.GetOutput()


def _translate_polydata(poly: vtk.vtkPolyData, tx, ty, tz) -> vtk.vtkPolyData:
    T = np.eye(4)
    T[0, 3] = tx; T[1, 3] = ty; T[2, 3] = tz
    vtk_t = _numpy_to_vtk_transform(T)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(vtk_t)
    f.Update()
    return f.GetOutput()


def _rot_z_matrix(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4)
    m[0, 0] = c; m[0, 1] = -s
    m[1, 0] = s; m[1, 1] = c
    return m


# ---------------------------------------------------------------------------
# Empty input guards
# ---------------------------------------------------------------------------

def test_empty_source_fails():
    empty = vtk.vtkPolyData()
    target = _sphere_polydata()
    result = SurfaceRegistrar().register(empty, target)
    assert not result.success
    assert "no points" in result.message.lower()


def test_empty_target_fails():
    source = _sphere_polydata()
    empty = vtk.vtkPolyData()
    result = SurfaceRegistrar().register(source, empty)
    assert not result.success


# ---------------------------------------------------------------------------
# Return types and structure
# ---------------------------------------------------------------------------

def test_result_is_icp_result():
    sphere = _sphere_polydata()
    result = SurfaceRegistrar().register(sphere, sphere)
    assert isinstance(result, ICPResult)


def test_transform_shape():
    sphere = _sphere_polydata()
    result = SurfaceRegistrar().register(sphere, sphere)
    assert result.transform.shape == (4, 4)


def test_identical_surfaces_near_zero_distance():
    """Registering a sphere onto itself should yield near-zero mean distance."""
    sphere = _sphere_polydata(radius=30.0)
    result = SurfaceRegistrar(max_mean_distance_mm=5.0).register(sphere, sphere)
    assert result.success, result.message
    assert result.mean_distance < 1.0


# ---------------------------------------------------------------------------
# Small translation recovery
# ---------------------------------------------------------------------------

def test_small_translation_converges():
    """Source shifted 5 mm → ICP should converge within threshold."""
    target = _sphere_polydata(radius=30.0, center=(0., 0., 0.))
    source = _sphere_polydata(radius=30.0, center=(5., 0., 0.))
    result = SurfaceRegistrar(max_mean_distance_mm=3.0).register(source, target)
    assert result.success, result.message
    assert result.mean_distance < 3.0


def test_initial_transform_applied():
    """A correct initial transform pre-aligns source; ICP refines to near-zero dist."""
    target = _sphere_polydata(radius=30.0, center=(0., 0., 0.))
    source = _sphere_polydata(radius=30.0, center=(20., 0., 0.))
    # Initial transform moves source back
    init = np.eye(4)
    init[0, 3] = -20.0
    result = SurfaceRegistrar(max_mean_distance_mm=2.0).register(
        source, target, initial_transform=init
    )
    assert result.success, result.message


# ---------------------------------------------------------------------------
# Source point count in result
# ---------------------------------------------------------------------------

def test_n_source_points_matches_input():
    sphere = _sphere_polydata()
    result = SurfaceRegistrar().register(sphere, sphere)
    assert result.n_source_points == sphere.GetNumberOfPoints()


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_numpy_vtk_matrix_roundtrip():
    m = np.array([
        [1., 0., 0., 5.],
        [0., 1., 0., 10.],
        [0., 0., 1., -3.],
        [0., 0., 0., 1.],
    ])
    vtk_t = _numpy_to_vtk_transform(m)
    vtk_m = vtk_t.GetMatrix()
    recovered = _vtk_matrix_to_numpy(vtk_m)
    np.testing.assert_allclose(recovered, m, atol=1e-9)


def test_mean_distance_identical_surfaces():
    sphere = _sphere_polydata(radius=20.0)
    dist = _mean_closest_point_distance(sphere, sphere, np.eye(4))
    assert dist < 1e-3


def test_mean_distance_shifted():
    target = _sphere_polydata(radius=20.0, center=(0., 0., 0.))
    source = _sphere_polydata(radius=20.0, center=(0., 0., 0.))
    T = np.eye(4); T[0, 3] = 100.0   # shift source far away
    dist = _mean_closest_point_distance(source, target, T)
    assert dist > 70.0   # must be much larger than sphere radius
