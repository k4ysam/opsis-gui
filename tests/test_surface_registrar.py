"""Tests for registration/surface_registrar.py (VTK-free stub)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from surgical_nav.registration.surface_registrar import SurfaceRegistrar, ICPResult


def test_register_returns_icp_result():
    result = SurfaceRegistrar().register(None, None)
    assert isinstance(result, ICPResult)


def test_register_returns_failure():
    result = SurfaceRegistrar().register(None, None)
    assert not result.success


def test_register_transform_shape():
    result = SurfaceRegistrar().register(None, None)
    assert result.transform.shape == (4, 4)


def test_register_message_mentions_vtk():
    result = SurfaceRegistrar().register(None, None)
    assert "vtk" in result.message.lower() or "unavailable" in result.message.lower()
