"""Tests for rendering/volume_viewer.py (VTK-backed VolumeViewer)."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import ModelNode


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def test_volume_viewer_creates(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    assert viewer is not None


def test_set_volume_does_not_crash(qapp):
    import vtkmodules.all as vtk
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    img = vtk.vtkImageData()
    img.SetDimensions(10, 10, 10)
    viewer.set_volume(img)


def test_add_model_does_not_crash(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    node = ModelNode(node_id="skin", vtk_poly_data=None, color=(1, 0.8, 0.7), opacity=0.6)
    viewer.add_model(node)


def test_remove_model_does_not_crash(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.remove_model("nonexistent")


def test_set_pointer_transform_does_not_crash(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.set_pointer_transform(np.eye(4))


def test_add_surface_does_not_crash(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.add_surface(None)


def test_get_renderer_returns_renderer(qapp):
    import vtkmodules.all as vtk
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    assert isinstance(viewer.get_renderer(), vtk.vtkRenderer)
