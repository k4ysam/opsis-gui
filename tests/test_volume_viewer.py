"""Headless tests for rendering/volume_viewer.py.

Tests VolumeViewer's public API without actually rendering (offscreen Qt).
"""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import vtkmodules.all as vtk
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import ModelNode

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _make_vtk_image(dims=(10, 10, 5)):
    img = vtk.vtkImageData()
    img.SetDimensions(*dims)
    img.SetSpacing(1.0, 1.0, 2.0)
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    img.GetPointData().GetScalars().Fill(100.0)
    return img


def _make_sphere_poly():
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(10.0)
    sphere.Update()
    return sphere.GetOutput()


def test_volume_viewer_creates(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    assert viewer is not None


def test_set_volume_does_not_crash(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.set_volume(_make_vtk_image())


def test_add_model_adds_actor(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    node = ModelNode(node_id="skin", vtk_poly_data=_make_sphere_poly(),
                     color=(1, 0.8, 0.7), opacity=0.6)
    viewer.add_model(node)
    assert "skin" in viewer._model_actors


def test_remove_model(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    node = ModelNode(node_id="target", vtk_poly_data=_make_sphere_poly())
    viewer.add_model(node)
    viewer.remove_model("target")
    assert "target" not in viewer._model_actors


def test_add_model_none_polydata_skipped(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    node = ModelNode(node_id="empty", vtk_poly_data=None)
    viewer.add_model(node)
    assert "empty" not in viewer._model_actors


def test_set_pointer_transform_shows_actor(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    T = np.eye(4)
    T[:3, 3] = [10, 20, 30]
    viewer.set_pointer_transform(T)
    assert viewer._pointer_actor.GetVisibility() == 1


def test_set_pointer_status_colors(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.set_pointer_status("SEEN")
    color = viewer._pointer_actor.GetProperty().GetColor()
    assert color[1] > 0.8   # green channel dominant for SEEN

    viewer.set_pointer_status("NOT_SEEN")
    color = viewer._pointer_actor.GetProperty().GetColor()
    assert color[0] > 0.7   # red/yellow for NOT_SEEN

    viewer.set_pointer_status("NEVER_SEEN")
    color = viewer._pointer_actor.GetProperty().GetColor()
    # grey: all channels roughly equal
    assert abs(color[0] - color[1]) < 0.1


def test_hide_pointer(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    viewer.set_pointer_transform(np.eye(4))
    viewer.hide_pointer()
    assert viewer._pointer_actor.GetVisibility() == 0


def test_update_model_visibility(qapp):
    from surgical_nav.rendering.volume_viewer import VolumeViewer
    viewer = VolumeViewer()
    node = ModelNode(node_id="m1", vtk_poly_data=_make_sphere_poly())
    viewer.add_model(node)
    viewer.update_model_visibility("m1", False)
    assert viewer._model_actors["m1"].GetVisibility() == 0
    viewer.update_model_visibility("m1", True)
    assert viewer._model_actors["m1"].GetVisibility() == 1
