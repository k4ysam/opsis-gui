"""Tests for rendering/reslice_driver.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop, QTimer

from surgical_nav.app.scene_graph import SceneGraph, TransformNode
from surgical_nav.rendering.reslice_driver import ResliceDriver


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


def _process_events(ms=50):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class _FakeViewer:
    """Records set_slice_position and set_crosshair calls."""
    def __init__(self):
        self.positions = []
        self.crosshairs = []

    def set_slice_position(self, x, y, z):
        self.positions.append((x, y, z))

    def set_crosshair(self, x, y, z):
        self.crosshairs.append((x, y, z))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_creates(qapp):
    driver = ResliceDriver(SceneGraph.instance(), [])
    driver._timer.stop()
    assert driver is not None


def test_not_frozen_initially(qapp):
    driver = ResliceDriver(SceneGraph.instance(), [])
    driver._timer.stop()
    assert not driver.is_frozen


# ---------------------------------------------------------------------------
# tip_in_image — no registration
# ---------------------------------------------------------------------------

def test_tip_none_without_registration(qapp):
    driver = ResliceDriver(SceneGraph.instance(), [])
    driver._timer.stop()
    driver.on_transform("PointerToTracker", np.eye(4))
    assert driver.tip_in_image() is None


# ---------------------------------------------------------------------------
# tip_in_image — with registration (identity)
# ---------------------------------------------------------------------------

def test_tip_identity_registration(qapp):
    """Identity cal + identity registration → tip = tracker origin."""
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))

    driver = ResliceDriver(sg, [])
    driver._timer.stop()

    T = np.eye(4)
    T[:3, 3] = [10., 20., 30.]
    driver.on_transform("PointerToTracker", T)
    tip = driver.tip_in_image()
    assert tip is not None
    np.testing.assert_allclose(tip, [10., 20., 30.], atol=1e-9)


def test_tip_with_calibration(qapp):
    """Calibration offset [0,0,100] → tip is 100 mm along tool Z."""
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))

    cal = np.eye(4); cal[:3, 3] = [0., 0., 100.]
    sg.add_node(TransformNode(node_id="POINTER_CALIBRATION", matrix=cal))

    driver = ResliceDriver(sg, [])
    driver._timer.stop()

    driver.on_transform("PointerToTracker", np.eye(4))
    tip = driver.tip_in_image()
    np.testing.assert_allclose(tip, [0., 0., 100.], atol=1e-9)


def test_tip_with_registration_translation(qapp):
    """IMAGE_REGISTRATION translates by [5,0,0] → tip shifted."""
    sg = SceneGraph.instance()
    reg = np.eye(4); reg[:3, 3] = [5., 0., 0.]
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=reg))

    driver = ResliceDriver(sg, [])
    driver._timer.stop()

    T = np.eye(4); T[:3, 3] = [10., 0., 0.]
    driver.on_transform("PointerToTracker", T)
    # tip_tracker = [10,0,0]; tip_image = reg_inv @ [10,0,0,1] = [5,0,0]
    tip = driver.tip_in_image()
    np.testing.assert_allclose(tip, [5., 0., 0.], atol=1e-9)


def test_tip_with_icp_refinement(qapp):
    """Refinement adds extra translation on top of landmark registration."""
    sg = SceneGraph.instance()
    reg = np.eye(4); reg[:3, 3] = [0., 0., 0.]
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=reg))
    ref = np.eye(4); ref[:3, 3] = [2., 0., 0.]
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION_REFINEMENT", matrix=ref))

    driver = ResliceDriver(sg, [])
    driver._timer.stop()

    T = np.eye(4); T[:3, 3] = [10., 0., 0.]
    driver.on_transform("PointerToTracker", T)
    # tip_image = refine_inv @ reg_inv @ [10,0,0] = [10-2, 0, 0] = [8,0,0]
    tip = driver.tip_in_image()
    np.testing.assert_allclose(tip, [8., 0., 0.], atol=1e-9)


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def test_freeze_stops_updates(qapp):
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))
    viewer = _FakeViewer()
    driver = ResliceDriver(sg, [viewer], render_hz=100.0)

    T = np.eye(4); T[:3, 3] = [1., 2., 3.]
    driver.on_transform("PointerToTracker", T)
    _process_events(50)

    n_before = len(viewer.positions)
    driver.set_frozen(True)

    T2 = np.eye(4); T2[:3, 3] = [9., 9., 9.]
    driver.on_transform("PointerToTracker", T2)
    _process_events(50)

    # No new position updates while frozen
    assert len(viewer.positions) == n_before
    driver._timer.stop()


def test_unfreeze_resumes(qapp):
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))
    viewer = _FakeViewer()
    driver = ResliceDriver(sg, [viewer], render_hz=100.0)

    driver.set_frozen(True)
    T = np.eye(4); T[:3, 3] = [5., 5., 5.]
    driver.on_transform("PointerToTracker", T)
    _process_events(50)
    n_frozen = len(viewer.positions)

    driver.set_frozen(False)
    driver.on_transform("PointerToTracker", T)
    _process_events(50)

    assert len(viewer.positions) > n_frozen
    driver._timer.stop()


# ---------------------------------------------------------------------------
# Viewer updates
# ---------------------------------------------------------------------------

def test_viewers_updated_on_flush(qapp):
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))
    viewer = _FakeViewer()
    driver = ResliceDriver(sg, [viewer], render_hz=100.0)

    T = np.eye(4); T[:3, 3] = [3., 7., 11.]
    driver.on_transform("PointerToTracker", T)
    _process_events(60)

    assert len(viewer.positions) >= 1
    last = viewer.positions[-1]
    np.testing.assert_allclose(last, (3., 7., 11.), atol=1e-9)
    driver._timer.stop()


def test_crosshair_updated(qapp):
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))
    viewer = _FakeViewer()
    driver = ResliceDriver(sg, [viewer], render_hz=100.0)

    T = np.eye(4); T[:3, 3] = [1., 2., 3.]
    driver.on_transform("PointerToTracker", T)
    _process_events(60)

    assert len(viewer.crosshairs) >= 1
    driver._timer.stop()


def test_non_pointer_transform_ignored(qapp):
    sg = SceneGraph.instance()
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4)))
    viewer = _FakeViewer()
    driver = ResliceDriver(sg, [viewer], render_hz=100.0)

    driver.on_transform("HeadFrameToTracker", np.eye(4))
    _process_events(60)
    assert viewer.positions == []
    driver._timer.stop()
