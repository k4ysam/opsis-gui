"""Headless tests for workflow/navigation_page.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import SceneGraph, TransformNode, FiducialSetNode
from surgical_nav.workflow.navigation_page import NavigationPage


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


class _FakeSliceViewer:
    def __init__(self):
        self.positions = []
        self.crosshairs = []

    def set_slice_position(self, x, y, z):
        self.positions.append((x, y, z))

    def set_crosshair(self, x, y, z):
        self.crosshairs.append((x, y, z))


class _FakeVolumeViewer:
    def __init__(self):
        self.trajectory_calls = []
        self.pointer_statuses = []

    def set_trajectory(self, entry, target):
        self.trajectory_calls.append((entry.copy(), target.copy()))

    def set_pointer_status(self, status):
        self.pointer_statuses.append(status)


def _make_page(qapp):
    viewers = [_FakeSliceViewer(), _FakeSliceViewer(), _FakeSliceViewer()]
    vol = _FakeVolumeViewer()
    page = NavigationPage(viewers, vol)
    return page, viewers, vol


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_creates(qapp):
    page, _, _ = _make_page(qapp)
    assert page is not None


def test_has_freeze_button(qapp):
    page, _, _ = _make_page(qapp)
    assert page._freeze_btn is not None
    assert not page._freeze_btn.isChecked()


# ---------------------------------------------------------------------------
# on_transform forwarded to ResliceDriver
# ---------------------------------------------------------------------------

def test_on_transform_ignored_without_registration(qapp):
    """tip_in_image is None without IMAGE_REGISTRATION → no viewer updates."""
    page, viewers, _ = _make_page(qapp)
    page.on_transform("PointerToTracker", np.eye(4))
    # No registration node → driver cannot compute tip → _dist_lbl shows dash
    page._update_distance()
    assert page._dist_lbl.text() == "Distance to target: — mm"


def test_on_transform_with_registration(qapp):
    SceneGraph.instance().add_node(
        TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4))
    )
    page, viewers, _ = _make_page(qapp)
    T = np.eye(4); T[:3, 3] = [1., 2., 3.]
    page.on_transform("PointerToTracker", T)
    tip = page._reslice_driver.tip_in_image()
    np.testing.assert_allclose(tip, [1., 2., 3.], atol=1e-9)


# ---------------------------------------------------------------------------
# Distance to target
# ---------------------------------------------------------------------------

def _add_trajectory(entry=(0., 0., 0.), target=(0., 0., 100.)):
    node = FiducialSetNode(node_id="TRAJECTORY_POINTS")
    node.points.append({"label": "Entry",  "position": np.array(entry)})
    node.points.append({"label": "Target", "position": np.array(target)})
    SceneGraph.instance().add_node(node)


def test_distance_label_updates(qapp):
    SceneGraph.instance().add_node(
        TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4))
    )
    _add_trajectory(entry=[0., 0., 0.], target=[0., 0., 100.])
    page, _, _ = _make_page(qapp)
    T = np.eye(4); T[:3, 3] = [0., 0., 50.]
    page.on_transform("PointerToTracker", T)
    page._update_distance()
    assert "50.0 mm" in page._dist_lbl.text()


def test_depth_label_updates(qapp):
    SceneGraph.instance().add_node(
        TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4))
    )
    _add_trajectory(entry=[0., 0., 0.], target=[0., 0., 100.])
    page, _, _ = _make_page(qapp)
    T = np.eye(4); T[:3, 3] = [0., 0., 30.]
    page.on_transform("PointerToTracker", T)
    page._update_distance()
    assert "30.0 mm" in page._depth_lbl.text()
    assert "100.0 mm" in page._depth_lbl.text()


def test_no_target_shows_dash(qapp):
    SceneGraph.instance().add_node(
        TransformNode(node_id="IMAGE_REGISTRATION", matrix=np.eye(4))
    )
    page, _, _ = _make_page(qapp)
    T = np.eye(4); T[:3, 3] = [5., 5., 5.]
    page.on_transform("PointerToTracker", T)
    page._update_distance()
    assert "no target" in page._dist_lbl.text()


# ---------------------------------------------------------------------------
# Trajectory refresh
# ---------------------------------------------------------------------------

def test_refresh_trajectory_calls_volume_viewer(qapp):
    _add_trajectory(entry=[1., 2., 3.], target=[4., 5., 6.])
    page, _, vol = _make_page(qapp)
    page._refresh_trajectory()
    assert len(vol.trajectory_calls) == 1
    np.testing.assert_allclose(vol.trajectory_calls[0][0], [1., 2., 3.])
    np.testing.assert_allclose(vol.trajectory_calls[0][1], [4., 5., 6.])


def test_refresh_trajectory_no_op_without_points(qapp):
    page, _, vol = _make_page(qapp)
    page._refresh_trajectory()
    assert vol.trajectory_calls == []


# ---------------------------------------------------------------------------
# Freeze toggle
# ---------------------------------------------------------------------------

def test_freeze_toggle_changes_button_text(qapp):
    page, _, _ = _make_page(qapp)
    page._freeze_btn.setChecked(True)
    assert page._freeze_btn.text() == "Unfreeze"
    page._freeze_btn.setChecked(False)
    assert page._freeze_btn.text() == "Freeze"


def test_freeze_propagates_to_driver(qapp):
    page, _, _ = _make_page(qapp)
    assert not page._reslice_driver.is_frozen
    page._on_freeze_toggled(True)
    assert page._reslice_driver.is_frozen
    page._on_freeze_toggled(False)
    assert not page._reslice_driver.is_frozen


# ---------------------------------------------------------------------------
# Pointer status forwarded to volume viewer
# ---------------------------------------------------------------------------

def test_set_pointer_status_forwarded(qapp):
    page, _, vol = _make_page(qapp)
    page.set_pointer_status("SEEN")
    assert vol.pointer_statuses == ["SEEN"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_on_enter_starts_timer(qapp):
    page, _, _ = _make_page(qapp)
    page._dist_timer.stop()  # make sure it's stopped
    page.on_enter()
    assert page._dist_timer.isActive()
    page._dist_timer.stop()


def test_on_leave_stops_timer(qapp):
    page, _, _ = _make_page(qapp)
    page.on_enter()
    page.on_leave()
    assert not page._dist_timer.isActive()
