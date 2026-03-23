"""Headless tests for workflow/planning_page.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import SimpleITK as sitk
import vtkmodules.all as vtk
from unittest.mock import patch
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import SceneGraph, VolumeNode, ModelNode, FiducialSetNode


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


def _push_volume():
    """Push a tiny synthetic volume into the SceneGraph."""
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[5:15, 5:15, 5:15] = 100.0
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(20, 20, 20)
    vtk_img.SetSpacing(1.0, 1.0, 1.0)
    vtk_img.AllocateScalars(vtk.VTK_FLOAT, 1)

    node = VolumeNode(node_id="ACTIVE_VOLUME", name="test",
                      sitk_image=sitk_img, vtk_image_data=vtk_img)
    SceneGraph.instance().add_node(node)
    return sitk_img, vtk_img


# ---------------------------------------------------------------------------
# Creation & initial state
# ---------------------------------------------------------------------------

def test_planning_page_creates(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    assert page is not None


def test_initial_step_is_1(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    assert page._step == 1


def test_only_step1_visible_initially(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.show()
    assert page._step1_box.isVisible()
    assert not page._step2_box.isVisible()
    assert not page._step3_box.isVisible()
    assert not page._step4_box.isVisible()


# ---------------------------------------------------------------------------
# Step navigation
# ---------------------------------------------------------------------------

def test_go_to_step2(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.show()
    page._go_to_step(2)
    assert not page._step1_box.isVisible()
    assert page._step2_box.isVisible()


def test_go_to_step4(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.show()
    page._go_to_step(4)
    assert page._step4_box.isVisible()
    for box in (page._step1_box, page._step2_box, page._step3_box):
        assert not box.isVisible()


# ---------------------------------------------------------------------------
# Step 1 — on_enter loads volume from SceneGraph
# ---------------------------------------------------------------------------

def test_on_enter_loads_sitk_image(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    sitk_img, _ = _push_volume()
    page = PlanningPage()
    page.on_enter()
    assert page._sitk_image is sitk_img


# ---------------------------------------------------------------------------
# Step 2 — seed_target_at_ijk
# ---------------------------------------------------------------------------

def test_seed_target_pushes_model_node(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    _push_volume()
    page = PlanningPage()
    page.on_enter()
    page.seed_target_at_ijk((10, 10, 10))
    node = SceneGraph.instance().get_node("TARGET_MODEL")
    assert isinstance(node, ModelNode)


def test_seed_target_emits_signal(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    _push_volume()
    page = PlanningPage()
    page.on_enter()
    received = []
    page.target_mesh_ready.connect(received.append)
    page.seed_target_at_ijk((10, 10, 10))
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Step 3 — trajectory fiducials
# ---------------------------------------------------------------------------

def test_place_entry_point(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.place_trajectory_point("entry", np.array([1.0, 2.0, 3.0]))
    node = SceneGraph.instance().get_node("TRAJECTORY_POINTS")
    labels = [p["label"] for p in node.points]
    assert "Entry" in labels


def test_place_target_point(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.place_trajectory_point("target", np.array([4.0, 5.0, 6.0]))
    node = SceneGraph.instance().get_node("TRAJECTORY_POINTS")
    labels = [p["label"] for p in node.points]
    assert "Target" in labels


def test_replace_existing_entry(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.place_trajectory_point("entry", np.array([1.0, 0.0, 0.0]))
    page.place_trajectory_point("entry", np.array([9.0, 0.0, 0.0]))
    node = SceneGraph.instance().get_node("TRAJECTORY_POINTS")
    entries = [p for p in node.points if p["label"] == "Entry"]
    assert len(entries) == 1
    np.testing.assert_allclose(entries[0]["position"], [9.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Step 4 — landmarks
# ---------------------------------------------------------------------------

def test_place_landmark_adds_to_table(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.place_landmark(np.array([1.0, 2.0, 3.0]))
    assert page._lm_table.rowCount() == 1


def test_three_landmarks_updates_status(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    for i in range(3):
        page.place_landmark(np.array([float(i), 0.0, 0.0]))
    assert "ready" in page._lm_status.text()


def test_complete_planning_requires_3_landmarks(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    fired = []
    page.stage_complete.connect(lambda: fired.append(True))
    page.place_landmark(np.array([0.0, 0.0, 0.0]))
    page.place_landmark(np.array([1.0, 0.0, 0.0]))
    with patch("surgical_nav.workflow.planning_page.QMessageBox.warning"):
        page._complete_planning()   # only 2 → should NOT fire
    assert fired == []


def test_complete_planning_fires_with_3_landmarks(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    fired = []
    page.stage_complete.connect(lambda: fired.append(True))
    for i in range(3):
        page.place_landmark(np.array([float(i), 0.0, 0.0]))
    page._complete_planning()
    assert fired == [True]


def test_delete_landmark(qapp):
    from surgical_nav.workflow.planning_page import PlanningPage
    page = PlanningPage()
    page.place_landmark(np.array([1.0, 0.0, 0.0]))
    page.place_landmark(np.array([2.0, 0.0, 0.0]))
    page._lm_table.selectRow(0)
    page._delete_landmark()
    assert page._lm_table.rowCount() == 1
