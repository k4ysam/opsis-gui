"""Headless tests for workflow/registration_page.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import SceneGraph, TransformNode, FiducialSetNode


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


def _push_landmarks(n=3):
    """Add n planning landmarks to the SceneGraph."""
    node = FiducialSetNode(node_id="PLANNING_LANDMARKS")
    for i in range(n):
        pos = np.array([float(i)*10, 0., 0.])
        node.points.append({"label": f"LM{i+1}", "position": pos})
    SceneGraph.instance().add_node(node)
    return node


def _make_pivot_samples(n=60):
    """Generate noiseless pivot transforms."""
    from surgical_nav.calibration.pivot_calibrator import PivotCalibrator
    p_tool = np.array([0., 0., 100.])
    p_ref  = np.array([10., 20., 30.])
    rng = np.random.default_rng(1)
    ax = np.array([0., 0., 1.])
    samples = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle2 = rng.uniform(0.3, np.pi)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3) + np.sin(angle2)*K + (1-np.cos(angle2))*(K@K)
        t = p_ref - R @ p_tool
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
        samples.append(T)
    return samples


# ---------------------------------------------------------------------------
# Creation and step navigation
# ---------------------------------------------------------------------------

def test_creates(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    assert page is not None


def test_initial_step_is_1(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    assert page._step == 1


def test_step_visibility(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page.show()
    assert page._step1_box.isVisible()
    for box in (page._step2_box, page._step3_box, page._step4_box,
                page._step5_box, page._step6_box, page._step7_box,
                page._step8_box, page._step9_box):
        assert not box.isVisible()


def test_navigate_to_step3(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page.show()
    page._go_to_step(3)
    assert page._step3_box.isVisible()
    assert not page._step1_box.isVisible()


# ---------------------------------------------------------------------------
# Pivot collection
# ---------------------------------------------------------------------------

def test_pivot_collection_increments(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page._start_pivot_collection()
    T = np.eye(4)
    page.receive_transform("PointerToTracker", T)
    assert page._pivot_cal.sample_count == 1


def test_non_pointer_ignored(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page._start_pivot_collection()
    page.receive_transform("HeadFrameToTracker", np.eye(4))
    assert page._pivot_cal.sample_count == 0


def test_pivot_calibration_stores_node(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    for T in _make_pivot_samples(150):
        page._pivot_cal.add_sample(T)
    page._run_pivot_calibration()
    node = SceneGraph.instance().get_node("POINTER_CALIBRATION")
    assert node is not None
    assert isinstance(node, TransformNode)


# ---------------------------------------------------------------------------
# Landmark loading
# ---------------------------------------------------------------------------

def test_on_enter_loads_image_landmarks(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    _push_landmarks(4)
    page = RegistrationPage()
    page.on_enter()
    assert len(page._image_landmarks) == 4


def test_add_patient_landmark_increments_pairs(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    _push_landmarks(3)
    page = RegistrationPage()
    page.on_enter()
    page.add_patient_landmark(np.array([1., 2., 3.]))
    assert page._landmark_reg.pair_count == 1


def test_add_patient_landmark_no_image_landmarks(qapp):
    """Should show warning and not crash if no planning landmarks loaded."""
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page.on_enter()   # no landmarks in scene
    with patch("surgical_nav.workflow.registration_page.QMessageBox.warning"):
        page.add_patient_landmark(np.array([0., 0., 0.]))
    assert page._landmark_reg.pair_count == 0


# ---------------------------------------------------------------------------
# Landmark registration
# ---------------------------------------------------------------------------

def test_landmark_registration_stores_node(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    from surgical_nav.registration.landmark_registrar import LandmarkRegistrar
    _push_landmarks(4)
    page = RegistrationPage()
    page.on_enter()

    # Inject 4 matching pairs directly (identity transform)
    rng = np.random.default_rng(5)
    for pos in page._image_landmarks:
        noise = rng.normal(scale=0.1, size=3)
        page._landmark_reg.add_pair(pos + noise, pos)

    page._run_landmark_registration()
    node = SceneGraph.instance().get_node("IMAGE_REGISTRATION")
    assert node is not None
    assert isinstance(node, TransformNode)


# ---------------------------------------------------------------------------
# Surface point collection
# ---------------------------------------------------------------------------

def test_add_surface_point(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    page.add_surface_point(np.array([1., 2., 3.]))
    assert len(page._surface_points) == 1


# ---------------------------------------------------------------------------
# Accept registration fires stage_complete
# ---------------------------------------------------------------------------

def test_accept_registration_fires_complete(qapp):
    from surgical_nav.workflow.registration_page import RegistrationPage
    page = RegistrationPage()
    # Push a registration node
    SceneGraph.instance().add_node(TransformNode(
        node_id="IMAGE_REGISTRATION", matrix=np.eye(4)
    ))
    fired = []
    page.stage_complete.connect(lambda: fired.append(True))
    page._accept_registration()
    assert fired == [True]
