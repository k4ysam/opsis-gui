"""Tests for app/scene_graph.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from surgical_nav.app.scene_graph import (
    SceneGraph, TransformNode, VolumeNode, FiducialSetNode, ModelNode,
)
from surgical_nav.utils.math_utils import identity, make_transform


@pytest.fixture(autouse=True)
def fresh_graph():
    """Each test gets a clean SceneGraph singleton."""
    SceneGraph.reset()
    yield
    SceneGraph.reset()


def sg():
    return SceneGraph.instance()


# -- Node add / get / remove ------------------------------------------

def test_add_and_get_node():
    node = TransformNode(node_id="T1")
    sg().add_node(node)
    assert sg().get_node("T1") is node


def test_get_missing_node_returns_none():
    assert sg().get_node("nope") is None


def test_remove_node():
    sg().add_node(TransformNode(node_id="T1"))
    sg().remove_node("T1")
    assert sg().get_node("T1") is None


def test_nodes_of_type():
    sg().add_node(TransformNode(node_id="T1"))
    sg().add_node(VolumeNode(node_id="V1"))
    transforms = sg().nodes_of_type(TransformNode)
    volumes = sg().nodes_of_type(VolumeNode)
    assert len(transforms) == 1
    assert len(volumes) == 1


# -- Transform tree -------------------------------------------------------

def test_world_transform_single_node():
    theta = np.pi / 3
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    T = make_transform(R, np.array([1.0, 2.0, 3.0]))
    sg().add_node(TransformNode(node_id="A", matrix=T))
    np.testing.assert_allclose(sg().get_world_transform("A"), T)


def test_world_transform_chain():
    """World = root @ middle @ leaf (left-to-right composition)."""
    T1 = identity(); T1[0, 3] = 10.0
    T2 = identity(); T2[1, 3] = 5.0
    T3 = identity(); T3[2, 3] = 2.0
    sg().add_node(TransformNode(node_id="root",   matrix=T1))
    sg().add_node(TransformNode(node_id="middle", matrix=T2))
    sg().add_node(TransformNode(node_id="leaf",   matrix=T3))
    sg().set_parent("middle", "root")
    sg().set_parent("leaf", "middle")
    world = sg().get_world_transform("leaf")
    # Composed: T1 @ T2 @ T3 = translate (10, 5, 2)
    assert world[0, 3] == pytest.approx(10.0)
    assert world[1, 3] == pytest.approx(5.0)
    assert world[2, 3] == pytest.approx(2.0)


def test_world_transform_no_parent_is_node_matrix():
    T = identity(); T[0, 3] = 7.0
    sg().add_node(TransformNode(node_id="solo", matrix=T))
    np.testing.assert_allclose(sg().get_world_transform("solo"), T)


def test_world_transform_non_transform_node():
    """A non-transform node with no parents returns identity."""
    sg().add_node(VolumeNode(node_id="Vol"))
    np.testing.assert_allclose(sg().get_world_transform("Vol"), identity())


def test_update_transform():
    sg().add_node(TransformNode(node_id="T1"))
    new_m = identity(); new_m[0, 3] = 99.0
    sg().update_transform("T1", new_m)
    assert sg().get_node("T1").matrix[0, 3] == pytest.approx(99.0)


def test_update_transform_wrong_type():
    sg().add_node(VolumeNode(node_id="V1"))
    with pytest.raises(KeyError):
        sg().update_transform("V1", identity())


def test_set_parent_to_none_detaches():
    sg().add_node(TransformNode(node_id="parent"))
    sg().add_node(TransformNode(node_id="child"))
    sg().set_parent("child", "parent")
    sg().set_parent("child", None)
    assert sg().get_parent("child") is None


def test_set_parent_missing_raises():
    sg().add_node(TransformNode(node_id="child"))
    with pytest.raises(KeyError):
        sg().set_parent("child", "ghost")


def test_remove_parent_detaches_children():
    sg().add_node(TransformNode(node_id="P"))
    sg().add_node(TransformNode(node_id="C"))
    sg().set_parent("C", "P")
    sg().remove_node("P")
    assert sg().get_parent("C") is None


# -- Observer pattern -----------------------------------------------------

def test_observer_node_added():
    received = []
    sg().add_observer("node_added", received.append)
    node = TransformNode(node_id="T1")
    sg().add_node(node)
    assert received == [node]


def test_observer_node_modified():
    sg().add_node(TransformNode(node_id="T1"))
    received = []
    sg().add_observer("node_modified", received.append)
    sg().update_transform("T1", identity())
    assert len(received) == 1


def test_observer_removed():
    received = []
    cb = received.append
    sg().add_observer("node_added", cb)
    sg().remove_observer("node_added", cb)
    sg().add_node(TransformNode(node_id="T1"))
    assert received == []


# -- Fiducial helpers ---------------------------------------------------

def test_add_fiducial_creates_node():
    sg().add_fiducial("LM", "Nasion", np.array([0, 0, 0]))
    node = sg().get_node("LM")
    assert isinstance(node, FiducialSetNode)
    assert len(node.points) == 1
    assert node.points[0]["label"] == "Nasion"


def test_clear_fiducials():
    sg().add_fiducial("LM", "A", np.array([1, 2, 3]))
    sg().add_fiducial("LM", "B", np.array([4, 5, 6]))
    sg().clear_fiducials("LM")
    assert sg().get_node("LM").points == []


def test_full_surgical_nav_chain():
    """Reproduce the SlicerOpenNav 4-level transform chain."""
    T_ptr = identity(); T_ptr[0, 3] = 1.0   # PointerToTracker
    T_cal = identity(); T_cal[0, 3] = 2.0   # POINTER_CALIBRATION
    T_reg = identity(); T_reg[0, 3] = 3.0   # IMAGE_REGISTRATION
    T_icp = identity(); T_icp[0, 3] = 4.0   # IMAGE_REGISTRATION_REFINEMENT

    sg().add_node(TransformNode(node_id="PointerToTracker",               matrix=T_ptr))
    sg().add_node(TransformNode(node_id="POINTER_CALIBRATION",            matrix=T_cal))
    sg().add_node(TransformNode(node_id="IMAGE_REGISTRATION",             matrix=T_reg))
    sg().add_node(TransformNode(node_id="IMAGE_REGISTRATION_REFINEMENT",  matrix=T_icp))

    sg().set_parent("POINTER_CALIBRATION",           "PointerToTracker")
    sg().set_parent("IMAGE_REGISTRATION",            "POINTER_CALIBRATION")
    sg().set_parent("IMAGE_REGISTRATION_REFINEMENT", "IMAGE_REGISTRATION")

    world = sg().get_world_transform("IMAGE_REGISTRATION_REFINEMENT")
    # T_ptr @ T_cal @ T_reg @ T_icp → translation sums to 10
    assert world[0, 3] == pytest.approx(10.0)
