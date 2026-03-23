"""Tests for persistence/case_manager.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surgical_nav.app.scene_graph import (
    SceneGraph, TransformNode, FiducialSetNode,
)
from surgical_nav.persistence.case_manager import CaseManager


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


@pytest.fixture
def tmp_mgr(tmp_path):
    return CaseManager(root_dir=tmp_path)


# ---------------------------------------------------------------------------
# list_cases
# ---------------------------------------------------------------------------

def test_list_cases_empty(tmp_mgr):
    assert tmp_mgr.list_cases() == []


def test_list_cases_after_save(tmp_mgr):
    tmp_mgr.save_case("Alpha", SceneGraph.instance())
    tmp_mgr.save_case("Beta",  SceneGraph.instance())
    assert tmp_mgr.list_cases() == ["Alpha", "Beta"]


def test_list_cases_ignores_dirs_without_metadata(tmp_mgr):
    (tmp_mgr.root / "Stray").mkdir(parents=True)
    tmp_mgr.save_case("Real", SceneGraph.instance())
    assert tmp_mgr.list_cases() == ["Real"]


# ---------------------------------------------------------------------------
# save / load — transforms
# ---------------------------------------------------------------------------

def test_save_creates_metadata(tmp_mgr):
    tmp_mgr.save_case("Test", SceneGraph.instance())
    meta_path = tmp_mgr.case_dir("Test") / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["case_name"] == "Test"
    assert "created" in meta
    assert "modified" in meta


def test_save_persists_transform_node(tmp_mgr):
    sg = SceneGraph.instance()
    M = np.eye(4); M[:3, 3] = [1., 2., 3.]
    sg.add_node(TransformNode(node_id="IMAGE_REGISTRATION", matrix=M))
    tmp_mgr.save_case("T1", sg)

    npy = tmp_mgr.case_dir("T1") / "transforms" / "IMAGE_REGISTRATION.npy"
    assert npy.exists()
    loaded = np.load(str(npy))
    np.testing.assert_array_equal(loaded, M)


def test_save_skips_missing_transform(tmp_mgr):
    tmp_mgr.save_case("T2", SceneGraph.instance())
    npy = tmp_mgr.case_dir("T2") / "transforms" / "IMAGE_REGISTRATION.npy"
    assert not npy.exists()


def test_load_restores_transform_node(tmp_mgr):
    sg = SceneGraph.instance()
    M = np.eye(4); M[:3, 3] = [5., 6., 7.]
    sg.add_node(TransformNode(node_id="POINTER_CALIBRATION", matrix=M))
    tmp_mgr.save_case("T3", sg)

    SceneGraph.reset()
    sg2 = SceneGraph.instance()
    tmp_mgr.load_case("T3", sg2)

    node = sg2.get_node("POINTER_CALIBRATION")
    assert isinstance(node, TransformNode)
    np.testing.assert_array_equal(node.matrix, M)


def test_load_all_three_transforms(tmp_mgr):
    sg = SceneGraph.instance()
    for key in ("POINTER_CALIBRATION", "IMAGE_REGISTRATION", "IMAGE_REGISTRATION_REFINEMENT"):
        sg.add_node(TransformNode(node_id=key, matrix=np.eye(4)))
    tmp_mgr.save_case("T4", sg)

    SceneGraph.reset()
    sg2 = SceneGraph.instance()
    tmp_mgr.load_case("T4", sg2)

    for key in ("POINTER_CALIBRATION", "IMAGE_REGISTRATION", "IMAGE_REGISTRATION_REFINEMENT"):
        assert isinstance(sg2.get_node(key), TransformNode)


# ---------------------------------------------------------------------------
# save / load — fiducials
# ---------------------------------------------------------------------------

def test_save_persists_fiducials(tmp_mgr):
    sg = SceneGraph.instance()
    node = FiducialSetNode(node_id="PLANNING_LANDMARKS")
    node.points.append({"label": "LM1", "position": np.array([1., 2., 3.])})
    sg.add_node(node)
    tmp_mgr.save_case("F1", sg)

    json_path = tmp_mgr.case_dir("F1") / "landmarks" / "PLANNING_LANDMARKS.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert len(data) == 1
    assert data[0]["label"] == "LM1"
    np.testing.assert_allclose(data[0]["position"], [1., 2., 3.])


def test_load_restores_fiducials(tmp_mgr):
    sg = SceneGraph.instance()
    node = FiducialSetNode(node_id="TRAJECTORY_POINTS")
    node.points.append({"label": "Entry",  "position": np.array([0., 0., 0.])})
    node.points.append({"label": "Target", "position": np.array([0., 0., 100.])})
    sg.add_node(node)
    tmp_mgr.save_case("F2", sg)

    SceneGraph.reset()
    sg2 = SceneGraph.instance()
    tmp_mgr.load_case("F2", sg2)

    restored = sg2.get_node("TRAJECTORY_POINTS")
    assert isinstance(restored, FiducialSetNode)
    assert len(restored.points) == 2
    assert restored.points[0]["label"] == "Entry"
    np.testing.assert_allclose(restored.points[1]["position"], [0., 0., 100.])


# ---------------------------------------------------------------------------
# load — missing case
# ---------------------------------------------------------------------------

def test_load_missing_case_raises(tmp_mgr):
    with pytest.raises(FileNotFoundError):
        tmp_mgr.load_case("DoesNotExist", SceneGraph.instance())


# ---------------------------------------------------------------------------
# metadata round-trip
# ---------------------------------------------------------------------------

def test_metadata_stage_persisted(tmp_mgr):
    tmp_mgr.save_case("Meta", SceneGraph.instance(), stage=3)
    meta = tmp_mgr.read_metadata("Meta")
    assert meta["stage"] == 3


def test_save_twice_preserves_created(tmp_mgr):
    tmp_mgr.save_case("Twice", SceneGraph.instance())
    created1 = tmp_mgr.read_metadata("Twice")["created"]
    tmp_mgr.save_case("Twice", SceneGraph.instance())
    created2 = tmp_mgr.read_metadata("Twice")["created"]
    assert created1 == created2


# ---------------------------------------------------------------------------
# delete_case
# ---------------------------------------------------------------------------

def test_delete_case_removes_directory(tmp_mgr):
    tmp_mgr.save_case("Del", SceneGraph.instance())
    assert tmp_mgr.case_dir("Del").exists()
    tmp_mgr.delete_case("Del")
    assert not tmp_mgr.case_dir("Del").exists()
    assert "Del" not in tmp_mgr.list_cases()


def test_delete_nonexistent_is_noop(tmp_mgr):
    tmp_mgr.delete_case("Ghost")   # should not raise
