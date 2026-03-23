"""Tests for workflow/landmark_manager_page.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import SceneGraph, FiducialSetNode
from surgical_nav.workflow.landmark_manager_page import LandmarkManagerPage


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


def _add_fiducials(node_id: str, n: int = 3):
    node = FiducialSetNode(node_id=node_id)
    for i in range(n):
        node.points.append({
            "label":    f"P{i+1}",
            "position": np.array([float(i), float(i*2), float(i*3)]),
        })
    SceneGraph.instance().add_node(node)
    return node


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_creates(qapp):
    page = LandmarkManagerPage()
    assert page is not None


def test_no_tabs_initially(qapp):
    page = LandmarkManagerPage()
    assert page._tabs.count() == 0


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------

def test_refresh_creates_tabs_per_set(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 3)
    _add_fiducials("TRAJECTORY_POINTS", 2)
    page = LandmarkManagerPage()
    page.refresh()
    assert page._tabs.count() == 2


def test_refresh_tab_names_match_node_ids(qapp):
    _add_fiducials("PLANNING_LANDMARKS")
    page = LandmarkManagerPage()
    page.refresh()
    assert page._tabs.tabText(0) == "PLANNING_LANDMARKS"


def test_refresh_table_row_count(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 4)
    page = LandmarkManagerPage()
    page.refresh()
    table = page._tabs.widget(0)
    assert table.rowCount() == 4


def test_refresh_empty_scene_no_tabs(qapp):
    page = LandmarkManagerPage()
    page.refresh()
    assert page._tabs.count() == 0


def test_on_enter_calls_refresh(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 2)
    page = LandmarkManagerPage()
    page.on_enter()
    assert page._tabs.count() == 1


# ---------------------------------------------------------------------------
# delete_selected_row
# ---------------------------------------------------------------------------

def test_delete_selected_row_removes_row(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 3)
    page = LandmarkManagerPage()
    page.refresh()
    table = page._tabs.widget(0)
    table.selectRow(0)
    page.delete_selected_row()
    assert table.rowCount() == 2


def test_delete_selected_row_updates_scene_graph(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 3)
    page = LandmarkManagerPage()
    page.refresh()
    table = page._tabs.widget(0)
    table.selectRow(1)
    page.delete_selected_row()
    node = SceneGraph.instance().get_node("PLANNING_LANDMARKS")
    assert len(node.points) == 2


def test_delete_with_no_selection_is_noop(qapp):
    _add_fiducials("PLANNING_LANDMARKS", 2)
    page = LandmarkManagerPage()
    page.refresh()
    page.delete_selected_row()   # no row selected → no crash
    assert page._tabs.widget(0).rowCount() == 2


# ---------------------------------------------------------------------------
# export_current
# ---------------------------------------------------------------------------

def test_export_creates_csv(qapp, tmp_path):
    _add_fiducials("PLANNING_LANDMARKS", 3)
    page = LandmarkManagerPage()
    page.refresh()
    out = tmp_path / "out.csv"
    result = page.export_current(str(out))
    assert result == str(out)
    assert out.exists()


def test_export_csv_has_header(qapp, tmp_path):
    _add_fiducials("PLANNING_LANDMARKS", 2)
    page = LandmarkManagerPage()
    page.refresh()
    out = tmp_path / "lm.csv"
    page.export_current(str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "label,x_mm,y_mm,z_mm"


def test_export_csv_row_count(qapp, tmp_path):
    _add_fiducials("PLANNING_LANDMARKS", 4)
    page = LandmarkManagerPage()
    page.refresh()
    out = tmp_path / "lm.csv"
    page.export_current(str(out))
    lines = out.read_text().splitlines()
    assert len(lines) == 5   # header + 4 data rows


def test_export_csv_values(qapp, tmp_path):
    node = FiducialSetNode(node_id="PLANNING_LANDMARKS")
    node.points.append({"label": "LM1", "position": np.array([1.0, 2.0, 3.0])})
    SceneGraph.instance().add_node(node)
    page = LandmarkManagerPage()
    page.refresh()
    out = tmp_path / "lm.csv"
    page.export_current(str(out))
    lines = out.read_text().splitlines()
    assert "LM1" in lines[1]
    assert "1.0" in lines[1]


def test_export_no_tabs_returns_none(qapp, tmp_path):
    page = LandmarkManagerPage()
    result = page.export_current(str(tmp_path / "out.csv"))
    assert result is None


# ---------------------------------------------------------------------------
# current_node_id
# ---------------------------------------------------------------------------

def test_current_node_id_no_tabs(qapp):
    page = LandmarkManagerPage()
    assert page.current_node_id() is None


def test_current_node_id_returns_tab_text(qapp):
    _add_fiducials("TRAJECTORY_POINTS", 2)
    page = LandmarkManagerPage()
    page.refresh()
    assert page.current_node_id() == "TRAJECTORY_POINTS"
