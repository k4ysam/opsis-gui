"""Headless tests for workflow/patients_page.py and workflow/base_page.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PySide6.QtWidgets import QApplication

from surgical_nav.app.scene_graph import SceneGraph

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)

@pytest.fixture(autouse=True)
def fresh_graph():
    SceneGraph.reset()
    yield
    SceneGraph.reset()


# -- WorkflowPage base class --------------------------------------------------

def test_base_page_creates(qapp):
    from surgical_nav.workflow.base_page import WorkflowPage
    # WorkflowPage is abstract only by convention; instantiate directly
    page = WorkflowPage("Test Stage")
    assert page._title == "Test Stage"


def test_base_page_has_scene_graph(qapp):
    from surgical_nav.workflow.base_page import WorkflowPage
    page = WorkflowPage("Test")
    assert page.scene_graph is SceneGraph.instance()


def test_base_page_stage_complete_signal(qapp):
    from surgical_nav.workflow.base_page import WorkflowPage
    page = WorkflowPage("Test")
    fired = []
    page.stage_complete.connect(lambda: fired.append(True))
    page.emit_complete()
    assert fired == [True]


def test_base_page_status_message_signal(qapp):
    from surgical_nav.workflow.base_page import WorkflowPage
    page = WorkflowPage("Test")
    messages = []
    page.status_message.connect(messages.append)
    page.emit_complete()
    assert any("Test" in m for m in messages)


# -- PatientsPage -------------------------------------------------------------

def test_patients_page_creates(qapp):
    from surgical_nav.workflow.patients_page import PatientsPage
    page = PatientsPage()
    assert page is not None


def test_load_btn_disabled_initially(qapp):
    from surgical_nav.workflow.patients_page import PatientsPage
    page = PatientsPage()
    assert not page._load_btn.isEnabled()


def test_load_btn_requires_series_and_name(qapp):
    from surgical_nav.workflow.patients_page import PatientsPage
    from surgical_nav.dicom.dicom_indexer import SeriesEntry
    page = PatientsPage()

    # Set case name only → still disabled
    page._case_edit.setText("MyCase")
    assert not page._load_btn.isEnabled()

    # Also set a selected series → enabled
    page._selected_series = SeriesEntry(
        series_uid="1", study_uid="2", patient_name="P",
        series_description="S", modality="CT", series_number=1,
        file_paths=["a.dcm"],
    )
    page._update_load_btn()
    assert page._load_btn.isEnabled()


def test_populate_table(qapp):
    from surgical_nav.workflow.patients_page import PatientsPage
    from surgical_nav.dicom.dicom_indexer import SeriesEntry
    page = PatientsPage()
    series = [
        SeriesEntry("u1", "s1", "Alice", "Axial T1", "MR", 1, ["a.dcm"]*10),
        SeriesEntry("u2", "s1", "Alice", "Sagittal", "MR", 2, ["b.dcm"]*5),
    ]
    page._populate_table(series)
    assert page._table.rowCount() == 2
    assert page._table.item(0, 1).text() == "Axial T1"
    assert page._table.item(1, 3).text() == "5"


def test_on_load_finished_pushes_to_scene_graph(qapp):
    """Simulate a successful load and verify VolumeNode is in the SceneGraph."""
    import vtkmodules.all as vtk
    import SimpleITK as sitk
    from surgical_nav.workflow.patients_page import PatientsPage
    from surgical_nav.app.scene_graph import VolumeNode

    page = PatientsPage()
    completed = []
    page.stage_complete.connect(lambda: completed.append(True))

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(5, 5, 3)
    vtk_img.AllocateScalars(vtk.VTK_FLOAT, 1)
    sitk_img = sitk.Image(5, 5, 3, sitk.sitkFloat32)

    page._on_load_finished(vtk_img, sitk_img, "TestCase")

    node = SceneGraph.instance().get_node("ACTIVE_VOLUME")
    assert isinstance(node, VolumeNode)
    assert node.name == "TestCase"
    assert completed == [True]
