"""Entry point for the standalone surgical navigation application."""

import sys
import os

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtWidgets import QApplication, QLabel, QWidget
from PySide6.QtCore import Qt

from surgical_nav.app.main_window import MainWindow
from surgical_nav.app.settings import AppSettings
from surgical_nav.app.scene_graph import SceneGraph
from surgical_nav.rendering.volume_viewer import VolumeViewer
from surgical_nav.rendering.slice_viewer import SliceViewer
from surgical_nav.rendering.layout_manager import LayoutManager
from surgical_nav.workflow.patients_page import PatientsPage
from surgical_nav.workflow.planning_page import PlanningPage
from surgical_nav.workflow.registration_page import RegistrationPage
from surgical_nav.workflow.navigation_page import NavigationPage
from surgical_nav.workflow.landmark_manager_page import LandmarkManagerPage
from surgical_nav.persistence.case_manager import CaseManager
from surgical_nav.tracking.mock_igtl_client import MockIGTLClient


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SurgicalNav")
    app.setOrganizationName("OpenNav")

    settings = AppSettings()
    case_mgr = CaseManager()

    window = MainWindow()

    # --- Shared rendering widgets (right-hand viewer panel) ---
    volume_viewer = VolumeViewer()
    axial    = SliceViewer("axial")
    coronal  = SliceViewer("coronal")
    sagittal = SliceViewer("sagittal")

    viewer_container = QWidget()
    lm = LayoutManager(viewer_container)
    lm.set_viewers(volume_viewer, axial, coronal, sagittal)
    lm.set_layout("6up")
    window.set_viewer_panel(viewer_container)

    # Shared state for auto-save
    _current_case:  list = [None]       # mutable cell
    _current_sitk:  list = [None]

    def _auto_save(stage: int):
        name = _current_case[0]
        if name:
            case_mgr.save_case(name, SceneGraph.instance(),
                               sitk_image=_current_sitk[0], stage=stage)
            settings.last_case = name

    # --- Stage 0: Patients ---
    patients_page = PatientsPage()

    def on_volume_loaded(vtk_image, sitk_image, case_name):
        if vtk_image is not None:
            volume_viewer.set_volume(vtk_image)
            axial.set_volume(vtk_image)
            coronal.set_volume(vtk_image)
            sagittal.set_volume(vtk_image)
        window.set_case_name(case_name)
        _current_case[0] = case_name
        _current_sitk[0] = sitk_image

    patients_page.volume_loaded.connect(on_volume_loaded)
    patients_page.stage_complete.connect(lambda: window.mark_stage_complete(0))
    patients_page.stage_complete.connect(lambda: _auto_save(0))
    patients_page.status_message.connect(window.statusBar().showMessage)

    window.add_page(patients_page)   # index 0

    # --- Stage 1: Planning ---
    planning_page = PlanningPage()
    planning_page.stage_complete.connect(lambda: window.mark_stage_complete(1))
    planning_page.stage_complete.connect(lambda: _auto_save(1))
    planning_page.status_message.connect(window.statusBar().showMessage)
    planning_page.skin_mesh_ready.connect(volume_viewer.add_surface)
    planning_page.target_mesh_ready.connect(volume_viewer.add_surface)

    window.add_page(planning_page)   # index 1

    patients_page.stage_complete.connect(
        lambda: (planning_page.on_enter(), window.set_page(1))
    )

    # --- Stage 2: Registration ---
    registration_page = RegistrationPage()
    registration_page.stage_complete.connect(lambda: window.mark_stage_complete(2))
    registration_page.stage_complete.connect(lambda: _auto_save(2))
    registration_page.status_message.connect(window.statusBar().showMessage)
    window.add_page(registration_page)   # index 2

    planning_page.stage_complete.connect(
        lambda: (registration_page.on_enter(), window.set_page(2))
    )

    # --- Stage 3: Navigation ---
    navigation_page = NavigationPage([axial, coronal, sagittal], volume_viewer)
    navigation_page.status_message.connect(window.statusBar().showMessage)
    window.add_page(navigation_page)   # index 3

    registration_page.stage_complete.connect(
        lambda: (navigation_page.on_enter(), window.set_page(3))
    )

    # --- Stage 4: Landmark Manager ---
    landmark_page = LandmarkManagerPage()
    landmark_page.status_message.connect(window.statusBar().showMessage)
    window.add_page(landmark_page)   # index 4

    navigation_page.stage_complete.connect(
        lambda: (landmark_page.on_enter(), window.set_page(4))
    )

    # --- Tracking: MockIGTLClient for development/testing ---
    tracker = MockIGTLClient(hz=10.0)

    def on_tool_status(name: str, status: str):
        tool_key = "Pointer" if "Pointer" in name else "HeadFrame"
        window.set_tool_status(tool_key, status)

    tracker.connected.connect(lambda: window.set_plus_status(True))
    tracker.disconnected.connect(lambda: window.set_plus_status(False))
    tracker.tool_status_changed.connect(on_tool_status)
    tracker.transform_received.connect(
        lambda name, m: volume_viewer.set_pointer_transform(m)
        if name == "PointerToTracker" else None
    )
    tracker.transform_received.connect(registration_page.receive_transform)
    tracker.transform_received.connect(navigation_page.on_transform)
    tracker.tool_status_changed.connect(
        lambda name, status: navigation_page.set_pointer_status(status)
        if "Pointer" in name else None
    )
    tracker.start()

    window.set_page(0)
    window.show()

    ret = app.exec()
    tracker.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
