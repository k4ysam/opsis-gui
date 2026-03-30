"""Entry point for the standalone surgical navigation application."""

import sys
import os
import time
import numpy as np

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer, QObject, Slot

from surgical_nav.app.main_window import MainWindow
from surgical_nav.app.settings import AppSettings
from surgical_nav.app.scene_graph import SceneGraph
from surgical_nav.rendering.qt_slice_preview import QtSlicePreview
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


def _debug_log(message: str):
    """Emit a startup log line immediately."""
    print(f"[startup] {message}", flush=True)


def _timing_log(message: str):
    """Emit a timing log line immediately."""
    print(f"[timing] {message}", flush=True)


class _TrackerBridge(QObject):
    """Deliver tracker signals onto the main Qt thread before touching UI/VTK."""

    def __init__(self, window, volume_viewer, registration_page, navigation_page):
        super().__init__()
        self._window = window
        self._volume_viewer = volume_viewer
        self._registration_page = registration_page
        self._navigation_page = navigation_page

    @Slot()
    def on_connected(self):
        self._window.set_plus_status(True)

    @Slot()
    def on_disconnected(self):
        self._window.set_plus_status(False)

    @Slot(str, str)
    def on_tool_status_changed(self, name: str, status: str):
        tool_key = "Pointer" if "Pointer" in name else "HeadFrame"
        self._window.set_tool_status(tool_key, status)
        if "Pointer" in name:
            self._navigation_page.set_pointer_status(status)

    @Slot(str, object)
    def on_transform_received(self, name: str, matrix):
        if name == "PointerToTracker":
            self._volume_viewer.set_pointer_transform(matrix)
        self._registration_page.receive_transform(name, matrix)
        self._navigation_page.on_transform(name, matrix)


def main():
    _debug_log("creating QApplication")
    app = QApplication(sys.argv)
    app.setApplicationName("SurgicalNav")
    app.setOrganizationName("OpenNav")
    app.aboutToQuit.connect(lambda: _debug_log("QApplication.aboutToQuit emitted"))

    _debug_log("creating shared app state")
    settings = AppSettings()
    case_mgr = CaseManager()

    _debug_log("creating MainWindow")
    window = MainWindow()
    window.destroyed.connect(lambda: _debug_log("MainWindow destroyed"))

    # --- Shared rendering widgets (right-hand viewer panel) ---
    _debug_log("creating viewer widgets")
    volume_viewer = VolumeViewer()
    axial    = SliceViewer("axial")
    coronal  = SliceViewer("coronal")
    sagittal = SliceViewer("sagittal")

    _debug_log("building viewer layout")
    preview_viewer = QtSlicePreview()
    window.set_viewer_panel(preview_viewer)

    # Shared state for auto-save
    _current_case:  list = [None]       # mutable cell
    _current_sitk:  list = [None]
    _pending_volume: list = [None]
    _pending_case_name: list = [None]
    _active_vtk_image: list = [None]

    def _initialize_loaded_volume():
        vtk_image = _pending_volume[0]
        case_name = _pending_case_name[0]
        if vtk_image is None or case_name is None:
            return

        window.set_case_name(case_name)
        t0 = time.perf_counter()
        preview_viewer.set_image(_current_sitk[0])
        _timing_log(f"qt preview init after load: {time.perf_counter() - t0:.2f}s")
        _active_vtk_image[0] = vtk_image
        _pending_volume[0] = None
        _pending_case_name[0] = None

    def _auto_save(stage: int):
        name = _current_case[0]
        if name:
            case_mgr.save_case(name, SceneGraph.instance(),
                               sitk_image=_current_sitk[0], stage=stage)
            settings.last_case = name

    # --- Stage 0: Patients ---
    _debug_log("creating workflow pages")
    patients_page = PatientsPage()

    def on_volume_loaded(vtk_image, sitk_image, case_name):
        _pending_volume[0] = vtk_image
        _pending_case_name[0] = case_name
        _current_case[0] = case_name
        _current_sitk[0] = sitk_image
        QTimer.singleShot(0, _initialize_loaded_volume)

    def on_preview_clicked(_plane: str, payload):
        planning_page.handle_preview_click(
            _plane, payload["ijk"], payload["ras"]
        )

    def on_preview_dragged(_plane: str, payload):
        planning_page.handle_preview_drag(
            _plane, payload["ijk"], payload["ras"]
        )

    preview_viewer.slice_clicked.connect(on_preview_clicked)
    preview_viewer.slice_dragged.connect(on_preview_dragged)

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
    planning_page.target_label_updated.connect(preview_viewer.set_target_label)
    planning_page.seed_label_updated.connect(preview_viewer.set_seed_label)
    planning_page.target_preview_updated.connect(preview_viewer.set_target_preview)
    planning_page.interaction_mode_changed.connect(preview_viewer.set_interaction_mode)
    planning_page.trajectory_updated.connect(preview_viewer.set_trajectory_points)
    planning_page.landmarks_updated.connect(preview_viewer.set_landmarks)

    _slice_viewers = [axial, coronal, sagittal]

    # Propagate interaction mode to all slice viewers
    planning_page.interaction_mode_changed.connect(
        lambda mode: [v.set_mode(mode) for v in _slice_viewers]
    )

    # Propagate placed points from any slice viewer to the planning page
    def _on_point_placed(mode: str, x: float, y: float, z: float):
        xyz = np.array([x, y, z])
        if mode in ("entry", "target"):
            planning_page.place_trajectory_point(mode, xyz)
        elif mode == "landmark":
            planning_page.place_landmark(xyz)

    for _v in _slice_viewers:
        _v.point_placed.connect(_on_point_placed)

    # Update markers on all slice viewers when trajectory points change
    planning_page.trajectory_points_updated.connect(
        lambda entry, target: [v.set_trajectory_points(entry, target) for v in _slice_viewers]
    )

    # Update landmark markers on all slice viewers when landmarks change
    planning_page.landmarks_updated.connect(
        lambda lms: [v.set_landmarks(lms) for v in _slice_viewers]
    )

    planning_page.go_back.connect(lambda: window.set_page(0))
    window.add_page(planning_page)   # index 1

    patients_page.stage_complete.connect(
        lambda: (planning_page.on_enter(), window.set_page(1))
    )

    # --- Stage 2: Registration ---
    registration_page = RegistrationPage()
    registration_page.stage_complete.connect(lambda: window.mark_stage_complete(2))
    registration_page.stage_complete.connect(lambda: _auto_save(2))
    registration_page.status_message.connect(window.statusBar().showMessage)
    registration_page.go_back.connect(lambda: window.set_page(1))
    window.add_page(registration_page)   # index 2

    planning_page.stage_complete.connect(
        lambda: (registration_page.on_enter(), window.set_page(2))
    )

    # --- Stage 3: Navigation ---
    navigation_page = NavigationPage([axial, coronal, sagittal], volume_viewer)
    navigation_page.status_message.connect(window.statusBar().showMessage)
    navigation_page.go_back.connect(lambda: window.set_page(2))
    window.add_page(navigation_page)   # index 3

    registration_page.stage_complete.connect(
        lambda: (navigation_page.on_enter(), window.set_page(3))
    )

    # --- Stage 4: Landmark Manager ---
    landmark_page = LandmarkManagerPage()
    landmark_page.status_message.connect(window.statusBar().showMessage)
    landmark_page.go_back.connect(lambda: window.set_page(3))
    window.add_page(landmark_page)   # index 4

    navigation_page.stage_complete.connect(
        lambda: (landmark_page.on_enter(), window.set_page(4))
    )

    # --- Tracking: MockIGTLClient for development/testing ---
    _debug_log("creating tracker")
    tracker = MockIGTLClient(hz=10.0)
    tracker_bridge = _TrackerBridge(
        window, volume_viewer, registration_page, navigation_page
    )

    tracker.connected.connect(tracker_bridge.on_connected)
    tracker.disconnected.connect(tracker_bridge.on_disconnected)
    tracker.tool_status_changed.connect(tracker_bridge.on_tool_status_changed)
    tracker.transform_received.connect(tracker_bridge.on_transform_received)
    _debug_log("mock tracker auto-start disabled")

    _debug_log("showing main window")
    window.set_page(0)
    window.show()
    QTimer.singleShot(0, window.raise_)
    QTimer.singleShot(0, window.activateWindow)
    QTimer.singleShot(
        0,
        lambda: _debug_log(
            f"window visible={window.isVisible()} size={window.size().width()}x{window.size().height()}"
        ),
    )

    _debug_log("entering Qt event loop")
    ret = app.exec()
    _debug_log(f"event loop exited with code {ret}")
    tracker.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
