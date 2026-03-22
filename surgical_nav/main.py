"""Entry point for the standalone surgical navigation application."""

import sys
import os

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt

from surgical_nav.app.main_window import MainWindow
from surgical_nav.rendering.volume_viewer import VolumeViewer
from surgical_nav.rendering.slice_viewer import SliceViewer
from surgical_nav.rendering.layout_manager import LayoutManager
from surgical_nav.workflow.patients_page import PatientsPage


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SurgicalNav")
    app.setOrganizationName("OpenNav")

    window = MainWindow()

    # --- Shared rendering widgets ---
    volume_viewer = VolumeViewer()
    axial    = SliceViewer("axial")
    coronal  = SliceViewer("coronal")
    sagittal = SliceViewer("sagittal")

    # Attach layout manager to main window's central area
    # (The layout manager lives inside the Patients/Planning/Navigation pages;
    #  for now we embed a 6-up view directly in a container page.)
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    viewer_container = QWidget()
    lm = LayoutManager(viewer_container)
    lm.set_viewers(volume_viewer, axial, coronal, sagittal)
    lm.set_layout("6up")

    # --- Stage 0: Patients ---
    patients_page = PatientsPage()

    def on_volume_loaded(vtk_image, sitk_image, case_name):
        volume_viewer.set_volume(vtk_image)
        axial.set_volume(vtk_image)
        coronal.set_volume(vtk_image)
        sagittal.set_volume(vtk_image)
        window.set_case_name(case_name)

    patients_page.volume_loaded.connect(on_volume_loaded)
    patients_page.stage_complete.connect(lambda: window.mark_stage_complete(0))
    patients_page.status_message.connect(window.statusBar().showMessage)

    window.add_page(patients_page)   # index 0

    # --- Stages 1–4: placeholder until implemented ---
    for text in ("Planning", "Registration", "Navigation", "Landmarks"):
        placeholder = QLabel(
            f"{text} — coming in a future phase",
            alignment=Qt.AlignmentFlag.AlignCenter
        )
        placeholder.setStyleSheet("font-size: 18px; color: #888;")
        window.add_page(placeholder)

    window.set_page(0)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
