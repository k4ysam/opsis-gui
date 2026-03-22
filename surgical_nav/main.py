"""Entry point for the standalone surgical navigation application."""

import sys
import os

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt

from surgical_nav.app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SurgicalNav")
    app.setOrganizationName("OpenNav")

    window = MainWindow()

    # Placeholder pages until full pages are implemented
    for text in ("Patients", "Planning", "Registration", "Navigation", "Landmarks"):
        placeholder = QLabel(f"{text} — coming soon", alignment=Qt.AlignmentFlag.AlignCenter)
        window.add_page(placeholder)

    window.set_page(0)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
