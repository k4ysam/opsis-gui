from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
import sys

app = QApplication(sys.argv)
w = QMainWindow()
w.setWindowTitle("Test")
w.setCentralWidget(QLabel("Hello! Qt is working."))
w.resize(400, 300)
w.show()
app.exec()
