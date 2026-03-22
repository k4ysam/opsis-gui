"""Headless smoke tests for MainWindow and LayoutManager.

Runs with the offscreen Qt platform so no display is needed.
"""

import sys
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

# One QApplication for the entire module
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_main_window_creates(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    assert win is not None
    assert win.windowTitle() == "Surgical Navigation"


def test_add_pages_and_switch(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    idx0 = win.add_page(QLabel("Page 0"))
    idx1 = win.add_page(QLabel("Page 1"))
    win.set_page(idx1)
    assert win.current_page() == idx1


def test_stage_actions_initial_state(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    actions = win._stage_actions
    assert actions[0].isEnabled()       # Patients always enabled
    for a in actions[1:]:
        assert not a.isEnabled()        # rest locked initially


def test_mark_stage_complete_unlocks_next(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    win.mark_stage_complete(0)
    assert win._stage_actions[1].isEnabled()   # Planning unlocked


def test_status_lights_set(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    win.set_plus_status(True)
    win.set_plus_status(False)
    win.set_tool_status("Pointer", "SEEN")
    win.set_tool_status("Pointer", "NOT_SEEN")
    win.set_tool_status("HeadFrame", "NEVER_SEEN")
    # No exception = pass


def test_set_case_name(qapp):
    from surgical_nav.app.main_window import MainWindow
    win = MainWindow()
    win.set_case_name("TestCase")
    assert "TestCase" in win._lbl_case.text()


def test_layout_manager_set_single(qapp):
    from surgical_nav.rendering.layout_manager import LayoutManager
    container = QWidget()
    lm = LayoutManager(container)
    v1, v2, v3, v4 = QWidget(), QWidget(), QWidget(), QWidget()
    lm.set_viewers(v1, v2, v3, v4)
    lm.set_layout("single")
    assert lm.current_layout() == "single"


def test_layout_manager_switch(qapp):
    from surgical_nav.rendering.layout_manager import LayoutManager
    container = QWidget()
    lm = LayoutManager(container)
    v1, v2, v3, v4 = QWidget(), QWidget(), QWidget(), QWidget()
    lm.set_viewers(v1, v2, v3, v4)
    lm.set_layout("single")
    lm.set_layout("2up")
    assert lm.current_layout() == "2up"
    lm.set_layout("6up")
    assert lm.current_layout() == "6up"


def test_layout_manager_invalid_raises(qapp):
    from surgical_nav.rendering.layout_manager import LayoutManager
    container = QWidget()
    lm = LayoutManager(container)
    with pytest.raises(ValueError):
        lm.set_layout("bogus")
