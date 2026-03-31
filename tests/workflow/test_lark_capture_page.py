"""Tests for LarkCapturePage."""
from __future__ import annotations

import numpy as np
import pytest


def test_lark_capture_page_instantiates(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    assert page is not None


def test_capture_disabled_by_default(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    assert not page._btn_capture.isEnabled()


def test_capture_enabled_when_seen(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    page.on_tool_status("PointerToTracker", "SEEN")
    assert page._btn_capture.isEnabled()


def test_on_transform_updates_pending(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    m = np.eye(4)
    m[:3, 3] = [1.0, 2.0, 3.0]
    page.on_transform("PointerToTracker", m)
    assert page._pending_transform is not None
    np.testing.assert_allclose(page._pending_transform[:3, 3], [1.0, 2.0, 3.0])


def test_on_transform_ignores_wrong_name(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    page.on_transform("SomeOtherTool", np.eye(4))
    assert page._pending_transform is None


def test_undo_removes_last(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    m = np.eye(4)
    m[:3, 3] = [10.0, 20.0, 30.0]
    store.update("PointerToTracker", m)
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    page.on_tool_status("PointerToTracker", "SEEN")
    page._do_capture()
    assert len(page._captured) == 1
    page._undo_last()
    assert len(page._captured) == 0
    assert page._table.rowCount() == 0


def test_clear_all(qtbot):
    from surgical_nav.tracking.transform_store import TransformStore
    from surgical_nav.workflow.lark_capture_page import LarkCapturePage
    store = TransformStore()
    m = np.eye(4)
    store.update("PointerToTracker", m)
    page = LarkCapturePage(tracker_store=store)
    qtbot.addWidget(page)
    page.on_tool_status("PointerToTracker", "SEEN")
    page._do_capture()
    page._do_capture()
    assert len(page._captured) == 2
    page._clear_all()
    assert len(page._captured) == 0
    assert page._table.rowCount() == 0
