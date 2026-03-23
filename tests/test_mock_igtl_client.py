"""Tests for tracking/mock_igtl_client.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop, QTimer

from surgical_nav.tracking.mock_igtl_client import MockIGTLClient
from surgical_nav.tracking.transform_store import TransformStore


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_event_loop_ms(qapp, ms: int):
    """Process Qt events for *ms* milliseconds."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_creates_with_defaults(qapp):
    client = MockIGTLClient()
    assert client is not None


def test_store_is_accessible(qapp):
    store = TransformStore()
    client = MockIGTLClient(store=store)
    assert client.store is store


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def test_start_then_stop(qapp):
    client = MockIGTLClient(hz=20.0)
    client.start()
    _run_event_loop_ms(qapp, 150)
    client.stop()
    _run_event_loop_ms(qapp, 50)  # allow thread to finish


def test_connected_signal_emitted(qapp):
    received = []
    client = MockIGTLClient(hz=20.0)
    client.connected.connect(lambda: received.append(True))
    client.start()
    _run_event_loop_ms(qapp, 100)
    client.stop()
    assert received == [True]


def test_disconnected_signal_emitted(qapp):
    received = []
    client = MockIGTLClient(hz=20.0)
    client.disconnected.connect(lambda: received.append(True))
    client.start()
    _run_event_loop_ms(qapp, 100)
    client.stop()
    _run_event_loop_ms(qapp, 100)
    assert received == [True]


# ---------------------------------------------------------------------------
# Transform emission
# ---------------------------------------------------------------------------

def test_pointer_transform_emitted(qapp):
    received = []
    client = MockIGTLClient(hz=20.0)
    client.transform_received.connect(
        lambda name, m: received.append(name) if name == "PointerToTracker" else None
    )
    client.start()
    _run_event_loop_ms(qapp, 200)
    client.stop()
    assert len(received) > 0


def test_headframe_transform_emitted(qapp):
    received = []
    client = MockIGTLClient(hz=20.0)
    client.transform_received.connect(
        lambda name, m: received.append(name) if name == "HeadFrameToTracker" else None
    )
    client.start()
    _run_event_loop_ms(qapp, 200)
    client.stop()
    assert len(received) > 0


def test_transform_matrix_is_4x4(qapp):
    matrices = []
    client = MockIGTLClient(hz=20.0)
    client.transform_received.connect(lambda name, m: matrices.append(m))
    client.start()
    _run_event_loop_ms(qapp, 100)
    client.stop()
    assert all(m.shape == (4, 4) for m in matrices)


def test_store_populated_after_start(qapp):
    store = TransformStore()
    client = MockIGTLClient(store=store, hz=20.0)
    client.start()
    _run_event_loop_ms(qapp, 200)
    client.stop()
    assert store.get("PointerToTracker") is not None
    assert store.get("HeadFrameToTracker") is not None


def test_pointer_traces_circular_path(qapp):
    """Pointer X/Y should vary across frames; Z stays near 0."""
    positions = []
    store = TransformStore()
    client = MockIGTLClient(store=store, hz=20.0, radius_mm=50.0)
    client.transform_received.connect(
        lambda name, m: positions.append(m[:3, 3].copy())
        if name == "PointerToTracker" else None
    )
    client.start()
    _run_event_loop_ms(qapp, 500)
    client.stop()

    assert len(positions) >= 5
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    # Position should vary (circular motion)
    assert max(xs) - min(xs) > 1.0
    assert max(ys) - min(ys) > 1.0
    # Z stays near 0
    zs = [p[2] for p in positions]
    assert max(abs(z) for z in zs) < 1e-6


def test_tool_status_seen_emitted(qapp):
    statuses = {}
    client = MockIGTLClient(hz=20.0)
    client.tool_status_changed.connect(lambda name, s: statuses.update({name: s}))
    client.start()
    _run_event_loop_ms(qapp, 200)
    client.stop()
    assert statuses.get("PointerToTracker") == "SEEN"
    assert statuses.get("HeadFrameToTracker") == "SEEN"


def test_stop_is_idempotent(qapp):
    client = MockIGTLClient(hz=20.0)
    client.start()
    _run_event_loop_ms(qapp, 100)
    client.stop()
    client.stop()  # second call must not raise
