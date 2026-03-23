"""Tests for tracking/transform_store.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import threading

import numpy as np
import pytest

from surgical_nav.tracking.transform_store import TransformStore


def _eye():
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic get / update
# ---------------------------------------------------------------------------

def test_get_missing_returns_none():
    store = TransformStore()
    assert store.get("Unknown") is None


def test_update_then_get():
    store = TransformStore()
    store.update("T", _eye())
    m = store.get("T")
    assert m is not None
    np.testing.assert_array_equal(m, _eye())


def test_get_returns_copy():
    store = TransformStore()
    store.update("T", _eye())
    m1 = store.get("T")
    m1[0, 3] = 999.0
    m2 = store.get("T")
    assert m2[0, 3] != 999.0


def test_update_overwrites():
    store = TransformStore()
    store.update("T", _eye())
    m2 = _eye(); m2[0, 3] = 42.0
    store.update("T", m2)
    assert store.get("T")[0, 3] == pytest.approx(42.0)


def test_wrong_shape_raises():
    store = TransformStore()
    with pytest.raises(ValueError):
        store.update("T", np.eye(3))


# ---------------------------------------------------------------------------
# names / clear
# ---------------------------------------------------------------------------

def test_names_empty():
    store = TransformStore()
    assert store.names() == []


def test_names_after_updates():
    store = TransformStore()
    store.update("A", _eye())
    store.update("B", _eye())
    assert set(store.names()) == {"A", "B"}


def test_clear():
    store = TransformStore()
    store.update("A", _eye())
    store.clear()
    assert store.get("A") is None
    assert store.names() == []


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def test_never_seen_is_stale():
    store = TransformStore()
    assert store.is_stale("X")


def test_fresh_is_not_stale():
    store = TransformStore()
    store.update("T", _eye())
    assert not store.is_stale("T", max_age_s=1.0)


def test_old_entry_is_stale():
    store = TransformStore()
    store.update("T", _eye())
    # Force staleness by inspecting with a very short max_age
    time.sleep(0.05)
    assert store.is_stale("T", max_age_s=0.01)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_updates():
    store = TransformStore()
    errors = []

    def writer(name):
        try:
            for _ in range(200):
                m = np.random.rand(4, 4)
                m[3] = [0, 0, 0, 1]
                store.update(name, m)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(f"T{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.names()) == 5
