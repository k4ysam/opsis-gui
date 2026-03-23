"""Tests for app/settings.py."""

import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings

from surgical_nav.app.settings import AppSettings, _ORG, _APP


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def clean_settings(qapp):
    """Wipe QSettings before and after each test."""
    QSettings(_ORG, _APP).clear()
    yield
    QSettings(_ORG, _APP).clear()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_plus_port(qapp):
    assert AppSettings().plus_port == 18944


def test_default_hu_skin_low(qapp):
    assert AppSettings().hu_skin_low == -500


def test_default_hu_skin_high(qapp):
    assert AppSettings().hu_skin_high == 500


def test_default_hu_target_low(qapp):
    assert AppSettings().hu_target_low == -100


def test_default_hu_target_high(qapp):
    assert AppSettings().hu_target_high == 400


def test_default_render_hz(qapp):
    assert AppSettings().render_hz == 20.0


def test_default_last_case_none(qapp):
    assert AppSettings().last_case is None


def test_default_dicom_dir_none(qapp):
    assert AppSettings().dicom_dir is None


# ---------------------------------------------------------------------------
# Round-trip read / write
# ---------------------------------------------------------------------------

def test_plus_port_roundtrip(qapp):
    s = AppSettings()
    s.plus_port = 12345
    assert AppSettings().plus_port == 12345


def test_hu_skin_low_roundtrip(qapp):
    s = AppSettings()
    s.hu_skin_low = -200
    assert AppSettings().hu_skin_low == -200


def test_hu_skin_high_roundtrip(qapp):
    s = AppSettings()
    s.hu_skin_high = 800
    assert AppSettings().hu_skin_high == 800


def test_render_hz_roundtrip(qapp):
    s = AppSettings()
    s.render_hz = 30.0
    assert AppSettings().render_hz == 30.0


def test_last_case_roundtrip(qapp):
    s = AppSettings()
    s.last_case = "MyCase"
    assert AppSettings().last_case == "MyCase"


def test_dicom_dir_roundtrip(qapp):
    s = AppSettings()
    s.dicom_dir = "/data/dicom"
    assert AppSettings().dicom_dir == "/data/dicom"


def test_plus_server_path_roundtrip(qapp):
    s = AppSettings()
    s.plus_server_path = "/usr/bin/PlusServer"
    assert AppSettings().plus_server_path == "/usr/bin/PlusServer"


# ---------------------------------------------------------------------------
# reset_to_defaults
# ---------------------------------------------------------------------------

def test_reset_clears_all(qapp):
    s = AppSettings()
    s.plus_port = 99999
    s.last_case = "Foo"
    s.reset_to_defaults()
    s2 = AppSettings()
    assert s2.plus_port == 18944
    assert s2.last_case is None


# ---------------------------------------------------------------------------
# all_values
# ---------------------------------------------------------------------------

def test_all_values_returns_dict(qapp):
    vals = AppSettings().all_values()
    assert isinstance(vals, dict)
    assert "plus_port" in vals
    assert "render_hz" in vals


def test_all_values_reflects_changes(qapp):
    s = AppSettings()
    s.render_hz = 60.0
    assert AppSettings().all_values()["render_hz"] == 60.0
