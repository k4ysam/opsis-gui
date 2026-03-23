"""AppSettings: QSettings-backed user preferences.

All read/write goes through this module so that defaults are centralised
and callers never have to know the raw key strings.

Usage::

    s = AppSettings()
    path = s.plus_server_path          # str | None
    s.plus_server_path = "/usr/bin/PlusServer"
    s.reset_to_defaults()
"""

from __future__ import annotations

import shutil
from typing import Optional

from PySide6.QtCore import QSettings

_ORG  = "OpenNav"
_APP  = "SurgicalNav"


class AppSettings:
    """Thin wrapper around ``QSettings`` with typed getters/setters."""

    # ----- defaults -------------------------------------------------------
    _DEFAULTS: dict = {
        "plus_server_path":    None,
        "dicom_dir":           None,
        "plus_port":           18944,
        "hu_skin_low":        -500,
        "hu_skin_high":        500,
        "hu_target_low":      -100,
        "hu_target_high":      400,
        "render_hz":           20.0,
        "last_case":           None,
    }

    def __init__(self):
        self._qs = QSettings(_ORG, _APP)

    # ------------------------------------------------------------------
    # Plus Server
    # ------------------------------------------------------------------

    @property
    def plus_server_path(self) -> Optional[str]:
        v = self._qs.value("plus_server_path", None)
        return v if v else shutil.which("PlusServer")

    @plus_server_path.setter
    def plus_server_path(self, path: Optional[str]):
        self._qs.setValue("plus_server_path", path or "")

    @property
    def plus_port(self) -> int:
        return int(self._qs.value("plus_port", self._DEFAULTS["plus_port"]))

    @plus_port.setter
    def plus_port(self, port: int):
        self._qs.setValue("plus_port", int(port))

    # ------------------------------------------------------------------
    # DICOM
    # ------------------------------------------------------------------

    @property
    def dicom_dir(self) -> Optional[str]:
        v = self._qs.value("dicom_dir", None)
        return v if v else None

    @dicom_dir.setter
    def dicom_dir(self, path: Optional[str]):
        self._qs.setValue("dicom_dir", path or "")

    # ------------------------------------------------------------------
    # HU thresholds
    # ------------------------------------------------------------------

    @property
    def hu_skin_low(self) -> int:
        return int(self._qs.value("hu_skin_low", self._DEFAULTS["hu_skin_low"]))

    @hu_skin_low.setter
    def hu_skin_low(self, v: int):
        self._qs.setValue("hu_skin_low", int(v))

    @property
    def hu_skin_high(self) -> int:
        return int(self._qs.value("hu_skin_high", self._DEFAULTS["hu_skin_high"]))

    @hu_skin_high.setter
    def hu_skin_high(self, v: int):
        self._qs.setValue("hu_skin_high", int(v))

    @property
    def hu_target_low(self) -> int:
        return int(self._qs.value("hu_target_low", self._DEFAULTS["hu_target_low"]))

    @hu_target_low.setter
    def hu_target_low(self, v: int):
        self._qs.setValue("hu_target_low", int(v))

    @property
    def hu_target_high(self) -> int:
        return int(self._qs.value("hu_target_high", self._DEFAULTS["hu_target_high"]))

    @hu_target_high.setter
    def hu_target_high(self, v: int):
        self._qs.setValue("hu_target_high", int(v))

    # ------------------------------------------------------------------
    # Render rate
    # ------------------------------------------------------------------

    @property
    def render_hz(self) -> float:
        return float(self._qs.value("render_hz", self._DEFAULTS["render_hz"]))

    @render_hz.setter
    def render_hz(self, hz: float):
        self._qs.setValue("render_hz", float(hz))

    # ------------------------------------------------------------------
    # Last case
    # ------------------------------------------------------------------

    @property
    def last_case(self) -> Optional[str]:
        v = self._qs.value("last_case", None)
        return v if v else None

    @last_case.setter
    def last_case(self, name: Optional[str]):
        self._qs.setValue("last_case", name or "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset_to_defaults(self):
        """Clear all stored settings (restores built-in defaults)."""
        self._qs.clear()

    def all_values(self) -> dict:
        """Return all current settings as a plain dict (for debugging)."""
        return {
            "plus_server_path": self.plus_server_path,
            "plus_port":        self.plus_port,
            "dicom_dir":        self.dicom_dir,
            "hu_skin_low":      self.hu_skin_low,
            "hu_skin_high":     self.hu_skin_high,
            "hu_target_low":    self.hu_target_low,
            "hu_target_high":   self.hu_target_high,
            "render_hz":        self.render_hz,
            "last_case":        self.last_case,
        }
