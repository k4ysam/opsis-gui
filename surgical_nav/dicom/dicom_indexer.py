"""DICOMIndexer: scans a directory tree and builds a series index.

Uses pydicom to read tags only (no pixel data) for speed.  Returns a list of
SeriesEntry dataclasses that callers can display in a table and then pass to
DICOMLoader for pixel loading.

Usage::

    indexer = DICOMIndexer()
    series = indexer.scan("/path/to/dicom/dir")
    for s in series:
        print(s.series_description, s.modality, len(s.file_paths))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pydicom


@dataclass
class SeriesEntry:
    """Metadata for one DICOM series (no pixel data loaded)."""
    series_uid: str
    study_uid: str
    patient_name: str
    series_description: str
    modality: str
    series_number: int
    file_paths: List[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        desc = self.series_description or f"Series {self.series_number}"
        return f"{desc} [{self.modality}] — {len(self.file_paths)} files"


class DICOMIndexer:
    """Walks a directory tree and indexes all DICOM series found.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with ``(files_scanned: int, total_files: int)`` as files are
        processed.  Useful for updating a progress bar.
    """

    _DICOM_EXTENSIONS = {".dcm", ".ima", ".dicom", ""}   # "" = no extension

    def __init__(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        self._progress_callback = progress_callback

    def scan(self, directory: str) -> List[SeriesEntry]:
        """Recursively scan *directory* and return one SeriesEntry per series.

        Files that cannot be read as DICOM are silently skipped.
        """
        candidates = self._collect_candidates(directory)
        series_map: Dict[str, SeriesEntry] = {}

        for i, path in enumerate(candidates):
            if self._progress_callback:
                self._progress_callback(i + 1, len(candidates))
            entry = self._read_tags(path)
            if entry is None:
                continue
            uid = entry["series_uid"]
            if uid not in series_map:
                series_map[uid] = SeriesEntry(
                    series_uid=uid,
                    study_uid=entry["study_uid"],
                    patient_name=entry["patient_name"],
                    series_description=entry["series_description"],
                    modality=entry["modality"],
                    series_number=entry["series_number"],
                )
            series_map[uid].file_paths.append(path)

        # Sort each series' files by InstanceNumber / filename
        for s in series_map.values():
            s.file_paths.sort(key=self._sort_key)

        return sorted(series_map.values(), key=lambda s: (s.study_uid, s.series_number))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_candidates(self, directory: str) -> List[str]:
        """Return all file paths that might be DICOM."""
        candidates = []
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self._DICOM_EXTENSIONS:
                    candidates.append(os.path.join(root, fname))
        return candidates

    @staticmethod
    def _read_tags(path: str) -> Optional[dict]:
        """Read DICOM tags from *path* without loading pixel data.

        Returns None if the file is not a valid DICOM file.
        """
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            return None

        # Must have a SeriesInstanceUID to be meaningful
        series_uid = getattr(ds, "SeriesInstanceUID", None)
        if series_uid is None:
            return None

        return {
            "series_uid":         str(series_uid),
            "study_uid":          str(getattr(ds, "StudyInstanceUID", "")),
            "patient_name":       str(getattr(ds, "PatientName", "Unknown")),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "modality":           str(getattr(ds, "Modality", "")),
            "series_number":      int(getattr(ds, "SeriesNumber", 0) or 0),
            "instance_number":    int(getattr(ds, "InstanceNumber", 0) or 0),
        }

    @staticmethod
    def _sort_key(path: str):
        """Sort files by InstanceNumber (from tags) then filename as fallback."""
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            return (int(getattr(ds, "InstanceNumber", 0) or 0), path)
        except Exception:
            return (0, path)
