"""Tests for dicom/dicom_indexer.py.

Uses pydicom's built-in test datasets — no external DICOM download needed.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import pydicom.uid

from surgical_nav.dicom.dicom_indexer import DICOMIndexer, SeriesEntry


# ---------------------------------------------------------------------------
# Helpers to create synthetic DICOM files
# ---------------------------------------------------------------------------

def _write_fake_dicom(path: str, series_uid: str, study_uid: str,
                      instance_number: int = 1,
                      modality: str = "CT",
                      series_description: str = "Test Series",
                      series_number: int = 1,
                      patient_name: str = "Test^Patient"):
    """Write a minimal valid DICOM file (no pixel data)."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.PatientName = patient_name
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.Modality = modality
    ds.SeriesDescription = series_description
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number

    pydicom.dcmwrite(path, ds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_empty_directory():
    with tempfile.TemporaryDirectory() as d:
        result = DICOMIndexer().scan(d)
    assert result == []


def test_scan_non_dicom_files_ignored():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "not_dicom.txt"), "w").close()
        open(os.path.join(d, "image.png"), "w").close()
        result = DICOMIndexer().scan(d)
    assert result == []


def test_scan_single_series():
    with tempfile.TemporaryDirectory() as d:
        series_uid = generate_uid()
        study_uid  = generate_uid()
        for i in range(5):
            _write_fake_dicom(
                os.path.join(d, f"slice_{i:03d}.dcm"),
                series_uid, study_uid, instance_number=i + 1
            )
        result = DICOMIndexer().scan(d)

    assert len(result) == 1
    assert result[0].series_uid == series_uid
    assert len(result[0].file_paths) == 5


def test_scan_two_series():
    with tempfile.TemporaryDirectory() as d:
        uid_a = generate_uid()
        uid_b = generate_uid()
        study_uid = generate_uid()
        for i in range(3):
            _write_fake_dicom(os.path.join(d, f"a_{i}.dcm"), uid_a, study_uid,
                              instance_number=i + 1, series_number=1)
        for i in range(4):
            _write_fake_dicom(os.path.join(d, f"b_{i}.dcm"), uid_b, study_uid,
                              instance_number=i + 1, series_number=2)
        result = DICOMIndexer().scan(d)

    assert len(result) == 2
    counts = {s.series_uid: len(s.file_paths) for s in result}
    assert counts[uid_a] == 3
    assert counts[uid_b] == 4


def test_scan_series_entry_fields():
    with tempfile.TemporaryDirectory() as d:
        uid = generate_uid()
        study = generate_uid()
        _write_fake_dicom(
            os.path.join(d, "s.dcm"), uid, study,
            modality="MR", series_description="Brain T1",
            series_number=3, patient_name="Doe^John"
        )
        result = DICOMIndexer().scan(d)

    s = result[0]
    assert s.modality == "MR"
    assert s.series_description == "Brain T1"
    assert s.series_number == 3
    assert "Doe" in s.patient_name


def test_scan_files_sorted_by_instance_number():
    with tempfile.TemporaryDirectory() as d:
        uid = generate_uid()
        study = generate_uid()
        # Write in reverse order
        for i in [5, 3, 1, 4, 2]:
            _write_fake_dicom(
                os.path.join(d, f"slice_{i}.dcm"), uid, study,
                instance_number=i
            )
        result = DICOMIndexer().scan(d)

        # Read back instance numbers while the temp dir still exists
        nums = []
        for p in result[0].file_paths:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            nums.append(int(ds.InstanceNumber))

    assert nums == [1, 2, 3, 4, 5]


def test_scan_subdirectory_recursive():
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "subdir")
        os.makedirs(sub)
        uid = generate_uid()
        study = generate_uid()
        _write_fake_dicom(os.path.join(sub, "s.dcm"), uid, study)
        result = DICOMIndexer().scan(d)

    assert len(result) == 1


def test_progress_callback():
    with tempfile.TemporaryDirectory() as d:
        uid = generate_uid()
        study = generate_uid()
        for i in range(3):
            _write_fake_dicom(os.path.join(d, f"s{i}.dcm"), uid, study,
                              instance_number=i + 1)

        calls = []
        DICOMIndexer(progress_callback=lambda done, total: calls.append((done, total))).scan(d)

    assert len(calls) == 3   # one call per file scanned
    assert calls[-1] == (3, 3)   # final call: done==total


def test_display_name_format():
    s = SeriesEntry(
        series_uid="1.2.3", study_uid="4.5.6",
        patient_name="P", series_description="Axial T2",
        modality="MR", series_number=2,
        file_paths=["a", "b", "c"]
    )
    assert "Axial T2" in s.display_name
    assert "MR" in s.display_name
    assert "3" in s.display_name


def test_display_name_fallback_no_description():
    s = SeriesEntry(
        series_uid="1", study_uid="2",
        patient_name="P", series_description="",
        modality="CT", series_number=5,
        file_paths=[]
    )
    assert "Series 5" in s.display_name
