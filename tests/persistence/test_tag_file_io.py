"""Tests for TagFileIO."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


FIXTURE_TAG = Path(__file__).parent.parent / "fixtures" / "sample_ground_truth.tag"


def test_read_tag_file_shape():
    from surgical_nav.persistence.tag_file_io import read_tag_file
    fixed, moving = read_tag_file(FIXTURE_TAG)
    assert fixed.shape == (5, 3)
    assert moving.shape == (5, 3)


def test_read_tag_file_values():
    from surgical_nav.persistence.tag_file_io import read_tag_file
    fixed, moving = read_tag_file(FIXTURE_TAG)
    np.testing.assert_allclose(fixed[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(moving[0], [11.0, 19.0, 31.0])


def test_write_read_roundtrip():
    from surgical_nav.persistence.tag_file_io import read_tag_file, write_tag_file
    fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    moving = fixed + 0.5
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.tag"
        write_tag_file(p, fixed, moving)
        f2, m2 = read_tag_file(p)
    np.testing.assert_allclose(f2, fixed, atol=1e-5)
    np.testing.assert_allclose(m2, moving, atol=1e-5)


def test_read_tag_file_skips_comments():
    from surgical_nav.persistence.tag_file_io import read_tag_file
    content = "# comment\n1.0 2.0 3.0 4.0 5.0 6.0\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.tag"
        p.write_text(content)
        fixed, moving = read_tag_file(p)
    assert fixed.shape == (1, 3)


def test_read_tag_file_bad_line():
    from surgical_nav.persistence.tag_file_io import read_tag_file
    content = "1.0 2.0 3.0\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.tag"
        p.write_text(content)
        with pytest.raises(ValueError, match="expected 6 floats"):
            read_tag_file(p)


def test_write_read_xfm_roundtrip():
    from surgical_nav.persistence.tag_file_io import read_xfm_file, write_xfm_file
    R = np.eye(3)
    T = np.array([1.0, 2.0, 3.0])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.xfm"
        write_xfm_file(p, R, T)
        R2, T2 = read_xfm_file(p)
    np.testing.assert_allclose(R2, R, atol=1e-5)
    np.testing.assert_allclose(T2, T, atol=1e-5)
