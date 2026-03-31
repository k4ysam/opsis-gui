"""Tests for ValidationEngine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_tag_file(fixed: np.ndarray, moving: np.ndarray) -> Path:
    from surgical_nav.persistence.tag_file_io import write_tag_file
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "gt.tag"
    write_tag_file(p, fixed, moving)
    return p


def test_validate_perfect_alignment():
    """When captured == ground truth moving points, FRE should be near zero."""
    from surgical_nav.registration.validation_engine import ValidationEngine
    fixed = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    moving = fixed.copy()  # trivial identity transform
    tag_path = _make_tag_file(fixed, moving)
    engine = ValidationEngine()
    result = engine.validate(captured_points=moving, ground_truth_tag=tag_path)
    assert result.success
    assert result.fre_rmse < 0.01


def test_validate_known_translation():
    """Captured points offset by a constant should register with low FRE."""
    from surgical_nav.registration.validation_engine import ValidationEngine
    fixed = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    offset = np.array([5.0, -3.0, 2.0])
    moving = fixed + offset  # purely translated
    tag_path = _make_tag_file(fixed, moving)
    engine = ValidationEngine()
    result = engine.validate(captured_points=moving, ground_truth_tag=tag_path)
    assert result.success
    assert result.fre_rmse < 0.1


def test_validate_too_few_points():
    """Fewer than 3 captured points should return failure."""
    from surgical_nav.registration.validation_engine import ValidationEngine
    fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    moving = fixed.copy()
    tag_path = _make_tag_file(fixed, moving)
    engine = ValidationEngine()
    result = engine.validate(captured_points=moving, ground_truth_tag=tag_path)
    assert not result.success
    assert "3" in result.message


def test_validate_bad_tag_file():
    """Non-existent .tag file should return failure with message."""
    from surgical_nav.registration.validation_engine import ValidationEngine
    engine = ValidationEngine()
    result = engine.validate(
        captured_points=np.zeros((4, 3)),
        ground_truth_tag="/nonexistent/path.tag",
    )
    assert not result.success
    assert "Failed to read" in result.message


def test_validate_held_out():
    """held_out_indices should separate fiducials from targets."""
    from surgical_nav.registration.validation_engine import ValidationEngine
    fixed = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 5.0, 5.0],
    ])
    moving = fixed.copy()
    tag_path = _make_tag_file(fixed, moving)
    engine = ValidationEngine()
    result = engine.validate(
        captured_points=moving,
        ground_truth_tag=tag_path,
        held_out_indices=[4],
    )
    assert result.success
    assert result.n_fiducial_pairs == 4
    assert result.n_target_pairs == 1
