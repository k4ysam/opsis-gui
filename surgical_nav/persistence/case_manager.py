"""CaseManager: save and load all surgical navigation case data.

Directory layout for a case named ``MyCase``::

    ~/OpenNav/Cases/MyCase/
        metadata.json          – case name, created/modified timestamps, stage
        volume.nii.gz          – loaded CT volume (SimpleITK)
        transforms/
            POINTER_CALIBRATION.npy
            IMAGE_REGISTRATION.npy
            IMAGE_REGISTRATION_REFINEMENT.npy
        landmarks/
            PLANNING_LANDMARKS.json
            TRAJECTORY_POINTS.json

Usage::

    mgr = CaseManager()
    mgr.save_case("MyCase", scene_graph, sitk_image)
    mgr.load_case("MyCase", scene_graph)   # repopulates scene graph
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from surgical_nav.app.scene_graph import (
    SceneGraph, TransformNode, FiducialSetNode, VolumeNode,
)

try:
    import SimpleITK as sitk
    _SITK_AVAILABLE = True
except ImportError:
    _SITK_AVAILABLE = False


_DEFAULT_ROOT = Path.home() / "OpenNav" / "Cases"

# Transform nodes persisted automatically
_TRANSFORM_KEYS = (
    "POINTER_CALIBRATION",
    "IMAGE_REGISTRATION",
    "IMAGE_REGISTRATION_REFINEMENT",
)

# Fiducial sets persisted automatically
_FIDUCIAL_KEYS = (
    "PLANNING_LANDMARKS",
    "TRAJECTORY_POINTS",
)


class CaseManager:
    """Saves and loads a complete navigation case to/from disk.

    Parameters
    ----------
    root_dir:
        Parent directory that contains all case sub-directories.
        Defaults to ``~/OpenNav/Cases``.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self._root = Path(root_dir) if root_dir else _DEFAULT_ROOT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def case_dir(self, case_name: str) -> Path:
        return self._root / case_name

    def list_cases(self) -> List[str]:
        """Return names of all saved cases (sub-directories with metadata.json)."""
        if not self._root.exists():
            return []
        return sorted(
            d.name
            for d in self._root.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )

    def save_case(
        self,
        case_name: str,
        scene_graph: SceneGraph,
        sitk_image=None,          # Optional[sitk.Image]
        stage: int = 0,
    ) -> Path:
        """Persist all scene-graph data for *case_name*.

        Returns the case directory path.
        """
        case_dir = self.case_dir(case_name)
        (case_dir / "transforms").mkdir(parents=True, exist_ok=True)
        (case_dir / "landmarks").mkdir(parents=True, exist_ok=True)

        # --- metadata ---
        meta_path = case_dir / "metadata.json"
        created = None
        if meta_path.exists():
            try:
                created = json.loads(meta_path.read_text())["created"]
            except Exception:
                pass
        meta = {
            "case_name": case_name,
            "created":   created or datetime.now(timezone.utc).isoformat(),
            "modified":  datetime.now(timezone.utc).isoformat(),
            "stage":     stage,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # --- volume ---
        if sitk_image is not None and _SITK_AVAILABLE:
            sitk.WriteImage(sitk_image, str(case_dir / "volume.nii.gz"))

        # --- transforms ---
        for key in _TRANSFORM_KEYS:
            node = scene_graph.get_node(key)
            if isinstance(node, TransformNode):
                np.save(str(case_dir / "transforms" / f"{key}.npy"), node.matrix)

        # --- fiducials ---
        for key in _FIDUCIAL_KEYS:
            node = scene_graph.get_node(key)
            if isinstance(node, FiducialSetNode):
                points = [
                    {
                        "label":    p.get("label", ""),
                        "position": list(np.asarray(p["position"]).tolist()),
                    }
                    for p in node.points
                ]
                (case_dir / "landmarks" / f"{key}.json").write_text(
                    json.dumps(points, indent=2)
                )

        return case_dir

    def load_case(
        self,
        case_name: str,
        scene_graph: SceneGraph,
    ):
        """Restore all persisted nodes into *scene_graph*.

        Returns the metadata dict, or raises ``FileNotFoundError`` if the
        case directory / metadata.json does not exist.
        """
        case_dir = self.case_dir(case_name)
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Case not found: {case_name!r} ({case_dir})")

        meta = json.loads(meta_path.read_text())

        # --- transforms ---
        tf_dir = case_dir / "transforms"
        if tf_dir.exists():
            for key in _TRANSFORM_KEYS:
                npy_path = tf_dir / f"{key}.npy"
                if npy_path.exists():
                    matrix = np.load(str(npy_path))
                    scene_graph.add_node(TransformNode(node_id=key, matrix=matrix))

        # --- fiducials ---
        lm_dir = case_dir / "landmarks"
        if lm_dir.exists():
            for key in _FIDUCIAL_KEYS:
                json_path = lm_dir / f"{key}.json"
                if json_path.exists():
                    raw = json.loads(json_path.read_text())
                    node = FiducialSetNode(node_id=key)
                    for p in raw:
                        node.points.append(
                            {
                                "label":    p.get("label", ""),
                                "position": np.array(p["position"], dtype=np.float64),
                            }
                        )
                    scene_graph.add_node(node)

        return meta

    def load_volume(self, case_name: str):
        """Return the saved SimpleITK image, or None if not present / no sitk."""
        if not _SITK_AVAILABLE:
            return None
        vol_path = self.case_dir(case_name) / "volume.nii.gz"
        if not vol_path.exists():
            return None
        return sitk.ReadImage(str(vol_path))

    def read_metadata(self, case_name: str) -> dict:
        meta_path = self.case_dir(case_name) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(case_name)
        return json.loads(meta_path.read_text())

    def delete_case(self, case_name: str) -> None:
        """Remove a case directory entirely."""
        import shutil
        case_dir = self.case_dir(case_name)
        if case_dir.exists():
            shutil.rmtree(case_dir)
