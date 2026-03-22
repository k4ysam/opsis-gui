"""Scene graph: typed node registry and transform chain.

Replaces 3D Slicer's MRML scene.  Nodes are plain dataclasses; the transform
tree is a parent-dict.  Observers fire synchronously on node modification.

Usage::

    sg = SceneGraph.instance()
    node = sg.add_node(TransformNode(node_id="POINTER_CALIBRATION"))
    sg.set_parent("POINTER_CALIBRATION", "PointerToTracker")
    T = sg.get_world_transform("POINTER_CALIBRATION")
    sg.add_observer("node_modified", callback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from surgical_nav.utils.math_utils import identity, compose, invert_transform


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    node_id: str
    name: str = ""
    save_with_scene: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = self.node_id


@dataclass
class TransformNode(SceneNode):
    """Holds a 4×4 rigid-body transform matrix."""
    matrix: np.ndarray = field(default_factory=identity)


@dataclass
class VolumeNode(SceneNode):
    """References a loaded medical image volume."""
    # vtkImageData stored externally; scene_graph holds metadata only
    spacing: tuple = (1.0, 1.0, 1.0)
    origin: tuple = (0.0, 0.0, 0.0)
    dimensions: tuple = (0, 0, 0)
    vtk_image_data: Any = field(default=None, repr=False)   # vtkImageData
    sitk_image: Any = field(default=None, repr=False)        # SimpleITK.Image


@dataclass
class FiducialSetNode(SceneNode):
    """An ordered list of labelled 3-D points (RAS mm)."""
    points: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {"label": str, "position": np.ndarray(3,)}


@dataclass
class ModelNode(SceneNode):
    """A surface mesh."""
    vtk_poly_data: Any = field(default=None, repr=False)     # vtkPolyData
    color: tuple = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    visible: bool = True


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------

class SceneGraph:
    """Singleton scene graph.

    Transform tree is stored as a parent-dict: ``_parents[child_id] = parent_id``.
    ``get_world_transform`` walks the chain from *node_id* to the root, composing
    matrices left-to-right (world = root … parent @ node).
    """

    _instance: Optional["SceneGraph"] = None

    def __init__(self):
        self._nodes: Dict[str, SceneNode] = {}
        self._parents: Dict[str, str] = {}           # child → parent node_id
        self._observers: Dict[str, List[Callable]] = {}

    # -- Singleton --------------------------------------------------------

    @classmethod
    def instance(cls) -> "SceneGraph":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (useful in tests)."""
        cls._instance = None

    # -- Node management --------------------------------------------------

    def add_node(self, node: SceneNode) -> SceneNode:
        """Add or replace a node; fires 'node_added'."""
        self._nodes[node.node_id] = node
        self._fire("node_added", node)
        return node

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str):
        """Remove a node and any parent links referencing it."""
        if node_id in self._nodes:
            del self._nodes[node_id]
        self._parents.pop(node_id, None)
        # Remove references where this node was a parent
        stale = [c for c, p in self._parents.items() if p == node_id]
        for c in stale:
            del self._parents[c]
        self._fire("node_removed", node_id)

    def all_nodes(self) -> List[SceneNode]:
        return list(self._nodes.values())

    def nodes_of_type(self, node_type: type) -> List[SceneNode]:
        return [n for n in self._nodes.values() if isinstance(n, node_type)]

    # -- Transform tree ---------------------------------------------------

    def set_parent(self, child_id: str, parent_id: Optional[str]):
        """Set the parent of a node in the transform tree.

        Pass *parent_id=None* to detach from the tree (make root-level).
        """
        if parent_id is None:
            self._parents.pop(child_id, None)
        else:
            if parent_id not in self._nodes:
                raise KeyError(f"Parent node '{parent_id}' not in scene graph")
            self._parents[child_id] = parent_id

    def get_parent(self, node_id: str) -> Optional[str]:
        return self._parents.get(node_id)

    def get_world_transform(self, node_id: str) -> np.ndarray:
        """Return the world-space 4×4 transform for a node.

        Walks the parent chain from the node up to the root, collecting
        TransformNode matrices, then composes them root-first.

        Non-TransformNode parents in the chain are skipped (they contribute
        the identity).
        """
        chain: List[np.ndarray] = []
        current = node_id
        visited = set()
        while current is not None:
            if current in visited:
                raise RuntimeError(f"Cycle detected in transform tree at '{current}'")
            visited.add(current)
            node = self._nodes.get(current)
            if isinstance(node, TransformNode):
                chain.append(node.matrix)
            current = self._parents.get(current)
        # chain[0] = current node, chain[-1] = root
        # world = root @ ... @ node  → reverse and compose
        chain.reverse()
        return compose(*chain) if chain else identity()

    def update_transform(self, node_id: str, matrix: np.ndarray):
        """Update a TransformNode's matrix and fire 'node_modified'."""
        node = self._nodes.get(node_id)
        if not isinstance(node, TransformNode):
            raise KeyError(f"'{node_id}' is not a TransformNode")
        node.matrix = matrix.copy()
        self._fire("node_modified", node)

    # -- Observers --------------------------------------------------------

    def add_observer(self, event: str, callback: Callable):
        self._observers.setdefault(event, []).append(callback)

    def remove_observer(self, event: str, callback: Callable):
        if event in self._observers:
            try:
                self._observers[event].remove(callback)
            except ValueError:
                pass

    def _fire(self, event: str, data: Any = None):
        for cb in self._observers.get(event, []):
            cb(data)

    # -- Convenience: fiducial helpers ------------------------------------

    def add_fiducial(self, node_id: str, label: str, position: np.ndarray):
        """Append a point to a FiducialSetNode (creates the node if absent)."""
        node = self._nodes.get(node_id)
        if node is None:
            node = self.add_node(FiducialSetNode(node_id=node_id))
        if not isinstance(node, FiducialSetNode):
            raise TypeError(f"'{node_id}' is not a FiducialSetNode")
        node.points.append({"label": label, "position": np.asarray(position, dtype=np.float64)})
        self._fire("node_modified", node)

    def clear_fiducials(self, node_id: str):
        node = self._nodes.get(node_id)
        if isinstance(node, FiducialSetNode):
            node.points.clear()
            self._fire("node_modified", node)
