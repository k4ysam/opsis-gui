"""VolumeViewer: 3-D volume renderer + surface actor manager.

Wraps a VTKWidget and adds:
  - GPU ray-cast volume rendering (vtkSmartVolumeMapper) with a CT bone preset
  - Surface mesh actors (ModelNodes) overlaid on the volume
  - A needle/pointer actor updated each frame from a 4×4 transform
  - Camera reset helpers

Usage::

    viewer = VolumeViewer(parent)
    viewer.set_volume(vtk_image_data)
    viewer.add_model(model_node)
    viewer.set_pointer_transform(matrix_4x4)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import vtkmodules.all as vtk
from PySide6.QtWidgets import QWidget, QVBoxLayout

from surgical_nav.rendering.vtk_widget import VTKWidget
from surgical_nav.app.scene_graph import ModelNode


class VolumeViewer(QWidget):
    """3-D volume renderer with surface overlay and pointer actor."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._vtk_widget = VTKWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)

        renderer = self._vtk_widget.get_renderer()
        renderer.SetBackground(0.05, 0.05, 0.05)

        # Volume rendering pipeline (actor not added until data is set)
        self._volume_mapper = vtk.vtkSmartVolumeMapper()
        self._volume_property = self._make_ct_bone_property()
        self._volume_actor = vtk.vtkVolume()
        self._volume_actor.SetMapper(self._volume_mapper)
        self._volume_actor.SetProperty(self._volume_property)
        self._volume_loaded = False

        # Pointer / needle actor (cylinder by default)
        self._pointer_actor = self._make_pointer_actor()
        self._pointer_transform = vtk.vtkTransform()
        self._pointer_actor.SetUserTransform(self._pointer_transform)
        self._pointer_actor.VisibilityOff()
        renderer.AddActor(self._pointer_actor)

        # Model actors keyed by node_id
        self._model_actors: Dict[str, vtk.vtkActor] = {}

    # ------------------------------------------------------------------
    # Public API — volume
    # ------------------------------------------------------------------

    def set_volume(self, vtk_image_data: vtk.vtkImageData):
        """Load a vtkImageData into the volume mapper and reset camera."""
        self._volume_mapper.SetInputData(vtk_image_data)
        self._vtk_widget.reset_camera()

    def clear_volume(self):
        self._volume_mapper.RemoveAllInputs()
        self._vtk_widget.render()

    # ------------------------------------------------------------------
    # Public API — surface models
    # ------------------------------------------------------------------

    def add_model(self, model_node: ModelNode):
        """Add or update a surface mesh actor for *model_node*."""
        if model_node.vtk_poly_data is None:
            return
        if model_node.node_id in self._model_actors:
            self.remove_model(model_node.node_id)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(model_node.vtk_poly_data)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*model_node.color)
        actor.GetProperty().SetOpacity(model_node.opacity)
        actor.SetVisibility(model_node.visible)

        self._vtk_widget.get_renderer().AddActor(actor)
        self._model_actors[model_node.node_id] = actor
        self._vtk_widget.render()

    def remove_model(self, node_id: str):
        actor = self._model_actors.pop(node_id, None)
        if actor:
            self._vtk_widget.get_renderer().RemoveActor(actor)
            self._vtk_widget.render()

    def update_model_visibility(self, node_id: str, visible: bool):
        if node_id in self._model_actors:
            self._model_actors[node_id].SetVisibility(visible)
            self._vtk_widget.render()

    def add_surface(self, poly_data: vtk.vtkPolyData,
                    color=(0.9, 0.75, 0.65), opacity: float = 0.6):
        """Add an anonymous surface actor (used by PlanningPage signals)."""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._vtk_widget.get_renderer().AddActor(actor)
        self._vtk_widget.render()

    # ------------------------------------------------------------------
    # Public API — pointer / needle
    # ------------------------------------------------------------------

    def set_pointer_transform(self, matrix_4x4: np.ndarray):
        """Update the needle actor position/orientation from a 4×4 numpy matrix."""
        vtk_m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_m.SetElement(r, c, float(matrix_4x4[r, c]))
        self._pointer_transform.SetMatrix(vtk_m)
        self._pointer_transform.Modified()
        self._pointer_actor.VisibilityOn()
        self._vtk_widget.render()

    def set_pointer_status(self, status: str):
        """Color the needle by tracking status: SEEN/NOT_SEEN/NEVER_SEEN."""
        colors = {
            "SEEN":       (0.2, 0.9, 0.2),
            "NOT_SEEN":   (0.9, 0.7, 0.0),
            "NEVER_SEEN": (0.5, 0.5, 0.5),
        }
        color = colors.get(status, (0.5, 0.5, 0.5))
        self._pointer_actor.GetProperty().SetColor(*color)
        self._vtk_widget.render()

    def hide_pointer(self):
        self._pointer_actor.VisibilityOff()
        self._vtk_widget.render()

    def set_trajectory(self, entry: np.ndarray, target: np.ndarray):
        """Draw (or update) a trajectory line from *entry* to *target* in 3-D."""
        entry  = np.asarray(entry,  dtype=float)
        target = np.asarray(target, dtype=float)

        if not hasattr(self, "_traj_actor"):
            self._traj_source = vtk.vtkLineSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self._traj_source.GetOutputPort())
            self._traj_actor = vtk.vtkActor()
            self._traj_actor.SetMapper(mapper)
            self._traj_actor.GetProperty().SetColor(1.0, 0.8, 0.0)   # yellow
            self._traj_actor.GetProperty().SetLineWidth(2.0)
            self._vtk_widget.get_renderer().AddActor(self._traj_actor)

        self._traj_source.SetPoint1(*entry)
        self._traj_source.SetPoint2(*target)
        self._traj_source.Update()
        self._traj_actor.VisibilityOn()
        self._vtk_widget.render()

    # ------------------------------------------------------------------
    # Public API — camera
    # ------------------------------------------------------------------

    def reset_camera(self):
        self._vtk_widget.reset_camera()

    def render(self):
        self._vtk_widget.render()

    def initialize(self):
        self._vtk_widget.initialize()

    def get_renderer(self) -> vtk.vtkRenderer:
        return self._vtk_widget.get_renderer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ct_bone_property() -> vtk.vtkVolumeProperty:
        """CT bone window-level transfer functions."""
        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(-1000, 0.00, 0.00, 0.00)  # air → black
        color_tf.AddRGBPoint(  -50, 0.20, 0.10, 0.05)  # soft tissue → dark brown
        color_tf.AddRGBPoint(  200, 0.80, 0.70, 0.60)  # bone → beige
        color_tf.AddRGBPoint( 1000, 1.00, 1.00, 0.90)  # dense bone → white

        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(-1000, 0.00)
        opacity_tf.AddPoint(  -50, 0.00)
        opacity_tf.AddPoint(  150, 0.15)
        opacity_tf.AddPoint(  500, 0.85)
        opacity_tf.AddPoint( 1000, 1.00)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        return prop

    @staticmethod
    def _make_pointer_actor() -> vtk.vtkActor:
        """Create a thin cylinder representing the tracked needle (100 mm long)."""
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(1.0)     # mm
        cylinder.SetHeight(100.0)   # mm
        cylinder.SetResolution(12)
        cylinder.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        return actor

    def showEvent(self, event):
        super().showEvent(event)
        self._vtk_widget.initialize()

    def closeEvent(self, event):
        super().closeEvent(event)
