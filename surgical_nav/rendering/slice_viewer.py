"""SliceViewer: single-plane MPR view (axial / coronal / sagittal).

Falls back to a plain placeholder label when VTK is unavailable
(set SURGICAL_NAV_NO_VTK=1 to force the stub path).
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

try:
    import vtkmodules.all as vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    _VTK = True
except ImportError:
    _VTK = False

# Plane definitions: (normal, view-up, label)
_PLANE_CONFIGS = {
    "axial":    {"normal": (0, 0, 1), "view_up": (0, 1, 0), "label": "Axial"},
    "coronal":  {"normal": (0, 1, 0), "view_up": (0, 0, 1), "label": "Coronal"},
    "sagittal": {"normal": (1, 0, 0), "view_up": (0, 0, 1), "label": "Sagittal"},
}


class SliceViewer(QWidget):
    """2-D slice viewer for one anatomical plane.

    Falls back to a plain placeholder when VTK is unavailable.
    """

    def __init__(self, plane: str = "axial", parent: Optional[QWidget] = None):
        super().__init__(parent)
        if plane not in _PLANE_CONFIGS:
            raise ValueError(f"plane must be one of {list(_PLANE_CONFIGS)}")

        cfg = _PLANE_CONFIGS[plane]
        self._plane = plane
        self._normal: Tuple[float, float, float] = cfg["normal"]
        self._view_up: Tuple[float, float, float] = cfg["view_up"]
        self._initialized = False
        self._vtk_image = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if _VTK:
            self._reslice = vtk.vtkImageReslice()
            self._reslice.SetOutputDimensionality(2)
            self._reslice.SetInterpolationModeToLinear()
            self._reslice.SetBackgroundLevel(-1000)

            # Blank placeholder so the pipeline has valid input before set_volume()
            _blank = vtk.vtkImageData()
            _blank.SetDimensions(2, 2, 1)
            _blank.AllocateScalars(vtk.VTK_SHORT, 1)
            self._reslice.SetInputData(_blank)

            self._lut = vtk.vtkWindowLevelLookupTable()
            self._lut.SetWindow(400)
            self._lut.SetLevel(40)
            self._lut.Build()

            self._color_map = vtk.vtkImageMapToColors()
            self._color_map.SetLookupTable(self._lut)
            self._color_map.SetInputConnection(self._reslice.GetOutputPort())

            self._image_actor = vtk.vtkImageActor()
            self._image_actor.GetMapper().SetInputConnection(self._color_map.GetOutputPort())

            self._renderer = vtk.vtkRenderer()
            self._renderer.SetBackground(0.0, 0.0, 0.0)
            self._renderer.AddActor(self._image_actor)

            self._crosshair_h = self._make_line_actor(color=(1, 1, 0))
            self._crosshair_v = self._make_line_actor(color=(1, 1, 0))
            self._renderer.AddActor(self._crosshair_h)
            self._renderer.AddActor(self._crosshair_v)

            self._corner_text = vtk.vtkCornerAnnotation()
            self._corner_text.SetText(2, cfg["label"])
            self._corner_text.GetTextProperty().SetColor(1, 1, 0)
            self._corner_text.GetTextProperty().SetFontSize(14)
            self._renderer.AddActor(self._corner_text)

            self._interactor = QVTKRenderWindowInteractor(self)
            self._interactor.GetRenderWindow().AddRenderer(self._renderer)
            layout.addWidget(self._interactor)
        else:
            self._renderer = None
            self._interactor = None
            lbl = QLabel(f"{cfg['label']}\n(VTK unavailable)", self)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #555; background: #000; font-size: 13px;")
            layout.addWidget(lbl)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self):
        if not _VTK or self._initialized:
            return
        self._interactor.Initialize()
        self._initialized = True
        cam = self._renderer.GetActiveCamera()
        cam.SetViewUp(*self._view_up)
        cam.ParallelProjectionOn()

    def showEvent(self, event):
        super().showEvent(event)
        self.initialize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_volume(self, vtk_image_data):
        if not _VTK:
            return
        self._vtk_image = vtk_image_data
        self._reslice.SetInputData(vtk_image_data)
        bounds = vtk_image_data.GetBounds()
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        self._set_reslice_axes(cx, cy, cz)
        self._renderer.ResetCamera()
        self.render()

    def set_window_level(self, window: float, level: float):
        if not _VTK:
            return
        self._lut.SetWindow(window)
        self._lut.SetLevel(level)
        self._lut.Build()
        self.render()

    def set_slice_position(self, world_x: float, world_y: float, world_z: float):
        if not _VTK:
            return
        self._set_reslice_axes(world_x, world_y, world_z)
        self.render()

    def set_crosshair(self, world_x: float, world_y: float, world_z: float):
        if not _VTK:
            return
        bounds = self._image_actor.GetBounds()
        if bounds[0] == bounds[1]:
            return
        if self._plane == "axial":
            px, py = world_x, world_y
            self._set_line(self._crosshair_h,
                           (bounds[0], py, 0), (bounds[1], py, 0))
            self._set_line(self._crosshair_v,
                           (px, bounds[2], 0), (px, bounds[3], 0))
        elif self._plane == "coronal":
            self._set_line(self._crosshair_h,
                           (bounds[0], 0, world_z), (bounds[1], 0, world_z))
            self._set_line(self._crosshair_v,
                           (world_x, 0, bounds[4]), (world_x, 0, bounds[5]))
        else:
            self._set_line(self._crosshair_h,
                           (0, bounds[2], world_z), (0, bounds[3], world_z))
            self._set_line(self._crosshair_v,
                           (0, world_y, bounds[4]), (0, world_y, bounds[5]))
        self.render()

    def render(self):
        if _VTK and self._initialized:
            self._interactor.GetRenderWindow().Render()

    def get_renderer(self):
        return self._renderer

    def get_interactor(self):
        return self._interactor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_reslice_axes(self, cx: float, cy: float, cz: float):
        axes = vtk.vtkMatrix4x4()
        axes.Identity()
        nx, ny, nz = self._normal
        vx, vy, vz = self._view_up
        ix = vy * nz - vz * ny
        iy = vz * nx - vx * nz
        iz = vx * ny - vy * nx
        axes.SetElement(0, 0, ix);  axes.SetElement(0, 1, iy);  axes.SetElement(0, 2, iz)
        axes.SetElement(1, 0, vx);  axes.SetElement(1, 1, vy);  axes.SetElement(1, 2, vz)
        axes.SetElement(2, 0, nx);  axes.SetElement(2, 1, ny);  axes.SetElement(2, 2, nz)
        axes.SetElement(0, 3, cx);  axes.SetElement(1, 3, cy);  axes.SetElement(2, 3, cz)
        self._reslice.SetResliceAxes(axes)
        self._reslice.Modified()

    @staticmethod
    def _make_line_actor(color=(1, 1, 0)):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(0, 0, 0)
        pts.InsertNextPoint(1, 0, 0)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(1.5)
        return actor

    @staticmethod
    def _set_line(actor, p1, p2):
        poly = actor.GetMapper().GetInput()
        pts = poly.GetPoints()
        pts.SetPoint(0, *p1)
        pts.SetPoint(1, *p2)
        pts.Modified()

    def closeEvent(self, event):
        if _VTK and self._interactor:
            self._interactor.Finalize()
        super().closeEvent(event)
