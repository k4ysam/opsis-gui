"""PaintBrush: voxel-level painting on a 2-D slice plane.

Converts a screen (display) coordinate to a world-space point on the slice
plane, then converts that to a voxel (IJK) index in a vtkImageData label
volume and paints a sphere of given radius.

The label volume is a separate vtkImageData (uint8) that is blended over the
anatomical slice view via vtkImageBlend.  Callers retrieve the modified
label array as a SimpleITK image via ``get_label_sitk()``.

Usage::

    brush = PaintBrush(label_vtk_image)
    # On mouse click at display (px, py) in a SliceViewer:
    brush.paint_at_display(renderer, display_x, display_y,
                           plane="axial", radius_voxels=3)
    label_sitk = brush.get_label_sitk()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    _VTK = True
except ImportError:
    vtk = None  # type: ignore
    vtk_to_numpy = numpy_to_vtk = None  # type: ignore
    _VTK = False


class PaintBrush:
    """Paints voxel spheres into a label vtkImageData.

    Parameters
    ----------
    label_image : vtkImageData
        Uint8 label volume, same geometry as the anatomical volume.
        Modified in-place by ``paint_at_world``.
    """

    def __init__(self, label_image: vtk.vtkImageData):
        self._label = label_image
        self._paint_value: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_paint_value(self, value: int):
        """Set the label value written by paint operations (default 1)."""
        self._paint_value = value

    def paint_at_world(
        self,
        world_x: float,
        world_y: float,
        world_z: float,
        radius_voxels: int = 3,
    ):
        """Paint a sphere centred at a world-space point.

        Parameters
        ----------
        world_x, world_y, world_z : float
            World (RAS) coordinates of the paint centre.
        radius_voxels : int
            Sphere radius in voxels.
        """
        ijk = self._world_to_ijk(world_x, world_y, world_z)
        if ijk is None:
            return
        self._paint_sphere(ijk, radius_voxels)

    def paint_at_display(
        self,
        renderer: vtk.vtkRenderer,
        display_x: float,
        display_y: float,
        plane: str,
        radius_voxels: int = 3,
    ):
        """Paint at a display (pixel) coordinate by unprojecting to world space.

        Parameters
        ----------
        renderer : vtkRenderer
            The renderer owning the camera (for unproject).
        display_x, display_y : float
            Mouse position in display coordinates.
        plane : str
            'axial' | 'coronal' | 'sagittal' — determines the Z-depth used
            for the unproject.
        radius_voxels : int
            Sphere radius in voxels.
        """
        world = self._display_to_world(renderer, display_x, display_y, plane)
        if world is None:
            return
        self.paint_at_world(*world, radius_voxels=radius_voxels)

    def erase_at_world(
        self,
        world_x: float,
        world_y: float,
        world_z: float,
        radius_voxels: int = 3,
    ):
        """Erase (set to 0) a sphere centred at a world-space point."""
        old_value = self._paint_value
        self._paint_value = 0
        self.paint_at_world(world_x, world_y, world_z, radius_voxels)
        self._paint_value = old_value

    def get_label_sitk(self) -> sitk.Image:
        """Return the current label volume as a SimpleITK uint8 image."""
        dims = self._label.GetDimensions()   # (nx, ny, nz)
        arr_flat = vtk_to_numpy(self._label.GetPointData().GetScalars())
        # VTK stores x-fastest; reshape to (nx, ny, nz) then transpose to (nz, ny, nx) for sitk
        arr = arr_flat.reshape(dims[0], dims[1], dims[2], order="F")
        arr = arr.transpose(2, 1, 0).astype(np.uint8)   # (nz, ny, nx)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(self._label.GetSpacing())
        img.SetOrigin(self._label.GetOrigin())
        return img

    def clear(self):
        """Erase all painted voxels."""
        self._label.GetPointData().GetScalars().Fill(0)
        self._label.Modified()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_ijk(
        self, wx: float, wy: float, wz: float
    ) -> Optional[Tuple[int, int, int]]:
        """Convert world (RAS) to voxel (i, j, k) index.

        Returns None if the point is outside the volume bounds.
        """
        origin  = np.array(self._label.GetOrigin())
        spacing = np.array(self._label.GetSpacing())
        dims    = self._label.GetDimensions()

        ijk_f = (np.array([wx, wy, wz]) - origin) / spacing
        ijk   = tuple(int(round(v)) for v in ijk_f)

        if (0 <= ijk[0] < dims[0] and
                0 <= ijk[1] < dims[1] and
                0 <= ijk[2] < dims[2]):
            return ijk
        return None

    def _paint_sphere(self, centre_ijk: Tuple[int, int, int], radius: int):
        """Set all voxels within ``radius`` of ``centre_ijk`` to ``_paint_value``."""
        dims = self._label.GetDimensions()
        cx, cy, cz = centre_ijk
        r2 = radius ** 2

        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > r2:
                        continue
                    ix = cx + dx
                    iy = cy + dy
                    iz = cz + dz
                    if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
                        flat_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
                        self._label.GetPointData().GetScalars().SetValue(
                            flat_idx, self._paint_value
                        )
        self._label.Modified()

    @staticmethod
    def _display_to_world(
        renderer: vtk.vtkRenderer,
        display_x: float,
        display_y: float,
        plane: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Unproject a display point to world coordinates on the slice plane.

        Uses the renderer's camera focal point depth for the Z-buffer value,
        placing the result on the current slice plane.
        """
        renderer.SetDisplayPoint(display_x, display_y, 0.5)
        renderer.DisplayToWorld()
        world = renderer.GetWorldPoint()
        if world[3] == 0:
            return None
        w = world[3]
        return (world[0] / w, world[1] / w, world[2] / w)

    # ------------------------------------------------------------------
    # Factory: create a blank label volume matching an anatomical volume
    # ------------------------------------------------------------------

    @classmethod
    def create_label_volume(cls, reference: vtk.vtkImageData) -> vtk.vtkImageData:
        """Create a blank uint8 vtkImageData with the same geometry as *reference*."""
        label = vtk.vtkImageData()
        label.SetDimensions(reference.GetDimensions())
        label.SetSpacing(reference.GetSpacing())
        label.SetOrigin(reference.GetOrigin())
        label.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        label.GetPointData().GetScalars().Fill(0)
        return label
