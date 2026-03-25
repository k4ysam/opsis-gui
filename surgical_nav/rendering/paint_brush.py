"""PaintBrush: voxel-level painting on a numpy uint8 label array.

Converts world-space coordinates to voxel (IJK) indices and paints a sphere
of given radius. The label array is a plain numpy uint8 3-D array (nx, ny, nz)
modified in-place; callers retrieve the result as a SimpleITK image via
``get_label_sitk()``.

Usage::

    arr = PaintBrush.create_label_volume((128, 128, 64))
    brush = PaintBrush(arr, spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
    brush.paint_at_world(64.0, 64.0, 64.0, radius_voxels=3)
    label_sitk = brush.get_label_sitk()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk


class PaintBrush:
    """Paints voxel spheres into a numpy uint8 label array.

    Parameters
    ----------
    label_array : np.ndarray
        3-D uint8 array of shape ``(nx, ny, nz)``, modified in-place.
    spacing : tuple of float
        Voxel spacing (sx, sy, sz) in mm. Default (1, 1, 1).
    origin : tuple of float
        Volume origin (ox, oy, oz) in mm. Default (0, 0, 0).
    """

    def __init__(
        self,
        label_array: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self._array = label_array
        self._spacing = np.array(spacing, dtype=float)
        self._origin = np.array(origin, dtype=float)
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
        """Paint a sphere centred at a world-space point."""
        ijk = self._world_to_ijk(world_x, world_y, world_z)
        if ijk is None:
            return
        self._paint_sphere(ijk, radius_voxels)

    def paint_at_display(
        self,
        renderer,
        display_x: float,
        display_y: float,
        plane: str,
        radius_voxels: int = 3,
    ):
        """No-op without a VTK renderer."""
        pass

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
        # array is (nx, ny, nz); sitk expects (nz, ny, nx)
        arr = self._array.transpose(2, 1, 0).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(tuple(float(s) for s in self._spacing))
        img.SetOrigin(tuple(float(o) for o in self._origin))
        return img

    def clear(self):
        """Erase all painted voxels."""
        self._array[:] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_ijk(
        self, wx: float, wy: float, wz: float
    ) -> Optional[Tuple[int, int, int]]:
        dims = self._array.shape  # (nx, ny, nz)
        ijk_f = (np.array([wx, wy, wz]) - self._origin) / self._spacing
        ijk = tuple(int(round(v)) for v in ijk_f)
        if (0 <= ijk[0] < dims[0] and
                0 <= ijk[1] < dims[1] and
                0 <= ijk[2] < dims[2]):
            return ijk
        return None

    def _paint_sphere(self, centre_ijk: Tuple[int, int, int], radius: int):
        dims = self._array.shape
        cx, cy, cz = centre_ijk
        r2 = radius ** 2
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > r2:
                        continue
                    ix, iy, iz = cx + dx, cy + dy, cz + dz
                    if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
                        self._array[ix, iy, iz] = self._paint_value

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_label_volume(
        cls,
        dims: Tuple[int, int, int],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Create a blank uint8 numpy array of shape *dims* = (nx, ny, nz)."""
        return np.zeros(dims, dtype=np.uint8)
