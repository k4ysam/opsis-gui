"""SliceViewer: single-plane MPR view using Qt/numpy (no VTK required).

Displays axial, coronal, or sagittal slices from a SimpleITK image.
Voxel spacing is respected so images have the correct physical aspect ratio.
A horizontal slider at the bottom lets the user scroll through slices.
Window/level is auto-computed from the 2nd–98th percentile on load.

Interaction modes (set via set_mode):
  "entry"    — next click places the entry point (green crosshair)
  "target"   — next click places the target point (red crosshair)
  "landmark" — next click places an anatomical landmark (yellow crosshair)
  ""         — view-only, clicks ignored
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QSlider,
)
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor

_PLANE_LABELS = {
    "axial":    "Axial",
    "coronal":  "Coronal",
    "sagittal": "Sagittal",
}

_ENTRY_COLOR  = QColor(50, 220, 50)
_TARGET_COLOR = QColor(220, 70, 70)
_LM_COLOR     = QColor(255, 200, 0)


class SliceViewer(QWidget):
    """2-D slice viewer for one anatomical plane."""

    # (mode, world_x, world_y, world_z) emitted on click in an active mode
    point_placed = Signal(str, float, float, float)

    def __init__(self, plane: str = "axial", parent: Optional[QWidget] = None):
        super().__init__(parent)
        if plane not in _PLANE_LABELS:
            raise ValueError(f"plane must be one of {list(_PLANE_LABELS)}")

        self._plane = plane
        self._volume: Optional[np.ndarray] = None   # (z, y, x) float32
        self._origin: Tuple[float, ...] = (0.0, 0.0, 0.0)
        self._spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
        self._slice_idx: int = 0
        self._window: float = 400.0
        self._level: float = 40.0
        self._mode: str = ""

        # Overlay markers
        self._entry_xyz:  Optional[np.ndarray] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._landmark_xyzs: list = []

        # Display-space scales (set by _update_display, used by click handler)
        # (scale_x, scale_y, off_x, off_y) from original image pixels → label pixels
        self._disp_transform: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)

        # ---- layout ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._title = QLabel(_PLANE_LABELS[plane])
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            "color: #aaa; background: #111; font-size: 11px; padding: 2px;"
        )
        self._title.setFixedHeight(18)
        layout.addWidget(self._title)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background: #000;")
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setMinimumSize(80, 80)
        layout.addWidget(self._image_label, stretch=1)

        # Slider row
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(4, 2, 4, 2)
        slider_row.setSpacing(6)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider, stretch=1)

        self._slice_lbl = QLabel("–/–")
        self._slice_lbl.setFixedWidth(52)
        self._slice_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._slice_lbl.setStyleSheet("color: #888; font-size: 10px;")
        slider_row.addWidget(self._slice_lbl)

        slider_container = QWidget()
        slider_container.setStyleSheet("background: #1a1a1a;")
        slider_container.setFixedHeight(28)
        slider_container.setLayout(slider_row)
        layout.addWidget(slider_container)

    # ------------------------------------------------------------------
    # Public API — image loading
    # ------------------------------------------------------------------

    def initialize(self): pass

    def set_sitk_image(self, sitk_image) -> None:
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk_image)
        self._volume  = arr.astype(np.float32)
        self._spacing = sitk_image.GetSpacing()
        self._origin  = sitk_image.GetOrigin()

        p2, p98 = np.percentile(self._volume, [2, 98])
        self._level  = float((p2 + p98) / 2.0)
        self._window = float(max(p98 - p2, 1.0))

        n = self._n_slices()
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(n - 1, 0))
        mid = n // 2
        self._slice_idx = mid
        self._slider.setValue(mid)
        self._slider.setEnabled(n > 1)
        self._slider.blockSignals(False)

        self._update_slice_label()
        self._update_display()

    def set_volume(self, vtk_image_data): pass

    def set_window_level(self, window: float, level: float) -> None:
        self._window = window
        self._level  = level
        self._update_display()

    def set_slice_position(self, world_x: float, world_y: float, world_z: float) -> None:
        if self._volume is None:
            return
        ox, oy, oz = self._origin
        sx, sy, sz = self._spacing
        if self._plane == "axial":
            idx = int(round((world_z - oz) / max(sz, 1e-6)))
        elif self._plane == "coronal":
            idx = int(round((world_y - oy) / max(sy, 1e-6)))
        else:
            idx = int(round((world_x - ox) / max(sx, 1e-6)))

        idx = max(0, min(idx, self._n_slices() - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._slice_idx = idx
        self._update_slice_label()
        self._update_display()

    def set_crosshair(self, world_x, world_y, world_z): pass
    def render(self): pass
    def get_renderer(self): return None
    def get_interactor(self): return None

    # ------------------------------------------------------------------
    # Public API — interaction mode & markers
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.unsetCursor()

    def set_trajectory_points(
        self,
        entry_xyz: Optional[np.ndarray],
        target_xyz: Optional[np.ndarray],
    ) -> None:
        self._entry_xyz  = entry_xyz
        self._target_xyz = target_xyz
        self._update_display()

    def set_landmarks(self, landmark_list: list) -> None:
        self._landmark_xyzs = list(landmark_list)
        self._update_display()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if not self._mode or self._volume is None:
            return super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        pos_in_label = self._image_label.mapFrom(self, event.position().toPoint())
        result = self._click_to_image_coords(pos_in_label.x(), pos_in_label.y())
        if result is None:
            return
        img_col, img_row = result
        world = self._image_to_world(img_col, img_row)
        self.point_placed.emit(self._mode, float(world[0]), float(world[1]), float(world[2]))

    def showEvent(self, event):
        super().showEvent(event)
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def closeEvent(self, event):
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Internal — slider
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        self._slice_idx = value
        self._update_slice_label()
        self._update_display()

    def _update_slice_label(self) -> None:
        n = self._n_slices()
        self._slice_lbl.setText("–/–" if n == 0 else f"{self._slice_idx + 1}/{n}")

    def _n_slices(self) -> int:
        if self._volume is None:
            return 0
        nz, ny, nx = self._volume.shape
        if self._plane == "axial":   return nz
        elif self._plane == "coronal": return ny
        return nx

    # ------------------------------------------------------------------
    # Internal — physical spacing helpers
    # ------------------------------------------------------------------

    def _plane_spacings(self) -> Tuple[float, float]:
        """Return (row_spacing, col_spacing) in mm for the current plane."""
        sx, sy, sz = self._spacing
        if self._plane == "axial":
            return sy, sx          # rows = Y, cols = X
        elif self._plane == "coronal":
            return sz, sx          # rows = Z, cols = X
        else:
            return sz, sy          # rows = Z, cols = Y

    # ------------------------------------------------------------------
    # Internal — geometry / coordinate mapping
    # ------------------------------------------------------------------

    def _get_slice_array(self) -> Optional[np.ndarray]:
        if self._volume is None:
            return None
        nz, ny, nx = self._volume.shape
        idx = self._slice_idx
        if self._plane == "axial":
            return self._volume[max(0, min(idx, nz-1)), :, :]
        elif self._plane == "coronal":
            return self._volume[:, max(0, min(idx, ny-1)), :]
        else:
            return self._volume[:, :, max(0, min(idx, nx-1))]

    def _click_to_image_coords(self, lx: int, ly: int) -> Optional[Tuple[int, int]]:
        """Label-space click → (img_col, img_row) in the flipped original slice."""
        scale_x, scale_y, off_x, off_y = self._disp_transform
        if scale_x <= 0 or scale_y <= 0:
            return None
        arr = self._get_slice_array()
        if arr is None:
            return None
        orig_h, orig_w = arr.shape

        # Convert label click → display-pixmap click (remove centering offset)
        px = lx - off_x
        py = ly - off_y

        disp_w = orig_w * scale_x
        disp_h = orig_h * scale_y
        if px < 0 or py < 0 or px >= disp_w or py >= disp_h:
            return None

        img_col = max(0, min(int(px / scale_x), orig_w - 1))
        img_row = max(0, min(int(py / scale_y), orig_h - 1))
        return img_col, img_row

    def _image_to_world(self, img_col: int, img_row: int) -> np.ndarray:
        """(img_col, img_row) in flipped-slice space → world mm."""
        ox, oy, oz = self._origin
        sx, sy, sz = self._spacing
        nz, ny, nx = self._volume.shape
        idx = self._slice_idx
        if self._plane == "axial":
            return np.array([ox + img_col * sx, oy + ((ny-1) - img_row) * sy, oz + idx * sz])
        elif self._plane == "coronal":
            return np.array([ox + img_col * sx, oy + idx * sy, oz + ((nz-1) - img_row) * sz])
        else:
            return np.array([ox + idx * sx, oy + img_col * sy, oz + ((nz-1) - img_row) * sz])

    def _world_to_display(self, world_xyz) -> Tuple[float, float]:
        """World mm → (img_col, img_row) in the flipped original slice."""
        ox, oy, oz = self._origin
        sx, sy, sz = self._spacing
        nz, ny, nx = self._volume.shape
        wx, wy, wz = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if self._plane == "axial":
            return (wx - ox) / sx, (ny-1) - (wy - oy) / sy
        elif self._plane == "coronal":
            return (wx - ox) / sx, (nz-1) - (wz - oz) / sz
        else:
            return (wy - oy) / sy, (nz-1) - (wz - oz) / sz

    # ------------------------------------------------------------------
    # Internal — rendering
    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        arr = self._get_slice_array()
        if arr is None:
            return

        arr = np.flipud(arr)

        lo = self._level - self._window / 2.0
        hi = self._level + self._window / 2.0
        arr_u8 = np.ascontiguousarray(
            (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6) * 255.0, dtype=np.uint8
        )
        orig_h, orig_w = arr_u8.shape

        qimg = QImage(arr_u8.tobytes(), orig_w, orig_h, orig_w,
                      QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        label_sz = self._image_label.size()
        if label_sz.width() <= 0 or label_sz.height() <= 0:
            self._image_label.setPixmap(pix)
            return

        # --- Physical aspect ratio correction ---
        # Each plane's rows and cols have different spacings in mm.
        # Scale the image to physical dimensions before fitting to the label,
        # so e.g. a coronal slice with 3mm Z voxels looks correct.
        row_sp, col_sp = self._plane_spacings()
        min_sp = min(row_sp, col_sp)
        phys_w = max(1, int(round(orig_w * col_sp / min_sp)))
        phys_h = max(1, int(round(orig_h * row_sp / min_sp)))

        if phys_w != orig_w or phys_h != orig_h:
            pix = pix.scaled(
                phys_w, phys_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # --- Fit to label (keep physical aspect ratio) ---
        pix = pix.scaled(
            label_sz,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Effective scale from *original image pixels* to *display pixels*
        scale_to_label = min(label_sz.width() / phys_w, label_sz.height() / phys_h)
        scale_x = (col_sp / min_sp) * scale_to_label   # orig col → display x
        scale_y = (row_sp / min_sp) * scale_to_label   # orig row → display y

        # Centering offset (label is bigger than the pixmap in one dimension)
        off_x = (label_sz.width()  - pix.width())  / 2.0
        off_y = (label_sz.height() - pix.height()) / 2.0

        self._disp_transform = (scale_x, scale_y, off_x, off_y)

        self._paint_markers(pix, scale_x, scale_y, orig_h, orig_w)
        self._image_label.setPixmap(pix)

    def _paint_markers(
        self,
        pix: QPixmap,
        scale_x: float,
        scale_y: float,
        orig_h: int,
        orig_w: int,
    ) -> None:
        markers = [
            (self._entry_xyz,  _ENTRY_COLOR),
            (self._target_xyz, _TARGET_COLOR),
        ]
        for xyz in self._landmark_xyzs:
            markers.append((xyz, _LM_COLOR))

        if all(xyz is None for xyz, _ in markers) and not self._landmark_xyzs:
            return

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        entry_disp = target_disp = None

        for world_xyz, color in markers:
            if world_xyz is None:
                continue
            ic, ir = self._world_to_display(world_xyz)
            if ic < 0 or ir < 0 or ic >= orig_w or ir >= orig_h:
                continue
            px, py = ic * scale_x, ir * scale_y

            if world_xyz is self._entry_xyz:
                entry_disp = (px, py)
            elif world_xyz is self._target_xyz:
                target_disp = (px, py)

            r, arm = 7, 12
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(px, py), r, r)
            painter.drawLine(QPointF(px - arm, py), QPointF(px - r - 1, py))
            painter.drawLine(QPointF(px + r + 1, py), QPointF(px + arm,  py))
            painter.drawLine(QPointF(px, py - arm), QPointF(px, py - r - 1))
            painter.drawLine(QPointF(px, py + r + 1), QPointF(px, py + arm))

        # Trajectory line between entry and target
        if entry_disp and target_disp:
            dash = QPen(QColor(255, 220, 0), 1)
            dash.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(dash)
            painter.drawLine(QPointF(*entry_disp), QPointF(*target_disp))

        painter.end()
