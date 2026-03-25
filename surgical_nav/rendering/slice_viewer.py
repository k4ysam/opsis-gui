"""SliceViewer: single-plane MPR view using Qt/numpy (no VTK required).

Displays axial, coronal, or sagittal slices from a SimpleITK image.
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
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor, QPointF

_PLANE_LABELS = {
    "axial":    "Axial",
    "coronal":  "Coronal",
    "sagittal": "Sagittal",
}

# Marker colours
_ENTRY_COLOR  = QColor(50, 220, 50)
_TARGET_COLOR = QColor(220, 70, 70)
_LM_COLOR     = QColor(255, 200, 0)


class SliceViewer(QWidget):
    """2-D slice viewer for one anatomical plane."""

    # Emitted when the user places a point in the active mode.
    # Arguments: mode ("entry" | "target" | "landmark"), world_x, world_y, world_z
    point_placed = Signal(str, float, float, float)

    def __init__(self, plane: str = "axial", parent: Optional[QWidget] = None):
        super().__init__(parent)
        if plane not in _PLANE_LABELS:
            raise ValueError(f"plane must be one of {list(_PLANE_LABELS)}")

        self._plane = plane
        self._volume: Optional[np.ndarray] = None   # (z, y, x) float32
        self._origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._slice_idx: int = 0
        self._window: float = 400.0
        self._level: float = 40.0

        # Interaction mode
        self._mode: str = ""

        # Stored 3-D world coordinates for overlay markers
        self._entry_xyz:  Optional[np.ndarray] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._landmark_xyzs: list = []     # list of np.ndarray

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

    def initialize(self):
        pass

    def set_sitk_image(self, sitk_image) -> None:
        """Load a SimpleITK image and display the middle slice."""
        import SimpleITK as sitk

        arr = sitk.GetArrayFromImage(sitk_image)
        self._volume = arr.astype(np.float32)
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

    def set_volume(self, vtk_image_data):
        """VTK path — unused without VTK."""
        pass

    def set_window_level(self, window: float, level: float) -> None:
        self._window = window
        self._level  = level
        self._update_display()

    def set_slice_position(
        self, world_x: float, world_y: float, world_z: float
    ) -> None:
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

    def set_crosshair(self, world_x: float, world_y: float, world_z: float) -> None:
        pass

    def render(self):       pass
    def get_renderer(self): return None
    def get_interactor(self): return None

    # ------------------------------------------------------------------
    # Public API — interaction mode & markers
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set interaction mode: "entry", "target", "landmark", or "" for none."""
        self._mode = mode
        if mode:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.unsetCursor()

    def set_trajectory_points(
        self,
        entry_xyz:  Optional[np.ndarray],
        target_xyz: Optional[np.ndarray],
    ) -> None:
        """Update entry/target world coordinates and redraw."""
        self._entry_xyz  = entry_xyz
        self._target_xyz = target_xyz
        self._update_display()

    def set_landmarks(self, landmark_list: list) -> None:
        """Update landmark world coordinates (list of np.ndarray)."""
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
        self.point_placed.emit(
            self._mode, float(world[0]), float(world[1]), float(world[2])
        )

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
        if n == 0:
            self._slice_lbl.setText("–/–")
        else:
            self._slice_lbl.setText(f"{self._slice_idx + 1}/{n}")

    def _n_slices(self) -> int:
        if self._volume is None:
            return 0
        nz, ny, nx = self._volume.shape
        if self._plane == "axial":   return nz
        elif self._plane == "coronal": return ny
        return nx

    # ------------------------------------------------------------------
    # Internal — geometry
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
        """Label-space click → (img_col, img_row) in the flipped slice."""
        if self._volume is None:
            return None
        arr = self._get_slice_array()
        if arr is None:
            return None
        orig_h, orig_w = arr.shape
        lbl_w = self._image_label.width()
        lbl_h = self._image_label.height()
        if lbl_w <= 0 or lbl_h <= 0 or orig_w <= 0 or orig_h <= 0:
            return None

        scale = min(lbl_w / orig_w, lbl_h / orig_h)
        pix_w = orig_w * scale
        pix_h = orig_h * scale
        off_x = (lbl_w - pix_w) / 2.0
        off_y = (lbl_h - pix_h) / 2.0

        px = lx - off_x
        py = ly - off_y
        if px < 0 or py < 0 or px >= pix_w or py >= pix_h:
            return None

        img_col = max(0, min(int(px / scale), orig_w - 1))
        img_row = max(0, min(int(py / scale), orig_h - 1))
        return img_col, img_row

    def _image_to_world(self, img_col: int, img_row: int) -> np.ndarray:
        """(img_col, img_row) in flipped-slice space → world (mm)."""
        ox, oy, oz = self._origin
        sx, sy, sz = self._spacing
        nz, ny, nx = self._volume.shape
        idx = self._slice_idx
        if self._plane == "axial":
            return np.array([
                ox + img_col * sx,
                oy + ((ny - 1) - img_row) * sy,
                oz + idx * sz,
            ])
        elif self._plane == "coronal":
            return np.array([
                ox + img_col * sx,
                oy + idx * sy,
                oz + ((nz - 1) - img_row) * sz,
            ])
        else:  # sagittal
            return np.array([
                ox + idx * sx,
                oy + img_col * sy,
                oz + ((nz - 1) - img_row) * sz,
            ])

    def _world_to_display(self, world_xyz) -> Tuple[float, float]:
        """World (mm) → (img_col, img_row) in the flipped slice image."""
        ox, oy, oz = self._origin
        sx, sy, sz = self._spacing
        nz, ny, nx = self._volume.shape
        wx, wy, wz = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if self._plane == "axial":
            return (wx - ox) / sx, (ny - 1) - (wy - oy) / sy
        elif self._plane == "coronal":
            return (wx - ox) / sx, (nz - 1) - (wz - oz) / sz
        else:
            return (wy - oy) / sy, (nz - 1) - (wz - oz) / sz

    # ------------------------------------------------------------------
    # Internal — rendering
    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        arr = self._get_slice_array()
        if arr is None:
            return

        arr = np.flipud(arr)

        lo   = self._level - self._window / 2.0
        hi   = self._level + self._window / 2.0
        span = max(hi - lo, 1e-6)
        arr_u8 = np.ascontiguousarray(
            (np.clip(arr, lo, hi) - lo) / span * 255.0, dtype=np.uint8
        )
        orig_h, orig_w = arr_u8.shape

        qimg = QImage(arr_u8.tobytes(), orig_w, orig_h, orig_w,
                      QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        label_sz = self._image_label.size()
        if label_sz.width() > 0 and label_sz.height() > 0:
            scale = min(label_sz.width() / orig_w, label_sz.height() / orig_h)
            pix = pix.scaled(
                label_sz,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._paint_markers(pix, scale, orig_h, orig_w)

        self._image_label.setPixmap(pix)

    def _paint_markers(
        self, pix: QPixmap, scale: float, orig_h: int, orig_w: int
    ) -> None:
        """Draw entry/target/landmark markers on *pix* (already scaled)."""
        markers = [
            (self._entry_xyz,  _ENTRY_COLOR),
            (self._target_xyz, _TARGET_COLOR),
        ]
        for xyz in self._landmark_xyzs:
            markers.append((xyz, _LM_COLOR))

        any_visible = False
        plotted: list = []   # (px, py) for trajectory line

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for world_xyz, color in markers:
            if world_xyz is None:
                continue
            ic, ir = self._world_to_display(world_xyz)
            if ic < 0 or ir < 0 or ic >= orig_w or ir >= orig_h:
                continue
            px, py = ic * scale, ir * scale
            any_visible = True
            plotted.append((px, py))

            r   = 7
            arm = 12
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(px, py), r, r)
            # Crosshair arms outside the circle
            painter.drawLine(QPointF(px - arm, py), QPointF(px - r - 1, py))
            painter.drawLine(QPointF(px + r + 1, py), QPointF(px + arm, py))
            painter.drawLine(QPointF(px, py - arm), QPointF(px, py - r - 1))
            painter.drawLine(QPointF(px, py + r + 1), QPointF(px, py + arm))

        # Trajectory line between entry and target (if both visible)
        if (self._entry_xyz is not None and self._target_xyz is not None
                and len(plotted) >= 2):
            ic_e, ir_e = self._world_to_display(self._entry_xyz)
            ic_t, ir_t = self._world_to_display(self._target_xyz)
            e_in = 0 <= ic_e < orig_w and 0 <= ir_e < orig_h
            t_in = 0 <= ic_t < orig_w and 0 <= ir_t < orig_h
            if e_in and t_in:
                dash_pen = QPen(QColor(255, 220, 0), 1)
                dash_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(
                    QPointF(ic_e * scale, ir_e * scale),
                    QPointF(ic_t * scale, ir_t * scale),
                )

        painter.end()
