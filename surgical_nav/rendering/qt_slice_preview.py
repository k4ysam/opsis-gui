"""Lightweight Qt-based multi-slice preview for non-VTK fallback mode."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _normalize_to_u8(image_2d: np.ndarray) -> np.ndarray:
    """Window a 2-D image robustly into uint8 for display."""
    arr = np.asarray(image_2d, dtype=np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


class _ClickableLabel(QLabel):
    clicked = Signal(float, float)
    dragged = Signal(float, float)
    zoom_requested = Signal(float, float, float)
    pan_requested = Signal(float, float)
    reset_requested = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panning = False
        self._left_dragging = False
        self._last_pos = None
        self._display_pixmap: QPixmap | None = None
        self._display_rect: tuple[float, float, float, float] | None = None

    def set_display_content(
        self,
        pixmap: QPixmap | None,
        rect: tuple[float, float, float, float] | None,
    ):
        self._display_pixmap = pixmap
        self._display_rect = rect
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._display_pixmap is None or self._display_pixmap.isNull() or self._display_rect is None:
            return
        x, y, width, height = self._display_rect
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(
            int(round(x)),
            int(round(y)),
            int(round(width)),
            int(round(height)),
            self._display_pixmap,
        )

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_dragging = True
            self.clicked.emit(float(event.position().x()), float(event.position().y()))
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._left_dragging:
            self.dragged.emit(float(event.position().x()), float(event.position().y()))
        if self._panning and self._last_pos is not None:
            pos = event.position()
            delta = pos - self._last_pos
            self._last_pos = pos
            self.pan_requested.emit(float(delta.x()), float(delta.y()))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_dragging = False
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._last_pos = None

    def wheelEvent(self, event):
        super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta:
            self.zoom_requested.emit(
                float(delta),
                float(event.position().x()),
                float(event.position().y()),
            )

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_requested.emit()


class _SlicePane(QWidget):
    """One image pane with its own slider and title."""

    clicked = Signal(str, object, object)
    dragged = Signal(str, object, object)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._volume_array: np.ndarray | None = None
        self._overlay_array: np.ndarray | None = None
        self._preview_overlay_array: np.ndarray | None = None
        self._axis = 0
        self._image_shape: tuple[int, int] | None = None
        self._markers: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        self._trajectory_line: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        self._outline_points: list[tuple[int, int, int]] = []
        self._outline_plane: str | None = None
        self._source_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_drag_voxel = None
        self._last_drag_pos: tuple[float, float] | None = None
        self._interaction_mode = "navigate"
        self._pan_timer = QTimer(self)
        self._pan_timer.setSingleShot(True)
        self._pan_timer.setInterval(16)
        self._pan_timer.timeout.connect(self._update_display)

        self._title_label = QLabel(title, self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color: #ddd; background: #222; padding: 6px;")

        self._image_label = _ClickableLabel("No image loaded", self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(240, 240)
        self._image_label.setStyleSheet(
            "background: #111; color: #bbb; font-size: 14px; padding: 12px;"
        )
        self._image_label.clicked.connect(self._on_image_clicked)
        self._image_label.dragged.connect(self._on_image_dragged)
        self._image_label.zoom_requested.connect(self._on_zoom_requested)
        self._image_label.pan_requested.connect(self._on_pan_requested)
        self._image_label.reset_requested.connect(self._reset_view)

        self._slice_label = QLabel("Slice: -", self)
        self._slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slice_label.setStyleSheet("color: #ddd; background: #222; padding: 4px;")

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._render_slice)

        self._prev_btn = QPushButton("<", self)
        self._prev_btn.setFixedWidth(28)
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(lambda: self._step_slice(-1))

        self._next_btn = QPushButton(">", self)
        self._next_btn.setFixedWidth(28)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(lambda: self._step_slice(1))

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.setSpacing(4)
        slider_row.addWidget(self._prev_btn)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._next_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title_label)
        layout.addWidget(self._image_label, stretch=1)
        layout.addWidget(self._slice_label)
        layout.addLayout(slider_row)

    def set_volume(self, volume_array: np.ndarray, axis: int):
        self._volume_array = volume_array
        self._axis = axis
        depth = int(volume_array.shape[axis])
        self._slider.setEnabled(depth > 0)
        self._prev_btn.setEnabled(depth > 1)
        self._next_btn.setEnabled(depth > 1)
        self._slider.setRange(0, max(0, depth - 1))
        self._slider.setValue(depth // 2 if depth else 0)
        self._reset_view()
        if depth:
            self._render_slice(self._slider.value())

    def _step_slice(self, delta: int):
        if not self._slider.isEnabled():
            return
        value = self._slider.value() + delta
        value = max(self._slider.minimum(), min(self._slider.maximum(), value))
        self._slider.setValue(value)

    def set_overlay(self, overlay_array: np.ndarray | None):
        self._overlay_array = overlay_array
        if self._volume_array is not None:
            self._render_slice(self._slider.value())

    def set_preview_overlay(self, overlay_array: np.ndarray | None):
        self._preview_overlay_array = overlay_array
        if self._volume_array is not None:
            self._render_slice(self._slider.value())

    def set_markers(self, markers: list[tuple[tuple[int, int, int], tuple[int, int, int]]]):
        self._markers = markers
        if self._volume_array is not None:
            self._render_slice(self._slider.value())

    def set_trajectory_line(self, line: tuple[tuple[int, int, int], tuple[int, int, int]] | None):
        self._trajectory_line = line
        if self._volume_array is not None:
            self._render_slice(self._slider.value())

    def set_outline(self, plane: str | None, points: list[tuple[int, int, int]]):
        self._outline_plane = plane
        self._outline_points = list(points)
        if self._volume_array is not None:
            self._render_slice(self._slider.value())

    def set_interaction_mode(self, mode: str):
        self._interaction_mode = mode

    def _extract_slice(self, index: int) -> np.ndarray:
        assert self._volume_array is not None
        if self._axis == 0:
            return self._volume_array[index, :, :]
        if self._axis == 1:
            return self._volume_array[:, index, :]
        return self._volume_array[:, :, index]

    def _render_slice(self, index: int):
        if self._volume_array is None:
            return

        image_8 = _normalize_to_u8(self._extract_slice(index))
        rgb = np.stack([image_8, image_8, image_8], axis=-1)
        if self._overlay_array is not None:
            overlay_slice = self._extract_overlay(index)
            if overlay_slice is not None:
                mask = overlay_slice > 0
                rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([64, 220, 96])).astype(np.uint8)
        if self._preview_overlay_array is not None:
            preview_slice = self._extract_preview_overlay(index)
            if preview_slice is not None:
                mask = preview_slice > 0
                rgb[mask] = (0.35 * rgb[mask] + 0.65 * np.array([255, 64, 64])).astype(np.uint8)
        self._draw_outline(rgb, index)
        self._draw_trajectory(rgb)
        self._draw_markers(rgb, index)
        self._image_shape = rgb.shape[:2]
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._source_pixmap = QPixmap.fromImage(qimage)
        self._update_display()
        self._slice_label.setText(
            f"{self._title} Slice: {index + 1} / {self._volume_array.shape[self._axis]}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _on_image_clicked(self, x: float, y: float):
        voxel = self._display_pos_to_voxel(x, y)
        self._last_drag_pos = (x, y)
        if voxel is not None:
            self._last_drag_voxel = voxel
            self.clicked.emit(self._title.lower(), voxel, self._slider.value())

    def _on_image_dragged(self, x: float, y: float):
        if self._interaction_mode != "paint":
            if self._last_drag_pos is not None:
                dx = x - self._last_drag_pos[0]
                dy = y - self._last_drag_pos[1]
                self._pan_x += dx
                self._pan_y += dy
                if not self._pan_timer.isActive():
                    self._pan_timer.start()
            self._last_drag_pos = (x, y)
            return
        voxel = self._display_pos_to_voxel(x, y)
        if voxel is not None:
            if voxel == self._last_drag_voxel:
                return
            self._last_drag_voxel = voxel
            self.dragged.emit(self._title.lower(), voxel, self._slider.value())

    def _display_pos_to_voxel(self, x: float, y: float):
        rect = self._display_rect()
        if self._volume_array is None or self._image_shape is None or rect is None:
            return None
        offset_x, offset_y, pix_w, pix_h = rect
        if not (offset_x <= x <= offset_x + pix_w and offset_y <= y <= offset_y + pix_h):
            return None

        img_h, img_w = self._image_shape
        img_x = int(np.clip((x - offset_x) * img_w / max(pix_w, 1), 0, img_w - 1))
        img_y = int(np.clip((y - offset_y) * img_h / max(pix_h, 1), 0, img_h - 1))

        zyx_shape = self._volume_array.shape
        slice_idx = self._slider.value()
        if self._axis == 0:
            return (img_x, img_y, slice_idx)
        if self._axis == 1:
            return (img_x, slice_idx, img_y)
        return (slice_idx, img_x, img_y)

    def _extract_overlay(self, index: int):
        if self._overlay_array is None:
            return None
        if self._axis == 0:
            return self._overlay_array[index, :, :]
        if self._axis == 1:
            return self._overlay_array[:, index, :]
        return self._overlay_array[:, :, index]

    def _extract_preview_overlay(self, index: int):
        if self._preview_overlay_array is None:
            return None
        if self._axis == 0:
            return self._preview_overlay_array[index, :, :]
        if self._axis == 1:
            return self._preview_overlay_array[:, index, :]
        return self._preview_overlay_array[:, :, index]

    def _draw_markers(self, rgb: np.ndarray, index: int):
        for ijk, color in self._markers:
            coords = self._marker_display_coords(ijk, index)
            if coords is None:
                continue
            x, y = coords
            self._draw_circle(rgb, x, y, radius=5, color=color)

    def _draw_trajectory(self, rgb: np.ndarray):
        if self._trajectory_line is None:
            return
        p1 = self._project_ijk(self._trajectory_line[0])
        p2 = self._project_ijk(self._trajectory_line[1])
        if p1 is None or p2 is None:
            return
        self._draw_line(rgb, p1, p2, (255, 180, 0))

    def _draw_outline(self, rgb: np.ndarray, index: int):
        if not self._outline_points or self._outline_plane != self._title.lower():
            return
        projected = []
        for point in self._outline_points:
            coords = self._marker_display_coords(point, index)
            if coords is None:
                return
            projected.append(coords)
        for p1, p2 in zip(projected, projected[1:]):
            self._draw_line(rgb, p1, p2, (64, 255, 255))

    def _marker_display_coords(self, ijk: tuple[int, int, int], slice_index: int):
        i, j, k = ijk
        if self._axis == 0:
            if k != slice_index:
                return None
            return i, j
        if self._axis == 1:
            if j != slice_index:
                return None
            return i, k
        if i != slice_index:
            return None
        return j, k

    def _project_ijk(self, ijk: tuple[int, int, int]):
        i, j, k = ijk
        if self._axis == 0:
            return i, j
        if self._axis == 1:
            return i, k
        return j, k

    @staticmethod
    def _draw_circle(rgb: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]):
        height, width, _ = rgb.shape
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > r2:
                    continue
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    rgb[y, x] = color

    @staticmethod
    def _draw_line(rgb: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]):
        x1, y1 = p1
        x2, y2 = p2
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for t in np.linspace(0.0, 1.0, steps + 1):
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[y, x] = color

    def _display_rect(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        src_w = self._source_pixmap.width()
        src_h = self._source_pixmap.height()
        label_w = max(1, self._image_label.width())
        label_h = max(1, self._image_label.height())
        fit_scale = min(label_w / src_w, label_h / src_h)
        scale = fit_scale * self._zoom
        pix_w = max(1, int(round(src_w * scale)))
        pix_h = max(1, int(round(src_h * scale)))
        offset_x = (label_w - pix_w) / 2 + self._pan_x
        offset_y = (label_h - pix_h) / 2 + self._pan_y
        return offset_x, offset_y, pix_w, pix_h

    def _update_display(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        rect = self._display_rect()
        if rect is None:
            return
        _, _, pix_w, pix_h = rect
        scaled = self._source_pixmap.scaled(
            pix_w,
            pix_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.set_display_content(scaled, rect)

    def _on_zoom_requested(self, delta: float, _x: float, _y: float):
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = float(np.clip(self._zoom * factor, 1.0, 8.0))
        self._update_display()

    def _on_pan_requested(self, dx: float, dy: float):
        if self._zoom <= 1.0:
            return
        self._pan_x += dx
        self._pan_y += dy
        self._update_display()

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_display()


class _OverviewPane(QWidget):
    """A simple overview image generated from a volume projection."""

    def __init__(self, title: str = "Overview", parent: QWidget | None = None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._volume_array: np.ndarray | None = None
        self._overview_volume: np.ndarray | None = None
        self._source_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._yaw = 25
        self._pitch = -20

        self._title_label = QLabel(title, self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color: #ddd; background: #222; padding: 6px;")

        self._mode_combo = QComboBox(self)
        self._mode_combo.addItems(["3D MIP", "Mean", "Axial MIP", "Coronal MIP", "Sagittal MIP"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(self._title_label, stretch=1)
        header.addWidget(self._mode_combo)

        self._yaw_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._yaw_slider.setRange(-90, 90)
        self._yaw_slider.setValue(self._yaw)
        self._yaw_slider.valueChanged.connect(self._on_rotation_changed)

        self._pitch_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._pitch_slider.setRange(-90, 90)
        self._pitch_slider.setValue(self._pitch)
        self._pitch_slider.valueChanged.connect(self._on_rotation_changed)

        rotation_row = QHBoxLayout()
        rotation_row.setContentsMargins(0, 0, 0, 0)
        rotation_row.setSpacing(4)
        rotation_row.addWidget(QLabel("Yaw", self))
        rotation_row.addWidget(self._yaw_slider, stretch=1)
        rotation_row.addWidget(QLabel("Pitch", self))
        rotation_row.addWidget(self._pitch_slider, stretch=1)
        self._rotation_controls = QWidget(self)
        self._rotation_controls.setLayout(rotation_row)

        self._image_label = _ClickableLabel("No image loaded", self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(240, 240)
        self._image_label.setStyleSheet(
            "background: #111; color: #bbb; font-size: 14px; padding: 12px;"
        )
        self._image_label.zoom_requested.connect(self._on_zoom_requested)
        self._image_label.pan_requested.connect(self._on_pan_requested)
        self._image_label.reset_requested.connect(self._reset_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._rotation_controls)
        layout.addWidget(self._image_label, stretch=1)
        self._on_mode_changed(self._mode_combo.currentText())

    def set_volume(self, volume_array: np.ndarray):
        self._volume_array = volume_array
        shrink = max(1, int(np.ceil(max(volume_array.shape) / 192)))
        self._overview_volume = volume_array[::shrink, ::shrink, ::shrink]
        self._reset_view()
        self._rebuild_projection()

    def _render(self):
        if self._image is None:
            return
        height, width = self._image.shape
        qimage = QImage(
            self._image.data,
            width,
            height,
            self._image.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._source_pixmap = QPixmap.fromImage(qimage)
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _rebuild_projection(self):
        if self._overview_volume is None:
            return
        mode = self._mode_combo.currentText()
        volume = self._overview_volume
        if mode == "3D MIP":
            rotated = ndimage.rotate(
                volume,
                float(self._pitch),
                axes=(1, 0),
                reshape=False,
                order=1,
                mode="nearest",
            )
            rotated = ndimage.rotate(
                rotated,
                float(self._yaw),
                axes=(2, 0),
                reshape=False,
                order=1,
                mode="nearest",
            )
            image = np.max(rotated, axis=0)
        elif mode == "Axial MIP":
            image = np.max(volume, axis=0)
        elif mode == "Coronal MIP":
            image = np.max(volume, axis=1)
        elif mode == "Sagittal MIP":
            image = np.max(volume, axis=2)
        else:
            image = np.mean(volume, axis=0)
        self._image = _normalize_to_u8(image)
        self._render()

    def _on_mode_changed(self, mode: str):
        self._rotation_controls.setVisible(mode == "3D MIP")
        self._rebuild_projection()

    def _on_rotation_changed(self, _value: int):
        self._yaw = self._yaw_slider.value()
        self._pitch = self._pitch_slider.value()
        if self._mode_combo.currentText() == "3D MIP":
            self._rebuild_projection()

    def _display_rect(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        src_w = self._source_pixmap.width()
        src_h = self._source_pixmap.height()
        label_w = max(1, self._image_label.width())
        label_h = max(1, self._image_label.height())
        fit_scale = min(label_w / src_w, label_h / src_h)
        scale = fit_scale * self._zoom
        pix_w = max(1, int(round(src_w * scale)))
        pix_h = max(1, int(round(src_h * scale)))
        offset_x = (label_w - pix_w) / 2 + self._pan_x
        offset_y = (label_h - pix_h) / 2 + self._pan_y
        return offset_x, offset_y, pix_w, pix_h

    def _update_display(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        rect = self._display_rect()
        if rect is None:
            return
        _, _, pix_w, pix_h = rect
        scaled = self._source_pixmap.scaled(
            pix_w,
            pix_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.set_display_content(scaled, rect)

    def _on_zoom_requested(self, delta: float, _x: float, _y: float):
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = float(np.clip(self._zoom * factor, 1.0, 8.0))
        self._update_display()

    def _on_pan_requested(self, dx: float, dy: float):
        if self._zoom <= 1.0:
            return
        self._pan_x += dx
        self._pan_y += dy
        self._update_display()

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_display()


class QtSlicePreview(QWidget):
    """Display axial, coronal, sagittal, and overview panes using only Qt."""

    slice_clicked = Signal(str, object)
    slice_dragged = Signal(str, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._volume_array: np.ndarray | None = None
        self._sitk_image: sitk.Image | None = None
        self._target_overlay: np.ndarray | None = None
        self._paint_preview_overlay: np.ndarray | None = None
        self._landmark_markers: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        self._trajectory_markers: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

        self._axial = _SlicePane("Axial", self)
        self._coronal = _SlicePane("Coronal", self)
        self._sagittal = _SlicePane("Sagittal", self)
        self._overview = _OverviewPane(parent=self)
        self._axial.clicked.connect(self._on_slice_clicked)
        self._coronal.clicked.connect(self._on_slice_clicked)
        self._sagittal.clicked.connect(self._on_slice_clicked)
        self._axial.dragged.connect(self._on_slice_dragged)
        self._coronal.dragged.connect(self._on_slice_dragged)
        self._sagittal.dragged.connect(self._on_slice_dragged)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(self._overview, 0, 0)
        grid.addWidget(self._axial, 0, 1)
        grid.addWidget(self._coronal, 1, 0)
        grid.addWidget(self._sagittal, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

    def set_image(self, image: sitk.Image):
        """Load a SimpleITK volume and display orthogonal slice previews."""
        self._sitk_image = image
        self._volume_array = sitk.GetArrayViewFromImage(image)
        self._overview.set_volume(self._volume_array)
        self._axial.set_volume(self._volume_array, axis=0)
        self._coronal.set_volume(self._volume_array, axis=1)
        self._sagittal.set_volume(self._volume_array, axis=2)
        self._target_overlay = None
        self._landmark_markers = []
        self._trajectory_markers = []
        self._refresh_annotations()

    def set_interaction_mode(self, mode: str):
        for pane in (self._axial, self._coronal, self._sagittal):
            pane.set_interaction_mode(mode)

    def set_target_label(self, label_image):
        if label_image is None:
            self._target_overlay = None
        elif isinstance(label_image, np.ndarray):
            self._target_overlay = label_image
        else:
            self._target_overlay = sitk.GetArrayViewFromImage(label_image)
        self._refresh_annotations()

    def set_paint_preview(self, label_image):
        if label_image is None:
            self._paint_preview_overlay = None
        elif isinstance(label_image, np.ndarray):
            self._paint_preview_overlay = label_image
        else:
            self._paint_preview_overlay = sitk.GetArrayViewFromImage(label_image)
        self._refresh_annotations()

    def set_landmarks(self, ras_points):
        self._landmark_markers = []
        for ras in ras_points:
            ijk = self._ras_to_ijk(ras)
            if ijk is not None:
                self._landmark_markers.append((ijk, (64, 220, 255)))
        self._refresh_annotations()

    def set_trajectory_points(self, entry_ras, target_ras):
        self._trajectory_markers = []
        line = None
        for ras, color in ((entry_ras, (64, 255, 64)), (target_ras, (255, 220, 64))):
            ijk = self._ras_to_ijk(ras)
            if ijk is not None:
                self._trajectory_markers.append((ijk, color))
        if len(self._trajectory_markers) == 2:
            line = (self._trajectory_markers[0][0], self._trajectory_markers[1][0])
        for pane in (self._axial, self._coronal, self._sagittal):
            pane.set_trajectory_line(line)
        self._refresh_annotations()

    def _on_slice_clicked(self, plane: str, ijk_xyz, _slice_index):
        if self._sitk_image is None:
            return
        i, j, k = (int(ijk_xyz[0]), int(ijk_xyz[1]), int(ijk_xyz[2]))
        lps = self._sitk_image.TransformIndexToPhysicalPoint((i, j, k))
        ras = (-float(lps[0]), -float(lps[1]), float(lps[2]))
        self.slice_clicked.emit(plane, {"ijk": (i, j, k), "ras": ras})

    def _on_slice_dragged(self, plane: str, ijk_xyz, _slice_index):
        if self._sitk_image is None:
            return
        i, j, k = (int(ijk_xyz[0]), int(ijk_xyz[1]), int(ijk_xyz[2]))
        lps = self._sitk_image.TransformIndexToPhysicalPoint((i, j, k))
        ras = (-float(lps[0]), -float(lps[1]), float(lps[2]))
        self.slice_dragged.emit(plane, {"ijk": (i, j, k), "ras": ras})

    def _ras_to_ijk(self, ras_point):
        if self._sitk_image is None or ras_point is None:
            return None
        lps = (-float(ras_point[0]), -float(ras_point[1]), float(ras_point[2]))
        try:
            return tuple(int(v) for v in self._sitk_image.TransformPhysicalPointToIndex(lps))
        except RuntimeError:
            return None

    def _refresh_annotations(self):
        markers = self._landmark_markers + self._trajectory_markers
        for pane in (self._axial, self._coronal, self._sagittal):
            pane.set_overlay(self._target_overlay)
            pane.set_preview_overlay(self._paint_preview_overlay)
            pane.set_markers(markers)
