"""TrackingViewer: right-panel widget for the Tracking Test page.

Contains:
- A matplotlib 3D trajectory canvas (refreshes every 500 ms)
- Up to 5 camera/video feed thumbnails
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea, QSplitter,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

def _check_mpl() -> bool:
    try:
        import matplotlib.figure  # noqa: F401
        return True
    except Exception:
        return False

_MPL_AVAILABLE = _check_mpl()

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Trajectory canvas
# ---------------------------------------------------------------------------

class _TrajectoryCanvas(QWidget):
    """3D scatter/line plot that accumulates tool positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._zs: list[float] = []
        self._dirty = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        self._fig.patch.set_facecolor('#1e1e1e')
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas)
        self._style_axes()

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor('#1e1e1e')
        ax.set_xlabel("X (mm)", color='#ccc', fontsize=8)
        ax.set_ylabel("Y (mm)", color='#ccc', fontsize=8)
        ax.set_zlabel("Z (mm)", color='#ccc', fontsize=8)
        ax.tick_params(colors='#aaa', labelsize=7)
        ax.title.set_color('#eee')
        ax.set_title("Tool Trajectory", fontsize=10)

    def add_point(self, x: float, y: float, z: float):
        self._xs.append(x)
        self._ys.append(y)
        self._zs.append(z)
        self._dirty = True

    def clear(self):
        self._xs.clear()
        self._ys.clear()
        self._zs.clear()
        self._dirty = True

    def refresh(self):
        if not self._dirty:
            return
        self._dirty = False
        self._ax.cla()
        self._style_axes()
        if len(self._xs) > 1:
            self._ax.plot(self._xs, self._ys, self._zs,
                          color='#4a9eff', linewidth=1.0, alpha=0.7)
        if self._xs:
            self._ax.scatter(
                [self._xs[-1]], [self._ys[-1]], [self._zs[-1]],
                c='#ff4444', s=40, zorder=5,
            )
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Video feed thumbnail
# ---------------------------------------------------------------------------

class _VideoFeed(QWidget):
    """Single video feed thumbnail."""

    def __init__(self, label: str, cap, parent=None):
        super().__init__(parent)
        self._cap = cap

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title = QLabel(label)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 11px; color: #ccc;")
        layout.addWidget(title)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(100)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setStyleSheet("background: #111; border: 1px solid #444;")
        layout.addWidget(self._image_label)

    def update_frame(self):
        if not _CV2_AVAILABLE:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        if not ret:
            return
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self._image_label.width(),
            self._image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(pixmap)

    def release(self):
        self._cap.release()


# ---------------------------------------------------------------------------
# TrackingViewer
# ---------------------------------------------------------------------------

class TrackingViewer(QWidget):
    """Right-panel viewer for the Tracking Test page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._feeds: list[_VideoFeed] = []

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(500)
        self._plot_timer.timeout.connect(self._refresh_plot)

        self._feed_timer = QTimer(self)
        self._feed_timer.setInterval(66)   # ~15 fps
        self._feed_timer.timeout.connect(self._update_feeds)

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left: trajectory ---
        if _MPL_AVAILABLE:
            self._traj = _TrajectoryCanvas()
        else:
            self._traj = QLabel(
                "matplotlib not installed\npip install matplotlib",
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            self._traj.setStyleSheet("color: #888; font-size: 13px;")
        splitter.addWidget(self._traj)

        # --- Right: camera feeds ---
        self._cam_container = QWidget()
        self._cam_container.setStyleSheet("background: #1e1e1e;")
        self._cam_layout = QVBoxLayout(self._cam_container)
        self._cam_layout.setContentsMargins(4, 4, 4, 4)
        self._cam_layout.setSpacing(6)

        cam_header = QLabel("Camera Feeds")
        cam_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #eee; padding: 4px;"
        )
        self._cam_layout.addWidget(cam_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none; background: #1e1e1e;")
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(6)
        self._no_feeds_lbl = QLabel("No videos loaded")
        self._no_feeds_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_feeds_lbl.setStyleSheet("color: #888; font-size: 12px;")
        self._scroll_layout.addWidget(self._no_feeds_lbl)
        self._scroll_layout.addStretch()
        scroll.setWidget(self._scroll_widget)
        self._cam_layout.addWidget(scroll)

        splitter.addWidget(self._cam_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        outer.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_video_feeds(self, video_paths: list[str]):
        """Open cv2 captures for the given video files."""
        self._close_feeds()
        if not _CV2_AVAILABLE:
            return
        self._no_feeds_lbl.setVisible(False)
        for i, path in enumerate(video_paths):
            cap = _cv2.VideoCapture(path)
            if cap.isOpened():
                feed = _VideoFeed(f"Camera {i + 1}", cap)
                self._feeds.append(feed)
                self._scroll_layout.insertWidget(
                    self._scroll_layout.count() - 1, feed
                )
        if self._feeds:
            self._feed_timer.start()
        else:
            self._no_feeds_lbl.setVisible(True)

    def close_video_feeds(self):
        self._feed_timer.stop()
        self._close_feeds()
        self._no_feeds_lbl.setVisible(True)

    def add_trajectory_point(self, x: float, y: float, z: float):
        if _MPL_AVAILABLE and isinstance(self._traj, _TrajectoryCanvas):
            self._traj.add_point(x, y, z)

    def clear_trajectory(self):
        if _MPL_AVAILABLE and isinstance(self._traj, _TrajectoryCanvas):
            self._traj.clear()
            self._traj.refresh()

    def start_timers(self):
        self._plot_timer.start()

    def stop_timers(self):
        self._plot_timer.stop()
        self._feed_timer.stop()

    # ------------------------------------------------------------------

    def _close_feeds(self):
        for feed in self._feeds:
            feed.release()
            self._scroll_layout.removeWidget(feed)
            feed.setParent(None)
        self._feeds.clear()

    def _refresh_plot(self):
        if _MPL_AVAILABLE and isinstance(self._traj, _TrajectoryCanvas):
            self._traj.refresh()

    def _update_feeds(self):
        for feed in self._feeds:
            feed.update_frame()

    def closeEvent(self, event):
        self.stop_timers()
        self._close_feeds()
        super().closeEvent(event)
