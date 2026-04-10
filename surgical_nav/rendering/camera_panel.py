"""CameraPanel: live camera feed display for up to 5 cameras.

Uses OpenCV VideoCapture to probe and display available cameras.
Falls back gracefully if OpenCV is not installed or no cameras are found.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


_FEED_IMAGE_HEIGHT = 120   # fixed px — prevents layout from growing on first frames


class _CameraFeed(QWidget):
    """Single camera feed: title label + live image label."""

    def __init__(self, index: int, cap):
        super().__init__()
        import cv2  # already confirmed available before constructing
        self._cv2 = cv2
        self._cap = cap
        self._index = index

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title = QLabel(f"Camera {index}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 11px; color: #ccc;")
        layout.addWidget(title)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setFixedHeight(_FEED_IMAGE_HEIGHT)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._image_label.setStyleSheet("background: #111; border: 1px solid #444;")
        layout.addWidget(self._image_label)

        # Fix the total widget height so the scroll layout never shifts
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def update_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            return
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
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


class CameraPanel(QWidget):
    """Panel showing live feeds from up to *max_cameras* cameras.

    Placed to the right of the four image outputs in the viewer area.
    """

    def __init__(self, max_cameras: int = 5, fps: int = 30):
        super().__init__()
        self._feeds: list[_CameraFeed] = []

        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1e1e1e;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        header = QLabel("Camera Feeds")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #eee; padding: 4px;"
        )
        outer.addWidget(header)

        # Scrollable area for the feed widgets
        self._feed_container = QWidget()
        self._feed_layout = QVBoxLayout(self._feed_container)
        self._feed_layout.setContentsMargins(0, 0, 0, 0)
        self._feed_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._feed_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none;")
        outer.addWidget(scroll)

        self._status_label = QLabel("Detecting cameras…")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        self._feed_layout.addWidget(self._status_label)
        self._feed_layout.addStretch()

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // fps)
        self._timer.timeout.connect(self._update_frames)

        # Probe cameras in background to avoid blocking the UI
        from PySide6.QtCore import QTimer as _QTimer
        _QTimer.singleShot(100, lambda: self._probe_cameras(max_cameras))

    # ------------------------------------------------------------------

    def _probe_cameras(self, max_cameras: int):
        try:
            import cv2
        except ImportError:
            self._status_label.setText("OpenCV not installed")
            return

        found = 0
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                feed = _CameraFeed(i, cap)
                self._feeds.append(feed)
                self._feed_layout.insertWidget(self._feed_layout.count() - 1, feed)
                found += 1

        if found > 0:
            self._status_label.setVisible(False)
            self._timer.start()
        else:
            self._status_label.setText("No cameras detected")

    def _update_frames(self):
        for feed in self._feeds:
            feed.update_frame()

    def closeEvent(self, event):
        self._timer.stop()
        for feed in self._feeds:
            feed.release()
        super().closeEvent(event)
