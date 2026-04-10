"""OrientationCube: interactive 3-D orientation gizmo drawn with QPainter.

No VTK or OpenGL required — purely Qt/numpy.

Behaviour
---------
- Drag anywhere on the widget to rotate the cube (trackball style).
- Double-click to reset to the default isometric view.
- Click a face label to snap to that standard anatomical orientation and emit
  ``view_snapped(axis, direction)`` so callers can respond (e.g. switch slice
  plane).

Signals
-------
orientation_changed(ndarray)  — emits the current 3x3 rotation matrix on
                                every drag update.
view_snapped(str, str)        — emits (axis, direction) when a face is clicked.
                                axis   ∈ {"X", "Y", "Z"}
                                direction ∈ {"pos", "neg"}
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QPolygon,
)
from PySide6.QtWidgets import QWidget


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

# 8 vertices of a unit cube  [-1, 1]^3
_VERTS = np.array([
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7
], dtype=float)

# Each face: (vertex indices CCW, outward normal, label, base QColor)
_FACES = [
    ([4, 5, 6, 7], np.array([ 0,  0,  1], float), "Top",   QColor(80,  185,  80)),
    ([3, 2, 1, 0], np.array([ 0,  0, -1], float), "Bot",   QColor(210,  80,  80)),
    ([1, 5, 6, 2], np.array([ 1,  0,  0], float), "Right", QColor(210, 100,  80)),
    ([4, 0, 3, 7], np.array([-1,  0,  0], float), "Left",  QColor( 80, 210, 210)),
    ([3, 7, 6, 2], np.array([ 0,  1,  0], float), "Front", QColor( 80,  80, 210)),
    ([0, 1, 5, 4], np.array([ 0, -1,  0], float), "Back",  QColor(210, 175,  80)),
]

# Unit axes for snapping
_SNAP_AXES: dict[str, tuple[str, str]] = {
    "Top":   ("Z", "pos"),  "Bot":   ("Z", "neg"),
    "Right": ("X", "pos"),  "Left":  ("X", "neg"),
    "Front": ("Y", "pos"),  "Back":  ("Y", "neg"),
}

# Canonical "nice" starting rotation — slight isometric tilt so 3 faces show
def _default_rot() -> np.ndarray:
    rx = _rot_x(np.radians(20))
    ry = _rot_y(np.radians(-35))
    return ry @ rx


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula → 3x3 matrix."""
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [    0, -axis[2],  axis[1]],
        [ axis[2],     0, -axis[0]],
        [-axis[1],  axis[0],     0],
    ], float)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _orthonorm(R: np.ndarray) -> np.ndarray:
    """Keep the rotation matrix numerically clean via SVD."""
    u, _, vh = np.linalg.svd(R)
    return u @ vh


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OrientationCube(QWidget):
    """Interactive orientation cube gizmo.

    Drop it into any layout or overlay it on top of a viewer panel.
    """

    orientation_changed = Signal(object)   # 3x3 ndarray
    view_snapped        = Signal(str, str) # (axis, direction)

    _VIEW_DIR = np.array([0.0, 0.0, 1.0])  # camera looks in +Z

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rot       = _default_rot()
        self._last_pos: QPoint | None = None

        # hit-test: label → face index, populated in paintEvent
        self._face_rects: list[tuple[QRect, str]] = []

        self.setMinimumSize(100, 100)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to rotate  •  Double-click to reset")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rotation(self) -> np.ndarray:
        return self._rot.copy()

    def set_rotation(self, rot: np.ndarray) -> None:
        self._rot = _orthonorm(np.asarray(rot, float))
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event):
        size = min(self.width(), self.height())
        cx   = self.width()  / 2
        cy   = self.height() / 2
        sc   = size * 0.36          # cube half-size in pixels

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent round background
        painter.setBrush(QBrush(QColor(15, 15, 15, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, self.width() - 4, self.height() - 4)

        # --- Project vertices (orthographic) ---
        rot_v = _VERTS @ self._rot.T          # shape (8, 3)
        sx = cx + sc * rot_v[:, 0]
        sy = cy - sc * rot_v[:, 1]            # screen Y flipped
        sz =             rot_v[:, 2]          # depth

        # --- Build face draw list ---
        face_list = []
        for vidx, normal, label, color in _FACES:
            rn    = self._rot @ normal
            dot   = float(np.dot(rn, self._VIEW_DIR))
            depth = float(np.mean(sz[vidx]))
            face_list.append((depth, vidx, label, color, dot, rn))

        # Painter's algorithm: back → front
        face_list.sort(key=lambda f: f[0])

        self._face_rects.clear()
        font = QFont()
        font.setBold(True)
        font.setPointSize(max(6, size // 18))

        for depth, vidx, label, color, dot, rn in face_list:
            pts = [(sx[i], sy[i]) for i in vidx]
            poly = QPolygon([QPoint(int(x), int(y)) for x, y in pts])

            if dot > 0.0:          # ---- visible face ----
                bright = 0.45 + 0.55 * dot
                fc = QColor(
                    int(color.red()   * bright),
                    int(color.green() * bright),
                    int(color.blue()  * bright),
                    230,
                )
                painter.setBrush(QBrush(fc))
                painter.setPen(QPen(QColor(30, 30, 30, 200), 1.2))
                painter.drawPolygon(poly)

                # Label
                fcx = int(sum(p[0] for p in pts) / 4)
                fcy = int(sum(p[1] for p in pts) / 4)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 255, 230))
                fm  = painter.fontMetrics()
                tw  = fm.horizontalAdvance(label)
                th  = fm.height()
                painter.drawText(fcx - tw // 2, fcy + th // 3, label)

                # Store hit rect for click detection
                half = max(14, size // 7)
                self._face_rects.append(
                    (QRect(fcx - half, fcy - half, half * 2, half * 2), label)
                )

            else:                  # ---- back face — edges only ----
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(70, 70, 70, 120), 0.8))
                painter.drawPolygon(poly)

        # Axis lines at cube center
        self._draw_axes(painter, cx, cy, sc * 0.55, rot_v)

        painter.end()

    def _draw_axes(self, p: QPainter, cx, cy, sc, rot_v):
        # draw X/Y/Z arrows through cube centre
        axes = [
            (np.array([1, 0, 0], float), QColor(220,  60,  60)),  # X red
            (np.array([0, 1, 0], float), QColor( 60, 200,  60)),  # Y green
            (np.array([0, 0, 1], float), QColor( 60, 120, 220)),  # Z blue
        ]
        for ax, col in axes:
            rotated = self._rot @ ax
            ex = int(cx + sc * rotated[0])
            ey = int(cy - sc * rotated[1])
            p.setPen(QPen(col, 1.5))
            p.drawLine(int(cx), int(cy), ex, ey)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        pos = event.position().toPoint()
        dx  = pos.x() - self._last_pos.x()
        dy  = pos.y() - self._last_pos.y()
        self._last_pos = pos

        dist = np.hypot(dx, dy)
        if dist < 0.5:
            return

        # Screen-space drag → world-space rotation axis
        # Drag right → rotate around screen-Y (+Y), drag up → rotate around screen-X (+X)
        axis = np.array([-dy, dx, 0.0])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return
        axis /= norm

        angle = dist * 0.012   # radians per pixel — feel
        R_delta = _rodrigues(axis, angle)
        self._rot = _orthonorm(R_delta @ self._rot)
        self.update()
        self.orientation_changed.emit(self._rot.copy())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Was it a click (no meaningful drag)?
            if self._last_pos is not None:
                pos = event.position().toPoint()
                if (abs(pos.x() - self._last_pos.x()) < 4 and
                        abs(pos.y() - self._last_pos.y()) < 4):
                    self._handle_click(pos)
            self._last_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, _event):
        self._rot = _default_rot()
        self.update()
        self.orientation_changed.emit(self._rot.copy())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_click(self, pos: QPoint):
        for rect, label in self._face_rects:
            if rect.contains(pos):
                axis, direction = _SNAP_AXES[label]
                self._snap(label)
                self.view_snapped.emit(axis, direction)
                return

    def _snap(self, label: str):
        """Animate-free snap to a standard face view."""
        targets = {
            "Top":   np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], float),
            "Bot":   np.array([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], float),
            "Front": np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]], float),
            "Back":  np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]], float),
            "Right": np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]], float),
            "Left":  np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]], float),
        }
        if label in targets:
            self._rot = targets[label]
            self.update()
            self.orientation_changed.emit(self._rot.copy())
