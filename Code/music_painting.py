"""A persistent artwork built from performed notes and tracked hand position."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1000


@dataclass(frozen=True)
class PaintMark:
    pitch: int
    velocity: int
    duration_ticks: int
    voices: int
    x: float
    y: float


class MusicPaintingView(QtWidgets.QOpenGLWidget):
    """Retain brush marks in a QImage and display them on an OpenGL surface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = QtGui.QImage(
            CANVAS_WIDTH, CANVAS_HEIGHT, QtGui.QImage.Format_ARGB32_Premultiplied
        )
        self._marks: list[PaintMark] = []
        self._last_point = None
        self._rng = random.Random(0xA27)
        self._paint_background()

    @property
    def mark_count(self) -> int:
        return len(self._marks)

    def _paint_background(self) -> None:
        painter = QtGui.QPainter(self._canvas)
        gradient = QtGui.QLinearGradient(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
        gradient.setColorAt(0, QtGui.QColor("#f8f4eb"))
        gradient.setColorAt(1, QtGui.QColor("#e8e8e2"))
        painter.fillRect(self._canvas.rect(), gradient)
        painter.end()

    def add_note(
        self,
        *,
        pitch: int,
        velocity: int,
        duration_ticks: int,
        voices: int,
        position: tuple[float, float],
    ) -> None:
        """Paint one lasting stroke when the generated music begins a note."""
        raw_x, raw_y = position
        x = float(raw_x)
        y = float(raw_y)
        if not math.isfinite(x) or not math.isfinite(y):
            return
        x = max(0.04, min(0.96, x))
        y = max(0.04, min(0.96, y))
        mark = PaintMark(
            pitch=int(pitch),
            velocity=max(0, min(127, int(velocity))),
            duration_ticks=max(1, int(duration_ticks)),
            voices=max(1, int(voices)),
            x=x,
            y=y,
        )
        self._marks.append(mark)
        self._draw_mark(mark)
        self.update()

    def _draw_mark(self, mark: PaintMark) -> None:
        painter = QtGui.QPainter(self._canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(mark.x * CANVAS_WIDTH, mark.y * CANVAS_HEIGHT)
        hue = (mark.pitch % 12) * 30
        color = QtGui.QColor.fromHsv(hue, 175, 205)
        width = 7.0 + mark.velocity * 0.14
        length = 22.0 + min(64, mark.duration_ticks) * 2.15

        previous = self._last_point
        if previous is not None:
            distance = math.hypot(center.x() - previous.x(), center.y() - previous.y())
        else:
            distance = 0.0
        if 18.0 < distance < 380.0:
            start = previous
            end = center
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            bend = min(42.0, distance * 0.2)
            control = QtCore.QPointF(
                (start.x() + end.x()) / 2 - dy / distance * bend,
                (start.y() + end.y()) / 2 + dx / distance * bend,
            )
        else:
            angle = math.radians((mark.pitch * 37 + self.mark_count * 19) % 360)
            start = QtCore.QPointF(
                center.x() - math.cos(angle) * length * 0.55,
                center.y() - math.sin(angle) * length * 0.55,
            )
            end = QtCore.QPointF(
                center.x() + math.cos(angle) * length * 0.55,
                center.y() + math.sin(angle) * length * 0.55,
            )
            control = QtCore.QPointF(
                center.x() - math.sin(angle) * length * 0.32,
                center.y() + math.cos(angle) * length * 0.32,
            )

        path = QtGui.QPainterPath(start)
        path.quadTo(control, end)
        soft = QtGui.QColor(color)
        soft.setAlpha(40)
        painter.setPen(
            QtGui.QPen(soft, width * 2.6, QtCore.Qt.SolidLine,
                       QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        )
        painter.drawPath(path)
        ink = QtGui.QColor(color)
        ink.setAlpha(90 + mark.velocity // 3)
        painter.setPen(
            QtGui.QPen(ink, width, QtCore.Qt.SolidLine,
                       QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        )
        painter.drawPath(path)

        painter.setPen(QtCore.Qt.NoPen)
        wash = QtGui.QColor(color)
        wash.setAlpha(38)
        painter.setBrush(wash)
        for index in range(min(4, mark.voices)):
            angle = 2 * math.pi * index / max(1, min(4, mark.voices))
            offset = 9 + index * 7
            painter.drawEllipse(
                QtCore.QPointF(
                    center.x() + math.cos(angle) * offset,
                    center.y() + math.sin(angle) * offset,
                ),
                length * 0.22,
                width * 1.5,
            )
        speckle = QtGui.QColor(color)
        speckle.setAlpha(125)
        painter.setBrush(speckle)
        for _ in range(min(9, 3 + mark.voices)):
            angle = self._rng.uniform(0, 2 * math.pi)
            radius = self._rng.uniform(8, length * 0.55)
            size = self._rng.uniform(2.5, 6.0)
            painter.drawEllipse(
                QtCore.QPointF(
                    center.x() + math.cos(angle) * radius,
                    center.y() + math.sin(angle) * radius,
                ),
                size,
                size,
            )
        painter.end()
        self._last_point = center

    def save_png(self, path: str | Path) -> bool:
        return self._canvas.save(str(path), "PNG")

    def paintGL(self):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#26323b"))
        scale = min(
            self.width() / self._canvas.width(),
            self.height() / self._canvas.height(),
        )
        width = self._canvas.width() * scale
        height = self._canvas.height() * scale
        target = QtCore.QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )
        painter.drawImage(target, self._canvas)
        painter.end()
