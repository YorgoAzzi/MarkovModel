"""Optional note-driven effects drawn over the live camera in an OpenGL widget."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from hand_movement import HAND_CONNECTIONS
except ModuleNotFoundError:
    from .hand_movement import HAND_CONNECTIONS


@dataclass(frozen=True)
class _Particle:
    born: float
    lifetime: float
    x: float  # normalized camera coordinates
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    size: float
    color: QtGui.QColor


@dataclass(frozen=True)
class _Pulse:
    born: float
    x: float
    y: float
    color: QtGui.QColor


class MusicPulseView(QtWidgets.QOpenGLWidget):
    """Paint camera frames and hand-anchored particles in one OpenGL surface.

    Qt's painter handles this first perspective effect within the OpenGL
    framebuffer. The camera and MediaPipe processing remain unchanged.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self._camera_image = None
        self._hand_overlays = ()
        self._chord_label = "Chord: waiting"
        self._effects_enabled = False
        self._hand_positions: tuple[tuple[float, float], ...] = ()
        self._hand_light_level: float | None = None
        self._particles: list[_Particle] = []
        self._pulses: list[_Pulse] = []
        self._rng = random.Random(0xC0DE)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

    def showEvent(self, event):
        super().showEvent(event)
        if self._particles or self._pulses:
            self._timer.start()
        self.update()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def set_camera_image(self, image: QtGui.QImage) -> None:
        """Receive an owned QImage copied from the most recent camera frame."""
        self._camera_image = image
        self.update()

    def set_hand_overlays(self, overlays) -> None:
        """Receive normalized MediaPipe points and finger states for drawing."""
        self._hand_overlays = tuple(overlays)
        self.update()

    def set_chord_label(self, label: str) -> None:
        label = str(label)
        if label != self._chord_label:
            self._chord_label = label
            self.update()

    def set_effects_enabled(self, enabled: bool) -> None:
        self._effects_enabled = bool(enabled)
        if not enabled:
            self._particles.clear()
            self._pulses.clear()
            self._timer.stop()
        self.update()

    def set_hand_positions(self, positions, *, distance_level=None) -> None:
        """Use MediaPipe palm centers and the shared volume control."""
        self._hand_positions = tuple(
            (
                max(0.0, min(1.0, float(x))),
                max(0.0, min(1.0, float(y))),
            )
            for x, y in positions
            if math.isfinite(float(x)) and math.isfinite(float(y))
        )
        self._hand_light_level = (
            max(0.0, min(1.0, float(distance_level)))
            if len(self._hand_positions) == 2
            and distance_level is not None
            and math.isfinite(float(distance_level))
            else None
        )
        self.update()

    def _paint_hand_light(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        """Draw a glowing connection whose size follows hand-distance volume."""
        if self._hand_light_level is None or len(self._hand_positions) != 2:
            return
        level = self._hand_light_level
        first, second = (
            QtCore.QPointF(
                target.left() + x * target.width(),
                target.top() + y * target.height(),
            )
            for x, y in self._hand_positions
        )
        center = QtCore.QPointF(
            (first.x() + second.x()) / 2,
            (first.y() + second.y()) / 2,
        )
        scale = min(target.width(), target.height())
        radius = scale * (0.075 + 0.19 * level)

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(
            QtGui.QColor(255, 225, 155, round(20 + 48 * level)),
            8 + 18 * level, QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
        ))
        painter.drawLine(first, second)
        painter.setPen(QtGui.QPen(
            QtGui.QColor(255, 245, 205, round(72 + 95 * level)),
            2.5 + 4.5 * level, QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
        ))
        painter.drawLine(first, second)

        glow = QtGui.QRadialGradient(center, radius)
        glow.setColorAt(0, QtGui.QColor(255, 251, 225, round(175 + 65 * level)))
        glow.setColorAt(0.18, QtGui.QColor(255, 229, 145, round(85 + 100 * level)))
        glow.setColorAt(0.55, QtGui.QColor(255, 195, 100, round(30 + 55 * level)))
        glow.setColorAt(1, QtGui.QColor(255, 175, 80, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, radius, radius)

    def pulse(self, *, pitch: int, velocity: int, voices: int = 1) -> None:
        """Burst at each tracked hand when generated music begins a note."""
        if not self._effects_enabled:
            return
        now = time.perf_counter()
        color = QtGui.QColor.fromHsv((int(pitch) * 29) % 360, 185, 255)
        anchors = self._hand_positions or ((0.5, 0.5),)
        count = min(34, 12 + 4 * max(1, int(voices)))
        energy = 0.6 + 0.65 * max(0, min(127, int(velocity))) / 127.0
        for anchor_x, anchor_y in anchors:
            self._pulses.append(_Pulse(now, anchor_x, anchor_y, color))
            for _ in range(max(8, count // len(anchors))):
                angle = self._rng.uniform(0, 2 * math.pi)
                speed = self._rng.uniform(0.35, 1.15) * energy
                self._particles.append(
                    _Particle(
                        born=now,
                        lifetime=self._rng.uniform(0.55, 1.25),
                        x=anchor_x + self._rng.uniform(-0.01, 0.01),
                        y=anchor_y + self._rng.uniform(-0.01, 0.01),
                        z=self._rng.uniform(-0.25, 0.55),
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed,
                        vz=self._rng.uniform(-0.7, 0.8),
                        size=self._rng.uniform(3.0, 8.0),
                        color=color,
                    )
                )
        self._particles = self._particles[-180:]
        self._pulses = self._pulses[-12:]
        if self.isVisible():
            self._timer.start()
        self.update()

    def paintGL(self):
        now = time.perf_counter()
        self._particles = [
            item for item in self._particles if now - item.born < item.lifetime
        ]
        self._pulses = [item for item in self._pulses if now - item.born < 0.65]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#111820"))

        image = self._camera_image
        if image is None or image.isNull():
            target = QtCore.QRectF(self.rect())
            painter.setPen(QtGui.QColor("#d9e3ea"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Waiting for camera...")
        else:
            scale = min(self.width() / image.width(), self.height() / image.height())
            image_width = image.width() * scale
            image_height = image.height() * scale
            target = QtCore.QRectF(
                (self.width() - image_width) / 2,
                (self.height() - image_height) / 2,
                image_width,
                image_height,
            )
            painter.drawImage(target, image)

        painter.setClipRect(target)
        self._paint_hand_light(painter, target)
        for points, finger_states, color_name in self._hand_overlays:
            if len(points) != 21:
                continue
            pixels = [
                QtCore.QPointF(
                    target.left() + x * target.width(),
                    target.top() + y * target.height(),
                )
                for x, y in points
            ]
            color = QtGui.QColor(color_name)
            painter.setPen(QtGui.QPen(color, 2.2))
            for start, end in HAND_CONNECTIONS:
                painter.drawLine(pixels[start], pixels[end])
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor("#ffffff"))
            for point in pixels:
                painter.drawEllipse(point, 2.7, 2.7)
            for extended, tip_index in zip(finger_states, (4, 8, 12, 16, 20)):
                painter.setBrush(QtGui.QColor("#4ae479" if extended else "#ed6666"))
                painter.drawEllipse(pixels[tip_index], 5.5, 5.5)

        painter.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.DemiBold))
        label_width = min(
            target.width() - 16,
            painter.fontMetrics().horizontalAdvance(self._chord_label) + 24,
        )
        label_box = QtCore.QRectF(
            target.center().x() - label_width / 2,
            target.top() + 10,
            label_width,
            29,
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(10, 20, 28, 185))
        painter.drawRoundedRect(label_box, 7, 7)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(label_box, QtCore.Qt.AlignCenter, self._chord_label)

        extent = min(target.width(), target.height()) * 0.42
        for pulse in (self._pulses if self._effects_enabled else ()):
            age = now - pulse.born
            fade = 1.0 - age / 0.65
            center = QtCore.QPointF(
                target.left() + pulse.x * target.width(),
                target.top() + pulse.y * target.height(),
            )
            glow = QtGui.QRadialGradient(center, extent * 0.45)
            inner = QtGui.QColor(pulse.color)
            inner.setAlpha(int(95 * fade))
            outer = QtGui.QColor(pulse.color)
            outer.setAlpha(0)
            glow.setColorAt(0, inner)
            glow.setColorAt(1, outer)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(center, extent * 0.45, extent * 0.45)
            ring = QtGui.QColor(pulse.color)
            ring.setAlpha(int(190 * fade))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(ring, 2.5))
            radius = 18 + age * extent * 1.35
            painter.drawEllipse(center, radius, radius)

        painter.setPen(QtCore.Qt.NoPen)
        for particle in (
            sorted(self._particles, key=lambda item: item.z)
            if self._effects_enabled else ()
        ):
            elapsed = now - particle.born
            z = particle.z + elapsed * particle.vz
            depth_scale = 1.0 / max(0.4, 1.0 + z * 0.45)
            x = target.left() + particle.x * target.width()
            y = target.top() + particle.y * target.height()
            x += elapsed * particle.vx * extent * depth_scale
            y += elapsed * particle.vy * extent * depth_scale
            radius = particle.size * depth_scale
            color = QtGui.QColor(particle.color)
            color.setAlpha(int(235 * (1.0 - elapsed / particle.lifetime)))
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)
        painter.end()
        if not self._particles and not self._pulses:
            self._timer.stop()
