"""Optional OpenGL-backed music visual for the integrated performance window."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class _Particle:
    born: float
    lifetime: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    size: float
    color: QtGui.QColor


class MusicPulseView(QtWidgets.QOpenGLWidget):
    """Draw note-triggered particles into a Qt OpenGL framebuffer.

    The first visual is deliberately modest: Qt's painter draws into the
    OpenGL widget, with simple perspective used for particle depth. It keeps
    the existing camera, MediaPipe, MIDI, and score paths independent.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._particles: list[_Particle] = []
        self._pulse_time = -10.0
        self._pulse_color = QtGui.QColor("#39d9d1")
        self._rng = random.Random(0xC0DE)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

    def showEvent(self, event):
        super().showEvent(event)
        if self._particles:
            self._timer.start()
        self.update()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def pulse(self, *, pitch: int, velocity: int, voices: int = 1) -> None:
        """Start one visual burst after an audible note onset, on the UI thread."""
        now = time.perf_counter()
        hue = (int(pitch) * 29) % 360
        color = QtGui.QColor.fromHsv(hue, 185, 255)
        self._pulse_time = now
        self._pulse_color = color
        amount = min(34, 12 + 4 * max(1, int(voices)))
        energy = 0.6 + 0.65 * max(0, min(127, int(velocity))) / 127.0
        for _ in range(amount):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(0.35, 1.15) * energy
            self._particles.append(
                _Particle(
                    born=now,
                    lifetime=self._rng.uniform(0.55, 1.25),
                    x=self._rng.uniform(-0.04, 0.04),
                    y=self._rng.uniform(-0.04, 0.04),
                    z=self._rng.uniform(-0.25, 0.55),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    vz=self._rng.uniform(-0.7, 0.8),
                    size=self._rng.uniform(3.0, 8.0),
                    color=color,
                )
            )
        self._particles = self._particles[-180:]
        if self.isVisible():
            self._timer.start()
        self.update()

    def paintGL(self):
        now = time.perf_counter()
        self._particles = [
            particle
            for particle in self._particles
            if now - particle.born < particle.lifetime
        ]
        width, height = self.width(), self.height()
        center_x, center_y = width * 0.5, height * 0.5
        extent = min(width, height) * 0.42
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        background = QtGui.QLinearGradient(0, 0, width, height)
        background.setColorAt(0, QtGui.QColor("#091825"))
        background.setColorAt(1, QtGui.QColor("#172035"))
        painter.fillRect(self.rect(), background)

        # A small vanishing-point grid gives the bursts visible depth.
        grid_pen = QtGui.QPen(QtGui.QColor(76, 128, 151, 55), 1)
        painter.setPen(grid_pen)
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            painter.drawLine(
                QtCore.QPointF(center_x, center_y),
                QtCore.QPointF(center_x + offset * width, height),
            )
        for fraction in (0.58, 0.69, 0.82, 0.96):
            y = center_y + (height - center_y) * fraction
            painter.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(width, y))

        age = now - self._pulse_time
        if 0 <= age < 0.65:
            fade = 1.0 - age / 0.65
            glow = QtGui.QRadialGradient(center_x, center_y, extent * 0.8)
            inner = QtGui.QColor(self._pulse_color)
            inner.setAlpha(int(95 * fade))
            outer = QtGui.QColor(self._pulse_color)
            outer.setAlpha(0)
            glow.setColorAt(0, inner)
            glow.setColorAt(1, outer)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(
                QtCore.QPointF(center_x, center_y), extent * 0.8, extent * 0.8
            )
            ring = QtGui.QColor(self._pulse_color)
            ring.setAlpha(int(190 * fade))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(ring, 2.5))
            radius = 18 + age * extent * 1.35
            painter.drawEllipse(QtCore.QPointF(center_x, center_y), radius, radius)

        painter.setPen(QtCore.Qt.NoPen)
        for particle in sorted(self._particles, key=lambda item: item.z):
            elapsed = now - particle.born
            z = particle.z + elapsed * particle.vz
            scale = 1.0 / max(0.4, 1.0 + z * 0.45)
            x = center_x + (particle.x + elapsed * particle.vx) * extent * scale
            y = center_y + (particle.y + elapsed * particle.vy) * extent * scale
            radius = particle.size * scale
            color = QtGui.QColor(particle.color)
            color.setAlpha(int(235 * (1.0 - elapsed / particle.lifetime)))
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)

        painter.setPen(QtGui.QColor("#c4d8e6"))
        painter.drawText(18, 30, "MUSIC VISUALS  |  NOTE PULSES")
        if not self._particles and age >= 0.65:
            painter.setPen(QtGui.QColor("#7d9aac"))
            painter.drawText(
                self.rect(), QtCore.Qt.AlignCenter, "Waiting for the next note..."
            )
        painter.end()
        if not self._particles and age >= 0.65:
            self._timer.stop()
