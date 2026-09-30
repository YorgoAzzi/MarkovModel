"""Colored-blob tracking and camera motion feature extraction.

OpenCV is imported only when the detector is constructed, which keeps the
timestamped motion math independently testable without a camera or GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalized(value: float, maximum: float, dead_zone: float) -> float:
    if maximum <= dead_zone:
        raise ValueError("maximum must be greater than dead_zone")
    return _clamp((float(value) - dead_zone) / (maximum - dead_zone), 0.0, 1.0)


def movement_to_music_controls(speed: float, y: float) -> dict[str, float]:
    """Map movement speed and vertical position to the two music controls."""
    return {
        "density": _clamp(speed, 0.0, 1.0),
        # Image coordinates grow downward, so the top of the frame is louder.
        "volume": 1.0 - _clamp(y, 0.0, 1.0),
    }


@dataclass(frozen=True)
class BlobDetection:
    x: float
    y: float
    area_ratio: float
    pixel_x: int
    pixel_y: int


@dataclass(frozen=True)
class MotionSample:
    timestamp: float
    detected: bool
    x: float
    y: float
    speed: float
    acceleration: float
    area_ratio: float


class MotionFeatureExtractor:
    """Turns normalized blob positions into stable speed and acceleration."""

    def __init__(
        self,
        *,
        max_speed: float = 1.5,
        max_acceleration: float = 8.0,
        speed_dead_zone: float = 0.03,
        acceleration_dead_zone: float = 0.15,
        velocity_time_constant: float = 0.08,
        acceleration_time_constant: float = 0.12,
        dropout_reset_seconds: float = 0.25,
        dropout_decay_seconds: float = 0.18,
    ):
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.speed_dead_zone = float(speed_dead_zone)
        self.acceleration_dead_zone = float(acceleration_dead_zone)
        self.velocity_time_constant = max(1e-6, float(velocity_time_constant))
        self.acceleration_time_constant = max(1e-6, float(acceleration_time_constant))
        self.dropout_reset_seconds = max(0.0, float(dropout_reset_seconds))
        self.dropout_decay_seconds = max(1e-6, float(dropout_decay_seconds))

        self._last_timestamp: Optional[float] = None
        self._last_position: Optional[np.ndarray] = None
        self._velocity = np.zeros(2, dtype=float)
        self._acceleration = np.zeros(2, dtype=float)
        self._speed_value = 0.0
        self._acceleration_value = 0.0
        self._last_x = 0.5
        self._last_y = 0.5

    @staticmethod
    def _alpha(dt: float, time_constant: float) -> float:
        return 1.0 - math.exp(-max(0.0, dt) / time_constant)

    def update(
        self,
        x: float,
        y: float,
        timestamp: float,
        area_ratio: float = 0.0,
    ) -> MotionSample:
        position = np.array([_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0)], dtype=float)
        timestamp = float(timestamp)

        if self._last_timestamp is None or self._last_position is None:
            self._velocity.fill(0.0)
            self._acceleration.fill(0.0)
        else:
            dt = max(1e-4, timestamp - self._last_timestamp)
            if dt > self.dropout_reset_seconds:
                self._velocity.fill(0.0)
                self._acceleration.fill(0.0)
            else:
                raw_velocity = (position - self._last_position) / dt
                velocity_alpha = self._alpha(dt, self.velocity_time_constant)
                previous_velocity = self._velocity.copy()
                self._velocity += velocity_alpha * (raw_velocity - self._velocity)

                raw_acceleration = (self._velocity - previous_velocity) / dt
                acceleration_alpha = self._alpha(dt, self.acceleration_time_constant)
                self._acceleration += acceleration_alpha * (
                    raw_acceleration - self._acceleration
                )

        speed_magnitude = float(np.linalg.norm(self._velocity))
        acceleration_magnitude = float(np.linalg.norm(self._acceleration))
        self._speed_value = _normalized(
            speed_magnitude, self.max_speed, self.speed_dead_zone
        )
        self._acceleration_value = _normalized(
            acceleration_magnitude,
            self.max_acceleration,
            self.acceleration_dead_zone,
        )
        self._last_position = position
        self._last_timestamp = timestamp
        self._last_x, self._last_y = float(position[0]), float(position[1])
        return MotionSample(
            timestamp=timestamp,
            detected=True,
            x=self._last_x,
            y=self._last_y,
            speed=self._speed_value,
            acceleration=self._acceleration_value,
            area_ratio=max(0.0, float(area_ratio)),
        )

    def missing(self, timestamp: float) -> MotionSample:
        timestamp = float(timestamp)
        if self._last_timestamp is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp - self._last_timestamp)
        decay = math.exp(-dt / self.dropout_decay_seconds)
        self._velocity *= decay
        self._acceleration *= decay
        self._speed_value *= decay
        self._acceleration_value *= decay
        if dt > self.dropout_reset_seconds:
            self._last_position = None
        return MotionSample(
            timestamp=timestamp,
            detected=False,
            x=self._last_x,
            y=self._last_y,
            speed=self._speed_value,
            acceleration=self._acceleration_value,
            area_ratio=0.0,
        )


class ColorBlobDetector:
    """Finds the largest blob near a calibrated HSV hue."""

    def __init__(
        self,
        *,
        hue: Optional[int] = None,
        hue_tolerance: int = 10,
        saturation_min: int = 80,
        value_min: int = 60,
        min_area_ratio: float = 0.001,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for camera tracking. Run: pip install -r requirements.txt"
            ) from exc
        self.cv2 = cv2
        self.hue = None if hue is None else int(hue) % 180
        self.hue_tolerance = max(1, int(hue_tolerance))
        self.saturation_min = max(0, min(255, int(saturation_min)))
        self.value_min = max(0, min(255, int(value_min)))
        self.min_area_ratio = max(0.0, float(min_area_ratio))
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    @property
    def calibrated(self) -> bool:
        return self.hue is not None

    def calibrate_pixel(self, bgr_pixel) -> int:
        pixel = np.asarray(bgr_pixel, dtype=np.uint8).reshape(1, 1, 3)
        hsv = self.cv2.cvtColor(pixel, self.cv2.COLOR_BGR2HSV)
        self.hue = int(hsv[0, 0, 0])
        return self.hue

    def _mask_for_hue(self, hsv_frame):
        low = self.hue - self.hue_tolerance
        high = self.hue + self.hue_tolerance

        def segment(start, end):
            return self.cv2.inRange(
                hsv_frame,
                np.array([start, self.saturation_min, self.value_min], dtype=np.uint8),
                np.array([end, 255, 255], dtype=np.uint8),
            )

        if low < 0:
            mask = self.cv2.bitwise_or(segment(0, high), segment(180 + low, 179))
        elif high > 179:
            mask = self.cv2.bitwise_or(segment(low, 179), segment(0, high - 180))
        else:
            mask = segment(low, high)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, self.kernel)
        return self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, self.kernel)

    def detect(self, frame) -> tuple[Optional[BlobDetection], np.ndarray]:
        height, width = frame.shape[:2]
        if not self.calibrated:
            return None, np.zeros((height, width), dtype=np.uint8)
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        mask = self._mask_for_hue(hsv)
        contours, _ = self.cv2.findContours(
            mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, mask
        contour = max(contours, key=self.cv2.contourArea)
        area = float(self.cv2.contourArea(contour))
        area_ratio = area / max(1.0, float(width * height))
        moments = self.cv2.moments(contour)
        if area_ratio < self.min_area_ratio or moments["m00"] <= 0:
            return None, mask
        pixel_x = int(moments["m10"] / moments["m00"])
        pixel_y = int(moments["m01"] / moments["m00"])
        return (
            BlobDetection(
                x=pixel_x / max(1, width - 1),
                y=pixel_y / max(1, height - 1),
                area_ratio=area_ratio,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
            ),
            mask,
        )

