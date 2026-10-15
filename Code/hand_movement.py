"""MediaPipe hand tracking and gesture features for movement-controlled music."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence

import numpy as np


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CHORD_GESTURES = {
    (False, True, False, False, False): 1,
    (False, True, True, False, False): 2,
    (False, True, True, True, False): 3,
    (False, True, True, True, True): 4,
    (True, True, True, True, True): 5,
    (True, False, False, False, False): 6,
    (True, True, False, False, False): 7,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandObservation:
    handedness: str
    confidence: float
    landmarks: tuple[Point3D, ...]
    world_landmarks: tuple[Point3D, ...] = ()


@dataclass(frozen=True)
class HandTrackingResult:
    timestamp: float
    hands: tuple[HandObservation, ...]


@dataclass(frozen=True)
class LeftHandControls:
    detected: bool
    melodic_activity: float
    tempo_bpm: float
    articulation: float
    palm_x: float
    palm_y: float
    horizontal_speed: float


@dataclass(frozen=True)
class ChordGesture:
    degree: int
    quality: str
    tilt_degrees: float
    fingers: tuple[bool, bool, bool, bool, bool]


def palm_center(hand: HandObservation) -> tuple[float, float]:
    """Average the wrist and four knuckles to obtain a stable control point."""
    points = [hand.landmarks[index] for index in (0, 5, 9, 13, 17)]
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def palm_width(hand: HandObservation) -> float:
    """Apparent palm width in normalized image coordinates (a depth proxy)."""
    index_knuckle = hand.landmarks[5]
    pinky_knuckle = hand.landmarks[17]
    return math.hypot(
        index_knuckle.x - pinky_knuckle.x,
        index_knuckle.y - pinky_knuckle.y,
    )


def height_to_unit(y: float, top: float = 0.25, bottom: float = 0.75) -> float:
    """Map image height to 0..1, with the top of the frame producing 1."""
    if bottom <= top:
        raise ValueError("bottom must be greater than top")
    return _clamp((bottom - float(y)) / (bottom - top), 0.0, 1.0)


def height_to_register(y: float, top: float = 0.25, bottom: float = 0.75) -> float:
    """Map image height to the engine register range -1..1."""
    return 2.0 * height_to_unit(y, top, bottom) - 1.0


def hand_tilt_degrees(hand: HandObservation) -> float:
    """Return palm-direction lean: upright=0, left negative, right positive."""
    wrist = hand.landmarks[0]
    middle_knuckle = hand.landmarks[9]
    dx = middle_knuckle.x - wrist.x
    dy_up = wrist.y - middle_knuckle.y
    return math.degrees(math.atan2(dx, dy_up))


def _distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _joint_angle(a: Point3D, joint: Point3D, c: Point3D) -> float:
    first = np.array([a.x - joint.x, a.y - joint.y, a.z - joint.z], dtype=float)
    second = np.array([c.x - joint.x, c.y - joint.y, c.z - joint.z], dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return 0.0
    cosine = _clamp(float(np.dot(first, second)) / denominator, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def hand_openness(hand: HandObservation) -> float:
    """Estimate continuous hand openness from rotation-independent finger shape."""
    points: Sequence[Point3D] = hand.world_landmarks or hand.landmarks
    if len(points) != 21:
        raise ValueError("A hand observation must contain 21 landmarks")

    palm_span = max(1e-9, _distance(points[5], points[17]))
    thumb_path = sum(_distance(points[a], points[b]) for a, b in ((1, 2), (2, 3), (3, 4)))
    thumb_straightness = _distance(points[1], points[4]) / max(1e-9, thumb_path)
    palm_points = [points[index] for index in (0, 5, 9, 13, 17)]
    center = Point3D(
        sum(point.x for point in palm_points) / len(palm_points),
        sum(point.y for point in palm_points) / len(palm_points),
        sum(point.z for point in palm_points) / len(palm_points),
    )
    thumb_spread = _distance(points[4], center) / palm_span
    amounts = [
        0.45 * _clamp((thumb_straightness - 0.55) / 0.35, 0.0, 1.0)
        + 0.55 * _clamp((thumb_spread - 0.30) / 0.75, 0.0, 1.0)
    ]
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        path = (
            _distance(points[mcp], points[pip])
            + _distance(points[pip], points[dip])
            + _distance(points[dip], points[tip])
        )
        straightness = _distance(points[mcp], points[tip]) / max(1e-9, path)
        amounts.append(_clamp((straightness - 0.55) / 0.35, 0.0, 1.0))
    return float(sum(amounts) / len(amounts))


def finger_extension_amount(hand: HandObservation, finger: str = "index") -> float:
    """Return a continuous 0..1 extension value for a non-thumb finger."""
    joints = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    name = str(finger).strip().lower()
    if name not in joints:
        raise ValueError("finger must be index, middle, ring, or pinky")
    points: Sequence[Point3D] = hand.world_landmarks or hand.landmarks
    if len(points) != 21:
        raise ValueError("A hand observation must contain 21 landmarks")
    mcp, pip, dip, tip = joints[name]
    path = (
        _distance(points[mcp], points[pip])
        + _distance(points[pip], points[dip])
        + _distance(points[dip], points[tip])
    )
    straightness = _distance(points[mcp], points[tip]) / max(1e-9, path)
    return _clamp((straightness - 0.55) / 0.35, 0.0, 1.0)


def is_pinching(hand: HandObservation, threshold: float = 0.42) -> bool:
    """Return whether the thumb and index fingertips form a scale-independent pinch."""
    points: Sequence[Point3D] = hand.world_landmarks or hand.landmarks
    if len(points) != 21:
        raise ValueError("A hand observation must contain 21 landmarks")
    span = max(1e-9, _distance(points[5], points[17]))
    return _distance(points[4], points[8]) / span <= max(0.05, float(threshold))


def extended_fingers(
    hand: HandObservation,
    *,
    straight_angle: float = 135.0,
    thumb_spread_ratio: float = 0.55,
) -> tuple[bool, bool, bool, bool, bool]:
    """Classify extended fingers from joint angles and overall straightness."""
    points: Sequence[Point3D] = hand.world_landmarks or hand.landmarks
    if len(points) != 21:
        raise ValueError("A hand observation must contain 21 landmarks")
    wrist = points[0]
    palm_points = [points[index] for index in (0, 5, 9, 13, 17)]
    palm_center_point = Point3D(
        sum(point.x for point in palm_points) / len(palm_points),
        sum(point.y for point in palm_points) / len(palm_points),
        sum(point.z for point in palm_points) / len(palm_points),
    )
    palm_width = max(1e-9, _distance(points[5], points[17]))

    thumb_path = (
        _distance(points[1], points[2])
        + _distance(points[2], points[3])
        + _distance(points[3], points[4])
    )
    thumb_straightness = _distance(points[1], points[4]) / max(1e-9, thumb_path)
    thumb_spread = _distance(points[4], palm_center_point) / palm_width
    thumb = (
        _joint_angle(points[2], points[3], points[4]) >= straight_angle
        and thumb_straightness >= 0.72
        and _distance(wrist, points[4]) > 0.95 * _distance(wrist, points[3])
        and thumb_spread >= max(0.1, float(thumb_spread_ratio))
    )
    fingers = [thumb]
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        finger_path = (
            _distance(points[mcp], points[pip])
            + _distance(points[pip], points[dip])
            + _distance(points[dip], points[tip])
        )
        straightness = _distance(points[mcp], points[tip]) / max(1e-9, finger_path)
        extended = (
            _joint_angle(points[mcp], points[pip], points[dip]) >= straight_angle
            and _joint_angle(points[pip], points[dip], points[tip]) >= straight_angle
            and straightness >= 0.72
            and _distance(wrist, points[tip]) > 0.95 * _distance(wrist, points[pip])
        )
        fingers.append(extended)
    return tuple(fingers)  # type: ignore[return-value]


class FingerStateFilter:
    """Debounce each finger separately so single-frame landmark errors are ignored."""

    def __init__(
        self,
        *,
        rise_time: float = 0.06,
        fall_time: float = 0.10,
        on_threshold: float = 0.62,
        off_threshold: float = 0.38,
        reset_seconds: float = 0.35,
    ):
        if not 0.0 <= off_threshold < on_threshold <= 1.0:
            raise ValueError("Finger thresholds must satisfy 0 <= off < on <= 1")
        self.rise_time = max(1e-6, float(rise_time))
        self.fall_time = max(1e-6, float(fall_time))
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.reset_seconds = max(0.0, float(reset_seconds))
        self._scores = np.zeros(5, dtype=float)
        self._states = [False] * 5
        self._last_timestamp: Optional[float] = None

    @property
    def states(self) -> tuple[bool, bool, bool, bool, bool]:
        return tuple(self._states)  # type: ignore[return-value]

    def update(
        self,
        raw_states: Sequence[bool],
        timestamp: float,
    ) -> tuple[bool, bool, bool, bool, bool]:
        if len(raw_states) != 5:
            raise ValueError("Expected five raw finger states")
        timestamp = float(timestamp)
        targets = np.asarray(raw_states, dtype=float)
        if self._last_timestamp is None:
            self._scores = targets.copy()
            self._states = [bool(value) for value in raw_states]
        else:
            dt = max(1e-4, timestamp - self._last_timestamp)
            for index, target in enumerate(targets):
                time_constant = self.rise_time if target > self._scores[index] else self.fall_time
                alpha = 1.0 - math.exp(-dt / time_constant)
                self._scores[index] += alpha * (target - self._scores[index])
                if not self._states[index] and self._scores[index] >= self.on_threshold:
                    self._states[index] = True
                elif self._states[index] and self._scores[index] <= self.off_threshold:
                    self._states[index] = False
        self._last_timestamp = timestamp
        return self.states

    def missing(self, timestamp: float) -> tuple[bool, bool, bool, bool, bool]:
        timestamp = float(timestamp)
        if (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > self.reset_seconds
        ):
            self._scores.fill(0.0)
            self._states = [False] * 5
            self._last_timestamp = None
        return self.states


def chord_degree_for_fingers(fingers: Sequence[bool]) -> Optional[int]:
    if len(fingers) != 5:
        raise ValueError("Expected thumb, index, middle, ring, and pinky states")
    return CHORD_GESTURES.get(tuple(bool(value) for value in fingers))


def chord_quality_for_tilt(tilt_degrees: float, threshold: float = 18.0) -> str:
    """Map left/upright/right lean to minor/diatonic/major."""
    threshold = max(1.0, float(threshold))
    if tilt_degrees <= -threshold:
        return "minor"
    if tilt_degrees >= threshold:
        return "major"
    return "diatonic"


class TiltQualityClassifier:
    """Three tilt zones with hysteresis at the minor/diatonic/major borders."""

    def __init__(self, threshold: float = 18.0, hysteresis: float = 4.0):
        self.threshold = max(1.0, float(threshold))
        self.hysteresis = _clamp(hysteresis, 0.0, self.threshold - 0.1)
        self.quality = "diatonic"

    def update(self, tilt_degrees: float) -> str:
        tilt = float(tilt_degrees)
        exit_threshold = self.threshold - self.hysteresis
        if self.quality == "major" and tilt >= exit_threshold:
            return self.quality
        if self.quality == "minor" and tilt <= -exit_threshold:
            return self.quality
        self.quality = chord_quality_for_tilt(tilt, self.threshold)
        return self.quality


def recognize_chord_gesture(
    hand: HandObservation,
    *,
    quality_tilt_threshold: float = 18.0,
) -> Optional[ChordGesture]:
    fingers = extended_fingers(hand)
    degree = chord_degree_for_fingers(fingers)
    if degree is None:
        return None
    tilt = hand_tilt_degrees(hand)
    return ChordGesture(
        degree=degree,
        quality=chord_quality_for_tilt(tilt, quality_tilt_threshold),
        tilt_degrees=tilt,
        fingers=fingers,
    )


class LeftHandControlExtractor:
    """Extract vertical position, tempo, and articulation from the left hand."""

    def __init__(
        self,
        *,
        min_tempo: float = 60.0,
        max_tempo: float = 140.0,
        max_horizontal_speed: float = 0.20,
        speed_dead_zone: float = 0.01,
        speed_time_constant: float = 0.15,
        activity_top: float = 0.25,
        activity_bottom: float = 0.75,
        dropout_decay_seconds: float = 0.20,
        dropout_reset_seconds: float = 0.40,
    ):
        if min_tempo <= 0 or max_tempo <= min_tempo:
            raise ValueError("Tempo range must be positive and increasing")
        if max_horizontal_speed <= speed_dead_zone:
            raise ValueError("max_horizontal_speed must exceed the dead zone")
        if activity_bottom <= activity_top:
            raise ValueError("activity_bottom must exceed activity_top")
        self.min_tempo = float(min_tempo)
        self.max_tempo = float(max_tempo)
        self.max_horizontal_speed = float(max_horizontal_speed)
        self.speed_dead_zone = float(speed_dead_zone)
        self.speed_time_constant = max(1e-6, float(speed_time_constant))
        self.activity_top = float(activity_top)
        self.activity_bottom = float(activity_bottom)
        self.dropout_decay_seconds = max(1e-6, float(dropout_decay_seconds))
        self.dropout_reset_seconds = max(0.0, float(dropout_reset_seconds))
        self._last_timestamp: Optional[float] = None
        self._last_seen_timestamp: Optional[float] = None
        self._last_x: Optional[float] = None
        self._horizontal_speed = 0.0
        self._melodic_activity = 0.5
        self._articulation = 0.9
        self._palm_x = 0.5
        self._palm_y = 0.5

    def _tempo_for_speed(self, speed: float) -> float:
        normalized = _clamp(
            (float(speed) - self.speed_dead_zone)
            / (self.max_horizontal_speed - self.speed_dead_zone),
            0.0,
            1.0,
        )
        return self.min_tempo + normalized * (self.max_tempo - self.min_tempo)

    def update(self, hand: HandObservation, timestamp: float) -> LeftHandControls:
        timestamp = float(timestamp)
        x, y = palm_center(hand)
        if self._last_timestamp is None or self._last_x is None:
            self._horizontal_speed = 0.0
        else:
            dt = max(1e-4, timestamp - self._last_timestamp)
            if dt > self.dropout_reset_seconds:
                self._horizontal_speed = 0.0
            else:
                raw_speed = abs(x - self._last_x) / dt
                alpha = 1.0 - math.exp(-dt / self.speed_time_constant)
                self._horizontal_speed += alpha * (
                    raw_speed - self._horizontal_speed
                )

        self._last_timestamp = timestamp
        self._last_seen_timestamp = timestamp
        self._last_x = x
        self._palm_x, self._palm_y = x, y
        self._melodic_activity = height_to_unit(
            y,
            self.activity_top,
            self.activity_bottom,
        )
        self._articulation = hand_openness(hand)
        return LeftHandControls(
            detected=True,
            melodic_activity=self._melodic_activity,
            tempo_bpm=self._tempo_for_speed(self._horizontal_speed),
            articulation=self._articulation,
            palm_x=x,
            palm_y=y,
            horizontal_speed=self._horizontal_speed,
        )

    def missing(self, timestamp: float) -> LeftHandControls:
        timestamp = float(timestamp)
        dt = 0.0 if self._last_timestamp is None else max(0.0, timestamp - self._last_timestamp)
        self._horizontal_speed *= math.exp(-dt / self.dropout_decay_seconds)
        missing_for = (
            0.0
            if self._last_seen_timestamp is None
            else max(0.0, timestamp - self._last_seen_timestamp)
        )
        if missing_for > self.dropout_reset_seconds:
            self._last_x = None
        self._last_timestamp = timestamp
        return LeftHandControls(
            detected=False,
            melodic_activity=self._melodic_activity,
            tempo_bpm=self._tempo_for_speed(self._horizontal_speed),
            articulation=self._articulation,
            palm_x=self._palm_x,
            palm_y=self._palm_y,
            horizontal_speed=self._horizontal_speed,
        )


class StableChordGesture:
    """Accept a chord gesture only after it remains unchanged for a short time."""

    def __init__(self, dwell_seconds: float = 0.20, dropout_grace_seconds: float = 0.12):
        self.dwell_seconds = max(0.0, float(dwell_seconds))
        self.dropout_grace_seconds = max(0.0, float(dropout_grace_seconds))
        self._candidate: Optional[ChordGesture] = None
        self._candidate_since = 0.0
        self._candidate_last_seen = 0.0
        self._accepted: Optional[ChordGesture] = None

    @staticmethod
    def _identity(gesture: Optional[ChordGesture]):
        return None if gesture is None else (gesture.degree, gesture.quality)

    @property
    def accepted(self) -> Optional[ChordGesture]:
        return self._accepted

    def update(
        self,
        gesture: Optional[ChordGesture],
        timestamp: float,
    ) -> tuple[Optional[ChordGesture], bool]:
        timestamp = float(timestamp)
        if gesture is None and self._candidate is not None:
            if timestamp - self._candidate_last_seen <= self.dropout_grace_seconds:
                return self._accepted, False
        if self._identity(gesture) != self._identity(self._candidate):
            self._candidate = gesture
            self._candidate_since = timestamp
            self._candidate_last_seen = timestamp
            return self._accepted, False
        if gesture is not None:
            self._candidate = gesture
            self._candidate_last_seen = timestamp
        if gesture is None or timestamp - self._candidate_since < self.dwell_seconds:
            return self._accepted, False
        changed = self._identity(gesture) != self._identity(self._accepted)
        if changed:
            self._accepted = gesture
        return self._accepted, changed


def hand_by_side(
    hands: Sequence[HandObservation],
    side: str,
    *,
    swap_hands: bool = False,
) -> Optional[HandObservation]:
    wanted = side.strip().lower()
    if swap_hands:
        wanted = "right" if wanted == "left" else "left"
    matches = [hand for hand in hands if hand.handedness.lower() == wanted]
    return max(matches, key=lambda hand: hand.confidence, default=None)


class MediaPipeHandTracker:
    """Asynchronous MediaPipe Hand Landmarker wrapper for OpenCV frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {model_path}. "
                "Download hand_landmarker.task into the models folder."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            if "tensorflow" in str(exc).lower() or "_pywrap_tensorflow" in repr(exc).lower():
                raise RuntimeError(
                    "MediaPipe could not start because an unused TensorFlow installation "
                    "failed to load. Remove it with: python -m pip uninstall tensorflow"
                ) from exc
            raise RuntimeError(
                "MediaPipe is required. Run: pip install -r requirements.txt"
            ) from exc

        self.mp = mp
        self._lock = Lock()
        self._latest = HandTrackingResult(0.0, ())
        self._last_submitted_ms = -1
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _points(values) -> tuple[Point3D, ...]:
        return tuple(
            Point3D(float(point.x), float(point.y), float(point.z))
            for point in values
        )

    def _on_result(self, result, _image, timestamp_ms: int) -> None:
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[index][0]
            world = (
                result.hand_world_landmarks[index]
                if index < len(result.hand_world_landmarks)
                else ()
            )
            hands.append(
                HandObservation(
                    handedness=str(category.category_name or "unknown"),
                    confidence=float(category.score or 0.0),
                    landmarks=self._points(landmarks),
                    world_landmarks=self._points(world),
                )
            )
        with self._lock:
            self._latest = HandTrackingResult(float(timestamp_ms) / 1000.0, tuple(hands))

    def submit_bgr(self, frame: np.ndarray, timestamp: float) -> None:
        timestamp_ms = max(self._last_submitted_ms + 1, int(float(timestamp) * 1000.0))
        self._last_submitted_ms = timestamp_ms
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(image, timestamp_ms)

    def latest(self) -> HandTrackingResult:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
