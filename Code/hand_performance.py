"""Control realtime music with two hands tracked by MediaPipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

from camera_performance import _audio_loop, _open_camera
from hand_movement import (
    FINGER_NAMES,
    HAND_CONNECTIONS,
    ChordGesture,
    FingerStateFilter,
    LeftHandControlExtractor,
    MediaPipeHandTracker,
    StableChordGesture,
    TiltQualityClassifier,
    chord_degree_for_fingers,
    extended_fingers,
    hand_by_side,
    hand_tilt_degrees,
    palm_center,
)
from realtime_engine import RealtimeChoraleEngine
from realtime_performance import MidiPlayer, NullPlayer


ROOT = Path(__file__).resolve().parents[1]


def _draw_hand(cv2, frame, hand, color, label, finger_states=None):
    height, width = frame.shape[:2]
    pixels = [
        (int(point.x * (width - 1)), int(point.y * (height - 1)))
        for point in hand.landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    if finger_states is not None:
        for extended, tip_index in zip(finger_states, (4, 8, 12, 16, 20)):
            tip_color = (0, 255, 0) if extended else (0, 0, 255)
            cv2.circle(frame, pixels[tip_index], 7, tip_color, -1, cv2.LINE_AA)
    x, y = palm_center(hand)
    center = (int(x * (width - 1)), int(y * (height - 1)))
    cv2.circle(frame, center, 7, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _put_lines(cv2, frame, lines):
    for index, (message, color) in enumerate(lines):
        cv2.putText(
            frame,
            message,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            color,
            2,
            cv2.LINE_AA,
        )


def _finger_text(fingers):
    active = [name for name, extended in zip(FINGER_NAMES, fingers) if extended]
    return "+".join(active) if active else "fist"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-hand MediaPipe control for realtime Markov music"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--key", default="C")
    parser.add_argument("--mode", default="major")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "hand_landmarker.task",
    )
    parser.add_argument(
        "--max-left-tilt",
        type=float,
        default=45.0,
        help="Left-hand tilt in degrees mapped to maximum density",
    )
    parser.add_argument(
        "--quality-tilt",
        type=float,
        default=18.0,
        help="Right-hand lean angle that enters minor or major",
    )
    parser.add_argument("--gesture-dwell", type=float, default=0.20)
    parser.add_argument(
        "--finger-angle",
        type=float,
        default=135.0,
        help="Minimum joint angle in degrees for an extended finger",
    )
    parser.add_argument(
        "--thumb-spread",
        type=float,
        default=1.2,
        help="Minimum thumb-tip distance from the palm, relative to palm width",
    )
    handedness = parser.add_mutually_exclusive_group()
    handedness.add_argument(
        "--swap-hands",
        dest="swap_hands",
        action="store_true",
        help="Swap MediaPipe left/right labels (the default for this setup)",
    )
    handedness.add_argument(
        "--no-swap-hands",
        dest="swap_hands",
        action="store_false",
        help="Use MediaPipe left/right labels without swapping",
    )
    parser.set_defaults(swap_hands=True)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--midi-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=16)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Run: pip install -r requirements.txt") from exc

    engine = RealtimeChoraleEngine.from_paths(
        tempo_bpm=args.tempo,
        key=args.key,
        mode=args.mode,
        seed=args.seed,
    )
    player = NullPlayer() if args.dry_run else MidiPlayer(args.midi_port)
    capture = _open_camera(cv2, args.camera, args.width, args.height)
    tracker = MediaPipeHandTracker(args.model, num_hands=2)
    left_controls = LeftHandControlExtractor(max_left_tilt=args.max_left_tilt)
    quality_classifier = TiltQualityClassifier(args.quality_tilt)
    finger_filter = FingerStateFilter()
    chord_stabilizer = StableChordGesture(args.gesture_dwell)
    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=_audio_loop,
        args=(engine, player, stop_event, args.log_every),
        daemon=True,
    )
    audio_thread.start()

    last_result_time = -1.0
    latest_left = left_controls.missing(time.perf_counter())
    candidate_gesture = None
    raw_fingers = (False,) * 5
    stable_fingers = (False,) * 5
    print("Left hand: height=volume, leftward tilt=density")
    print("Right hand: finger pattern=degree, left/upright/right tilt=minor/diatonic/major")
    print("Press Q or Escape to stop.")

    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            tracker.submit_bgr(frame, now)
            result = tracker.latest()

            left_hand = hand_by_side(result.hands, "left", swap_hands=args.swap_hands)
            right_hand = hand_by_side(result.hands, "right", swap_hands=args.swap_hands)
            if result.timestamp > last_result_time:
                last_result_time = result.timestamp
                if left_hand is None:
                    latest_left = left_controls.missing(result.timestamp)
                else:
                    latest_left = left_controls.update(left_hand, result.timestamp)
                engine.update_controls(
                    volume=latest_left.volume,
                    density=latest_left.density,
                )

                candidate_gesture = None
                if right_hand is not None:
                    raw_fingers = extended_fingers(
                        right_hand,
                        straight_angle=args.finger_angle,
                        thumb_spread_ratio=args.thumb_spread,
                    )
                    stable_fingers = finger_filter.update(raw_fingers, result.timestamp)
                    degree = chord_degree_for_fingers(stable_fingers)
                    tilt = hand_tilt_degrees(right_hand)
                    quality = quality_classifier.update(tilt)
                    if degree is not None:
                        candidate_gesture = ChordGesture(
                            degree=degree,
                            quality=quality,
                            tilt_degrees=tilt,
                            fingers=stable_fingers,
                        )
                else:
                    raw_fingers = (False,) * 5
                    stable_fingers = finger_filter.missing(result.timestamp)
                accepted, changed = chord_stabilizer.update(
                    candidate_gesture,
                    result.timestamp,
                )
                if changed and accepted is not None:
                    engine.update_chord_control(accepted.degree, accepted.quality)
                    print(
                        f"Chord gesture accepted: degree={accepted.degree} "
                        f"quality={accepted.quality}"
                    )

            if left_hand is not None:
                _draw_hand(cv2, frame, left_hand, (255, 200, 0), "LEFT: volume/density")
            if right_hand is not None:
                _draw_hand(
                    cv2,
                    frame,
                    right_hand,
                    (0, 255, 0),
                    "RIGHT: chord",
                    stable_fingers,
                )

            target = engine.chord_control_target()
            if candidate_gesture is None:
                right_status = "Right: pattern not mapped; hold the intended gesture"
            else:
                right_status = (
                    f"Right candidate: degree {candidate_gesture.degree} "
                    f"{candidate_gesture.quality} tilt={candidate_gesture.tilt_degrees:+.0f}deg"
                )
            finger_status = (
                f"Fingers raw={_finger_text(raw_fingers)} "
                f"stable={_finger_text(stable_fingers)}"
            )
            selected_status = (
                "Selected chord: learned progression"
                if target is None
                else f"Selected chord: degree {target.degree} {target.quality}"
            )
            lines = [
                (
                    f"Left volume={latest_left.volume:.2f} density={latest_left.density:.2f} "
                    f"tilt={latest_left.tilt_degrees:+.0f}deg",
                    (255, 200, 0),
                ),
                (right_status, (0, 255, 0)),
                (finger_status, (0, 255, 0)),
                (selected_status, (255, 255, 255)),
                (f"Key={engine.key} mode={engine.mode} | Q/Esc: quit", (255, 255, 255)),
            ]
            _put_lines(cv2, frame, lines)
            cv2.imshow("MediaPipe Hands - Movement to Music", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                stop_event.set()
    finally:
        stop_event.set()
        audio_thread.join(timeout=2.0)
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
