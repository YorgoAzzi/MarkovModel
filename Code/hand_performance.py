"""Control realtime music with two hands tracked by MediaPipe."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
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
    finger_extension_amount,
    hand_by_side,
    hand_tilt_degrees,
    height_to_register,
    is_pinching,
    palm_center,
)
from live_score import LiveScoreDisplay
from magenta_router import (
    BACKENDS,
    DEFAULT_MAGENTA_PORT,
    INPUT_MODES,
    INPUT_SOPRANO,
    MagentaRouter,
)
from piano_roll import SopranoPianoRoll
from performance_session import MODE_FULL, MODE_SOPRANO, PerformanceSession
from performance_ui import IntegratedPerformanceWindow
from realtime_engine import RealtimeChoraleEngine
from realtime_performance import MidiPlayer, NullPlayer
from soprano_performance import SopranoGestureController


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


def _meter(value, width=10):
    filled = round(max(0.0, min(1.0, float(value))) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-hand MediaPipe control for realtime Markov music"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--tempo",
        type=float,
        default=60.0,
        help="Tempo when the left hand is stationary",
    )
    parser.add_argument("--max-tempo", type=float, default=140.0)
    parser.add_argument("--key", default="C")
    parser.add_argument("--mode", default="major")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "hand_landmarker.task",
    )
    parser.add_argument(
        "--max-horizontal-speed",
        type=float,
        default=0.20,
        help="Left-hand screen widths/second mapped to maximum tempo",
    )
    parser.add_argument(
        "--height-top",
        type=float,
        default=0.25,
        help="Normalized screen height mapped to maximum activity/register",
    )
    parser.add_argument(
        "--height-bottom",
        type=float,
        default=0.75,
        help="Normalized screen height mapped to minimum activity/register",
    )
    parser.add_argument(
        "--quality-tilt",
        type=float,
        default=18.0,
        help="Right-hand lean angle that enters minor or major",
    )
    parser.add_argument("--gesture-dwell", type=float, default=0.20)
    parser.add_argument(
        "--performance-mode",
        choices=(MODE_FULL, MODE_SOPRANO),
        default=MODE_FULL,
        help="Initial interface mode; it can also be changed in the camera window",
    )
    parser.add_argument(
        "--soprano-step",
        type=float,
        default=0.032,
        help="Vertical hand travel needed for one successive scale note",
    )
    parser.add_argument(
        "--pinch-distance",
        type=float,
        default=0.42,
        help="Thumb-index distance relative to palm width for soprano sustain",
    )
    parser.add_argument(
        "--trill-shake-speed",
        type=float,
        default=0.35,
        help="Right-hand horizontal screen widths/second needed for a trill",
    )
    parser.add_argument(
        "--timbre-dwell",
        type=float,
        default=0.22,
        help="Seconds a finger count must remain stable before changing sound",
    )
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
    parser.add_argument(
        "--no-live-score",
        action="store_true",
        help="Hide the integrated live-score panel",
    )
    parser.add_argument(
        "--score-bars",
        type=int,
        default=4,
        help="Number of recent measures shown in the live score",
    )
    parser.add_argument(
        "--magenta",
        action="store_true",
        help="Start with local Magenta accompaniment enabled",
    )
    parser.add_argument(
        "--magenta-input",
        choices=INPUT_MODES,
        default=INPUT_SOPRANO,
        help="Send only soprano or all four generated voices to Magenta",
    )
    parser.add_argument(
        "--magenta-backend",
        choices=BACKENDS,
        default="midi",
        help="Use the real Magenta MIDI input or an in-process mock",
    )
    parser.add_argument(
        "--magenta-midi-port",
        default=DEFAULT_MAGENTA_PORT,
        help="Virtual MIDI destination created by the Magenta standalone app",
    )
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
        ticks_per_beat=16,
    )
    # The hand interface no longer controls these values. Neutral density keeps
    # the learned Markov duration probabilities unchanged; volume stays fixed.
    engine.update_controls(density=0.5, volume=0.7)
    soprano = SopranoGestureController(
        key=args.key,
        mode=args.mode,
        movement_step=args.soprano_step,
        shake_speed=args.trill_shake_speed,
        timbre_dwell_seconds=args.timbre_dwell,
    )
    session = PerformanceSession(engine, soprano, mode=args.performance_mode)
    magenta = MagentaRouter(
        enabled=args.magenta,
        input_mode=args.magenta_input,
        backend=args.magenta_backend,
        port_name=args.magenta_midi_port,
    ).start()
    player = NullPlayer() if args.dry_run else MidiPlayer(args.midi_port)
    capture = _open_camera(cv2, args.camera, args.width, args.height)
    tracker = MediaPipeHandTracker(args.model, num_hands=2)
    left_controls = LeftHandControlExtractor(
        min_tempo=args.tempo,
        max_tempo=args.max_tempo,
        max_horizontal_speed=args.max_horizontal_speed,
        activity_top=args.height_top,
        activity_bottom=args.height_bottom,
    )
    quality_classifier = TiltQualityClassifier(args.quality_tilt)
    finger_filter = FingerStateFilter()
    chord_stabilizer = StableChordGesture(args.gesture_dwell)
    live_score = None
    piano_roll = None
    if not args.no_live_score:
        live_score = LiveScoreDisplay(
            ticks_per_beat=session.ticks_per_beat,
            beats_per_bar=session.beats_per_bar,
            key_name=session.key,
            mode=engine.mode,
            visible_bars=args.score_bars,
            port=0,
        ).start(open_browser=False, serve=False)
        piano_roll = SopranoPianoRoll(
            ticks_per_beat=session.ticks_per_beat,
            beats_per_bar=session.beats_per_bar,
            visible_bars=args.score_bars,
        )
    observers = [magenta.observe]
    if live_score is not None:
        observers.append(live_score.observe)
    if piano_roll is not None:
        observers.append(
            lambda frame: piano_roll.observe(
                frame,
                enabled=session.mode == MODE_SOPRANO,
            )
        )
    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=_audio_loop,
        args=(session, player, stop_event, args.log_every, observers),
        daemon=True,
    )
    audio_thread.start()

    last_result_time = -1.0
    latest_left = left_controls.missing(time.perf_counter())
    candidate_gesture = None
    raw_fingers = (False,) * 5
    stable_fingers = (False,) * 5
    right_register = 0.0
    print("Modes: click the window buttons or press 1/2. Space pauses or resumes.")
    print("Full SATB: left controls activity/tempo/articulation; right controls register/chords.")
    print("Soprano Performance: left controls accompaniment register/tempo/articulation; right performs soprano.")
    print("Soprano fingers: 1=piano, 2=violin, 3=choir, 4=flute, 5=synth pad; shake=trill, index tap=mordent.")
    print("Magenta: use the window controls to enable accompaniment and select soprano/all input.")
    print("Press Q or Escape to stop.")

    pending_actions = deque()
    try:
        from PyQt5 import QtWidgets
    except ImportError as exc:
        raise RuntimeError("PyQt5 is required. Run: pip install -r requirements.txt") from exc
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = IntegratedPerformanceWindow(show_score=live_score is not None)
    window.mode_requested.connect(pending_actions.append)
    window.transport_requested.connect(lambda: pending_actions.append("transport"))
    window.magenta_enabled_requested.connect(
        lambda enabled: pending_actions.append(("magenta_enabled", bool(enabled)))
    )
    window.magenta_input_requested.connect(
        lambda input_mode: pending_actions.append(("magenta_input", input_mode))
    )
    window.close_requested.connect(stop_event.set)
    window.set_state(
        mode=session.mode,
        playing=session.is_playing,
        summary="Initializing hand tracking...",
    )
    window.set_magenta_state(
        enabled=magenta.enabled,
        input_mode=magenta.input_mode,
        status=magenta.status(),
    )
    window.show()

    try:
        while not stop_event.is_set():
            qt_app.processEvents()
            if window.closed:
                stop_event.set()
                break
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            tracker.submit_bgr(frame, now)
            result = tracker.latest()

            while pending_actions:
                action = pending_actions.popleft()
                if action in {MODE_FULL, MODE_SOPRANO}:
                    if action == session.mode:
                        continue
                    session.set_mode(action)
                    player.stop_all()
                    magenta.reset_output()
                    finger_filter = FingerStateFilter()
                    quality_classifier = TiltQualityClassifier(args.quality_tilt)
                    chord_stabilizer = StableChordGesture(args.gesture_dwell)
                    candidate_gesture = None
                    print(f"Performance mode: {action}")
                elif action == "transport":
                    playing = session.toggle_playing()
                    magenta.set_playing(playing)
                    if not playing:
                        player.stop_all()
                    print("Playing" if playing else "Paused")
                elif isinstance(action, tuple) and action[0] == "magenta_enabled":
                    enabled = magenta.set_enabled(action[1])
                    print("Magenta accompaniment on" if enabled else "Magenta accompaniment off")
                elif isinstance(action, tuple) and action[0] == "magenta_input":
                    input_mode = magenta.set_input_mode(action[1])
                    print(f"Magenta input: {input_mode}")

            left_hand = hand_by_side(result.hands, "left", swap_hands=args.swap_hands)
            right_hand = hand_by_side(result.hands, "right", swap_hands=args.swap_hands)
            if result.timestamp > last_result_time:
                last_result_time = result.timestamp
                if left_hand is None:
                    latest_left = left_controls.missing(result.timestamp)
                else:
                    latest_left = left_controls.update(left_hand, result.timestamp)
                if session.mode == MODE_FULL:
                    engine.update_controls(
                        rotation=latest_left.melodic_activity,
                        articulation=latest_left.articulation,
                        brightness=0.5,
                    )
                else:
                    accompaniment_register = height_to_register(
                        latest_left.palm_y,
                        args.height_top,
                        args.height_bottom,
                    )
                    engine.update_controls(
                        tilt=accompaniment_register,
                        rotation=0.5,
                        articulation=latest_left.articulation,
                        brightness=0.5,
                    )
                engine.update_tempo(latest_left.tempo_bpm)

                candidate_gesture = None
                if right_hand is not None:
                    right_x, right_y = palm_center(right_hand)
                    right_register = height_to_register(
                        right_y,
                        args.height_top,
                        args.height_bottom,
                    )
                    raw_fingers = extended_fingers(
                        right_hand,
                        straight_angle=args.finger_angle,
                        thumb_spread_ratio=args.thumb_spread,
                    )
                    stable_fingers = finger_filter.update(raw_fingers, result.timestamp)
                    if session.mode == MODE_FULL:
                        engine.update_controls(tilt=right_register)
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
                        soprano_input = dict(
                            x=right_x,
                            y=right_y,
                            fingers=stable_fingers,
                            pinching=is_pinching(
                                right_hand, threshold=args.pinch_distance
                            ),
                            index_extension=finger_extension_amount(
                                right_hand, "index"
                            ),
                            timestamp=result.timestamp,
                        )
                        if session.is_playing:
                            soprano.update(**soprano_input)
                        else:
                            soprano.synchronize(**soprano_input)
                else:
                    raw_fingers = (False,) * 5
                    stable_fingers = finger_filter.missing(result.timestamp)
                    if session.mode == MODE_SOPRANO:
                        soprano.missing(result.timestamp)
                if session.mode == MODE_FULL:
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
                left_label = (
                    "LEFT: activity/tempo"
                    if session.mode == MODE_FULL
                    else "LEFT: accompaniment"
                )
                _draw_hand(cv2, frame, left_hand, (255, 200, 0), left_label)
            if right_hand is not None:
                _draw_hand(
                    cv2,
                    frame,
                    right_hand,
                    (0, 255, 0),
                    (
                        "RIGHT: chord/register"
                        if session.mode == MODE_FULL
                        else "RIGHT: soprano"
                    ),
                    stable_fingers,
                )

            finger_status = (
                f"Fingers raw={_finger_text(raw_fingers)} "
                f"stable={_finger_text(stable_fingers)}"
            )
            if session.mode == MODE_FULL:
                target = engine.chord_control_target()
                if candidate_gesture is None:
                    right_status = "Right: pattern not mapped; hold the intended gesture"
                else:
                    right_status = (
                        f"Right candidate: degree {candidate_gesture.degree} "
                        f"{candidate_gesture.quality} tilt={candidate_gesture.tilt_degrees:+.0f}deg"
                    )
                selected_status = (
                    "Selected chord: learned progression"
                    if target is None
                    else f"Selected chord: degree {target.degree} {target.quality}"
                )
                left_status = (
                    f"Left activity {_meter(latest_left.melodic_activity)} "
                    f"tempo={latest_left.tempo_bpm:.0f} BPM"
                )
                register_status = (
                    f"Right register {_meter((right_register + 1.0) / 2.0)} "
                    f"{right_register:+.2f}"
                )
            else:
                soprano_gesture, soprano_pitch = soprano.status()
                _program, timbre_name = soprano.timbre()
                left_register = height_to_register(
                    latest_left.palm_y, args.height_top, args.height_bottom
                )
                right_status = (
                    f"Soprano: {soprano_gesture} | MIDI pitch "
                    f"{soprano_pitch if soprano_pitch is not None else 'REST'}"
                )
                selected_status = "Harmony: learned Markov progression (no chord gestures)"
                left_status = (
                    f"Left accompaniment register {_meter((left_register + 1.0) / 2.0)} "
                    f"tempo={latest_left.tempo_bpm:.0f} BPM"
                )
                register_status = (
                    f"Timbre={timbre_name} | shake=trill | index tap=mordent"
                )
            lines = [
                (
                    f"MODE: {'FULL SATB' if session.mode == MODE_FULL else 'SOPRANO PERFORMANCE'} "
                    f"| {'PLAYING' if session.is_playing else 'PAUSED'}",
                    (255, 255, 255),
                ),
                (
                    left_status,
                    (255, 200, 0),
                ),
                (
                    f"Left articulation={latest_left.articulation:.2f} "
                    f"x-speed={latest_left.horizontal_speed:.3f}",
                    (255, 200, 0),
                ),
                (right_status, (0, 255, 0)),
                (finger_status, (0, 255, 0)),
                (register_status, (0, 255, 0)),
                (selected_status, (255, 255, 255)),
                (f"Key={engine.key} mode={engine.mode} | Q/Esc: quit", (255, 255, 255)),
            ]
            _put_lines(cv2, frame, lines)
            window.set_camera_frame(frame)
            window.set_state(
                mode=session.mode,
                playing=session.is_playing,
                summary=(
                    f"{right_status} | {left_status} | "
                    f"articulation={latest_left.articulation:.2f}"
                ),
            )
            window.set_magenta_state(
                enabled=magenta.enabled,
                input_mode=magenta.input_mode,
                status=magenta.status(),
            )
            if live_score is not None:
                window.update_score(live_score.snapshot())
            if piano_roll is not None:
                window.update_piano_roll(piano_roll.snapshot())
            qt_app.processEvents()
    finally:
        stop_event.set()
        audio_thread.join(timeout=2.0)
        magenta.close()
        if live_score is not None:
            live_score.stop()
        tracker.close()
        capture.release()
        window.close()
        qt_app.processEvents()


if __name__ == "__main__":
    main()
