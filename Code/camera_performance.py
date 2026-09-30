"""Control the realtime SATB generator by tracking a colored ribbon."""

from __future__ import annotations

import argparse
import os
import threading
import time

from camera_movement import (
    ColorBlobDetector,
    MotionFeatureExtractor,
    movement_to_music_controls,
)
from realtime_engine import RealtimeChoraleEngine
from realtime_performance import MidiPlayer, NullPlayer


def _audio_loop(engine, player, stop_event, log_every):
    deadline = time.perf_counter()
    generated = 0
    try:
        while not stop_event.is_set():
            started = time.perf_counter()
            frame = engine.generate_tick()
            generation_ms = (time.perf_counter() - started) * 1000.0
            player.render(frame)
            if log_every > 0 and generated % log_every == 0:
                controls = frame.controls
                print(
                    f"audio tick={frame.tick_index:04d} generation={generation_ms:.1f}ms "
                    f"density={controls.density:.2f} volume={controls.volume:.2f} "
                    f"rhythm={frame.sampled_duration_beats:g}beat"
                )
            generated += 1
            deadline += engine.tick_seconds
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                stop_event.wait(remaining)
            else:
                deadline = time.perf_counter()
    except Exception as exc:
        print(f"Audio loop stopped because of an error: {exc}")
        stop_event.set()
    finally:
        player.close()


def _open_camera(cv2, camera_index, width, height):
    if os.name == "nt":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
    else:
        capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    return capture


def _draw_overlay(cv2, frame, detector, detection, sample, controls):
    if detection is not None:
        radius = max(8, int((detection.area_ratio * frame.shape[0] * frame.shape[1]) ** 0.5 / 2))
        cv2.circle(frame, (detection.pixel_x, detection.pixel_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (detection.pixel_x, detection.pixel_y), 4, (255, 255, 255), -1)
    status = "TRACKING" if sample.detected else "RIBBON LOST"
    if not detector.calibrated:
        status = "CLICK THE COLORED RIBBON TO CALIBRATE"
    lines = [
        status,
        "SLOW < density > FAST",
        "DOWN < volume > UP",
        f"speed={sample.speed:.2f}  height={1.0 - sample.y:.2f}",
        f"density={controls['density']:.2f}  volume={controls['volume']:.2f}  hue={detector.hue}",
        "Click ribbon: recalibrate | Q or Esc: quit",
    ]
    for index, text in enumerate(lines):
        color = (0, 255, 0) if index == 0 and sample.detected else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Colored-ribbon camera control for realtime music")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--hue", type=int, default=None, help="OpenCV hue 0..179; click ribbon if omitted")
    parser.add_argument("--hue-tolerance", type=int, default=10)
    parser.add_argument("--min-area", type=float, default=0.001, help="Minimum blob area as frame fraction")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.5,
        help="Screen widths per second mapped to maximum density",
    )
    parser.add_argument("--midi-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc

    detector = ColorBlobDetector(
        hue=args.hue,
        hue_tolerance=args.hue_tolerance,
        min_area_ratio=args.min_area,
    )
    features = MotionFeatureExtractor(
        max_speed=args.max_speed,
    )
    capture = _open_camera(cv2, args.camera, args.width, args.height)
    engine = RealtimeChoraleEngine.from_paths(tempo_bpm=args.tempo, seed=args.seed)
    player = NullPlayer() if args.dry_run else MidiPlayer(args.midi_port)
    stop_event = threading.Event()
    latest_frame = [None]
    camera_controls = movement_to_music_controls(0.0, 0.5)

    def on_mouse(event, x, y, _flags, _data):
        if event != cv2.EVENT_LBUTTONDOWN or latest_frame[0] is None:
            return
        frame = latest_frame[0]
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            hue = detector.calibrate_pixel(frame[y, x])
            print(f"Calibrated ribbon hue={hue}")

    window_name = "Movement to Music - Colored Ribbon"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    audio_thread = threading.Thread(
        target=_audio_loop,
        args=(engine, player, stop_event, args.log_every),
        daemon=True,
    )
    audio_thread.start()

    print("Click the colored ribbon in the camera window, then move it. Press Q to stop.")
    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            latest_frame[0] = frame
            timestamp = time.perf_counter()
            detection, mask = detector.detect(frame)
            if detection is None:
                sample = features.missing(timestamp)
            else:
                sample = features.update(
                    detection.x,
                    detection.y,
                    timestamp,
                    area_ratio=detection.area_ratio,
                )

            if detector.calibrated:
                camera_controls = movement_to_music_controls(sample.speed, sample.y)
                engine.update_controls(**camera_controls)

            _draw_overlay(
                cv2,
                frame,
                detector,
                detection,
                sample,
                camera_controls,
            )
            cv2.imshow(window_name, frame)
            cv2.imshow("Ribbon mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                stop_event.set()
    finally:
        stop_event.set()
        audio_thread.join(timeout=2.0)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
