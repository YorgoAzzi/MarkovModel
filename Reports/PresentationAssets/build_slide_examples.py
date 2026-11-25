"""Create presentation examples from a saved score and the current Qt app."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import verovio
from music21 import converter, stream
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_SCORE = ROOT / "generated_realtime_live.musicxml"
SOURCE_PHOTO = HERE / "illustrative_performer.png"
SCORE_PNG = HERE / "generated_score_excerpt.png"
APP_PNG = HERE / "hand_app_interface_preview.png"

sys.path.insert(0, str(ROOT / "Code"))
from performance_ui import IntegratedPerformanceWindow, qt_compatible_svg  # noqa: E402
from performance_session import MODE_FULL  # noqa: E402


CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def score_excerpt_svg(*, app_preview: bool = False) -> str:
    original = converter.parse(SOURCE_SCORE)
    excerpt = stream.Score()
    for original_part in original.parts:
        part = stream.Part()
        for measure in list(original_part.getElementsByClass(stream.Measure))[:4]:
            part.append(deepcopy(measure))
        excerpt.append(part)
    excerpt_path = HERE / "generated_score_excerpt.musicxml"
    excerpt.write("musicxml", fp=excerpt_path)

    toolkit = verovio.toolkit()
    toolkit.setOptions({
        "pageWidth": 1800 if app_preview else 3000,
        "pageHeight": 1200 if app_preview else 2400,
        "scale": 42,
        "adjustPageHeight": app_preview,
        "breaks": "none" if app_preview else "auto",
        "header": "none",
        "footer": "none",
    })
    if not toolkit.loadFile(str(excerpt_path)):
        raise RuntimeError("Could not render saved MusicXML score")
    return toolkit.renderToSVG(1)


def save_svg_png(svg: str, path: Path, width: int = 1600) -> None:
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(qt_compatible_svg(svg)))
    natural = renderer.defaultSize()
    height = round(width * natural.height() / natural.width())
    image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    renderer.render(painter, QtCore.QRectF(0, 0, width, height))
    painter.end()
    image.save(str(path))
    canvas = Image.open(path).convert("RGB")
    dark = np.any(np.asarray(canvas) < 235, axis=2)
    ys, xs = np.where(dark)
    box = (
        max(0, int(xs.min()) - 35), max(0, int(ys.min()) - 35),
        min(canvas.width, int(xs.max()) + 36),
        min(canvas.height, int(ys.max()) + 36),
    )
    canvas.crop(box).save(path)


def tracked_frame() -> np.ndarray:
    frame = cv2.imread(str(SOURCE_PHOTO))
    if frame is None:
        raise FileNotFoundError(SOURCE_PHOTO)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ROOT / "models" / "hand_landmarker.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
    if len(result.hand_landmarks) != 2:
        raise RuntimeError("MediaPipe did not detect both hands in example frame")
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for landmarks in result.hand_landmarks:
        points = [
            (round(point.x * width), round(point.y * height))
            for point in landmarks
        ]
        for start, end in CONNECTIONS:
            cv2.line(overlay, points[start], points[end], (176, 232, 80), 5,
                     cv2.LINE_AA)
        for point in points:
            cv2.circle(overlay, point, 7, (5, 57, 242), -1, cv2.LINE_AA)
            cv2.circle(overlay, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)


def label_offscreen_window(path: Path) -> None:
    """Qt offscreen renders controls without glyphs on this Windows host."""
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    regular = ImageFont.truetype(font_path, 16)
    bold = ImageFont.truetype(bold_path, 17)
    small = ImageFont.truetype(font_path, 14)

    def centered(value: str, box, font=regular, color="#234e70"):
        left, top, right, bottom = box
        b = draw.textbbox((0, 0), value, font=font)
        draw.text(((left + right - (b[2] - b[0])) / 2,
                   (top + bottom - (b[3] - b[1])) / 2 - b[1]),
                  value, font=font, fill=color)

    draw.text((23, 20), "Camera and gesture tracking", font=bold, fill="#234e70")
    draw.text((932, 20), "Live SATB score", font=bold, fill="#234e70")
    centered("4 generated measures - live notation", (1130, 14, 1480, 51),
             small, "#44515a")
    centered("Follow live", (1480, 14, 1805, 51), small)
    for value, box in (
        ("Full SATB  [1]", (20, 583, 234, 621)),
        ("Soprano Performance  [2]", (240, 583, 455, 621)),
        ("Conductor  [3]", (462, 583, 678, 621)),
        ("Pause  [Space]", (684, 583, 900, 621)),
    ):
        centered(value, box, small,
                 "#ffffff" if value.startswith("Full") else "#234e70")
    centered("Swap hand roles  [H]", (20, 625, 900, 663), small)
    centered("Music visuals  [V]", (20, 668, 900, 706), small)
    centered("Magenta Off", (20, 711, 232, 750), small)
    draw.text((258, 720), "Input to Magenta:", font=small, fill="#234e70")
    centered("Soprano only", (531, 711, 820, 750), small)
    draw.text((25, 770),
              "Tracking both hands  |  Key: C major  |  Tempo: 120 BPM",
              font=small, fill="#44515a")
    image.save(path)


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    svg = score_excerpt_svg()
    save_svg_png(svg, SCORE_PNG)

    window = IntegratedPerformanceWindow(show_score=True)
    window.set_state(
        mode=MODE_FULL,
        playing=True,
        summary="Tracking both hands  |  Key: C major  |  Tempo: 120 BPM",
    )
    window.update_score({
        "status": "4 generated measures · live notation",
        "page_versions": [1],
        "bars_per_page": 4,
        "svgs": [score_excerpt_svg(app_preview=True)],
    })
    window.show()
    app.processEvents()
    window.set_camera_frame(tracked_frame())
    app.processEvents()
    window.grab().save(str(APP_PNG))
    window.close()
    label_offscreen_window(APP_PNG)
    print(SCORE_PNG)
    print(APP_PNG)


if __name__ == "__main__":
    main()
