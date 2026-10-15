"""Generate SVG and PNG architecture diagrams for the movement-to-music system."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2800
HEIGHT = 1800
OUTPUT_DIR = Path(__file__).resolve().parent

BACKGROUND = "#F3F6F9"
INK = "#183042"
MUTED = "#557083"
ARROW = "#647D8D"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class Section:
    x: int
    y: int
    width: int
    height: int
    title: str
    subtitle: str
    color: str
    fill: str


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int
    title: str
    lines: tuple[str, ...]
    color: str
    fill: str
    title_size: int = 29
    body_size: int = 23


SECTIONS = (
    Section(70, 235, 470, 1300, "1  VISION", "camera and landmark extraction", "#2877A5", "#EAF5FB"),
    Section(585, 235, 575, 1300, "2  GESTURE MAPPING", "movement becomes musical control", "#27856F", "#EAF7F3"),
    Section(1205, 235, 730, 1300, "3  MUSIC GENERATION", "persistent Markov and SATB state", "#C2781B", "#FFF5E5"),
    Section(1980, 235, 750, 1300, "4  OUTPUT", "sound and notation from one event", "#765BA8", "#F2EEFA"),
)


BOXES = (
    Box(125, 350, 360, 145, "Laptop camera", ("Live RGB video frames",), "#2877A5", WHITE),
    Box(125, 570, 360, 175, "OpenCV", ("Capture and mirror frames", "Display tracking overlay"), "#2877A5", WHITE),
    Box(125, 825, 360, 215, "MediaPipe Hand Landmarker", ("Detect two hands", "Return 21 landmarks per hand", "Label left and right hands"), "#2877A5", WHITE),
    Box(125, 1125, 360, 170, "Hand observations", ("Position, shape and depth", "with timestamps"), "#2877A5", WHITE),

    Box(635, 350, 475, 425, "Left hand — expression", (
        "Height  →  melodic activity",
        "Horizontal speed  →  tempo",
        "Openness  →  articulation",
        "Forward / backward  →  brightness",
    ), "#27856F", WHITE),
    Box(635, 855, 475, 355, "Right hand — harmony", (
        "Height  →  register",
        "Finger pattern  →  scale degree 1–7",
        "Tilt  →  minor / diatonic / major",
    ), "#27856F", WHITE),
    Box(635, 1290, 475, 155, "Stabilization", ("Smoothing, finger filtering", "and gesture dwell time"), "#27856F", WHITE),

    Box(1260, 350, 620, 175, "Realtime control interface", (
        "Smoothed continuous controls",
        "Tempo clock + stable chord requests",
    ), "#C2781B", WHITE),
    Box(1260, 600, 285, 180, "Melody model", (
        "Second-order Markov",
        "pitch intervals + rhythm",
    ), "#C2781B", WHITE, 27, 21),
    Box(1595, 600, 285, 180, "Harmony model", (
        "Chord Markov model",
        "progression blocks",
    ), "#C2781B", WHITE, 27, 21),
    Box(1260, 850, 620, 350, "RealtimeChoraleEngine", (
        "Maintains persistent pitch, rhythm and chord history",
        "Biases learned choices using hand controls",
        "Applies performer-selected chord degree and quality",
        "Runs constrained SATB beam search and voice leading",
    ), "#C2781B", WHITE),
    Box(1260, 1280, 620, 165, "GeneratedTick", (
        "SATB pitches • rhythm • chord • tempo",
        "velocity • articulation • brightness",
    ), "#C2781B", "#FFFDF8"),

    Box(2040, 350, 630, 135, "Realtime scheduler", ("Audio clock receives every GeneratedTick",), "#A85B22", WHITE),
    Box(2040, 545, 630, 145, "MIDI player", ("Note on/off • expression • timbre controls",), "#A85B22", WHITE),
    Box(2040, 750, 630, 145, "MIDI synthesizer", ("Four SATB channels  →  audible music",), "#A85B22", "#FFF7EE"),
    Box(2040, 985, 630, 115, "Asynchronous notation queue", ("Only new note onsets are queued",), "#765BA8", WHITE, 27, 21),
    Box(2040, 1160, 295, 135, "music21", ("Build SATB MusicXML",), "#765BA8", WHITE, 27, 21),
    Box(2375, 1160, 295, 135, "Verovio", ("Engrave MusicXML as SVG",), "#765BA8", WHITE, 27, 21),
    Box(2040, 1360, 630, 125, "Browser live score", ("Current section + scrollable performance history",), "#765BA8", "#F8F5FD", 28, 21),
)


# Each connector is (points, label, color, dashed).
CONNECTORS = (
    (((305, 495), (305, 570)), "frames", "#2877A5", False),
    (((305, 745), (305, 825)), "BGR frames", "#2877A5", False),
    (((305, 1040), (305, 1125)), "landmarks", "#2877A5", False),
    (((485, 1210), (555, 1210), (555, 560), (635, 560)), "left hand", "#27856F", False),
    (((485, 1210), (570, 1210), (570, 1030), (635, 1030)), "right hand", "#27856F", False),
    (((1110, 560), (1180, 560), (1180, 420), (1260, 420)), "", "#27856F", False),
    (((1110, 1030), (1190, 1030), (1190, 475), (1260, 475)), "", "#27856F", False),
    (((872, 775), (1125, 775), (1125, 1367), (1110, 1367)), "", "#27856F", True),
    (((872, 1210), (872, 1290)), "", "#27856F", True),
    (((1570, 525), (1570, 815), (1500, 815), (1500, 850)), "", "#C2781B", False),
    (((1402, 780), (1402, 850)), "learned state", "#C2781B", False),
    (((1737, 780), (1737, 850)), "harmony", "#C2781B", False),
    (((1570, 1200), (1570, 1280)), "one clock event", "#C2781B", False),
    (((1880, 1362), (1960, 1362), (1960, 418), (2040, 418)), "same event", "#A85B22", False),
    (((1880, 1398), (1995, 1398), (1995, 1042), (2040, 1042)), "", "#765BA8", False),
    (((2355, 485), (2355, 545)), "schedule", "#A85B22", False),
    (((2355, 690), (2355, 750)), "MIDI", "#A85B22", False),
    (((2355, 1100), (2355, 1125), (2187, 1125), (2187, 1160)), "events", "#765BA8", False),
    (((2335, 1227), (2375, 1227)), "", "#765BA8", False),
    (((2522, 1295), (2522, 1360)), "SVG", "#765BA8", False),
)


def _font_path(bold: bool = False) -> str:
    windows = Path("C:/Windows/Fonts")
    preferred = windows / ("seguisb.ttf" if bold else "segoeui.ttf")
    fallback = windows / ("arialbd.ttf" if bold else "arial.ttf")
    return str(preferred if preferred.exists() else fallback)


def _font(size: int, bold: bool = False, scale: int = 1):
    return ImageFont.truetype(_font_path(bold), size * scale)


def _draw_arrow(draw, points, color, scale=1, dashed=False):
    pts = [(int(x * scale), int(y * scale)) for x, y in points]
    width = 5 * scale
    if dashed:
        for start, end in zip(pts, pts[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = max(1, int((dx * dx + dy * dy) ** 0.5))
            for offset in range(0, length, 22 * scale):
                a = offset / length
                b = min(length, offset + 12 * scale) / length
                draw.line(
                    (
                        start[0] + dx * a,
                        start[1] + dy * a,
                        start[0] + dx * b,
                        start[1] + dy * b,
                    ),
                    fill=color,
                    width=width,
                )
    else:
        draw.line(pts, fill=color, width=width, joint="curve")
    start, end = pts[-2], pts[-1]
    angle = atan2(end[1] - start[1], end[0] - start[0])
    length = 18 * scale
    spread = pi / 6
    head = [
        end,
        (end[0] - length * cos(angle - spread), end[1] - length * sin(angle - spread)),
        (end[0] - length * cos(angle + spread), end[1] - length * sin(angle + spread)),
    ]
    draw.polygon(head, fill=color)


def _label_position(points):
    segments = list(zip(points, points[1:]))
    start, end = max(
        segments,
        key=lambda pair: abs(pair[1][0] - pair[0][0]) + abs(pair[1][1] - pair[0][1]),
    )
    return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 13)


def render_png(path: Path) -> None:
    scale = 2
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((80 * scale, 55 * scale), "Movement-Controlled Markov Music System", font=_font(54, True, scale), fill=INK)
    draw.text(
        (82 * scale, 130 * scale),
        "Realtime architecture: vision → gestures → symbolic generation → sound and live notation",
        font=_font(27, False, scale),
        fill=MUTED,
    )

    for section in SECTIONS:
        xy = tuple(value * scale for value in (section.x, section.y, section.x + section.width, section.y + section.height))
        draw.rounded_rectangle(xy, radius=24 * scale, fill=section.fill, outline=section.color, width=3 * scale)
        draw.rounded_rectangle(
            tuple(value * scale for value in (section.x, section.y, section.x + section.width, section.y + 82)),
            radius=22 * scale,
            fill=section.color,
        )
        draw.rectangle(
            tuple(value * scale for value in (section.x, section.y + 58, section.x + section.width, section.y + 82)),
            fill=section.color,
        )
        draw.text(((section.x + 24) * scale, (section.y + 18) * scale), section.title, font=_font(29, True, scale), fill=WHITE)
        draw.text(((section.x + 22) * scale, (section.y + 87) * scale), section.subtitle, font=_font(18, False, scale), fill=section.color)

    for points, _label, color, dashed in CONNECTORS:
        _draw_arrow(draw, points, color, scale, dashed)

    for box in BOXES:
        xy = tuple(value * scale for value in (box.x, box.y, box.x + box.width, box.y + box.height))
        draw.rounded_rectangle(xy, radius=20 * scale, fill=box.fill, outline=box.color, width=3 * scale)
        draw.rounded_rectangle(
            tuple(value * scale for value in (box.x, box.y, box.x + 12, box.y + box.height)),
            radius=6 * scale,
            fill=box.color,
        )
        draw.text(((box.x + 30) * scale, (box.y + 22) * scale), box.title, font=_font(box.title_size, True, scale), fill=INK)
        y = box.y + 73
        for line in box.lines:
            draw.text(((box.x + 31) * scale, y * scale), line, font=_font(box.body_size, False, scale), fill=MUTED)
            y += box.body_size + 17

    for points, label, color, _dashed in CONNECTORS:
        if not label:
            continue
        x, y = _label_position(points)
        font = _font(17, True, scale)
        bbox = draw.textbbox((0, 0), label, font=font)
        pad = 7 * scale
        box = (
            x * scale - (bbox[2] - bbox[0]) / 2 - pad,
            y * scale - pad,
            x * scale + (bbox[2] - bbox[0]) / 2 + pad,
            y * scale + (bbox[3] - bbox[1]) + pad,
        )
        draw.rounded_rectangle(box, radius=8 * scale, fill=BACKGROUND)
        draw.text((x * scale, y * scale), label, font=font, fill=color, anchor="ma")

    footer = "One GeneratedTick drives both outputs. The live score reads symbolic notes directly and never transcribes the audio."
    draw.rounded_rectangle((420 * scale, 1600 * scale, 2380 * scale, 1710 * scale), radius=24 * scale, fill="#E4EBF0")
    draw.text((1400 * scale, 1655 * scale), footer, font=_font(25, True, scale), fill=INK, anchor="mm")

    image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(path, quality=95)


def _svg_text(x, y, text, size, color, bold=False, anchor="start"):
    weight = "600" if bold else "400"
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def render_svg(path: Path) -> None:
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-opacity="0.10"/></filter>',
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="context-stroke"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        _svg_text(80, 100, "Movement-Controlled Markov Music System", 54, INK, True),
        _svg_text(82, 155, "Realtime architecture: vision → gestures → symbolic generation → sound and live notation", 27, MUTED),
    ]

    for section in SECTIONS:
        chunks.append(
            f'<rect x="{section.x}" y="{section.y}" width="{section.width}" height="{section.height}" rx="24" fill="{section.fill}" stroke="{section.color}" stroke-width="3"/>'
        )
        chunks.append(
            f'<path d="M{section.x + 24},{section.y} H{section.x + section.width - 24} Q{section.x + section.width},{section.y} {section.x + section.width},{section.y + 24} V{section.y + 82} H{section.x} V{section.y + 24} Q{section.x},{section.y} {section.x + 24},{section.y} Z" fill="{section.color}"/>'
        )
        chunks.append(_svg_text(section.x + 24, section.y + 54, section.title, 29, WHITE, True))
        chunks.append(_svg_text(section.x + 22, section.y + 105, section.subtitle, 18, section.color))

    for points, _label, color, dashed in CONNECTORS:
        point_text = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="12 10"' if dashed else ""
        chunks.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'
        )

    for box in BOXES:
        chunks.append(
            f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="20" fill="{box.fill}" stroke="{box.color}" stroke-width="3" filter="url(#shadow)"/>'
        )
        chunks.append(f'<rect x="{box.x}" y="{box.y}" width="12" height="{box.height}" rx="6" fill="{box.color}"/>')
        chunks.append(_svg_text(box.x + 30, box.y + 51, box.title, box.title_size, INK, True))
        y = box.y + 93
        for line in box.lines:
            chunks.append(_svg_text(box.x + 31, y, line, box.body_size, MUTED))
            y += box.body_size + 17

    for points, label, color, _dashed in CONNECTORS:
        if not label:
            continue
        x, y = _label_position(points)
        width = max(74, len(label) * 9 + 18)
        chunks.append(f'<rect x="{x - width / 2}" y="{y - 19}" width="{width}" height="28" rx="8" fill="{BACKGROUND}"/>')
        chunks.append(_svg_text(x, y + 2, label, 17, color, True, "middle"))

    chunks.append('<rect x="420" y="1600" width="1960" height="110" rx="24" fill="#E4EBF0"/>')
    chunks.append(
        _svg_text(
            1400,
            1664,
            "One GeneratedTick drives both outputs. The live score reads symbolic notes directly and never transcribes the audio.",
            25,
            INK,
            True,
            "middle",
        )
    )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    svg_path = OUTPUT_DIR / "movement_to_music_architecture.svg"
    png_path = OUTPUT_DIR / "movement_to_music_architecture.png"
    render_svg(svg_path)
    render_png(png_path)
    print(f"Created {svg_path}")
    print(f"Created {png_path}")


if __name__ == "__main__":
    main()
