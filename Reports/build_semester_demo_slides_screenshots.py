"""Build the semester demo from real app screenshots and generated score."""

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "PresentationAssets"
OUTPUT = ROOT / "Yorgo_Azzi_Movement_to_Sound_Semester_Demo_Screenshots.pptx"
SCORE = ASSETS / "generated_score_excerpt.png"
PAINTING = ASSETS / "app_music_painting.png"
ROLL = ASSETS / "app_piano_roll.png"
EFFECTS = ASSETS / "app_camera_visuals.png"
NAVY = RGBColor(14, 35, 47)
INK = RGBColor(24, 49, 61)
TEAL = RGBColor(35, 122, 112)
MUTED = RGBColor(85, 104, 111)
PALE = RGBColor(226, 234, 233)
WHITE = RGBColor(255, 255, 255)
PAPER = RGBColor(248, 250, 249)
GOLD = RGBColor(235, 184, 93)


def box(slide, x, y, w, h, color, rounded=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def txt(slide, value, x, y, w, h, size=20, color=INK, bold=False,
        align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0)
    frame.margin_top = frame.margin_bottom = Inches(0)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def image(slide, path, x, y, w, h):
    with Image.open(path) as source:
        iw, ih = source.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    slide.shapes.add_picture(
        str(path), Inches(x + (w - dw) / 2), Inches(y + (h - dh) / 2),
        width=Inches(dw), height=Inches(dh),
    )


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value.strip()


def heading(slide, number, title, subtitle):
    box(slide, 0, 0, 13.333, 7.5, PAPER)
    txt(slide, number, 0.62, 0.36, 0.55, 0.34, 15, TEAL, True)
    txt(slide, title, 1.18, 0.26, 11.5, 0.58, 30, NAVY, True)
    txt(slide, subtitle, 1.20, 0.87, 11.30, 0.40, 16, MUTED)
    box(slide, 0.62, 1.37, 12.08, 0.025, PALE)


def make_deck():
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]

    # Cover: second attachment, a real music-painting screenshot.
    slide = deck.slides.add_slide(blank)
    box(slide, 0, 0, 13.333, 7.5, NAVY)
    txt(slide, "MOVEMENT TO SOUND", 0.62, 0.37, 5.6, 0.31, 16, GOLD, True)
    txt(slide, "Music you can conduct and paint", 0.62, 0.76, 12, 0.66, 33, WHITE, True)
    box(slide, 0.58, 1.62, 9.62, 5.20, WHITE, True)
    image(slide, PAINTING, 0.66, 1.68, 9.46, 5.06)
    txt(slide, "A live performance system", 10.46, 1.92, 2.18, 0.75, 22, WHITE, True)
    box(slide, 10.46, 2.85, 1.26, 0.055, GOLD)
    txt(slide, "Hand movement steers a Markov music generator.",
        10.46, 3.18, 2.18, 1.25, 19, WHITE)
    txt(slide, "The same performance produces sound and a visual trace.",
        10.46, 4.87, 2.18, 1.25, 18, PALE)
    txt(slide, "Yorgo Azzi  |  Semester demo", 0.65, 7.06, 8, 0.24, 13, PALE)
    notes(slide, """0:00–0:45. This is a real screenshot of the current app, with
    MediaPipe hand tracking on the left and music painting on the right. The
    project began as a Markov generator of symbolic music. I turned it into a
    live system where hand gestures control musical decisions and visuals.""")

    # First pipeline stage: real generated score and a marker-photo placeholder.
    slide = deck.slides.add_slide(blank)
    heading(slide, "01", "From generated score to movement input",
            "First, the Markov model generated music. Then a colored camera marker changed its volume.")
    txt(slide, "MARKOV SCORE", 0.74, 1.66, 5.62, 0.34, 16, TEAL, True)
    txt(slide, "COLORED-MARKER TRACKING", 7.00, 1.66, 5.62, 0.34, 16, TEAL, True)
    box(slide, 0.70, 2.08, 5.70, 3.33, WHITE, True)
    image(slide, SCORE, 0.83, 2.20, 5.43, 3.05)
    placeholder = box(slide, 6.94, 2.08, 5.70, 3.33, WHITE)
    placeholder.line.color.rgb = TEAL
    placeholder.line.width = Pt(2)
    placeholder.line.dash_style = 3
    txt(slide, "Insert marker-detection\nscreenshot here", 7.45, 2.94,
        4.68, 0.90, 23, MUTED, True, PP_ALIGN.CENTER)
    txt(slide, "Colored ribbon / blob tracked by OpenCV", 7.36, 4.07,
        4.86, 0.38, 16, MUTED, align=PP_ALIGN.CENTER)
    txt(slide, "Generated four-part score", 0.78, 5.59, 5.6, 0.38, 18, NAVY, True)
    txt(slide, "Marker height controls volume", 7.01, 5.59, 5.60, 0.38, 18, NAVY, True)
    box(slide, 0.72, 6.36, 11.91, 0.72, NAVY, True)
    txt(slide, "Markov generation  →  realtime MIDI  →  camera movement control",
        1.05, 6.53, 11.27, 0.35, 18, WHITE, align=PP_ALIGN.CENTER)
    notes(slide, """0:45–2:20. The left image is an excerpt from music the
    original Markov model generated. It learned pitch motion, rhythm and chord
    transitions and produced a four-part score. I made generation persistent
    and used MIDI for live playback. The first movement interface used OpenCV
    to detect a colored ribbon as a blob; its height controlled volume. The
    photo space is intentionally empty for a real marker-detection capture.""")

    # Second pipeline stage: before/after captures of the actual application.
    slide = deck.slides.add_slide(blank)
    heading(slide, "02", "From hand tracking to a visual performance",
            "MediaPipe hand landmarks control the music; the app displays and visualizes its output.")
    txt(slide, "PIANO ROLL", 0.76, 1.65, 5.65, 0.34, 16, TEAL, True)
    txt(slide, "CAMERA EFFECTS + PAINTING", 6.91, 1.65, 5.70, 0.34, 16, TEAL, True)
    box(slide, 0.68, 2.09, 5.87, 3.52, WHITE, True)
    box(slide, 6.78, 2.09, 5.87, 3.52, WHITE, True)
    image(slide, ROLL, 0.78, 2.19, 5.67, 3.32)
    image(slide, EFFECTS, 6.88, 2.19, 5.67, 3.32)
    txt(slide, "Hands shape the music; the soprano appears as notes play.",
        0.81, 5.80, 5.56, 0.70, 17, INK)
    txt(slide, "Sound also creates light, particles and chord-colored marks.",
        6.90, 5.80, 5.60, 0.70, 17, INK)
    box(slide, 0.72, 6.73, 11.91, 0.035, TEAL)
    txt(slide, "OpenCV capture  →  MediaPipe tracking  →  Markov engine  →  MIDI + visuals",
        0.77, 6.91, 11.80, 0.29, 17, NAVY, align=PP_ALIGN.CENTER)
    notes(slide, """2:20–3:55. Both images are real app screenshots. The left
    shows MediaPipe hand landmarks and a soprano piano roll, without camera
    effects. The right shows the later camera effects and music painting.
    OpenCV captures frames; MediaPipe estimates hand landmarks. Hand features
    update the Markov engine and MIDI performance, and the same music events
    drive the live visual output. Next I will show the three performance modes.""")

    # On-screen guide for the live demonstration.
    slide = deck.slides.add_slide(blank)
    box(slide, 0, 0, 13.333, 7.5, NAVY)
    txt(slide, "LIVE DEMO", 0.66, 0.35, 2.3, 0.31, 16, GOLD, True)
    txt(slide, "Three ways to perform", 0.66, 0.78, 12, 0.66, 35, WHITE, True)
    modes = [
        ("1", "Full SATB", "The model generates all four voices.",
         "Hands shape chords, register, tempo and articulation."),
        ("2", "Soprano Performance", "The model supplies three lower voices.",
         "The performer controls the soprano melody."),
        ("3", "Conductor", "A downward hand stroke triggers each beat.",
         "Finger gestures choose chords and shape the melody."),
    ]
    for index, (number, title, first, second) in enumerate(modes):
        x = 0.66 + index * 4.22
        box(slide, x, 1.87, 3.96, 4.29, WHITE, True)
        box(slide, x, 1.87, 3.96, 0.09, TEAL)
        txt(slide, number, x + 0.25, 2.20, 0.63, 0.60, 35, TEAL, True)
        txt(slide, title, x + 0.25, 2.96, 3.45, 0.66, 23, NAVY, True)
        box(slide, x + 0.25, 3.77, 0.84, 0.04, GOLD)
        txt(slide, first, x + 0.25, 4.03, 3.43, 0.76, 18, INK)
        txt(slide, second, x + 0.25, 5.00, 3.43, 0.84, 17, MUTED)
    txt(slide, "1 / 2 / 3 switch modes   ·   Space pauses   ·   P: piano roll / painting   ·   V: camera effects",
        0.75, 6.69, 11.94, 0.47, 15, WHITE, align=PP_ALIGN.CENTER)
    notes(slide, """3:55–7:30. Press 1 for Full SATB: the model generates all
    voices while gestures shape its controls. Press 2 for Soprano Performance:
    the model supplies the lower three voices and the right hand performs the
    soprano. Press 3 for Conductor: a shared downward stroke triggers a chord
    beat, right fingers select a chord degree, and right-hand height shapes
    the melody between beats. Left fingers choose melody steps; a left fist
    holds the last soprano note. P switches piano roll and painting. V toggles
    camera effects. The light between hands responds to their distance.""")

    deck.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(make_deck())
