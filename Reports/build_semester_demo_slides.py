"""Build the four-slide movement-to-sound semester presentation."""

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Yorgo_Azzi_Movement_to_Sound_Semester_Demo_Revised.pptx"
COVER = HERE / "PresentationAssets" / "movement_to_sound_cover.png"
SCORE = HERE / "PresentationAssets" / "generated_score_excerpt.png"
APP = HERE / "PresentationAssets" / "hand_app_interface_preview.png"

NAVY = RGBColor(13, 35, 50)
INK = RGBColor(25, 51, 67)
TEAL = RGBColor(35, 144, 142)
LIGHT_TEAL = RGBColor(113, 225, 213)
AMBER = RGBColor(236, 169, 97)
MUTED = RGBColor(87, 108, 120)
PALE = RGBColor(234, 239, 240)
WHITE = RGBColor(255, 255, 255)
OFF_WHITE = RGBColor(248, 249, 247)


def rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def text(slide, value, left, top, width, height, *, size=20, color=INK,
         bold=False, align=PP_ALIGN.LEFT, font="Arial"):
    shape = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0)
    frame.margin_right = Inches(0)
    frame.margin_top = Inches(0)
    frame.margin_bottom = Inches(0)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def notes(slide, content):
    slide.notes_slide.notes_text_frame.text = content.strip()


def image_fit(slide, path, left, top, width, height):
    with Image.open(path) as source:
        image_width, image_height = source.size
    scale = min(width / image_width, height / image_height)
    draw_width, draw_height = image_width * scale, image_height * scale
    slide.shapes.add_picture(
        str(path), Inches(left + (width - draw_width) / 2),
        Inches(top + (height - draw_height) / 2),
        width=Inches(draw_width), height=Inches(draw_height),
    )


def make_deck():
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]

    # Slide 1: one concept and presenter. The image is explicitly illustrative.
    slide = deck.slides.add_slide(blank)
    slide.shapes.add_picture(str(COVER), 0, 0, width=deck.slide_width,
                             height=deck.slide_height)
    text(slide, "Movement to sound", 0.68, 2.02, 5.9, 0.85,
         size=43, color=WHITE, bold=True)
    text(slide, "Conducting a realtime music generator", 0.71, 2.93,
         5.7, 1.18, size=23, color=LIGHT_TEAL)
    text(slide, "Yorgo Azzi  |  Semester project", 0.71, 6.75,
         6.5, 0.34, size=16, color=WHITE)
    text(slide, "Concept illustration", 10.78, 6.96,
         1.9, 0.18, size=9, color=PALE, align=PP_ALIGN.RIGHT)
    notes(slide, """
    0:00-0:40. This semester I adapted my existing Markov music generator into
    an instrument controlled by movement. The webcam observes hand motion;
    gestures control when music happens and what it plays. The cover image is
    an illustration, not a screenshot of the application. I will show the
    actual application on the final slide's demo.
    """)

    # Slide 2: the actual generated score beside an interface preview.
    slide = deck.slides.add_slide(blank)
    rect(slide, 0, 0, 13.333, 7.5, OFF_WHITE)
    text(slide, "From generated score to hand-controlled music", 0.70, 0.54,
         12.0, 0.64, size=32, color=NAVY, bold=True)
    rect(slide, 0.72, 1.28, 11.90, 0.035, TEAL)
    text(slide, "MARKOV GENERATOR", 0.78, 1.60, 5.68, 0.36,
         size=18, color=TEAL, bold=True)
    text(slide, "MOVEMENT-CONTROLLED APP", 6.87, 1.60, 5.68, 0.36,
         size=18, color=TEAL, bold=True)
    rect(slide, 0.76, 2.04, 5.76, 3.75, WHITE)
    rect(slide, 6.80, 2.04, 5.76, 3.75, WHITE)
    image_fit(slide, SCORE, 0.83, 2.14, 5.62, 3.50)
    image_fit(slide, APP, 6.87, 2.14, 5.62, 3.50)
    text(slide, "Generated four-part score", 0.78, 5.90, 5.68, 0.42,
         size=18, color=INK, bold=True)
    text(slide, "Camera + MediaPipe hand landmarks + live score", 6.87, 5.90,
         5.69, 0.52, size=18, color=INK, bold=True)
    text(slide, "A persistent realtime engine connects gestures to the music.",
         0.78, 6.70, 11.78, 0.38, size=20, color=NAVY)
    text(slide, "Interface preview; camera frame illustrative", 9.35, 7.18,
         3.19, 0.18, size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    notes(slide, """
    0:40-2:15. The left image is a four-measure excerpt from a score actually
    generated by the original Markov system. It learned pitch motion, rhythm
    and chord progressions, then generated the four voices. I made that engine
    persistent and changed playback to scheduled MIDI so new controls could
    take effect while it played. The right image shows the current Qt app
    interface with camera tracking, the three mode buttons and live score.
    MediaPipe supplies hand and finger landmarks; movement then influences
    timing, harmony and melody. The pictured camera frame and landmark overlay
    are illustrative because a live webcam was unavailable while building the
    presentation. Use the running app for the real demonstration on slide 4.
    """)

    # Slide 3: evolution pipeline and its three selectable modes.
    slide = deck.slides.add_slide(blank)
    rect(slide, 0, 0, 13.333, 7.5, OFF_WHITE)
    text(slide, "How the system evolved", 0.70, 0.54, 11.9, 0.64,
         size=34, color=NAVY, bold=True)
    rect(slide, 0.72, 1.28, 11.90, 0.035, TEAL)

    stages = [
        ("Markov model", "Generated\nscore"),
        ("Realtime MIDI", "Continuous\nplayback"),
        ("OpenCV ribbon", "First movement\ninput"),
        ("MediaPipe", "Two-hand\ntracking"),
        ("Live app", "Music +\nvisuals"),
    ]
    for index, (heading, subtitle) in enumerate(stages):
        x = 0.73 + index * 2.50
        rect(slide, x, 1.82, 2.12, 1.36, WHITE)
        rect(slide, x, 1.82, 2.12, 0.06, TEAL)
        text(slide, heading, x + 0.14, 2.00, 1.84, 0.38,
             size=18, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        text(slide, subtitle, x + 0.14, 2.43, 1.84, 0.53,
             size=15, color=MUTED, align=PP_ALIGN.CENTER)
        if index < len(stages) - 1:
            text(slide, "→", x + 2.16, 2.23, 0.28, 0.44,
                 size=27, color=TEAL, align=PP_ALIGN.CENTER)

    text(slide, "Three performance modes", 0.75, 3.76, 11.9, 0.48,
         size=25, color=NAVY, bold=True)
    rect(slide, 0.75, 4.34, 11.84, 0.025, PALE)
    modes = [
        ("01", "Full SATB", "Hands shape the four generated voices."),
        ("02", "Soprano Performance", "Hand plays melody; model supplies three voices."),
        ("03", "Conductor", "Downstroke triggers beats; hands shape chords and melody."),
    ]
    for index, (number, heading, body) in enumerate(modes):
        x = 0.76 + index * 4.02
        text(slide, number, x, 4.70, 0.55, 0.38,
             size=18, color=TEAL, bold=True)
        text(slide, heading, x + 0.57, 4.68, 3.27, 0.43,
             size=20, color=NAVY, bold=True)
        text(slide, body, x + 0.57, 5.30, 3.17, 0.90,
             size=17, color=INK)
    text(slide, "All three modes share the same camera, music engine and MIDI output.",
         0.75, 6.72, 11.9, 0.40, size=18, color=MUTED)
    notes(slide, """
    2:15-3:45. The project moved through five stages. The original Markov
    model produced complete symbolic scores. I first made its generation
    persistent and played notes directly over MIDI. An OpenCV colored-ribbon
    prototype established a movement-to-control connection. MediaPipe then
    supplied two-hand landmarks and finger tracking, integrated in one Qt
    performance app. There are three modes: Full SATB, where gestures change
    all four generated voices; Soprano Performance, where the user plays the
    melody and the model supplies the lower voices; and Conductor, where a
    downstroke advances each beat and gestures shape chords and melody.
    The app also displays a live score or piano roll, offers music painting,
    and can route notes to Magenta RealTime on a compatible machine.
    """)

    # Slide 4: run the live demonstration while this slide stays on screen.
    slide = deck.slides.add_slide(blank)
    rect(slide, 0, 0, 13.333, 7.5, NAVY)
    text(slide, "Live demo and takeaways", 0.70, 0.54, 11.9, 0.64,
         size=34, color=WHITE, bold=True)
    rect(slide, 0.72, 1.28, 11.90, 0.035, LIGHT_TEAL)
    rect(slide, 8.13, 1.58, 0.022, 5.12, RGBColor(81, 115, 130))

    steps = [
        ("1  Conduct a chord",
         "Left fist + both hands down: one beat. Right fingers choose the chord degree."),
        ("2  Shape the melody",
         "Right-hand height moves the melody; left fingers choose the jump size."),
        ("3  Reveal the output",
         "Press P for music painting and V for particles over the camera image."),
    ]
    for index, (heading, body) in enumerate(steps):
        y = 1.62 + index * 1.53
        text(slide, heading, 0.76, y, 6.87, 0.48,
             size=23, color=LIGHT_TEAL, bold=True)
        text(slide, body, 0.78, y + 0.56, 6.96, 0.72,
             size=18, color=WHITE)

    text(slide, "What changed", 8.48, 1.64, 4.00, 0.48,
         size=23, color=AMBER, bold=True)
    text(slide, "The performer decides when harmony advances.\n\n"
                "The melody can change between beats.\n\n"
                "The same note events drive sound and visuals.",
         8.48, 2.28, 4.00, 3.85, size=19, color=WHITE)
    notes(slide, """
    3:45-7:30. Before presenting, open the app in Conductor mode with MIDI output
    and camera visible. Demonstrate a left fist and synchronized downward stroke:
    one accompaniment beat is generated at the gesture, not after a full bar.
    Change right-hand finger pattern between beats to choose scale degree 1-7.
    Move the right hand vertically to step the soprano up or down; extend left
    fingers to change interval size. Hold a left fist to sustain the last note.
    Press P to show how generated notes accumulate into a painting; press V for
    particles centered on tracked hands. If Magenta is running on a compatible
    computer, briefly enable it and compare soprano-only with all-voice input.
    If not, omit Magenta from the live demo. The project was tested with Magenta
    on a compatible machine. Close with limitations: finger tracking can fail
    under occlusion or fast motion, and end-to-end audible latency has not yet
    been measured with a performer. Total talk/demo target: about 7-8 minutes.
    """)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    deck.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(make_deck())
