import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from PyQt5 import QtCore, QtGui, QtWidgets
from magenta_router import INPUT_ALL
from music_painting import color_for_chord
from performance_session import MODE_CONDUCTOR, MODE_FULL, MODE_SOPRANO
from piano_roll import PianoRollNote
from performance_ui import (
    IntegratedPerformanceWindow, PIANO_HIGH, PIANO_LOW,
    PIANO_ROW_HEIGHT, qt_compatible_svg,
)


class IntegratedWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_camera_score_and_controls_share_one_window(self):
        window = IntegratedPerformanceWindow(show_score=True)
        try:
            window.set_camera_frame(np.zeros((120, 160, 3), dtype=np.uint8))
            window.set_state(mode=MODE_SOPRANO, playing=False, summary="test")
            window.set_magenta_state(
                enabled=True,
                input_mode=INPUT_ALL,
                status="Connected (mock)",
            )
            window.update_score(
                {
                    "status": "Measure 1",
                    "page_versions": (1,),
                    "svgs": (
                        '<svg xmlns="http://www.w3.org/2000/svg" '
                        'width="100" height="50"></svg>',
                    ),
                    "bars_per_page": 4,
                }
            )
            window.update_piano_roll(
                {
                    "status": "Measure 1 | 1 soprano note",
                    "notes": (PianoRollNote(0, 8, 72, active=True),),
                    "current_tick": 7,
                    "ticks_per_beat": 16,
                    "ticks_per_bar": 64,
                    "visible_bars": 4,
                    "version": 1,
                }
            )
            self.assertIsNotNone(window._effects_view._camera_image)
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
            window.set_chord_label("Chord: C")
            window.set_hand_overlays((
                (tuple((0.5, 0.5) for _ in range(21)),
                 (True, False, False, False, False), "#00e070"),
            ))
            self.assertEqual(window._effects_view._chord_label, "Chord: C")
            self.assertEqual(len(window._effects_view._hand_overlays), 1)
            self.assertTrue(window.soprano_button.isChecked())
            self.assertIn("Play", window.transport_button.text())
            self.assertEqual(window.visual_title.text(), "Soprano piano roll")
            self.assertEqual(window.score_status.text(), "Measure 1 | 1 soprano note")
            self.assertIs(window.visual_stack.currentWidget(), window.piano_roll)
            self.assertEqual(len(window._score_labels), 1)
            self.assertTrue(window.magenta_button.isChecked())
            self.assertEqual(window.magenta_input.currentData(), INPUT_ALL)
            self.assertEqual(window.magenta_status.text(), "Connected (mock)")

            window.set_state(mode=MODE_FULL, playing=True, summary="full")
            window.update_score(
                {
                    "status": "Measure 1",
                    "page_versions": (1,),
                    "svgs": (
                        '<svg xmlns="http://www.w3.org/2000/svg" '
                        'width="100" height="50"></svg>',
                    ),
                    "bars_per_page": 4,
                }
            )
            self.assertEqual(window.visual_title.text(), "Live SATB score")
            self.assertEqual(window.score_status.text(), "Measure 1")
            self.assertIs(window.visual_stack.currentWidget(), window.score_scroll)

            window.set_state(mode=MODE_CONDUCTOR, playing=True, summary="conduct")
            window.set_swap_roles(True)
            self.assertTrue(window.conductor_button.isChecked())
            self.assertTrue(window.swap_roles_button.isEnabled())
            self.assertTrue(window.swap_roles_button.isChecked())
            self.assertEqual(window.visual_title.text(), "Conducted melody")
            self.assertIs(window.visual_stack.currentWidget(), window.piano_roll)
            self.assertFalse(window.visual_effects_enabled)
            window.set_visual_hand_positions(((0.25, 0.75),))
            window.set_visual_effects_enabled(True)
            self.assertTrue(window.visual_effects_enabled)
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
            self.assertIs(window.visual_stack.currentWidget(), window.piano_roll)
            self.assertIsNotNone(window._effects_view._camera_image)
            window.on_music_event(pitch=72, velocity=100, voices=4)
            self.assertGreater(len(window._effects_view._particles), 0)
            self.assertEqual(window._effects_view._pulses[-1].x, 0.25)
            self.assertEqual(window._effects_view._pulses[-1].y, 0.75)
            window.set_state(mode=MODE_FULL, playing=True, summary="full")
            self.assertIs(window.visual_stack.currentWidget(), window.score_scroll)
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
            window.set_visual_effects_enabled(False)
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
            self.assertIs(window.visual_stack.currentWidget(), window.score_scroll)
        finally:
            window.close()

    def test_verovio_references_are_expanded_for_qt_svg(self):
        source = """<svg xmlns="http://www.w3.org/2000/svg"
            xmlns:xlink="http://www.w3.org/1999/xlink" color="black">
            <defs><g id="note"><path d="M0 0 L10 0 L10 10 Z" /></g></defs>
            <style>path { stroke: currentColor }</style>
            <g color="#D32F2F" fill="#D32F2F">
                <use xlink:href="#note" transform="translate(5,5)" />
            </g>
        </svg>"""

        converted = qt_compatible_svg(source)
        root = ET.fromstring(converted)

        self.assertFalse(any(element.tag.endswith("use") for element in root.iter()))
        self.assertFalse(any(element.tag.endswith("defs") for element in root.iter()))
        paths = [element for element in root.iter() if element.tag.endswith("path")]
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].attrib["stroke"], "#D32F2F")
        self.assertEqual(paths[0].attrib["fill"], "#D32F2F")

    def test_visual_effects_can_open_when_live_score_is_disabled(self):
        window = IntegratedPerformanceWindow(show_score=False)
        try:
            self.assertTrue(window.score_panel.isHidden())
            window.set_visual_effects_enabled(True)
            self.assertTrue(window.score_panel.isHidden())
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
            window.set_visual_effects_enabled(False)
            self.assertTrue(window.score_panel.isHidden())
            self.assertIs(window.camera_stack.currentWidget(), window._effects_view)
        finally:
            window.close()

    def test_hand_distance_light_grows_and_hides_when_tracking_is_lost(self):
        window = IntegratedPerformanceWindow(show_score=False)
        try:
            def render(level, positions):
                window.set_visual_hand_positions(positions, distance_level=level)
                image = QtGui.QImage(640, 480, QtGui.QImage.Format_ARGB32)
                image.fill(QtGui.QColor("black"))
                painter = QtGui.QPainter(image)
                window._effects_view._paint_hand_light(
                    painter, QtCore.QRectF(0, 0, 640, 480)
                )
                painter.end()
                return image

            hands = ((0.25, 0.5), (0.75, 0.5))
            small = render(0.0, hands)
            large = render(1.0, hands)
            sample = (320, 320)  # 80 pixels below the midpoint, off the beam.
            self.assertEqual(small.pixelColor(*sample), QtGui.QColor("black"))
            self.assertNotEqual(large.pixelColor(*sample), QtGui.QColor("black"))
            missing = render(1.0, hands[:1])
            self.assertEqual(missing.pixelColor(320, 240), QtGui.QColor("black"))
        finally:
            window.close()

    def test_piano_roll_covers_the_full_performed_pitch_range(self):
        window = IntegratedPerformanceWindow(show_score=True)
        try:
            self.assertEqual((PIANO_LOW, PIANO_HIGH), (48, 96))
            self.assertEqual(window.piano_roll.canvas.height(),
                             (PIANO_HIGH - PIANO_LOW + 1) * PIANO_ROW_HEIGHT)
            self.assertEqual(window.piano_roll.canvas._note_y(PIANO_HIGH), 0)
            self.assertEqual(window.piano_roll.canvas._note_y(PIANO_LOW),
                             (PIANO_HIGH - PIANO_LOW) * PIANO_ROW_HEIGHT)
        finally:
            window.close()

    def test_music_painting_keeps_marks_and_exports_png(self):
        window = IntegratedPerformanceWindow(show_score=True)
        try:
            window.set_state(mode=MODE_SOPRANO, playing=True, summary="painting")
            window.set_visual_hand_positions((), paint_position=(0.3, 0.65))
            before = window.music_painting._canvas.pixelColor(480, 650)
            window.on_music_event(
                pitch=72, velocity=100, voices=1,
                duration_ticks=16, mode=MODE_SOPRANO, chord=(0, "maj"),
            )
            self.assertEqual(window.music_painting.mark_count, 1)
            self.assertNotEqual(
                window.music_painting._canvas.pixelColor(480, 650), before
            )
            self.assertEqual(window.music_painting._marks[0].x, 0.3)
            self.assertEqual(window.music_painting._marks[0].duration_ticks, 16)
            window.set_visual_hand_positions((), paint_position=(0.6, 0.65))
            window.on_music_event(
                pitch=76, velocity=80, voices=1,
                duration_ticks=8, mode=MODE_SOPRANO, chord=(0, "maj"),
            )
            window.on_music_event(
                pitch=79, velocity=80, voices=1,
                duration_ticks=8, mode=MODE_SOPRANO, chord=(7, "maj"),
            )
            marks = window.music_painting._marks
            self.assertEqual(color_for_chord(marks[0].chord), color_for_chord(marks[1].chord))
            self.assertNotEqual(color_for_chord(marks[1].chord), color_for_chord(marks[2].chord))
            self.assertNotEqual(color_for_chord((0, "maj")), color_for_chord((0, "min")))
            window.set_painting_enabled(True)
            self.assertIs(window.visual_stack.currentWidget(), window.music_painting)
            self.assertEqual(window.visual_title.text(), "Music painting")

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "painting.png"
                self.assertTrue(window.music_painting.save_png(path))
                self.assertEqual(QtGui.QImage(str(path)).width(), 1600)
                self.assertTrue(window.clear_painting_button.isEnabled())
                window.clear_painting_button.click()
                self.assertEqual(window.music_painting.mark_count, 0)
                self.assertEqual(
                    window.music_painting._canvas.pixelColor(480, 650), before
                )
                self.assertNotEqual(QtGui.QImage(str(path)).pixelColor(480, 650), before)
                self.assertFalse(window.clear_painting_button.isEnabled())
                window.on_music_event(
                    pitch=72, velocity=100, voices=1,
                    mode=MODE_SOPRANO, chord=(0, "maj"),
                    event_time=window._painting_clear_time - 0.01,
                )
                self.assertEqual(window.music_painting.mark_count, 0)
                window.on_music_event(
                    pitch=72, velocity=100, voices=1,
                    mode=MODE_SOPRANO, chord=(0, "maj"),
                    event_time=window._painting_clear_time + 0.01,
                )
                self.assertEqual(window.music_painting.mark_count, 1)
                self.assertTrue(window.clear_painting_button.isEnabled())

            window.set_state(mode=MODE_FULL, playing=True, summary="full")
            self.assertIs(window.visual_stack.currentWidget(), window.score_scroll)
            window.set_state(mode=MODE_SOPRANO, playing=True, summary="again")
            self.assertIs(window.visual_stack.currentWidget(), window.music_painting)
            window.set_painting_enabled(False)
            self.assertIs(window.visual_stack.currentWidget(), window.piano_roll)
            self.assertEqual(window.music_painting.mark_count, 1)
        finally:
            window.close()

    def test_chord_only_event_changes_color_without_drawing(self):
        window = IntegratedPerformanceWindow(show_score=True)
        try:
            window.set_state(mode=MODE_CONDUCTOR, playing=True, summary="")
            window.set_visual_hand_positions((), paint_position=(0.4, 0.5))
            window.on_music_event(pitch=60, velocity=90, voices=3,
                                  chord=(0, "maj"), soprano_onset=False)
            self.assertEqual(window.music_painting.mark_count, 0)
            window.on_music_event(pitch=72, velocity=90, voices=1,
                                  soprano_onset=True)
            self.assertEqual(window.music_painting._marks[-1].chord, (0, "maj"))
            window.on_music_event(pitch=64, velocity=90, voices=3,
                                  chord=(7, "maj"), soprano_onset=False)
            self.assertEqual(window.music_painting.mark_count, 1)
            window.on_music_event(pitch=74, velocity=90, voices=1,
                                  soprano_onset=True)
            self.assertEqual(window.music_painting._marks[-1].chord, (7, "maj"))
            self.assertEqual(window.music_painting.mark_count, 2)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
