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

from PyQt5 import QtGui, QtWidgets
from magenta_router import INPUT_ALL
from music_painting import color_for_chord
from performance_session import MODE_CONDUCTOR, MODE_FULL, MODE_SOPRANO
from piano_roll import PianoRollNote
from performance_ui import IntegratedPerformanceWindow, qt_compatible_svg


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
            self.assertIsNotNone(window.camera_label.pixmap())
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
            self.assertIs(window.camera_stack.currentWidget(), window.camera_label)
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
            self.assertIs(window.camera_stack.currentWidget(), window.camera_label)
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

            window.set_state(mode=MODE_FULL, playing=True, summary="full")
            self.assertIs(window.visual_stack.currentWidget(), window.score_scroll)
            window.set_state(mode=MODE_SOPRANO, playing=True, summary="again")
            self.assertIs(window.visual_stack.currentWidget(), window.music_painting)
            window.set_painting_enabled(False)
            self.assertIs(window.visual_stack.currentWidget(), window.piano_roll)
            self.assertEqual(window.music_painting.mark_count, 3)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
