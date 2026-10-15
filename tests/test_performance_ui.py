import os
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from PyQt5 import QtWidgets
from performance_session import MODE_SOPRANO
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
            self.assertIsNotNone(window.camera_label.pixmap())
            self.assertTrue(window.soprano_button.isChecked())
            self.assertIn("Play", window.transport_button.text())
            self.assertEqual(window.score_status.text(), "Measure 1")
            self.assertEqual(len(window._score_labels), 1)
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


if __name__ == "__main__":
    unittest.main()
