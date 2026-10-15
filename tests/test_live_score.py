import sys
import json
import time
import unittest
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from live_score import LiveScoreDisplay, RollingNotation


def frame(
    tick,
    duration,
    pitches,
    chord=(0, "maj", 4),
    articulation=0.8,
    new_voices=None,
    released_voices=(),
):
    values = dict(
        new_notes=True,
        tick_index=tick,
        planned_duration_ticks=duration,
        duration_seconds=0.125,
        pitches=pitches,
        chord=chord,
        controls=SimpleNamespace(articulation=articulation),
    )
    if new_voices is not None:
        values["new_voices"] = frozenset(new_voices)
        values["voice_duration_ticks"] = {
            voice: duration for voice in new_voices
        }
    values["released_voices"] = frozenset(released_voices)
    return SimpleNamespace(**values)


class LiveScoreTests(unittest.TestCase):
    def setUp(self):
        self.first_pitches = {
            "bass": 48,
            "tenor": 55,
            "alto": 60,
            "soprano": 64,
        }
        self.second_pitches = {
            "bass": 50,
            "tenor": 57,
            "alto": 62,
            "soprano": 65,
        }

    def test_onset_finalizes_previous_duration_and_ignores_clock_ticks(self):
        notation = RollingNotation(ticks_per_beat=4, beats_per_bar=4)
        notation.add_frame(frame(0, 16, self.first_pitches))
        ignored = SimpleNamespace(new_notes=False)

        self.assertFalse(notation.add_frame(ignored))
        notation.add_frame(frame(6, 4, self.second_pitches))

        self.assertEqual(notation.events[0].duration_ticks, 6)
        self.assertEqual(notation.events[1].duration_ticks, 4)

    def test_two_staff_musicxml_renders_with_verovio(self):
        import verovio

        notation = RollingNotation(
            ticks_per_beat=4,
            beats_per_bar=4,
            visible_bars=4,
            key_name="C",
            mode="major",
        )
        # The first sonority crosses a bar line, which exercises ties.
        notation.add_frame(frame(0, 20, self.first_pitches))
        notation.add_frame(
            frame(
                20,
                4,
                self.second_pitches,
                chord=(2, "min", 4),
                articulation=0.2,
            )
        )

        musicxml = notation.musicxml()
        toolkit = verovio.toolkit()

        self.assertIn("<score-partwise", musicxml)
        self.assertIn("<tie type=\"start\"", musicxml)
        self.assertIn("<stem>up</stem>", musicxml)
        self.assertIn("<stem>down</stem>", musicxml)
        self.assertNotIn("Soprano / Alto", musicxml)
        self.assertNotIn("Tenor / Bass", musicxml)
        expected_stems = {"1": "up", "2": "down", "3": "up", "4": "down"}
        seen_voices = set()
        for xml_note in ET.fromstring(musicxml).iter("note"):
            if xml_note.find("rest") is not None:
                continue
            voice = xml_note.findtext("voice")
            seen_voices.add(voice)
            self.assertEqual(xml_note.findtext("stem"), expected_stems[voice])
        self.assertEqual(seen_voices, set(expected_stems))
        self.assertTrue(toolkit.loadData(musicxml))
        self.assertIn("<svg", toolkit.renderToSVG(1))

    def test_soprano_events_do_not_split_held_accompaniment_notes(self):
        notation = RollingNotation(ticks_per_beat=16, beats_per_bar=4)
        notation.add_frame(
            frame(
                0,
                16,
                self.first_pitches,
                new_voices=("bass", "tenor", "alto", "soprano"),
            )
        )
        pitches = dict(self.first_pitches)
        pitches["soprano"] = 67
        notation.add_frame(
            frame(4, 4, pitches, new_voices=("soprano",))
        )
        pitches["soprano"] = 69
        notation.add_frame(
            frame(8, 4, pitches, new_voices=("soprano",))
        )

        alto_spans = notation._voice_spans("alto")
        soprano_spans = notation._voice_spans("soprano")
        self.assertEqual(len(alto_spans), 1)
        self.assertEqual((alto_spans[0].start_tick, alto_spans[0].end_tick), (0, 16))
        self.assertEqual(len(soprano_spans), 3)

    def test_background_renderer_serves_updated_score(self):
        display = LiveScoreDisplay(
            ticks_per_beat=4,
            beats_per_bar=4,
            key_name="C",
            mode="major",
            visible_bars=4,
            port=0,
        ).start(open_browser=False)
        try:
            display.observe(frame(0, 4, self.first_pitches))
            deadline = time.monotonic() + 5.0
            state = {"version": 0}
            while time.monotonic() < deadline and state["version"] == 0:
                with urllib.request.urlopen(
                    display.url + "state.json", timeout=1
                ) as response:
                    state = json.load(response)
                time.sleep(0.02)
            with urllib.request.urlopen(
                display.url + "score.svg", timeout=1
            ) as response:
                svg = response.read().decode("utf-8")

            self.assertGreater(state["version"], 0)
            self.assertIn("Measure 1", state["status"])
            self.assertEqual(len(state["page_versions"]), 1)
            self.assertIn("<svg", svg)

            # Advance into measure 5. The first four-measure section must
            # remain available while a second live section is created.
            for tick in (16, 32, 48, 64):
                display.observe(frame(tick, 4, self.second_pitches))
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with urllib.request.urlopen(
                    display.url + "state.json", timeout=1
                ) as response:
                    state = json.load(response)
                if len(state["page_versions"]) == 2:
                    break
                time.sleep(0.02)

            with urllib.request.urlopen(
                display.url + "page/0.svg", timeout=1
            ) as response:
                old_svg = response.read().decode("utf-8")
            with urllib.request.urlopen(
                display.url + "page/1.svg", timeout=1
            ) as response:
                current_svg = response.read().decode("utf-8")

            self.assertEqual(len(state["page_versions"]), 2)
            self.assertTrue(all(version > 0 for version in state["page_versions"]))
            self.assertEqual(len(display.notation.events), 5)
            self.assertIn("<svg", old_svg)
            self.assertIn("<svg", current_svg)
        finally:
            display.stop()

    def test_embedded_renderer_updates_without_starting_web_server(self):
        display = LiveScoreDisplay(
            ticks_per_beat=4,
            beats_per_bar=4,
            key_name="C",
            mode="major",
            port=0,
        ).start(open_browser=False, serve=False)
        try:
            self.assertEqual(display.url, "")
            display.observe(frame(0, 4, self.first_pitches))
            deadline = time.monotonic() + 5.0
            snapshot = display.snapshot()
            while time.monotonic() < deadline and snapshot["page_versions"][0] == 0:
                time.sleep(0.02)
                snapshot = display.snapshot()
            self.assertGreater(snapshot["page_versions"][0], 0)
            self.assertIn("Measure 1", snapshot["status"])
            self.assertIn("<svg", snapshot["svgs"][0])
        finally:
            display.stop()


if __name__ == "__main__":
    unittest.main()
