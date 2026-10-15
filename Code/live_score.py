"""Rolling SATB notation display for the realtime generator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Optional
import webbrowser


VOICE_ORDER = ("soprano", "alto", "tenor", "bass")
STEM_DIRECTIONS = {
    "soprano": "up",
    "alto": "down",
    "tenor": "up",
    "bass": "down",
}
PITCH_CLASS_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


@dataclass
class NotationEvent:
    tick_index: int
    duration_ticks: int
    pitches: dict[str, int]
    chord: tuple[int, str, float]
    articulation: float
    tempo_bpm: float
    new_voices: frozenset[str]
    released_voices: frozenset[str]
    voice_duration_ticks: dict[str, int]


@dataclass
class VoiceSpan:
    start_tick: int
    end_tick: int
    pitch: int
    articulation: float
    event_index: int


class RollingNotation:
    """Keep realtime onsets and convert selected measures to a SATB score."""

    def __init__(
        self,
        *,
        ticks_per_beat: int = 4,
        beats_per_bar: int = 4,
        visible_bars: int = 4,
        key_name: str = "C",
        mode: str = "major",
    ):
        self.ticks_per_beat = max(1, int(ticks_per_beat))
        self.beats_per_bar = max(1, int(beats_per_bar))
        self.ticks_per_bar = self.ticks_per_beat * self.beats_per_bar
        self.visible_bars = max(1, int(visible_bars))
        self.key_name = str(key_name)
        self.mode = str(mode)
        self.events: list[NotationEvent] = []

    def add_frame(self, frame) -> bool:
        released = frozenset(getattr(frame, "released_voices", ()))
        if not frame.new_notes and not released:
            return False
        if self.events:
            self.events[-1].duration_ticks = max(
                1,
                int(frame.tick_index) - self.events[-1].tick_index,
            )
        tempo_bpm = 60.0 / max(1e-9, frame.duration_seconds * self.ticks_per_beat)
        pitches = {
            voice: int(pitch)
            for voice, pitch in frame.pitches.items()
            if voice in VOICE_ORDER
        }
        new_voices = frozenset(
            getattr(frame, "new_voices", pitches.keys() if frame.new_notes else ())
        )
        durations = {
            str(voice): max(1, int(duration))
            for voice, duration in getattr(frame, "voice_duration_ticks", {}).items()
        }
        self.events.append(
            NotationEvent(
                tick_index=int(frame.tick_index),
                duration_ticks=max(1, int(frame.planned_duration_ticks)),
                pitches=pitches,
                chord=(int(frame.chord[0]), str(frame.chord[1]), float(frame.chord[2])),
                articulation=float(frame.controls.articulation),
                tempo_bpm=tempo_bpm,
                new_voices=new_voices,
                released_voices=released,
                voice_duration_ticks=durations,
            )
        )
        return True

    @property
    def current_bar(self) -> int:
        if not self.events:
            return 1
        return self.events[-1].tick_index // self.ticks_per_bar + 1

    def _window(self) -> tuple[int, int, int]:
        current_bar_zero = self.current_bar - 1
        start_bar = max(0, current_bar_zero - self.visible_bars + 1)
        start_tick = start_bar * self.ticks_per_bar
        end_tick = (current_bar_zero + 1) * self.ticks_per_bar
        return start_bar, start_tick, end_tick

    def _event_slices(self, measure_start: int, measure_end: int):
        for event_index, event in enumerate(self.events):
            if event.tick_index >= measure_end:
                break
            event_end = event.tick_index + event.duration_ticks
            start = max(measure_start, event.tick_index)
            end = min(measure_end, event_end)
            if end > start:
                yield event_index, event, start, end, event_end

    def _voice_spans(self, voice_name: str) -> list[VoiceSpan]:
        """Merge snapshots into notes, preserving independent voice onsets."""
        spans: list[VoiceSpan] = []
        current: Optional[VoiceSpan] = None
        for event_index, event in enumerate(self.events):
            tick = event.tick_index
            released = voice_name in event.released_voices or voice_name not in event.pitches
            starts = voice_name in event.new_voices
            if released:
                if current is not None:
                    current.end_tick = max(current.start_tick + 1, tick)
                    spans.append(current)
                    current = None
                continue
            if starts or current is None:
                if current is not None:
                    current.end_tick = max(current.start_tick + 1, tick)
                    spans.append(current)
                duration = event.voice_duration_ticks.get(
                    voice_name, event.duration_ticks
                )
                current = VoiceSpan(
                    start_tick=tick,
                    end_tick=tick + max(1, int(duration)),
                    pitch=event.pitches[voice_name],
                    articulation=event.articulation,
                    event_index=event_index,
                )
            elif current is not None:
                # Another voice attacked. Keep this voice sounding through it.
                current.end_tick = max(current.end_tick, tick + 1)
        if current is not None:
            spans.append(current)
        return spans

    def _voice_slices(self, voice_name: str, measure_start: int, measure_end: int):
        for span in self._voice_spans(voice_name):
            if span.start_tick >= measure_end:
                break
            start = max(measure_start, span.start_tick)
            end = min(measure_end, span.end_tick)
            if end > start:
                yield span, start, end

    def _make_voice(self, voice_name: str, measure_start: int, measure_end: int, voice_id: int):
        from music21 import articulations, note, stream, tie

        voice = stream.Voice(id=voice_id)
        cursor = measure_start
        spans = self._voice_spans(voice_name)
        last_event_index = spans[-1].event_index if spans else -1
        for span, start, end in self._voice_slices(voice_name, measure_start, measure_end):
            if start > cursor:
                voice.append(
                    note.Rest(quarterLength=(start - cursor) / self.ticks_per_beat)
                )
            item = note.Note(
                span.pitch,
                quarterLength=(end - start) / self.ticks_per_beat,
            )
            item.stemDirection = STEM_DIRECTIONS[voice_name]
            continues_from_previous = span.start_tick < start
            continues_to_next = span.end_tick > end
            if continues_from_previous and continues_to_next:
                item.tie = tie.Tie("continue")
            elif continues_from_previous:
                item.tie = tie.Tie("stop")
            elif continues_to_next:
                item.tie = tie.Tie("start")
            if span.articulation < 0.40:
                item.articulations.append(articulations.Staccato())
            if span.event_index == last_event_index:
                item.style.color = "#D32F2F"
            voice.append(item)
            cursor = end
        if cursor < measure_end:
            voice.append(
                note.Rest(quarterLength=(measure_end - cursor) / self.ticks_per_beat)
            )
        return voice

    @staticmethod
    def _chord_label(chord: tuple[int, str, float]) -> str:
        root, quality, _duration = chord
        suffix = {
            "maj": "",
            "min": "m",
            "dim": "dim",
            "7": "7",
            "maj7": "maj7",
            "min7": "m7",
        }.get(quality, quality)
        return f"{PITCH_CLASS_NAMES[root % 12]}{suffix}"

    def build_score(self, *, start_bar: Optional[int] = None, measure_count: Optional[int] = None):
        from music21 import clef, expressions, key, layout, metadata, meter, stream, tempo

        score = stream.Score(id="LiveSATB")
        score.metadata = metadata.Metadata(title="Live SATB Generation")
        treble = stream.PartStaff(id="Treble")
        bass_staff = stream.PartStaff(id="Bass")
        if start_bar is None:
            start_bar, start_tick, end_tick = self._window()
        else:
            start_bar = max(0, int(start_bar))
            measure_count = max(1, int(measure_count or self.visible_bars))
            start_tick = start_bar * self.ticks_per_bar
            end_tick = (start_bar + measure_count) * self.ticks_per_bar
        previous_chord = None
        tempo_event = next(
            (
                event
                for event in self.events
                if event.tick_index < end_tick
                and event.tick_index + event.duration_ticks > start_tick
            ),
            None,
        )

        for local_measure, measure_start in enumerate(
            range(start_tick, end_tick, self.ticks_per_bar)
        ):
            measure_end = measure_start + self.ticks_per_bar
            treble_measure = stream.Measure(number=start_bar + local_measure + 1)
            bass_measure = stream.Measure(number=start_bar + local_measure + 1)
            if local_measure == 0:
                signature = meter.TimeSignature(f"{self.beats_per_bar}/4")
                treble_measure.insert(0, signature)
                bass_measure.insert(0, meter.TimeSignature(f"{self.beats_per_bar}/4"))
                treble_measure.insert(0, clef.TrebleClef())
                bass_measure.insert(0, clef.BassClef())
                try:
                    treble_measure.insert(0, key.Key(self.key_name, self.mode))
                    bass_measure.insert(0, key.Key(self.key_name, self.mode))
                except Exception:
                    pass
                if tempo_event is not None:
                    treble_measure.insert(
                        0,
                        tempo.MetronomeMark(number=round(tempo_event.tempo_bpm)),
                    )

            treble_measure.insert(
                0,
                self._make_voice("soprano", measure_start, measure_end, 1),
            )
            treble_measure.insert(
                0,
                self._make_voice("alto", measure_start, measure_end, 2),
            )
            bass_measure.insert(
                0,
                self._make_voice("tenor", measure_start, measure_end, 3),
            )
            bass_measure.insert(
                0,
                self._make_voice("bass", measure_start, measure_end, 4),
            )

            for _index, event, start, _end, _event_end in self._event_slices(
                measure_start, measure_end
            ):
                if start != event.tick_index or event.chord == previous_chord:
                    continue
                label = expressions.TextExpression(self._chord_label(event.chord))
                label.placement = "above"
                treble_measure.insert(
                    (start - measure_start) / self.ticks_per_beat,
                    label,
                )
                previous_chord = event.chord

            treble.append(treble_measure)
            bass_staff.append(bass_measure)

        score.insert(0, treble)
        score.insert(0, bass_staff)
        score.insert(
            0,
            layout.StaffGroup([treble, bass_staff], symbol="brace", barTogether=True),
        )
        return score

    def musicxml(
        self,
        *,
        start_bar: Optional[int] = None,
        measure_count: Optional[int] = None,
    ) -> str:
        from music21.musicxml.m21ToXml import GeneralObjectExporter

        score = self.build_score(start_bar=start_bar, measure_count=measure_count)
        return GeneralObjectExporter(score).parse().decode("utf-8")


class LiveScoreDisplay:
    """Render notation off the audio thread and expose it through localhost."""

    def __init__(
        self,
        *,
        ticks_per_beat: int,
        beats_per_bar: int,
        key_name: str,
        mode: str,
        visible_bars: int = 4,
        port: int = 8765,
    ):
        try:
            import verovio
        except ImportError as exc:
            raise RuntimeError(
                "Verovio is required for live notation. Run: pip install -r requirements.txt"
            ) from exc
        self.verovio = verovio
        # Initialize Verovio where the display is created. On Windows, its
        # bundled font resources may fail to initialize if the first toolkit
        # instance is constructed inside a worker thread.
        self.toolkit = verovio.toolkit()
        self.notation = RollingNotation(
            ticks_per_beat=ticks_per_beat,
            beats_per_bar=beats_per_bar,
            visible_bars=visible_bars,
            key_name=key_name,
            mode=mode,
        )
        self.requested_port = max(0, int(port))
        self._queue: queue.Queue = queue.Queue()
        self._lock = Lock()
        self._svgs = [self._waiting_svg()]
        self._page_versions = [0]
        self._rendered_pages: set[int] = set()
        self._current_page: Optional[int] = None
        self._version = 0
        self._status = "Waiting for the first generated notes..."
        self._running = False
        self._worker: Optional[Thread] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[Thread] = None

    @staticmethod
    def _waiting_svg() -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 260">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="500" y="135" text-anchor="middle" '
            'font-family="sans-serif" font-size="28" fill="#455a64">'
            "Waiting for generated notes...</text></svg>"
        )

    @property
    def url(self) -> str:
        if self._server is None:
            return ""
        return f"http://127.0.0.1:{self._server.server_address[1]}/"

    def snapshot(self):
        """Return a thread-safe copy for an embedded desktop score view."""
        with self._lock:
            return {
                "svgs": tuple(self._svgs),
                "page_versions": tuple(self._page_versions),
                "status": self._status,
                "bars_per_page": self.notation.visible_bars,
            }

    def observe(self, frame) -> None:
        if (frame.new_notes or getattr(frame, "released_voices", ())) and self._running:
            self._queue.put(frame)

    def _render_page(self, page_index: int) -> str:
        xml = self.notation.musicxml(
            start_bar=page_index * self.notation.visible_bars,
            measure_count=self.notation.visible_bars,
        )
        self.toolkit.setOptions(
            {
                "adjustPageHeight": True,
                # Split dense four-measure sections into readable systems.
                "breaks": "auto",
                "footer": "none",
                "header": "none",
                "pageWidth": 1800,
                "scale": 42,
            }
        )
        if not self.toolkit.loadData(xml):
            raise RuntimeError("Verovio could not load generated MusicXML")
        return self.toolkit.renderToSVG(1)

    def _render(self) -> None:
        current_page = (self.notation.current_bar - 1) // self.notation.visible_bars
        dirty_pages = {current_page}
        if self._current_page is not None and self._current_page != current_page:
            # The first onset on a new page finalizes the last note on the
            # previous page, so refresh that completed page once.
            dirty_pages.add(self._current_page)
        dirty_pages.update(
            page
            for page in range(current_page + 1)
            if page not in self._rendered_pages
        )

        rendered = {
            page_index: self._render_page(page_index)
            for page_index in sorted(dirty_pages)
        }
        with self._lock:
            for page_index, svg in rendered.items():
                while len(self._svgs) <= page_index:
                    self._svgs.append(self._waiting_svg())
                    self._page_versions.append(0)
                self._svgs[page_index] = svg
                self._page_versions[page_index] += 1
                self._rendered_pages.add(page_index)
            self._current_page = current_page
            self._version += 1
            self._status = (
                f"Measure {self.notation.current_bar} | "
                f"{len(self._svgs)} score section(s) | "
                f"{self.notation.events[-1].tempo_bpm:.0f} BPM"
            )

    def _work(self) -> None:
        while self._running:
            frame = self._queue.get()
            if frame is None:
                break
            frames = [frame]
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if pending is None:
                    self._running = False
                    break
                frames.append(pending)
            try:
                changed = False
                for pending_frame in frames:
                    changed = self.notation.add_frame(pending_frame) or changed
                if changed:
                    self._render()
            except Exception as exc:
                with self._lock:
                    self._status = f"Notation error: {exc}"

    @staticmethod
    def _html() -> bytes:
        return b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Live SATB Score</title>
<style>
body{margin:0;background:#eef2f5;color:#263238;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;padding:14px 24px;background:#234e70;color:white;
display:flex;gap:24px;align-items:center;box-shadow:0 2px 8px #0003}
h1{font-size:20px;margin:0}#status{opacity:.9;flex:1}button{padding:7px 12px;border:0;
border-radius:4px;background:white;color:#234e70;font-weight:600;cursor:pointer}
#pages{padding:18px}.score-page{margin:0 auto 22px;max-width:1800px}.score-page h2{
font-size:15px;font-weight:600;color:#546e7a;margin:0 0 7px 4px}
img{display:block;width:100%;object-fit:contain;background:white;box-shadow:0 2px 12px #0002;
border-radius:6px;min-height:180px}
</style></head><body><header><h1>Live SATB Score</h1><span id="status">Connecting...</span>
<button id="follow">Follow live</button></header><main id="pages"></main>
<script>
let versions=[];let followLive=true;const pages=document.getElementById('pages');
function nearBottom(){return window.innerHeight+window.scrollY>=document.documentElement.scrollHeight-120;}
window.addEventListener('scroll',()=>{if(!nearBottom())followLive=false;},{passive:true});
document.getElementById('follow').onclick=()=>{followLive=true;
const last=pages.lastElementChild;if(last)last.scrollIntoView({behavior:'smooth',block:'end'});};
function ensurePage(i,bars){let section=document.getElementById('page-'+i);if(section)return section;
section=document.createElement('section');section.className='score-page';section.id='page-'+i;
const first=i*bars+1;section.innerHTML='<h2>Measures '+first+'--'+(first+bars-1)+
'</h2><img alt="Score measures '+first+' to '+(first+bars-1)+'">';pages.appendChild(section);
return section;}
async function refresh(){try{const r=await fetch('/state.json',{cache:'no-store'});const s=await r.json();
document.getElementById('status').textContent=s.status;const shouldFollow=followLive&&nearBottom();
s.page_versions.forEach((v,i)=>{const section=ensurePage(i,s.bars_per_page);
if(versions[i]!==v){versions[i]=v;section.querySelector('img').src='/page/'+i+'.svg?v='+v;}});
if(shouldFollow){requestAnimationFrame(()=>window.scrollTo({top:document.documentElement.scrollHeight,
behavior:'smooth'}));}}catch(e){}
setTimeout(refresh,150);}refresh();
</script></body></html>"""

    def _handler_class(self):
        display = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path.startswith("/page/") and path.endswith(".svg"):
                    try:
                        page_index = int(path.removeprefix("/page/").removesuffix(".svg"))
                    except ValueError:
                        page_index = -1
                    with display._lock:
                        if 0 <= page_index < len(display._svgs):
                            svg = display._svgs[page_index]
                        else:
                            svg = display._waiting_svg()
                        body = svg.encode("utf-8")
                    content_type = "image/svg+xml; charset=utf-8"
                elif path == "/score.svg":
                    # Compatibility endpoint: return the newest four-measure section.
                    with display._lock:
                        body = display._svgs[-1].encode("utf-8")
                    content_type = "image/svg+xml; charset=utf-8"
                elif path == "/state.json":
                    with display._lock:
                        body = json.dumps(
                            {
                                "version": display._version,
                                "status": display._status,
                                "page_versions": display._page_versions,
                                "bars_per_page": display.notation.visible_bars,
                            }
                        ).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                else:
                    body = display._html()
                    content_type = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        return Handler

    def start(
        self,
        *,
        open_browser: bool = True,
        serve: bool = True,
    ) -> "LiveScoreDisplay":
        if self._running:
            return self
        self._running = True
        self._worker = Thread(target=self._work, name="live-score-renderer", daemon=True)
        self._worker.start()
        if not serve:
            print("Live score: embedded in the performance window")
            return self
        try:
            self._server = ThreadingHTTPServer(
                ("127.0.0.1", self.requested_port),
                self._handler_class(),
            )
        except OSError:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._server_thread = Thread(
            target=self._server.serve_forever,
            name="live-score-server",
            daemon=True,
        )
        self._server_thread.start()
        print(f"Live score: {self.url}")
        if open_browser:
            webbrowser.open(self.url)
        return self

    def stop(self) -> None:
        if not self._running and self._server is None:
            return
        self._running = False
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
