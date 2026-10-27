"""Qt camera and live-score window for the hand performance."""

from __future__ import annotations

from copy import deepcopy
import xml.etree.ElementTree as ET

try:
    from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets
except ImportError as exc:
    raise RuntimeError(
        "PyQt5 is required for the integrated camera/score window. "
        "Run: pip install -r requirements.txt"
    ) from exc

try:
    from performance_session import MODE_FULL, MODE_SOPRANO
except ModuleNotFoundError:
    from .performance_session import MODE_FULL, MODE_SOPRANO

try:
    from magenta_router import INPUT_ALL, INPUT_SOPRANO
except ModuleNotFoundError:
    from .magenta_router import INPUT_ALL, INPUT_SOPRANO


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
SVG_USE = f"{{{SVG_NS}}}use"
SVG_DEFS = f"{{{SVG_NS}}}defs"
SVG_STYLE = f"{{{SVG_NS}}}style"
PAINTED_SVG_TAGS = {
    f"{{{SVG_NS}}}ellipse",
    f"{{{SVG_NS}}}path",
    f"{{{SVG_NS}}}polygon",
    f"{{{SVG_NS}}}polyline",
    f"{{{SVG_NS}}}rect",
}

PIANO_LOW = 60
PIANO_HIGH = 84
PIANO_ROW_HEIGHT = 24
PIANO_ROLL_HEIGHT = (PIANO_HIGH - PIANO_LOW + 1) * PIANO_ROW_HEIGHT
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def qt_compatible_svg(svg: str) -> bytes:
    """Expand SVG features that QtSvg does not render reliably.

    Verovio stores noteheads, clefs, rests, and accidentals in ``defs`` and
    places them with ``use``. It also colors staff geometry with the CSS value
    ``currentColor``. QtSvg 5 silently omits those items. Expanding each use
    into ordinary paths and assigning explicit paint values preserves the
    complete score in a QLabel/QPixmap.
    """
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    root = ET.fromstring(svg)
    definitions = {
        element_id: element
        for element in root.iter()
        if (element_id := element.attrib.get("id"))
    }

    for parent in list(root.iter()):
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag != SVG_USE:
                continue
            href = child.attrib.get(f"{{{XLINK_NS}}}href") or child.attrib.get("href")
            source = definitions.get((href or "").lstrip("#"))
            if source is None:
                continue
            replacement = deepcopy(source)
            replacement.attrib.pop("id", None)
            use_transform = child.attrib.get("transform")
            source_transform = replacement.attrib.get("transform")
            if use_transform and source_transform:
                replacement.set("transform", f"{use_transform} {source_transform}")
            elif use_transform:
                replacement.set("transform", use_transform)
            for attribute in ("color", "fill", "opacity"):
                if attribute in child.attrib:
                    replacement.set(attribute, child.attrib[attribute])
            parent.remove(child)
            parent.insert(index, replacement)

    def assign_paint(element, inherited_color="black", inherited_fill=None):
        color = element.attrib.get("color", inherited_color)
        fill = element.attrib.get("fill", inherited_fill)
        if element.tag in PAINTED_SVG_TAGS:
            if element.attrib.get("stroke") in (None, "currentColor"):
                element.set("stroke", color)
            if "fill" not in element.attrib:
                element.set("fill", fill or color)
        for child in element:
            assign_paint(child, color, fill)

    assign_paint(root)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in (SVG_DEFS, SVG_STYLE):
                parent.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _pitch_label(pitch: int) -> str:
    return f"{PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"


class PianoKeyboardWidget(QtWidgets.QWidget):
    """Fixed pitch labels aligned with the scrolling soprano roll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(66, PIANO_ROLL_HEIGHT)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        for pitch in range(PIANO_HIGH, PIANO_LOW - 1, -1):
            y = (PIANO_HIGH - pitch) * PIANO_ROW_HEIGHT
            black = pitch % 12 in BLACK_PITCH_CLASSES
            painter.fillRect(
                0,
                y,
                self.width(),
                PIANO_ROW_HEIGHT,
                QtGui.QColor("#34434d" if black else "#ffffff"),
            )
            painter.setPen(QtGui.QColor("#dce5ea" if black else "#455a64"))
            painter.drawRect(0, y, self.width() - 1, PIANO_ROW_HEIGHT)
            if pitch % 12 == 0 or not black:
                painter.drawText(
                    5,
                    y,
                    self.width() - 10,
                    PIANO_ROW_HEIGHT,
                    QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
                    _pitch_label(pitch),
                )
        painter.end()


class PianoRollCanvas(QtWidgets.QWidget):
    """Draw a complete, horizontally scrollable soprano history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snapshot = {
            "notes": (),
            "current_tick": 0,
            "ticks_per_beat": 16,
            "ticks_per_bar": 64,
            "visible_bars": 4,
            "version": 0,
        }
        self._pixels_per_tick = 3.0
        self._viewport_width = 720
        self.setFixedHeight(PIANO_ROLL_HEIGHT)
        self._resize_timeline()

    def set_viewport_width(self, width: int) -> None:
        width = max(240, int(width))
        if width != self._viewport_width:
            self._viewport_width = width
            self._resize_timeline()

    def set_snapshot(self, snapshot) -> bool:
        changed = int(snapshot["version"]) != int(self._snapshot.get("version", -1))
        self._snapshot = snapshot
        self._resize_timeline()
        self.update()
        return changed

    def _resize_timeline(self) -> None:
        visible_ticks = max(
            1,
            int(self._snapshot.get("ticks_per_bar", 64))
            * int(self._snapshot.get("visible_bars", 4)),
        )
        self._pixels_per_tick = max(1.5, self._viewport_width / visible_ticks)
        end_tick = int(self._snapshot.get("current_tick", 0)) + 5
        width = max(
            self._viewport_width,
            int(round(end_tick * self._pixels_per_tick)) + 16,
        )
        self.resize(width, PIANO_ROLL_HEIGHT)

    def _note_y(self, pitch: int) -> int:
        return (PIANO_HIGH - pitch) * PIANO_ROW_HEIGHT

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        clip = event.rect()

        for pitch in range(PIANO_HIGH, PIANO_LOW - 1, -1):
            y = self._note_y(pitch)
            color = "#edf1f3" if pitch % 12 in BLACK_PITCH_CLASSES else "#fafcfd"
            painter.fillRect(clip.left(), y, clip.width(), PIANO_ROW_HEIGHT, QtGui.QColor(color))
            painter.setPen(QtGui.QColor("#dde4e8"))
            painter.drawLine(clip.left(), y + PIANO_ROW_HEIGHT, clip.right(), y + PIANO_ROW_HEIGHT)

        ticks_per_beat = max(1, int(self._snapshot.get("ticks_per_beat", 16)))
        ticks_per_bar = max(1, int(self._snapshot.get("ticks_per_bar", 64)))
        first_tick = max(0, int(clip.left() / self._pixels_per_tick) - 1)
        last_tick = int(clip.right() / self._pixels_per_tick) + 1
        first_grid = first_tick - first_tick % ticks_per_beat
        for tick in range(first_grid, last_tick + 1, ticks_per_beat):
            x = round(tick * self._pixels_per_tick)
            bar_line = tick % ticks_per_bar == 0
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor("#90a4ae" if bar_line else "#cfd8dc"),
                    2 if bar_line else 1,
                )
            )
            painter.drawLine(x, 0, x, self.height())
            if bar_line:
                painter.setPen(QtGui.QColor("#607d8b"))
                painter.drawText(x + 4, 15, str(tick // ticks_per_bar + 1))

        for note in self._snapshot.get("notes", ()):
            if note.pitch < PIANO_LOW or note.pitch > PIANO_HIGH:
                continue
            x = round(note.start_tick * self._pixels_per_tick) + 1
            right = round(note.end_tick * self._pixels_per_tick) - 1
            width = max(4, right - x)
            if x + width < clip.left() or x > clip.right():
                continue
            y = self._note_y(note.pitch) + 3
            color = "#18a999" if note.active else ("#ef8b2c" if note.ornament else "#3978b8")
            painter.setPen(QtGui.QPen(QtGui.QColor("#17506d"), 1))
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(x, y, width, PIANO_ROW_HEIGHT - 6, 3, 3)

        playhead_x = round(
            int(self._snapshot.get("current_tick", 0)) * self._pixels_per_tick
        )
        painter.setPen(QtGui.QPen(QtGui.QColor("#d32f2f"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())
        painter.end()


class SopranoPianoRollView(QtWidgets.QWidget):
    """Piano keyboard plus the scrollable note canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.keyboard = PianoKeyboardWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.canvas = PianoRollCanvas()
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.keyboard, 0, QtCore.Qt.AlignTop)
        layout.addWidget(self.scroll, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.set_viewport_width(self.scroll.viewport().width())

    def update_snapshot(self, snapshot) -> bool:
        self.canvas.set_viewport_width(self.scroll.viewport().width())
        return self.canvas.set_snapshot(snapshot)

    def follow_live(self) -> None:
        bar = self.scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())


class IntegratedPerformanceWindow(QtWidgets.QMainWindow):
    """Show camera tracking, transport controls, and score in one window."""

    mode_requested = QtCore.pyqtSignal(str)
    transport_requested = QtCore.pyqtSignal()
    magenta_enabled_requested = QtCore.pyqtSignal(bool)
    magenta_input_requested = QtCore.pyqtSignal(str)
    close_requested = QtCore.pyqtSignal()

    def __init__(self, *, show_score: bool = True):
        super().__init__()
        self.closed = False
        self._versions = []
        self._score_labels = []
        self._follow_live = True
        self._mode = MODE_FULL
        self.setWindowTitle("Movement to Music Performance")
        self.resize(1500 if show_score else 760, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        camera_panel = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout(camera_panel)
        heading = QtWidgets.QLabel("Camera and gesture tracking")
        heading.setObjectName("heading")
        camera_layout.addWidget(heading)
        self.camera_label = QtWidgets.QLabel("Waiting for camera...")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet(
            "background:#111820;color:#d9e3ea;border-radius:6px;"
        )
        camera_layout.addWidget(self.camera_label, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.full_button = QtWidgets.QPushButton("Full SATB  [1]")
        self.soprano_button = QtWidgets.QPushButton("Soprano Performance  [2]")
        self.transport_button = QtWidgets.QPushButton("Pause  [Space]")
        for button in (
            self.full_button,
            self.soprano_button,
            self.transport_button,
        ):
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Fixed,
            )
        for button in (self.full_button, self.soprano_button):
            button.setCheckable(True)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.full_button)
        group.addButton(self.soprano_button)
        buttons.addWidget(self.full_button)
        buttons.addWidget(self.soprano_button)
        buttons.addWidget(self.transport_button)
        camera_layout.addLayout(buttons)

        magenta_controls = QtWidgets.QHBoxLayout()
        self.magenta_button = QtWidgets.QPushButton("Magenta Off")
        self.magenta_button.setCheckable(True)
        self.magenta_input = QtWidgets.QComboBox()
        self.magenta_input.addItem("Soprano only", INPUT_SOPRANO)
        self.magenta_input.addItem("All four voices", INPUT_ALL)
        self.magenta_status = QtWidgets.QLabel("Off")
        self.magenta_status.setWordWrap(True)
        self.magenta_status.setStyleSheet("color:#60727d;padding:3px;")
        magenta_controls.addWidget(self.magenta_button)
        magenta_controls.addWidget(QtWidgets.QLabel("Input to Magenta:"))
        magenta_controls.addWidget(self.magenta_input)
        magenta_controls.addWidget(self.magenta_status, 1)
        camera_layout.addLayout(magenta_controls)
        self.summary_label = QtWidgets.QLabel("Initializing performance...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color:#44515a;padding:5px;")
        camera_layout.addWidget(self.summary_label)
        splitter.addWidget(camera_panel)

        self.score_panel = QtWidgets.QWidget()
        score_layout = QtWidgets.QVBoxLayout(self.score_panel)
        score_header = QtWidgets.QHBoxLayout()
        self.visual_title = QtWidgets.QLabel("Live score")
        self.visual_title.setObjectName("heading")
        self.score_status = QtWidgets.QLabel("Waiting for generated notes...")
        self.follow_button = QtWidgets.QPushButton("Follow live")
        self.score_status.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.follow_button.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Fixed,
        )
        score_header.addWidget(self.visual_title)
        score_header.addWidget(self.score_status, 1)
        score_header.addWidget(self.follow_button)
        score_layout.addLayout(score_header)
        self.score_scroll = QtWidgets.QScrollArea()
        self.score_scroll.setWidgetResizable(True)
        self.score_container = QtWidgets.QWidget()
        self.score_pages = QtWidgets.QVBoxLayout(self.score_container)
        self.score_pages.setAlignment(QtCore.Qt.AlignTop)
        self.score_scroll.setWidget(self.score_container)
        self.piano_roll = SopranoPianoRollView()
        self.visual_stack = QtWidgets.QStackedWidget()
        self.visual_stack.addWidget(self.score_scroll)
        self.visual_stack.addWidget(self.piano_roll)
        score_layout.addWidget(self.visual_stack, 1)
        splitter.addWidget(self.score_panel)
        self.score_panel.setVisible(show_score)
        splitter.setSizes([650, 830])

        self.full_button.clicked.connect(
            lambda: self.mode_requested.emit(MODE_FULL)
        )
        self.soprano_button.clicked.connect(
            lambda: self.mode_requested.emit(MODE_SOPRANO)
        )
        self.transport_button.clicked.connect(self.transport_requested.emit)
        self.magenta_button.toggled.connect(self.magenta_enabled_requested.emit)
        self.magenta_input.currentIndexChanged.connect(
            lambda _index: self.magenta_input_requested.emit(
                str(self.magenta_input.currentData())
            )
        )
        self.follow_button.clicked.connect(self.follow_live)
        self.score_scroll.verticalScrollBar().actionTriggered.connect(
            self._manual_scroll
        )
        self.piano_roll.scroll.horizontalScrollBar().actionTriggered.connect(
            self._manual_scroll
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("1"),
            self,
            lambda: self.mode_requested.emit(MODE_FULL),
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("2"),
            self,
            lambda: self.mode_requested.emit(MODE_SOPRANO),
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Space"), self, self.transport_requested.emit
        )
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, self.close)

        self.setStyleSheet(
            "QMainWindow{background:#eef2f5;}"
            "QLabel#heading{font-size:18px;font-weight:600;color:#234e70;padding:4px;}"
            "QPushButton{padding:9px 12px;border:1px solid #9aa8b2;border-radius:5px;"
            "background:white;color:#234e70;font-weight:600;}"
            "QPushButton:checked{background:#2e7d4c;color:white;border-color:#2e7d4c;}"
            "QScrollArea{background:white;border:1px solid #ccd5db;border-radius:5px;}"
        )

    def _manual_scroll(self, _action):
        self._follow_live = False

    def follow_live(self):
        self._follow_live = True
        if self._mode == MODE_SOPRANO:
            self.piano_roll.follow_live()
        else:
            bar = self.score_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

    def set_state(self, *, mode: str, playing: bool, summary: str) -> None:
        mode_changed = mode != self._mode
        self._mode = mode
        self.full_button.setChecked(mode == MODE_FULL)
        self.soprano_button.setChecked(mode == MODE_SOPRANO)
        self.visual_stack.setCurrentWidget(
            self.piano_roll if mode == MODE_SOPRANO else self.score_scroll
        )
        self.visual_title.setText(
            "Soprano piano roll" if mode == MODE_SOPRANO else "Live SATB score"
        )
        if mode_changed and self._follow_live:
            QtCore.QTimer.singleShot(0, self.follow_live)
        self.transport_button.setText(
            "Pause  [Space]" if playing else "Play  [Space]"
        )
        self.summary_label.setText(summary)

    def set_magenta_state(
        self,
        *,
        enabled: bool,
        input_mode: str,
        status: str,
    ) -> None:
        self.magenta_button.blockSignals(True)
        self.magenta_button.setChecked(bool(enabled))
        self.magenta_button.setText("Magenta On" if enabled else "Magenta Off")
        self.magenta_button.blockSignals(False)
        index = self.magenta_input.findData(input_mode)
        if index >= 0 and index != self.magenta_input.currentIndex():
            self.magenta_input.blockSignals(True)
            self.magenta_input.setCurrentIndex(index)
            self.magenta_input.blockSignals(False)
        self.magenta_status.setText(status)

    def set_camera_frame(self, frame) -> None:
        height, width, channels = frame.shape
        image = QtGui.QImage(
            frame.data,
            width,
            height,
            channels * width,
            QtGui.QImage.Format_BGR888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _new_score_page(self, index: int, bars_per_page: int):
        section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(4, 4, 4, 12)
        first = index * bars_per_page + 1
        title = QtWidgets.QLabel(
            f"Measures {first}-{first + bars_per_page - 1}"
        )
        title.setStyleSheet("color:#60727d;font-weight:600;")
        image = QtWidgets.QLabel()
        image.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        image.setStyleSheet("background:white;")
        image.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Fixed,
        )
        layout.addWidget(title)
        layout.addWidget(image)
        self.score_pages.addWidget(section)
        self._score_labels.append(image)

    def _render_svg(self, svg: str):
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(qt_compatible_svg(svg)))
        natural = renderer.defaultSize()
        target_width = max(
            620,
            min(900, self.score_scroll.viewport().width() - 28),
        )
        if natural.width() <= 0 or natural.height() <= 0:
            natural = QtCore.QSize(1200, 500)
        target_height = max(
            220,
            round(target_width * natural.height() / natural.width()),
        )
        pixmap = QtGui.QPixmap(target_width, target_height)
        pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap

    def update_score(self, snapshot) -> None:
        if self._mode == MODE_FULL:
            self.score_status.setText(snapshot["status"])
        versions = list(snapshot["page_versions"])
        changed = len(versions) != len(self._versions)
        while len(self._score_labels) < len(versions):
            self._new_score_page(
                len(self._score_labels), snapshot["bars_per_page"]
            )
        for index, version in enumerate(versions):
            old_version = self._versions[index] if index < len(self._versions) else -1
            if version != old_version:
                changed = True
                self._score_labels[index].setPixmap(
                    self._render_svg(snapshot["svgs"][index])
                )
        self._versions = versions
        if self._follow_live and changed:
            if self._mode == MODE_FULL:
                QtCore.QTimer.singleShot(0, self.follow_live)

    def update_piano_roll(self, snapshot) -> None:
        """Refresh the soprano-only roll from a thread-safe model snapshot."""
        if self._mode == MODE_SOPRANO:
            self.score_status.setText(snapshot["status"])
        changed = self.piano_roll.update_snapshot(snapshot)
        if self._follow_live and changed and self._mode == MODE_SOPRANO:
            QtCore.QTimer.singleShot(0, self.follow_live)

    def closeEvent(self, event):
        self.closed = True
        self.close_requested.emit()
        event.accept()
