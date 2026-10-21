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
    from magenta_protocol import INPUT_ALL, INPUT_SOPRANO
except ModuleNotFoundError:
    from .magenta_protocol import INPUT_ALL, INPUT_SOPRANO


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
        score_title = QtWidgets.QLabel("Live score")
        score_title.setObjectName("heading")
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
        score_header.addWidget(score_title)
        score_header.addWidget(self.score_status, 1)
        score_header.addWidget(self.follow_button)
        score_layout.addLayout(score_header)
        self.score_scroll = QtWidgets.QScrollArea()
        self.score_scroll.setWidgetResizable(True)
        self.score_container = QtWidgets.QWidget()
        self.score_pages = QtWidgets.QVBoxLayout(self.score_container)
        self.score_pages.setAlignment(QtCore.Qt.AlignTop)
        self.score_scroll.setWidget(self.score_container)
        score_layout.addWidget(self.score_scroll, 1)
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
        bar = self.score_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_state(self, *, mode: str, playing: bool, summary: str) -> None:
        self.full_button.setChecked(mode == MODE_FULL)
        self.soprano_button.setChecked(mode == MODE_SOPRANO)
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
            QtCore.QTimer.singleShot(0, self.follow_live)

    def closeEvent(self, event):
        self.closed = True
        self.close_requested.emit()
        event.accept()
