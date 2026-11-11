"""Qt UI for the BLE-based SOSLamp controller.

This file has two classes:
  * ScreenColorSampler - samples the screen colour on a timer (Ambilight-style).
  * SOSLampWindow - the main window: device list, power/colour/effects panels and
    the (long) Music Reactive panel.

Threading rule to remember: Qt widgets may only be touched from the
GUI thread, but BLE work runs on the asyncio thread. So async code never edits
widgets directly - it emits a Qt signal (see UiSignals) and the GUI thread handles it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

from collections import OrderedDict

from soslamp.audio.device_utils import AudioDeviceInfo, list_input_devices
from soslamp.audio.reactive_engine import MusicReactiveWorker
from soslamp.services.ble_manager import (
    BleControllerManager,
    BleDeviceInfo,
    EFFECTS,
    WEEK_DAYS,
)
from soslamp.utils.async_bridge import AsyncExecutor
from soslamp.utils.config import (
    AUDIO_BUFFER_SIZE,
    AUDIO_GAIN,
    AUDIO_VALUE_BOOST,
    AUDIO_VALUE_FLOOR,
    AUDIO_SILENCE_GATE,
    COLOR_THEME,
    MUSIC_BASE_BRIGHTNESS,
    MUSIC_MAX_BRIGHTNESS,
)


class NoWheelSlider(QtWidgets.QSlider):
    """Slider that ignores the mouse wheel.

    The panel is inside a scroll area, so without this, scrolling the page over a
    slider would change its value by accident.
    """

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        event.ignore()


# Label + bit mask for each weekday checkbox (meant for a schedule panel).
DAY_MAPPING = [
    ("Mon", WEEK_DAYS.monday),
    ("Tue", WEEK_DAYS.tuesday),
    ("Wed", WEEK_DAYS.wednesday),
    ("Thu", WEEK_DAYS.thursday),
    ("Fri", WEEK_DAYS.friday),
    ("Sat", WEEK_DAYS.saturday),
    ("Sun", WEEK_DAYS.sunday),
]


class UiSignals(QtCore.QObject):
    """Signals used to safely send results from background threads to the GUI thread."""

    ble_devices = QtCore.Signal(list)
    status_message = QtCore.Signal(str)
    connection_status = QtCore.Signal(str)
    music_color = QtCore.Signal(tuple)
    audio_error = QtCore.Signal(str)


DEFAULT_EFFECT_CODE = 0x00  # device "power-on" effect (first entry in the effect dropdown)


class ScreenColorSampler(QtCore.QObject):
    """Grabs the screen and pushes the dominant/average color to the controller.

    A QTimer calls _sample() every interval_ms. Each sample: grab the screen ->
    shrink it -> pick a colour -> brighten/smooth it -> send it to the strip.
    """

    def __init__(
        self,
        executor: AsyncExecutor,
        manager: BleControllerManager,
        interval_ms: int = 120,
        boost: float = 1.4,
        target_luma: float = 140.0,
        max_gain: float = 2.8,
        min_value: int = 28,
        mode: str = "dominant",
        edge_emphasis: float = 0.6,
        smoothing: float = 0.25,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.executor = executor
        self.manager = manager
        # Tuning values (the GUI changes smoothing/edge/mode/region at runtime).
        self.interval_ms = interval_ms
        self.boost = boost
        self.target_luma = target_luma
        self.max_gain = max_gain
        self.min_value = min_value
        self.mode = mode
        self.edge_emphasis = edge_emphasis
        self.smoothing = smoothing
        self.monitor_index: int | None = None
        self.region: str = "full"
        # QTimer fires on the GUI thread, which is what we need for grabbing the screen.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._sample)
        self._last_rgb: tuple[int, int, int] | None = None

    def start(self) -> None:
        self._last_rgb = None
        if not self.timer.isActive():
            self.timer.start(self.interval_ms)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def stop(self) -> None:
        self.timer.stop()

    def set_monitor(self, idx: int | None) -> None:
        self.monitor_index = idx if idx is not None and idx >= 0 else None

    def set_region(self, region: str) -> None:
        self.region = region

    def set_smoothing(self, amt: float) -> None:
        self.smoothing = float(np.clip(amt, 0.0, 0.95))

    def set_edge_emphasis(self, amt: float) -> None:
        self.edge_emphasis = float(np.clip(amt, 0.0, 2.0))

    def _sample(self) -> None:
        """Take one screenshot sample and send the resulting colour to the strip."""
        screens = QtWidgets.QApplication.screens()
        if not screens:
            return
        # Use the chosen monitor, or the first one if the index is missing/out of range.
        screen = screens[self.monitor_index] if self.monitor_index is not None and self.monitor_index < len(screens) else screens[0]
        geo = screen.geometry()
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        # Optionally only look at one half of the screen.
        if self.region == "left":
            w //= 2
        elif self.region == "right":
            half = w // 2
            x += half
            w = w - half
        elif self.region == "top":
            h //= 2
        elif self.region == "bottom":
            half = h // 2
            y += half
            h = h - half
        pixmap = screen.grabWindow(0, x, y, w, h)
        if pixmap.isNull():
            return
        # Shrink to 80x80 first: we only want a colour, and fewer pixels = much faster maths.
        img = (
            pixmap.toImage()
            .convertToFormat(QtGui.QImage.Format_RGBA8888)
            .scaled(80, 80, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        )
        if img.isNull():
            return
        # sizeInBytes() isn't available in every Qt binding version, so fall back to the manual calculation.
        try:
            byte_len = img.sizeInBytes()
        except AttributeError:
            byte_len = img.bytesPerLine() * img.height()
        if byte_len <= 0:
            return
        # View the image memory as a numpy array (H x W x 4 RGBA), then drop alpha.
        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=byte_len)
        if arr.size == 0:
            return
        arr = arr.reshape(img.height(), img.width(), 4)[..., :3].astype(np.float32)
        h, w = arr.shape[:2]
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        # Edge weighting: pixels near the screen border count more (that's what's around the strip).
        wx = np.minimum(xs, 1 - xs)
        wy = np.minimum(ys, 1 - ys)
        edge_weight = 1.0 + self.edge_emphasis * (1.0 - np.outer(wy, wx))
        # Two modes: "standout" = the single most vivid pixel, otherwise a weighted average.
        if self.mode == "standout":
            maxc = arr.max(axis=2)
            minc = arr.min(axis=2)
            # Saturation x brightness = how "vibrant" a pixel is (1e-3 avoids dividing by zero).
            sat = (maxc - minc) / (maxc + 1e-3)
            val = maxc / 255.0
            prominence = sat * val * edge_weight
            idx = np.unravel_index(int(np.argmax(prominence)), prominence.shape)
            base_rgb = arr[idx[0], idx[1], :]
        else:
            weighted = arr * edge_weight[..., None]
            norm = np.sum(edge_weight) + 1e-6
            base_rgb = weighted.sum(axis=(0, 1)) / norm
        # Brightness compensation: target a minimum luma and apply a limited gain.
        # (luma = perceived brightness; the weights are the standard Rec.709 coefficients)
        luma = float(0.2126 * base_rgb[0] + 0.7152 * base_rgb[1] + 0.0722 * base_rgb[2])
        gain = max(self.boost, self.target_luma / max(luma, 1e-3))
        gain = min(gain, self.max_gain)
        boosted = base_rgb * gain
        peak = float(np.max(boosted))
        if peak < self.min_value:
            scale = self.min_value / max(peak, 1e-3)
            boosted *= scale
        rgb_arr = boosted
        # Blend with the previous colour so the lamp doesn't flicker on every frame change.
        if self._last_rgb is not None:
            prev = np.array(self._last_rgb, dtype=np.float32)
            rgb_arr = (1 - self.smoothing) * rgb_arr + self.smoothing * prev
        rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb_arr)
        self._last_rgb = rgb_arr
        # Fire-and-forget: the BLE write runs on the asyncio thread so the GUI never waits on it.
        try:
            self.executor.submit(self.manager.set_color(rgb))
        except Exception:
            pass


class SOSLampWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self, executor: AsyncExecutor) -> None:
        super().__init__()
        self.executor = executor
        self.manager = BleControllerManager()
        # Pass the *method* (not its result) so the worker always asks for the current controller.
        self.music_worker = MusicReactiveWorker(self.manager.controller)
        self.audio_devices: list[AudioDeviceInfo] = []
        self.selected_audio_device: AudioDeviceInfo | None = None
        # Settings saved when "Fast mode" is switched on, so switching it off can restore them.
        self._fast_snapshot: dict[str, int | bool] = {}
        self.screen_sampler = ScreenColorSampler(self.executor, self.manager, parent=self)
        # Wire each signal to the GUI method that handles it.
        self.signals = UiSignals()
        self.signals.ble_devices.connect(self._populate_devices)
        self.signals.status_message.connect(self._set_status)
        self.signals.connection_status.connect(self._update_connection_label)
        self.signals.music_color.connect(self._update_music_preview)
        self.signals.audio_error.connect(self._set_status)
        # These callbacks fire on other threads, so they only emit signals (never touch widgets).
        self.music_worker.set_observer(lambda data, b: self.signals.music_color.emit((data, b)))
        self.music_worker.set_drop_notifier(lambda msg: self.signals.status_message.emit(msg))
        self.music_worker.engine.set_error_handler(lambda msg: self.signals.audio_error.emit(msg))
        self._screens: list[QtGui.QScreen] = []
        self._build_ui()
        # Kick off a scan straight away and auto-connect to the first strip we find.
        self._set_status("SOSLamp module loaded – scanning for devices…")
        self._refresh_devices(auto_connect=True)

    def _build_ui(self) -> None:
        """Create the layout: left card = devices, right card = control panels."""
        self.setWindowTitle("SOSLamp")
        self.resize(1100, 650)
        # Place the window near the top-centre of the screen instead of wherever the OS puts it.
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            x = max(
                avail.left() + 20,
                min(avail.right() - self.width() - 20, avail.center().x() - self.width() // 2),
            )
            y = max(avail.top() + 20, avail.center().y() - self.height() // 2 - 120)
            self.move(x, y)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central_layout = QtWidgets.QVBoxLayout(central)

        # Everything lives in a scroll area because the Music panel is very tall.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        central_layout.addWidget(scroll)

        root_layout = QtWidgets.QVBoxLayout(content)

        # objectName lets the stylesheet in _apply_theme target this specific label.
        hero = QtWidgets.QLabel("SOSLamp")
        hero.setObjectName("HeroLabel")
        hero.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        root_layout.addWidget(hero)

        layout = QtWidgets.QHBoxLayout()
        root_layout.addLayout(layout)

        self.device_list = QtWidgets.QListWidget()
        self.device_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        device_panel = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("BLE LED Devices")
        title.setObjectName("SectionLabel")
        device_panel.addWidget(title)
        device_panel.addWidget(self.device_list)

        btn_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda: self._refresh_devices(auto_connect=True))
        btn_row.addWidget(refresh_btn)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_selected)
        btn_row.addWidget(self.connect_btn)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._disconnect)
        btn_row.addWidget(self.disconnect_btn)
        device_panel.addLayout(btn_row)

        self.connection_label = QtWidgets.QLabel("Not connected")
        device_panel.addWidget(self.connection_label)

        device_frame = QtWidgets.QFrame()
        device_frame.setObjectName("CardFrame")
        device_frame.setLayout(device_panel)
        # The numbers (2 and 3) are stretch factors: the right card gets 3/5 of the width.
        layout.addWidget(device_frame, 2)

        controls_panel = QtWidgets.QVBoxLayout()
        controls_panel.addWidget(self._build_power_group())
        controls_panel.addWidget(self._build_color_group())
        controls_panel.addWidget(self._build_effect_group())
        controls_panel.addWidget(self._build_music_group())
        controls_panel.addStretch(1)
        controls_frame = QtWidgets.QFrame()
        controls_frame.setObjectName("CardFrame")
        controls_frame.setLayout(controls_panel)
        layout.addWidget(controls_frame, 3)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the dark theme with a Qt stylesheet (like CSS) built from COLOR_THEME."""
        accent = COLOR_THEME.accent
        accent_alt = COLOR_THEME.accent_alt
        panel = COLOR_THEME.panel
        background = COLOR_THEME.background
        self.setStyleSheet(
            # Doubled braces {{ }} are needed because this is an f-string.
            f"""
            QWidget {{
                background-color: {background};
                color: white;
                font-size: 14px;
            }}
            QFrame#CardFrame {{
                background-color: {panel};
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 16px;
            }}
            QLabel#HeroLabel {{
                font-size: 28px;
                font-weight: 700;
                color: {accent};
                margin-bottom: 12px;
            }}
            QLabel#SectionLabel {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
            }}
            QPushButton {{
                background-color: {accent};
                border: none;
                border-radius: 20px;
                padding: 8px 18px;
                font-weight: 600;
            }}
            QPushButton:checked {{
                background-color: {accent_alt};
            }}
            QPushButton:hover {{
                background-color: {accent_alt};
            }}
            QListWidget {{
                background-color: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 10px;
            }}
            QComboBox, QSpinBox, QLineEdit {{
                background-color: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 8px;
                padding: 4px 8px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                border-radius: 3px;
                background: rgba(255,255,255,0.15);
            }}
            QSlider::handle:horizontal {{
                background: {accent};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QStatusBar {{
                background-color: {panel};
                border-top: 1px solid rgba(255,255,255,0.08);
            }}
            """
        )

    # --- UI builders: each returns a QGroupBox panel that _build_ui adds to the layout ---
    def _build_power_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Power")
        layout = QtWidgets.QHBoxLayout(group)
        on_btn = QtWidgets.QPushButton("Power On")
        # The lambda creates the coroutine at click time; _run_async sends it to the asyncio thread.
        on_btn.clicked.connect(lambda: self._run_async(self.manager.power_on(), "Powered on"))
        off_btn = QtWidgets.QPushButton("Power Off")
        off_btn.clicked.connect(lambda: self._run_async(self.manager.power_off(), "Powered off"))
        layout.addWidget(on_btn)
        layout.addWidget(off_btn)
        return group

    def _build_color_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Color & White Balance")
        layout = QtWidgets.QVBoxLayout(group)

        color_btn = QtWidgets.QPushButton("Pick Color…")
        color_btn.clicked.connect(self._pick_color)
        layout.addWidget(color_btn)

        self.brightness_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(5, 100)
        self.brightness_slider.setValue(80)
        # sliderReleased = only send to the lamp when the user lets go (not on every tiny movement,
        # which would flood the Bluetooth connection).
        self.brightness_slider.sliderReleased.connect(self._apply_brightness)
        layout.addWidget(QtWidgets.QLabel("Brightness"))
        layout.addWidget(self.brightness_slider)

        self.temp_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.temp_slider.setRange(2700, 6500)
        self.temp_slider.setSingleStep(100)
        self.temp_slider.setValue(5000)
        self.temp_slider.sliderReleased.connect(self._apply_color_temp)
        self.temp_value_label = QtWidgets.QLabel("Color temp: 5000K")
        self.temp_slider.valueChanged.connect(
            lambda value: self.temp_value_label.setText(f"Color temp: {value}K")
        )
        layout.addWidget(self.temp_value_label)
        layout.addWidget(self.temp_slider)
        return group

    def _build_effect_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Effects")
        layout = QtWidgets.QGridLayout(group)

        # Name shown in the dropdown -> effect code sent to the device.
        # ("Crossfade Rainbow" uses the raw code 0x01, which is not in the EFFECTS table.)
        presets = OrderedDict(
            [
                ("Power On", DEFAULT_EFFECT_CODE),
                ("Crossfade Rainbow", 0x01),
                ("Jump RGB", EFFECTS.jump_red_green_blue),
                ("Jump Rainbow", EFFECTS.jump_red_green_blue_yellow_cyan_magenta_white),
                ("Crossfade RGB", EFFECTS.crossfade_red_green_blue),
                ("Crossfade Warm", EFFECTS.crossfade_red_green),
                ("Crossfade Cool", EFFECTS.crossfade_green_blue),
                ("Blink Red", EFFECTS.blink_red),
                ("Blink Cyan", EFFECTS.blink_cyan),
                ("Blink Rainbow", EFFECTS.blink_red_green_blue_yellow_cyan_magenta_white),
                ("Single White", EFFECTS.crossfade_white),
            ]
        )
        self.effect_map = presets
        layout.addWidget(QtWidgets.QLabel("Effect preset"), 0, 0)
        self.effect_combo = QtWidgets.QComboBox()
        self.effect_combo.addItems(self.effect_map.keys())
        self.effect_combo.currentIndexChanged.connect(self._update_effect_code_label)
        layout.addWidget(self.effect_combo, 0, 1)
        self.effect_code_label = QtWidgets.QLabel("Code: --")
        layout.addWidget(self.effect_code_label, 0, 2)

        self.effect_speed = NoWheelSlider(QtCore.Qt.Horizontal)
        self.effect_speed.setRange(1, 100)
        self.effect_speed.setValue(10)
        # Lower number = faster on the device, hence the "fast <-> slow" label.
        self.effect_speed_label = QtWidgets.QLabel("Speed (fast ←→ slow)")
        self.effect_speed_value = QtWidgets.QLabel("")
        self.effect_speed.valueChanged.connect(self._on_effect_speed_changed)
        self.effect_speed.sliderReleased.connect(self._apply_effect_speed)
        layout.addWidget(self.effect_speed_label, 1, 0)
        layout.addWidget(self.effect_speed, 1, 1)
        layout.addWidget(self.effect_speed_value, 1, 2)

        self.effect_brightness = NoWheelSlider(QtCore.Qt.Horizontal)
        self.effect_brightness.setRange(10, 100)
        self.effect_brightness.setValue(80)
        self.effect_brightness_value = QtWidgets.QLabel("")
        self.effect_brightness.valueChanged.connect(self._on_effect_brightness_changed)
        self.effect_brightness.sliderReleased.connect(self._apply_effect_brightness)
        layout.addWidget(QtWidgets.QLabel("Effect Brightness"), 2, 0)
        layout.addWidget(self.effect_brightness, 2, 1)
        layout.addWidget(self.effect_brightness_value, 2, 2)

        self.custom_effect_spin = QtWidgets.QSpinBox()
        self.custom_effect_spin.setRange(0, 255)
        # Hex spin box so any effect code can be tried while exploring the protocol.
        self.custom_effect_spin.setPrefix("0x")
        self.custom_effect_spin.setDisplayIntegerBase(16)
        custom_btn = QtWidgets.QPushButton("Send Custom Code")
        custom_btn.clicked.connect(self._send_custom_effect)
        layout.addWidget(QtWidgets.QLabel("Custom Effect"), 3, 0)
        layout.addWidget(self.custom_effect_spin, 3, 1)
        layout.addWidget(custom_btn, 3, 2)

        apply_btn = QtWidgets.QPushButton("Apply Effect")
        apply_btn.clicked.connect(self._apply_effect)
        layout.addWidget(apply_btn, 4, 0, 1, 3)
        self._update_effect_code_label()
        self._on_effect_speed_changed()
        self._on_effect_brightness_changed(self.effect_brightness.value())
        return group

    def _build_music_group(self) -> QtWidgets.QGroupBox:
        """Build the big Music Reactive panel (audio source, tuning, presets, screen sync).

        It's long because every tuning option has its own widget. Pattern used for each one:
        create widget -> connect its signal to an _apply_* method -> add it to the layout.
        """
        group = QtWidgets.QGroupBox("Music Reactive")
        layout = QtWidgets.QVBoxLayout(group)
        device_row = QtWidgets.QHBoxLayout()
        self.audio_device_combo = QtWidgets.QComboBox()
        device_refresh = QtWidgets.QPushButton("Refresh Inputs")
        device_refresh.clicked.connect(self._refresh_audio_devices)
        self.audio_device_combo.currentIndexChanged.connect(self._apply_audio_device_selection)
        device_row.addWidget(QtWidgets.QLabel("Audio Source"))
        device_row.addWidget(self.audio_device_combo)
        device_row.addWidget(device_refresh)
        layout.addLayout(device_row)

        # Loopback = record what the PC is playing instead of using the microphone.
        self.loopback_checkbox = QtWidgets.QCheckBox("Capture speaker output (loopback)")
        self.loopback_checkbox.toggled.connect(lambda _: self._apply_audio_device_selection())
        layout.addWidget(self.loopback_checkbox)

        # Audio analysis settings. Initial values come from utils/config.py; tooltips explain each one.
        self.music_buffer_spin = QtWidgets.QSpinBox()
        self.music_buffer_spin.setRange(64, 4096)
        self.music_buffer_spin.setSingleStep(64)
        self.music_buffer_spin.setValue(AUDIO_BUFFER_SIZE)
        self.music_buffer_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_buffer_spin.setToolTip("Audio buffer size. Lower = faster response, higher CPU. Example: 128 for snappy beats.")

        self.music_gain_spin = QtWidgets.QDoubleSpinBox()
        self.music_gain_spin.setRange(0.1, 10.0)
        self.music_gain_spin.setSingleStep(0.1)
        self.music_gain_spin.setDecimals(2)
        self.music_gain_spin.setValue(AUDIO_GAIN)
        self.music_gain_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_gain_spin.setToolTip("FFT gain multiplier. Higher = more reactive. Example: 2.5 for quiet songs.")

        self.music_floor_spin = QtWidgets.QDoubleSpinBox()
        self.music_floor_spin.setRange(0.0, 1.0)
        self.music_floor_spin.setSingleStep(0.05)
        self.music_floor_spin.setDecimals(2)
        self.music_floor_spin.setValue(AUDIO_VALUE_FLOOR)
        self.music_floor_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_floor_spin.setToolTip("Minimum channel value before color. Raise to avoid dim flicker. Example: 0.3 keeps mids visible.")

        self.music_boost_spin = QtWidgets.QDoubleSpinBox()
        self.music_boost_spin.setRange(0.1, 10.0)
        self.music_boost_spin.setSingleStep(0.1)
        self.music_boost_spin.setDecimals(2)
        self.music_boost_spin.setValue(AUDIO_VALUE_BOOST)
        self.music_boost_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_boost_spin.setToolTip("Value boost per band. Higher = brighter peaks. Example: 3.0 for energetic pop.")

        self.music_silence_spin = QtWidgets.QDoubleSpinBox()
        self.music_silence_spin.setRange(0.0, 0.05)
        self.music_silence_spin.setSingleStep(0.001)
        self.music_silence_spin.setDecimals(3)
        self.music_silence_spin.setValue(AUDIO_SILENCE_GATE)
        self.music_silence_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_silence_spin.setToolTip("Silence gate. Below this RMS, audio is treated as silence. Example: 0.003 to ignore room noise.")

        self.music_channels_spin = QtWidgets.QSpinBox()
        self.music_channels_spin.setRange(1, 8)
        self.music_channels_spin.setValue(2)
        self.music_channels_spin.valueChanged.connect(self._apply_music_audio_params)
        self.music_channels_spin.setToolTip("Force channel count. 1=mono, 2=stereo. Use 1 if your device rejects 2ch.")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Buffer size"), 0, 0)
        grid.addWidget(self.music_buffer_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("FFT gain"), 1, 0)
        grid.addWidget(self.music_gain_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Value floor"), 2, 0)
        grid.addWidget(self.music_floor_spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Value boost"), 3, 0)
        grid.addWidget(self.music_boost_spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Silence gate"), 4, 0)
        grid.addWidget(self.music_silence_spin, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Force channels"), 5, 0)
        grid.addWidget(self.music_channels_spin, 5, 1)
        layout.addLayout(grid)

        self.music_dynamic_checkbox = QtWidgets.QCheckBox("Scale brightness with song energy")
        self.music_dynamic_checkbox.setChecked(True)
        self.music_dynamic_checkbox.toggled.connect(self._apply_music_brightness_settings)
        self.music_dynamic_checkbox.setToolTip("When on, brightness follows song energy. Turn off for constant brightness.")
        layout.addWidget(self.music_dynamic_checkbox)

        self.music_keep_lit_checkbox = QtWidgets.QCheckBox("Never go black (hold last color)")
        self.music_keep_lit_checkbox.setChecked(False)
        self.music_keep_lit_checkbox.toggled.connect(self._apply_music_keep_lit)
        self.music_keep_lit_checkbox.setToolTip("Prevents full black during silence; holds last nonzero color at base brightness.")
        layout.addWidget(self.music_keep_lit_checkbox)

        self.music_ambient_checkbox = QtWidgets.QCheckBox("Fill silence with ambient sweep")
        self.music_ambient_checkbox.setChecked(False)
        self.music_ambient_checkbox.toggled.connect(self._apply_music_ambient_fill)
        self.music_ambient_checkbox.setToolTip("During silence, rotate through colors instead of holding still.")
        layout.addWidget(self.music_ambient_checkbox)

        # Keep the config defaults inside the slider range, and make sure max >= base.
        base_default = max(5, min(MUSIC_BASE_BRIGHTNESS, 100))
        max_default = max(base_default, min(MUSIC_MAX_BRIGHTNESS, 100))

        self.music_base_label = QtWidgets.QLabel(f"Base brightness (floor on quiet): {base_default}%")
        self.music_base_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_base_slider.setRange(5, 100)
        self.music_base_slider.setValue(base_default)
        self.music_base_slider.valueChanged.connect(self._on_music_base_changed)
        self.music_base_slider.sliderReleased.connect(self._apply_music_brightness_settings)
        self.music_base_slider.setToolTip("Minimum brightness. Example: 20% to keep lights visible on quiet songs.")

        self.music_max_label = QtWidgets.QLabel(f"Max brightness (cap peaks): {max_default}%")
        self.music_max_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_max_slider.setRange(5, 100)
        self.music_max_slider.setValue(max_default)
        self.music_max_slider.valueChanged.connect(self._on_music_max_changed)
        self.music_max_slider.sliderReleased.connect(self._apply_music_brightness_settings)
        self.music_max_slider.setToolTip("Maximum brightness cap. Example: 85% to avoid full white-outs.")

        self.music_smoothing_label = QtWidgets.QLabel("Color smoothing (blend speed): 30%")
        self.music_smoothing_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_smoothing_slider.setRange(0, 95)
        self.music_smoothing_slider.setValue(30)
        self.music_smoothing_slider.valueChanged.connect(self._on_music_smoothing_changed)
        self.music_smoothing_slider.sliderReleased.connect(self._apply_music_smoothing)
        self.music_smoothing_slider.setToolTip("Color smoothing. Higher = smoother, slower changes. Example: 50% for chill tracks.")

        self.music_transient_label = QtWidgets.QLabel("Transient boost (kick pop): 120%")
        self.music_transient_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_transient_slider.setRange(0, 300)
        self.music_transient_slider.setSingleStep(5)
        self.music_transient_slider.setValue(120)
        self.music_transient_slider.valueChanged.connect(self._on_music_transient_changed)
        self.music_transient_slider.sliderReleased.connect(self._apply_music_transient)
        self.music_transient_slider.setToolTip("Beat/transient boost. Higher = more pop on kicks/snares. Example: 160% for punchy EDM.")

        self.music_bass_label = QtWidgets.QLabel("Bass weight (lows): 100%")
        self.music_bass_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_bass_slider.setRange(50, 200)
        self.music_bass_slider.setValue(100)
        self.music_bass_slider.valueChanged.connect(self._on_music_band_changed)
        self.music_bass_slider.sliderReleased.connect(self._apply_music_bands)
        self.music_bass_slider.setToolTip("Bass weight. Higher = more response to lows. Example: 140% for hip-hop.")

        self.music_mid_label = QtWidgets.QLabel("Mid weight (vocals/gtrs): 100%")
        self.music_mid_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_mid_slider.setRange(50, 200)
        self.music_mid_slider.setValue(100)
        self.music_mid_slider.valueChanged.connect(self._on_music_band_changed)
        self.music_mid_slider.sliderReleased.connect(self._apply_music_bands)
        self.music_mid_slider.setToolTip("Mid weight. Controls vocals/guitars. Example: 90% to de-emphasize mids.")

        self.music_high_label = QtWidgets.QLabel("High weight (hi-hats): 100%")
        self.music_high_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_high_slider.setRange(50, 200)
        self.music_high_slider.setValue(100)
        self.music_high_slider.valueChanged.connect(self._on_music_band_changed)
        self.music_high_slider.sliderReleased.connect(self._apply_music_bands)
        self.music_high_slider.setToolTip("High weight. Cymbals/hi-hats. Example: 110% to sparkle more.")

        self.music_limiter_label = QtWidgets.QLabel("Peak limiter (compress peaks): 30%")
        self.music_limiter_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_limiter_slider.setRange(0, 100)
        self.music_limiter_slider.setValue(30)
        self.music_limiter_slider.valueChanged.connect(self._on_music_limiter_changed)
        self.music_limiter_slider.sliderReleased.connect(self._apply_music_limiter)
        self.music_limiter_slider.setToolTip("Soft limiter on brightness peaks. Higher = more compression. Example: 40% for loud tracks.")

        self.music_agc_checkbox = QtWidgets.QCheckBox("Auto gain (leveling)")
        self.music_agc_checkbox.setChecked(False)
        self.music_agc_checkbox.toggled.connect(self._apply_music_agc)
        self.music_agc_target = QtWidgets.QDoubleSpinBox()
        self.music_agc_target.setRange(0.05, 0.6)
        self.music_agc_target.setSingleStep(0.01)
        self.music_agc_target.setDecimals(2)
        self.music_agc_target.setValue(0.18)
        self.music_agc_target.valueChanged.connect(self._apply_music_agc)

        self.music_gate_checkbox = QtWidgets.QCheckBox("Beat gate (emphasize kicks)")
        self.music_gate_checkbox.setChecked(False)
        self.music_gate_checkbox.toggled.connect(self._apply_music_beat_gate)

        self.music_gate_label = QtWidgets.QLabel("Gate sensitivity (kick-only bias): 12%")
        self.music_gate_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_gate_slider.setRange(1, 50)
        self.music_gate_slider.setValue(12)
        self.music_gate_slider.valueChanged.connect(self._on_music_gate_changed)
        self.music_gate_slider.sliderReleased.connect(self._apply_music_beat_gate)
        self.music_gate_slider.setToolTip("Beat gate sensitivity. Lower = opens only on strong kicks. Example: 10% for EDM.")

        # NOTE: music_melody_label is created again further down; this first copy is redundant.
        self.music_melody_label = QtWidgets.QLabel("Melody focus (tonal boost): 25%")

        self.music_strobe_checkbox = QtWidgets.QCheckBox("Strobe on strong hits")
        self.music_strobe_checkbox.setChecked(False)
        self.music_strobe_checkbox.toggled.connect(self._apply_music_strobe)
        self.music_strobe_checkbox.setToolTip("Flash brighter on strong transients. Use sparingly with high brightness caps.")

        self.music_strobe_strength_label = QtWidgets.QLabel("Strobe strength (flash level): 50%")
        self.music_strobe_strength_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_strobe_strength_slider.setRange(0, 100)
        self.music_strobe_strength_slider.setValue(50)
        self.music_strobe_strength_slider.valueChanged.connect(self._on_music_strobe_changed)
        self.music_strobe_strength_slider.sliderReleased.connect(self._apply_music_strobe)
        self.music_strobe_strength_slider.setToolTip("How bright the strobe gets. Example: 60% for noticeable flashes.")

        self.music_strobe_thresh_label = QtWidgets.QLabel("Strobe sensitivity (trigger): 25%")
        self.music_strobe_thresh_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_strobe_thresh_slider.setRange(5, 80)
        self.music_strobe_thresh_slider.setValue(25)
        self.music_strobe_thresh_slider.valueChanged.connect(self._on_music_strobe_thresh_changed)
        self.music_strobe_thresh_slider.sliderReleased.connect(self._apply_music_strobe)
        self.music_strobe_thresh_slider.setToolTip("Strobe trigger threshold. Lower = strobes more often. Example: 20% for frequent hits.")

        self.music_flash_checkbox = QtWidgets.QCheckBox("Flash colors (no fades)")
        self.music_flash_checkbox.setChecked(False)
        self.music_flash_checkbox.toggled.connect(self._apply_music_flash)
        self.music_flash_checkbox.setToolTip("When on, colors step-change instead of smooth fades.")

        self.music_flicker_checkbox = QtWidgets.QCheckBox("Add flicker jitter")
        self.music_flicker_checkbox.setChecked(False)
        self.music_flicker_checkbox.toggled.connect(self._apply_music_flicker)
        self.music_flicker_amount = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_flicker_amount.setRange(0, 60)
        self.music_flicker_amount.setValue(0)
        self.music_flicker_amount.valueChanged.connect(self._on_music_flicker_changed)
        self.music_flicker_amount.sliderReleased.connect(self._apply_music_flicker)

        self.music_centroid_checkbox = QtWidgets.QCheckBox("Hue follows spectrum")
        self.music_centroid_checkbox.setChecked(False)
        self.music_centroid_checkbox.toggled.connect(self._apply_music_centroid_hue)
        self.music_centroid_label = QtWidgets.QLabel("")
        self.music_centroid_amount = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_centroid_amount.setRange(0, 50)
        self.music_centroid_amount.setValue(15)
        self.music_centroid_amount.valueChanged.connect(self._on_music_centroid_changed)
        self.music_centroid_amount.sliderReleased.connect(self._apply_music_centroid_hue)

        self.music_melody_checkbox = QtWidgets.QCheckBox("Melody focus")
        self.music_melody_checkbox.setChecked(False)
        self.music_melody_checkbox.toggled.connect(self._apply_music_melody_focus)
        self.music_melody_amount = NoWheelSlider(QtCore.Qt.Horizontal)
        self.music_melody_amount.setRange(0, 100)
        self.music_melody_amount.setValue(25)
        self.music_melody_amount.valueChanged.connect(self._apply_music_melody_focus)
        self.music_melody_amount.sliderReleased.connect(self._apply_music_melody_focus)
        self.music_melody_label = QtWidgets.QLabel("Melody focus (tonal boost): 25%")

        self.music_vertical_checkbox = QtWidgets.QCheckBox("Bottom-up pulse")
        self.music_vertical_checkbox.setChecked(False)
        self.music_vertical_checkbox.toggled.connect(self._apply_music_vertical_pulse)
        self.music_vertical_checkbox.setToolTip("Fill the LEDs from bottom to top with the music level.")

        self.music_fast_checkbox = QtWidgets.QCheckBox("Fast mode (low latency)")
        self.music_fast_checkbox.setChecked(False)
        self.music_fast_checkbox.toggled.connect(self._apply_music_fast_mode)
        self.music_fast_checkbox.setToolTip("Use smaller buffers and snappier settings for tight beat response.")

        self.music_drop_checkbox = QtWidgets.QCheckBox("Detect drops")
        self.music_drop_checkbox.setChecked(False)
        self.music_drop_checkbox.toggled.connect(self._apply_music_drop)
        self.music_drop_ratio = QtWidgets.QDoubleSpinBox()
        self.music_drop_ratio.setRange(0.1, 0.9)
        self.music_drop_ratio.setSingleStep(0.05)
        self.music_drop_ratio.setDecimals(2)
        self.music_drop_ratio.setValue(0.35)
        self.music_drop_ratio.valueChanged.connect(self._apply_music_drop)
        self.music_drop_min = QtWidgets.QSpinBox()
        self.music_drop_min.setRange(1, 200)
        self.music_drop_min.setValue(10)
        self.music_drop_min.valueChanged.connect(self._apply_music_drop)
        self.music_drop_cooldown = QtWidgets.QSpinBox()
        self.music_drop_cooldown.setRange(1, 500)
        self.music_drop_cooldown.setValue(60)
        self.music_drop_cooldown.valueChanged.connect(self._apply_music_drop)
        palette_row = QtWidgets.QHBoxLayout()
        palette_row.addWidget(QtWidgets.QLabel("Palette"))
        self.music_palette_combo = QtWidgets.QComboBox()
        self.music_palette_combo.addItems(
            ["Default", "Sunset", "Neon", "Cool", "Aurora", "Candy", "Ocean", "Currents", "Slow Rush"]
        )
        self.music_palette_combo.currentIndexChanged.connect(self._apply_music_palette)
        palette_row.addWidget(self.music_palette_combo, 1)

        # One-click presets: each dict holds every setting for a genre/mood. _apply_music_preset()
        # copies the values into the widgets. Missing keys fall back to defaults (see .get calls there).
        self.music_presets = OrderedDict(
            {
                "Balanced Live": {
                    "buffer": 192,
                    "gain": 2.2,
                    "floor": 0.26,
                    "boost": 3.0,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.28,
                    "transient": 1.4,
                    "bass": 1.15,
                    "mid": 1.1,
                    "high": 1.0,
                    "limiter": 0.28,
                    "gate_enabled": True,
                    "gate_threshold": 0.12,
                    "palette": "Currents",
                    "strobe_enabled": True,
                    "strobe_strength": 0.45,
                    "strobe_threshold": 0.24,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                },
                "Never Dark+": {
                    "buffer": 192,
                    "gain": 2.0,
                    "floor": 0.32,
                    "boost": 2.8,
                    "silence": 0.0025,
                    "channels": 2,
                    "smoothing": 0.35,
                    "transient": 1.3,
                    "bass": 1.1,
                    "mid": 1.15,
                    "high": 1.05,
                    "limiter": 0.3,
                    "gate_enabled": True,
                    "gate_threshold": 0.12,
                    "palette": "Slow Rush",
                    "strobe_enabled": False,
                    "strobe_strength": 0.4,
                    "strobe_threshold": 0.26,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                    "keep_lit": True,
                    "ambient_fill": True,
                },
                "Punchy": {
                    "buffer": 128,
                    "gain": 2.6,
                    "floor": 0.22,
                    "boost": 3.2,
                    "silence": 0.002,
                    "channels": 2,
                    "smoothing": 0.2,
                    "transient": 1.6,
                    "bass": 1.2,
                    "mid": 1.0,
                    "high": 0.9,
                    "limiter": 0.25,
                    "gate_enabled": True,
                    "gate_threshold": 0.12,
                    "palette": "Neon",
                    "strobe_enabled": True,
                    "strobe_strength": 0.6,
                    "strobe_threshold": 0.25,
                    "color_mode": "gradient",
                },
                "Smooth": {
                    "buffer": 256,
                    "gain": 1.5,
                    "floor": 0.35,
                    "boost": 2.0,
                    "silence": 0.004,
                    "channels": 2,
                    "smoothing": 0.5,
                    "transient": 1.0,
                    "bass": 1.0,
                    "mid": 1.0,
                    "high": 1.0,
                    "limiter": 0.35,
                    "gate_enabled": False,
                    "gate_threshold": 0.18,
                    "palette": "Cool",
                    "strobe_enabled": False,
                    "strobe_strength": 0.4,
                    "strobe_threshold": 0.30,
                    "color_mode": "gradient",
                },
                "Tame Impala": {
                    "buffer": 192,
                    "gain": 2.1,
                    "floor": 0.28,
                    "boost": 2.6,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.45,
                    "transient": 1.2,
                    "bass": 1.0,
                    "mid": 1.1,
                    "high": 1.2,
                    "limiter": 0.30,
                    "gate_enabled": False,
                    "gate_threshold": 0.16,
                    "palette": "Sunset",
                    "strobe_enabled": True,
                    "strobe_strength": 0.35,
                    "strobe_threshold": 0.26,
                    "color_mode": "gradient",
                },
                "EDM Rave": {
                    "buffer": 128,
                    "gain": 3.0,
                    "floor": 0.22,
                    "boost": 3.5,
                    "silence": 0.002,
                    "channels": 2,
                    "smoothing": 0.25,
                    "transient": 1.8,
                    "bass": 1.3,
                    "mid": 1.0,
                    "high": 1.1,
                    "limiter": 0.20,
                    "gate_enabled": True,
                    "gate_threshold": 0.10,
                    "palette": "Neon",
                    "strobe_enabled": True,
                    "strobe_strength": 0.70,
                    "strobe_threshold": 0.22,
                    "color_mode": "gradient",
                },
                "Lo-fi": {
                    "buffer": 256,
                    "gain": 1.3,
                    "floor": 0.35,
                    "boost": 1.8,
                    "silence": 0.004,
                    "channels": 1,
                    "smoothing": 0.65,
                    "transient": 0.9,
                    "bass": 1.1,
                    "mid": 1.0,
                    "high": 0.9,
                    "limiter": 0.40,
                    "gate_enabled": False,
                    "gate_threshold": 0.20,
                    "palette": "Cool",
                    "strobe_enabled": False,
                    "strobe_strength": 0.40,
                    "strobe_threshold": 0.30,
                    "color_mode": "gradient",
                },
                "Rock Live": {
                    "buffer": 160,
                    "gain": 2.4,
                    "floor": 0.25,
                    "boost": 3.0,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.35,
                    "transient": 1.4,
                    "bass": 1.1,
                    "mid": 1.2,
                    "high": 1.0,
                    "limiter": 0.25,
                    "gate_enabled": True,
                    "gate_threshold": 0.13,
                    "palette": "Default",
                    "strobe_enabled": True,
                    "strobe_strength": 0.45,
                    "strobe_threshold": 0.24,
                    "color_mode": "gradient",
                },
                "Sub-only": {
                    "buffer": 192,
                    "gain": 2.8,
                    "floor": 0.18,
                    "boost": 3.5,
                    "silence": 0.003,
                    "channels": 1,
                    "smoothing": 0.25,
                    "transient": 1.4,
                    "bass": 1.5,
                    "mid": 0.8,
                    "high": 0.6,
                    "limiter": 0.2,
                    "gate_enabled": True,
                    "gate_threshold": 0.14,
                    "palette": "Sunset",
                    "strobe_enabled": True,
                    "strobe_strength": 0.5,
                    "strobe_threshold": 0.28,
                    "color_mode": "dominant",
                },
                "Fast EDM": {
                    "buffer": 128,
                    "gain": 2.8,
                    "floor": 0.20,
                    "boost": 3.4,
                    "silence": 0.002,
                    "channels": 2,
                    "smoothing": 0.18,
                    "transient": 1.7,
                    "bass": 1.25,
                    "mid": 1.05,
                    "high": 1.05,
                    "limiter": 0.22,
                    "gate_enabled": True,
                    "gate_threshold": 0.11,
                    "palette": "Neon",
                    "strobe_enabled": True,
                    "strobe_strength": 0.65,
                    "strobe_threshold": 0.22,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                },
                "Drum & Bass": {
                    "buffer": 160,
                    "gain": 2.2,
                    "floor": 0.24,
                    "boost": 3.2,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.22,
                    "transient": 1.5,
                    "bass": 1.35,
                    "mid": 1.1,
                    "high": 1.0,
                    "limiter": 0.3,
                    "gate_enabled": True,
                    "gate_threshold": 0.12,
                    "palette": "Cool",
                    "strobe_enabled": True,
                    "strobe_strength": 0.55,
                    "strobe_threshold": 0.24,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                },
                "House Groove": {
                    "buffer": 192,
                    "gain": 2.0,
                    "floor": 0.26,
                    "boost": 2.8,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.3,
                    "transient": 1.3,
                    "bass": 1.2,
                    "mid": 1.0,
                    "high": 0.95,
                    "limiter": 0.28,
                    "gate_enabled": True,
                    "gate_threshold": 0.13,
                    "palette": "Aurora",
                    "strobe_enabled": False,
                    "strobe_strength": 0.4,
                    "strobe_threshold": 0.26,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                },
                "Chill Vocal": {
                    "buffer": 224,
                    "gain": 1.7,
                    "floor": 0.32,
                    "boost": 2.2,
                    "silence": 0.004,
                    "channels": 2,
                    "smoothing": 0.45,
                    "transient": 1.0,
                    "bass": 1.0,
                    "mid": 1.2,
                    "high": 1.05,
                    "limiter": 0.35,
                    "gate_enabled": False,
                    "gate_threshold": 0.16,
                    "palette": "Candy",
                    "strobe_enabled": False,
                    "strobe_strength": 0.3,
                    "strobe_threshold": 0.3,
                    "color_mode": "gradient",
                    "vertical_pulse": False,
                },
                "Cinematic Rise": {
                    "buffer": 256,
                    "gain": 2.4,
                    "floor": 0.28,
                    "boost": 3.0,
                    "silence": 0.003,
                    "channels": 2,
                    "smoothing": 0.4,
                    "transient": 1.6,
                    "bass": 1.1,
                    "mid": 1.15,
                    "high": 1.2,
                    "limiter": 0.32,
                    "gate_enabled": True,
                    "gate_threshold": 0.14,
                    "palette": "Ocean",
                    "strobe_enabled": True,
                    "strobe_strength": 0.6,
                    "strobe_threshold": 0.23,
                    "color_mode": "gradient",
                    "vertical_pulse": True,
                },
            }
        )

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Color mode"))
        self.music_color_mode = QtWidgets.QComboBox()
        self.music_color_mode.addItems(["Gradient sweep", "Dominant band"])
        self.music_color_mode.currentIndexChanged.connect(self._apply_music_color_mode)
        mode_row.addWidget(self.music_color_mode, 1)
        mode_row.addWidget(self.music_melody_checkbox)

        preview_row = QtWidgets.QHBoxLayout()
        self.music_color_preview = QtWidgets.QFrame()
        self.music_color_preview.setFixedSize(72, 72)
        self.music_color_preview.setStyleSheet(
            "background-color: rgb(0,0,0); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px;"
        )
        self.music_color_label = QtWidgets.QLabel("Live: #000000 @ 0%")
        preview_row.addWidget(self.music_color_preview)
        preview_row.addWidget(self.music_color_label, 1)

        # Apply bracketed descriptions with smaller font.
        # (Repetitive, but each label needs its own text; a helper dict could tidy this up later.)
        self._set_label_with_desc(
            self.music_base_label, "Base brightness", "floor on quiet", self.music_base_slider.value()
        )
        self._set_label_with_desc(
            self.music_max_label, "Max brightness", "cap peaks", self.music_max_slider.value()
        )
        self._set_label_with_desc(
            self.music_smoothing_label, "Color smoothing", "blend speed", self.music_smoothing_slider.value()
        )
        self._set_label_with_desc(
            self.music_transient_label, "Transient boost", "kick pop", self.music_transient_slider.value()
        )
        self._set_label_with_desc(
            self.music_bass_label, "Bass weight", "lows", self.music_bass_slider.value()
        )
        self._set_label_with_desc(
            self.music_mid_label, "Mid weight", "vocals/gtrs", self.music_mid_slider.value()
        )
        self._set_label_with_desc(
            self.music_high_label, "High weight", "hi-hats", self.music_high_slider.value()
        )
        self._set_label_with_desc(
            self.music_limiter_label, "Peak limiter", "compress peaks", self.music_limiter_slider.value()
        )
        self._set_label_with_desc(
            self.music_gate_label, "Gate sensitivity", "kick-only bias", self.music_gate_slider.value()
        )
        self._set_label_with_desc(
            self.music_strobe_strength_label, "Strobe strength", "flash level", self.music_strobe_strength_slider.value()
        )
        self._set_label_with_desc(
            self.music_strobe_thresh_label, "Strobe sensitivity", "trigger", self.music_strobe_thresh_slider.value()
        )
        self._set_label_with_desc(
            self.music_centroid_label, "Hue follows spectrum", "bias amount", self.music_centroid_amount.value()
        )

        # Now add everything to the panel in display order.
        layout.addWidget(self.music_base_label)
        layout.addWidget(self.music_base_slider)
        layout.addWidget(self.music_max_label)
        layout.addWidget(self.music_max_slider)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Presets"))
        for name in self.music_presets.keys():
            # n=name binds the current name; without it every button would use the last preset.
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._apply_music_preset(n))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)
        layout.addLayout(mode_row)
        layout.addWidget(self.music_vertical_checkbox)
        layout.addWidget(self.music_fast_checkbox)
        layout.addWidget(self.music_smoothing_label)
        layout.addWidget(self.music_smoothing_slider)
        layout.addWidget(self.music_transient_label)
        layout.addWidget(self.music_transient_slider)
        layout.addWidget(self.music_bass_label)
        layout.addWidget(self.music_bass_slider)
        layout.addWidget(self.music_mid_label)
        layout.addWidget(self.music_mid_slider)
        layout.addWidget(self.music_high_label)
        layout.addWidget(self.music_high_slider)
        layout.addWidget(self.music_limiter_label)
        layout.addWidget(self.music_limiter_slider)
        layout.addWidget(self.music_melody_label)
        layout.addWidget(self.music_melody_amount)
        agc_row = QtWidgets.QHBoxLayout()
        agc_row.addWidget(self.music_agc_checkbox)
        agc_row.addWidget(QtWidgets.QLabel("Target"))
        agc_row.addWidget(self.music_agc_target)
        layout.addLayout(agc_row)
        layout.addWidget(self.music_gate_checkbox)
        layout.addWidget(self.music_gate_label)
        layout.addWidget(self.music_gate_slider)
        layout.addWidget(self.music_strobe_checkbox)
        layout.addWidget(self.music_strobe_strength_label)
        layout.addWidget(self.music_strobe_strength_slider)
        layout.addWidget(self.music_strobe_thresh_label)
        layout.addWidget(self.music_strobe_thresh_slider)
        layout.addWidget(self.music_flash_checkbox)
        layout.addWidget(self.music_flicker_checkbox)
        layout.addWidget(self.music_flicker_amount)
        layout.addWidget(self.music_centroid_checkbox)
        layout.addWidget(self.music_centroid_label)
        layout.addWidget(self.music_centroid_amount)
        drop_row1 = QtWidgets.QHBoxLayout()
        drop_row1.addWidget(self.music_drop_checkbox)
        drop_row1.addWidget(QtWidgets.QLabel("Ratio"))
        drop_row1.addWidget(self.music_drop_ratio)
        drop_row2 = QtWidgets.QHBoxLayout()
        drop_row2.addWidget(QtWidgets.QLabel("Min frames"))
        drop_row2.addWidget(self.music_drop_min)
        drop_row2.addWidget(QtWidgets.QLabel("Cooldown"))
        drop_row2.addWidget(self.music_drop_cooldown)
        layout.addLayout(drop_row1)
        layout.addLayout(drop_row2)
        layout.addLayout(palette_row)
        layout.addLayout(preview_row)

        # Checkable button = acts as an on/off switch; toggled(bool) fires when it flips.
        self.music_toggle = QtWidgets.QPushButton("Enable Audio Reactive")
        self.music_toggle.setCheckable(True)
        self.music_toggle.toggled.connect(self._toggle_music)
        layout.addWidget(self.music_toggle)
        layout.addWidget(
            QtWidgets.QLabel("Uses microphone FFT to push colors directly to the connected strip.")
        )
        # --- Screen colour sync controls (they drive ScreenColorSampler) ---
        self.screen_sync_toggle = QtWidgets.QPushButton("Enable Screen Color Sync")
        self.screen_sync_toggle.setCheckable(True)
        self.screen_sync_toggle.toggled.connect(self._toggle_screen_sync)
        layout.addWidget(self.screen_sync_toggle)
        layout.addWidget(QtWidgets.QLabel("Matches LEDs to the dominant color of your display in real time."))
        self.screen_monitor_combo = QtWidgets.QComboBox()
        self.screen_monitor_combo.currentIndexChanged.connect(self._on_screen_monitor_changed)
        self.screen_region_combo = QtWidgets.QComboBox()
        self.screen_region_combo.addItems(["Full", "Left", "Right", "Top", "Bottom"])
        self.screen_region_combo.currentIndexChanged.connect(self._on_screen_region_changed)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Screen mode"))
        self.screen_mode_combo = QtWidgets.QComboBox()
        self.screen_mode_combo.addItems(["Dominant (average)", "Standout (most vibrant)"])
        self.screen_mode_combo.currentIndexChanged.connect(self._on_screen_mode_changed)
        mode_row.addWidget(self.screen_mode_combo)
        mode_row.addWidget(QtWidgets.QLabel("Monitor"))
        mode_row.addWidget(self.screen_monitor_combo, 1)
        mode_row.addWidget(QtWidgets.QLabel("Region"))
        mode_row.addWidget(self.screen_region_combo)
        layout.addLayout(mode_row)
        resp_row = QtWidgets.QHBoxLayout()
        resp_row.addWidget(QtWidgets.QLabel("Responsiveness"))
        self.screen_smoothing_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.screen_smoothing_slider.setRange(0, 90)
        self.screen_smoothing_slider.setValue(25)
        self.screen_smoothing_slider.valueChanged.connect(self._on_screen_smoothing_changed)
        resp_row.addWidget(self.screen_smoothing_slider, 1)
        resp_row.addWidget(QtWidgets.QLabel("Snap"))
        resp_row.addWidget(QtWidgets.QLabel("Smooth"))
        layout.addLayout(resp_row)
        edge_row = QtWidgets.QHBoxLayout()
        edge_row.addWidget(QtWidgets.QLabel("Edge emphasis"))
        self.screen_edge_slider = NoWheelSlider(QtCore.Qt.Horizontal)
        self.screen_edge_slider.setRange(0, 200)
        self.screen_edge_slider.setValue(60)
        self.screen_edge_slider.valueChanged.connect(self._on_screen_edge_changed)
        edge_row.addWidget(self.screen_edge_slider, 1)
        layout.addLayout(edge_row)
        # Push every widget's initial value into the worker so the UI and engine start in sync.
        self._refresh_audio_devices()
        self._apply_music_audio_params()
        self._apply_music_brightness_settings()
        self._apply_music_smoothing()
        self._apply_music_transient()
        self._apply_music_bands()
        self._apply_music_limiter()
        self._apply_music_beat_gate()
        self._apply_music_strobe()
        self._apply_music_flash()
        self._apply_music_flicker()
        self._apply_music_agc()
        self._apply_music_centroid_hue()
        self._apply_music_vertical_pulse()
        self._apply_music_drop()
        self._apply_music_palette()
        self._apply_music_ambient_fill()
        self._apply_music_keep_lit()
        self._refresh_screens()
        return group

    # --- Device scanning / connecting ---
    # Pattern: submit a coroutine to the asyncio thread, then handle the result in a done-callback.
    def _refresh_devices(self, auto_connect: bool = True) -> None:
        future = self.executor.submit(self.manager.refresh(auto_connect=auto_connect))
        future.add_done_callback(self._handle_device_refresh)

    def _handle_device_refresh(self, future) -> None:
        # Runs on the asyncio thread, hence signals instead of direct widget updates.
        try:
            devices = future.result()
        except Exception as exc:  # pragma: no cover - GUI path
            self.signals.status_message.emit(f"Device refresh failed: {exc}")
            return
        self.signals.ble_devices.emit(devices)
        summary = f"Scan complete – {len(devices)} compatible device(s) found"
        self.signals.status_message.emit(summary)
        connected = self.manager.state.connected
        if self.manager.controller() and connected:
            self.signals.connection_status.emit(f"Connected to {connected.name}")
            self.signals.status_message.emit(f"✓ Auto-connected to {connected.name}")
        self._refresh_screens()

    def _populate_devices(self, devices: list[BleDeviceInfo]) -> None:
        self.device_list.clear()
        # Store the address inside each list item so we know which device was picked later.
        for dev in devices:
            item = QtWidgets.QListWidgetItem(dev.label())
            item.setData(QtCore.Qt.UserRole, dev.address)
            self.device_list.addItem(item)

    def _connect_selected(self) -> None:
        item = self.device_list.currentItem()
        if not item:
            self._set_status("Select a device first")
            return
        address = item.data(QtCore.Qt.UserRole)
        future = self.executor.submit(self.manager.connect(address))
        future.add_done_callback(lambda f, addr=address: self._handle_connect(f, addr))

    def _refresh_screens(self) -> None:
        """Fill the monitor dropdown for screen sync (blockSignals avoids triggering handlers)."""
        self._screens = QtWidgets.QApplication.screens()
        self.screen_monitor_combo.blockSignals(True)
        self.screen_monitor_combo.clear()
        for idx, screen in enumerate(self._screens):
            self.screen_monitor_combo.addItem(f"{idx}: {screen.name()}")
        self.screen_monitor_combo.blockSignals(False)
        if self._screens:
            self.screen_monitor_combo.setCurrentIndex(0)
            self.screen_sampler.set_monitor(0)

    def _handle_connect(self, future, address: str) -> None:
        try:
            future.result()
        except Exception as exc:
            self.signals.status_message.emit(f"Connect failed: {exc}")
            return
        info = self.manager.state.connected
        label = info.name if info else address
        self.signals.connection_status.emit(f"Connected to {label}")
        self.signals.status_message.emit("Connected")
        # connect() powers the strip on, so switch it back off; the user decides when to turn it on.
        self._silent_power_off()

    def _disconnect(self) -> None:
        future = self.executor.submit(self.manager.disconnect())
        future.add_done_callback(self._handle_disconnect)

    def _handle_disconnect(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            self.signals.status_message.emit(f"Disconnect failed: {exc}")
            return
        self.signals.connection_status.emit("Not connected")
        self.signals.status_message.emit("Disconnected")

    # --- Colour / effect handlers ---
    def _pick_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor("#ffffff"), self, "Choose color")
        if not color.isValid():
            return
        rgb = (color.red(), color.green(), color.blue())

        # Small coroutine so colour and brightness are sent one after the other.
        async def task():
            await self.manager.set_color(rgb)
            await self.manager.set_brightness(self.brightness_slider.value())
        self._run_async(task(), f"Color set to {rgb}")

    def _apply_brightness(self) -> None:
        value = self.brightness_slider.value()
        self._run_async(self.manager.set_brightness(value), f"Brightness {value}%")

    def _apply_color_temp(self) -> None:
        kelvin = self.temp_slider.value()
        self._run_async(self.manager.set_color_temp(kelvin), f"Color temp {kelvin}K")

    def _apply_effect(self) -> None:
        name = self.effect_combo.currentText()
        code = self.effect_map.get(name)
        speed = self.effect_speed.value()
        if code is None:
            return
        device_speed = self._to_device_speed(speed)
        self._run_async(self.manager.apply_effect(code, device_speed), f"Effect {name}")
        self._update_effect_code_label()

    def _apply_effect_speed(self) -> None:
        speed = self._to_device_speed(self.effect_speed.value())
        self._run_async(self.manager.set_effect_speed_value(speed), f"Effect speed {self.effect_speed.value()}")

    def _apply_effect_brightness(self) -> None:
        value = self.effect_brightness.value()
        self._run_async(self.manager.set_brightness(value), f"Effect brightness {value}%")

    def _on_effect_speed_changed(self, value: int | None = None) -> None:
        slider_value = value if value is not None else self.effect_speed.value()
        device_value = self._to_device_speed(slider_value)
        self.effect_speed_value.setText(f"{slider_value}% (device {device_value})")

    def _on_effect_brightness_changed(self, value: int) -> None:
        self.effect_brightness_value.setText(f"{value}%")

    def _send_custom_effect(self) -> None:
        code = self.custom_effect_spin.value()
        speed = self._to_device_speed(self.effect_speed.value())
        self._run_async(self.manager.apply_effect(code, speed), f"Custom effect 0x{code:02X}")
        self._update_effect_code_label()

    def _update_effect_code_label(self, *_) -> None:
        code = self._current_effect_code()
        if code is None:
            self.effect_code_label.setText("Code: --")
        else:
            self.effect_code_label.setText(f"Code: 0x{code:02X}")

    def _current_effect_code(self) -> int | None:
        name = self.effect_combo.currentText()
        return self.effect_map.get(name)

    @staticmethod
    def _to_device_speed(slider_value: int) -> int:
        # Currently 1:1 with the slider (clamped); kept as a function in case the mapping changes.
        return max(0, min(100, slider_value))

    # NOTE: this reads self.schedule_widgets, but no schedule panel is built yet, so nothing
    # calls it. The BLE layer already supports schedules; the GUI part is still TODO.
    def _apply_schedule(self, when: str) -> None:
        widgets = self.schedule_widgets[when]
        time_edit: QtWidgets.QTimeEdit = widgets["time"]
        enable_cb: QtWidgets.QCheckBox = widgets["enable"]
        day_checks: list[QtWidgets.QCheckBox] = widgets["days"]
        hour = time_edit.time().hour()
        minute = time_edit.time().minute()
        mask = 0
        for cb, (_, value) in zip(day_checks, DAY_MAPPING):
            if cb.isChecked():
                mask |= value
        # No day ticked = treat it as every day.
        if mask == 0:
            mask = WEEK_DAYS.all
        self._run_async(
            self.manager.set_schedule(when, mask, hour, minute, enable_cb.isChecked()),
            f"Schedule ({when}) saved",
        )

    # --- Music Reactive handlers ---
    # Most _apply_music_* methods just read their widgets and forward the values to music_worker.
    # The matching _on_music_*_changed methods update the label text while a slider moves.
    def _refresh_audio_devices(self) -> None:
        try:
            self.audio_devices = list_input_devices()
        except Exception as exc:
            self._set_status(f"Audio device refresh failed: {exc}")
            return
        self.audio_device_combo.blockSignals(True)
        self.audio_device_combo.clear()
        for info in self.audio_devices:
            self.audio_device_combo.addItem(info.label(), userData=info.index)
        self.audio_device_combo.blockSignals(False)
        if self.audio_devices:
            self.audio_device_combo.setCurrentIndex(0)
            self._apply_audio_device_selection()

    def _apply_audio_device_selection(self) -> None:
        if not self.audio_devices:
            return
        idx = self.audio_device_combo.currentIndex()
        if idx < 0 or idx >= len(self.audio_devices):
            return
        device_info = self.audio_devices[idx]
        # Output devices can only be captured as loopback, so tick the box for the user.
        if device_info.supports_loopback and not self.loopback_checkbox.isChecked():
            self.loopback_checkbox.blockSignals(True)
            self.loopback_checkbox.setChecked(True)
            self.loopback_checkbox.blockSignals(False)
        loopback = self.loopback_checkbox.isChecked()
        if device_info.supports_loopback:
            loopback = True
        elif loopback and device_info.max_input_channels <= 0:
            loopback = False
        self.selected_audio_device = device_info
        self._run_async(
            self.music_worker.set_audio_source(device_info.index, loopback),
            f"Audio source set to {device_info.name}",
        )
        self._apply_music_audio_params()
        if self.music_toggle.isChecked():
            self._run_async(self.music_worker.start(), None)

    def _apply_music_audio_params(self) -> None:
        buffer_size = self.music_buffer_spin.value()
        gain = float(self.music_gain_spin.value())
        value_floor = float(self.music_floor_spin.value())
        value_boost = float(self.music_boost_spin.value())
        silence_gate = float(self.music_silence_spin.value())
        forced_channels = int(self.music_channels_spin.value())
        self._run_async(
            self.music_worker.apply_audio_params(
                buffer_size=buffer_size,
                gain=gain,
                value_floor=value_floor,
                value_boost=value_boost,
                silence_gate=silence_gate,
                forced_channels=forced_channels,
            ),
            None,
        )

    def _apply_music_brightness_settings(self) -> None:
        base = self.music_base_slider.value()
        ceiling = self.music_max_slider.value()
        adaptive = self.music_dynamic_checkbox.isChecked()
        self.music_worker.configure_brightness(base, ceiling, adaptive)

    def _apply_music_keep_lit(self) -> None:
        self.music_worker.configure_keep_lit(self.music_keep_lit_checkbox.isChecked())

    def _apply_music_ambient_fill(self) -> None:
        self.music_worker.configure_ambient_fill(self.music_ambient_checkbox.isChecked())

    def _on_music_base_changed(self, value: int) -> None:
        # Keep base <= max: pushing base above max drags the max slider up with it.
        if value > self.music_max_slider.value():
            self.music_max_slider.blockSignals(True)
            self.music_max_slider.setValue(value)
            self.music_max_slider.blockSignals(False)
            self._on_music_max_changed(value)
        # While the user is dragging, wait until release to apply (avoids spamming the lamp).
        if not self.music_base_slider.isSliderDown():
            self._apply_music_brightness_settings()
        self._set_label_with_desc(self.music_base_label, "Base brightness", "floor on quiet", value)

    def _on_music_max_changed(self, value: int) -> None:
        # Mirror of the base handler: lowering max below base drags base down.
        if value < self.music_base_slider.value():
            self.music_base_slider.blockSignals(True)
            self.music_base_slider.setValue(value)
            self.music_base_slider.blockSignals(False)
            self._on_music_base_changed(value)
        if not self.music_max_slider.isSliderDown():
            self._apply_music_brightness_settings()
        self._set_label_with_desc(self.music_max_label, "Max brightness", "cap peaks", value)

    def _apply_music_smoothing(self) -> None:
        amount = self.music_smoothing_slider.value() / 100.0
        self.music_worker.configure_smoothing(amount)

    def _on_music_smoothing_changed(self, value: int) -> None:
        self._set_label_with_desc(self.music_smoothing_label, "Color smoothing", "blend speed", value)
        if not self.music_smoothing_slider.isSliderDown():
            self._apply_music_smoothing()

    def _apply_music_transient(self) -> None:
        boost = self.music_transient_slider.value() / 100.0
        self.music_worker.configure_transient_boost(boost)

    def _on_music_transient_changed(self, value: int) -> None:
        self._set_label_with_desc(self.music_transient_label, "Transient boost", "kick pop", value)
        if not self.music_transient_slider.isSliderDown():
            self._apply_music_transient()

    def _apply_music_bands(self) -> None:
        bass = self.music_bass_slider.value() / 100.0
        mid = self.music_mid_slider.value() / 100.0
        high = self.music_high_slider.value() / 100.0
        self.music_worker.configure_band_weights(bass, mid, high)

    def _on_music_band_changed(self, _value: int) -> None:
        self._set_label_with_desc(
            self.music_bass_label, "Bass weight", "lows", self.music_bass_slider.value()
        )
        self._set_label_with_desc(
            self.music_mid_label, "Mid weight", "vocals/gtrs", self.music_mid_slider.value()
        )
        self._set_label_with_desc(
            self.music_high_label, "High weight", "hi-hats", self.music_high_slider.value()
        )
        if not any(
            slider.isSliderDown()
            for slider in (self.music_bass_slider, self.music_mid_slider, self.music_high_slider)
        ):
            self._apply_music_bands()

    def _apply_music_limiter(self) -> None:
        amount = self.music_limiter_slider.value() / 100.0
        self.music_worker.configure_soft_limiter(amount)

    def _on_music_limiter_changed(self, value: int) -> None:
        self._set_label_with_desc(self.music_limiter_label, "Peak limiter", "compress peaks", value)
        if not self.music_limiter_slider.isSliderDown():
            self._apply_music_limiter()

    def _apply_music_beat_gate(self) -> None:
        enabled = self.music_gate_checkbox.isChecked()
        threshold = self.music_gate_slider.value() / 100.0
        self.music_worker.configure_beat_gate(enabled, threshold)

    def _on_music_gate_changed(self, value: int) -> None:
        self._set_label_with_desc(self.music_gate_label, "Gate sensitivity", "kick-only bias", value)
        if not self.music_gate_slider.isSliderDown():
            self._apply_music_beat_gate()

    def _apply_music_palette(self) -> None:
        name = self.music_palette_combo.currentText()
        self.music_worker.configure_palette(name)

    def _apply_music_strobe(self) -> None:
        enabled = self.music_strobe_checkbox.isChecked()
        strength = self.music_strobe_strength_slider.value() / 100.0
        threshold = self.music_strobe_thresh_slider.value() / 100.0
        self.music_worker.configure_strobe(enabled, strength, threshold)

    def _on_music_strobe_changed(self, value: int) -> None:
        self._set_label_with_desc(
            self.music_strobe_strength_label, "Strobe strength", "flash level", value
        )
        if not self.music_strobe_strength_slider.isSliderDown():
            self._apply_music_strobe()

    def _on_music_strobe_thresh_changed(self, value: int) -> None:
        self._set_label_with_desc(
            self.music_strobe_thresh_label, "Strobe sensitivity", "trigger", value
        )
        if not self.music_strobe_thresh_slider.isSliderDown():
            self._apply_music_strobe()

    def _apply_music_flash(self) -> None:
        enabled = self.music_flash_checkbox.isChecked()
        self.music_worker.configure_flash(enabled)

    def _apply_music_flicker(self) -> None:
        enabled = self.music_flicker_checkbox.isChecked()
        amount = self.music_flicker_amount.value() / 100.0
        self.music_worker.configure_flicker(enabled, amount)

    def _apply_music_agc(self) -> None:
        enabled = self.music_agc_checkbox.isChecked()
        target = float(self.music_agc_target.value())
        self.music_worker.configure_auto_gain(enabled, target)

    def _apply_music_centroid_hue(self) -> None:
        enabled = self.music_centroid_checkbox.isChecked()
        amount = self.music_centroid_amount.value() / 100.0
        self.music_worker.configure_centroid_hue(enabled, amount)

    def _apply_music_melody_focus(self) -> None:
        enabled = self.music_melody_checkbox.isChecked()
        amount = self.music_melody_amount.value() / 100.0
        self._set_label_with_desc(self.music_melody_label, "Melody focus", "boost mids/highs when tonal", int(self.music_melody_amount.value()))
        self.music_worker.configure_melody_focus(enabled, amount)

    def _on_music_centroid_changed(self, value: int) -> None:
        self._set_label_with_desc(self.music_centroid_label, "Hue follows spectrum", "bias amount", value)
        if not self.music_centroid_amount.isSliderDown():
            self._apply_music_centroid_hue()

    def _on_music_flicker_changed(self, value: int) -> None:
        if not self.music_flicker_amount.isSliderDown():
            self._apply_music_flicker()

    def _apply_music_drop(self) -> None:
        enabled = self.music_drop_checkbox.isChecked()
        ratio = float(self.music_drop_ratio.value())
        min_frames = int(self.music_drop_min.value())
        cooldown = int(self.music_drop_cooldown.value())
        self.music_worker.configure_drop_detector(enabled, ratio, min_frames, cooldown)

    def _apply_music_color_mode(self) -> None:
        mode = "gradient" if self.music_color_mode.currentIndex() == 0 else "dominant"
        self.music_worker.configure_color_mode(mode)

    def _apply_music_vertical_pulse(self) -> None:
        enabled = self.music_vertical_checkbox.isChecked()
        self.music_worker.configure_vertical_pulse(enabled)

    def _apply_music_fast_mode(self) -> None:
        """Toggle low-latency mode: save current settings, apply snappy ones, restore on disable."""
        enabled = self.music_fast_checkbox.isChecked()

        # Set a widget's value without firing its valueChanged signal (we apply everything at the end).
        def set_value(widget, value):
            block = getattr(widget, "blockSignals", lambda _: None)
            block(True)
            widget.setValue(value)
            block(False)

        if enabled:
            # Only snapshot once, so toggling twice can't overwrite the originals with fast values.
            if not self._fast_snapshot:
                self._fast_snapshot = {
                    "buffer": self.music_buffer_spin.value(),
                    "smoothing": self.music_smoothing_slider.value(),
                    "transient": self.music_transient_slider.value(),
                    "limiter": self.music_limiter_slider.value(),
                    "gate_enabled": self.music_gate_checkbox.isChecked(),
                    "gate_threshold": self.music_gate_slider.value(),
                }
            set_value(self.music_buffer_spin, 128)
            set_value(self.music_smoothing_slider, 20)
            set_value(self.music_transient_slider, 150)
            set_value(self.music_limiter_slider, 25)
            self.music_gate_checkbox.blockSignals(True)
            self.music_gate_checkbox.setChecked(True)
            self.music_gate_checkbox.blockSignals(False)
            set_value(self.music_gate_slider, 12)
            self._apply_music_audio_params()
            self._apply_music_smoothing()
            self._on_music_smoothing_changed(self.music_smoothing_slider.value())
            self._apply_music_transient()
            self._on_music_transient_changed(self.music_transient_slider.value())
            self._apply_music_limiter()
            self._on_music_limiter_changed(self.music_limiter_slider.value())
            self._apply_music_beat_gate()
            self._on_music_gate_changed(self.music_gate_slider.value())
            self._set_status("Fast mode enabled: low buffer, low smoothing, strong transient and gate.")
        else:
            if self._fast_snapshot:
                set_value(self.music_buffer_spin, int(self._fast_snapshot.get("buffer", 256)))
                set_value(self.music_smoothing_slider, int(self._fast_snapshot.get("smoothing", 30)))
                set_value(self.music_transient_slider, int(self._fast_snapshot.get("transient", 120)))
                set_value(self.music_limiter_slider, int(self._fast_snapshot.get("limiter", 30)))
                self.music_gate_checkbox.blockSignals(True)
                self.music_gate_checkbox.setChecked(bool(self._fast_snapshot.get("gate_enabled", False)))
                self.music_gate_checkbox.blockSignals(False)
                set_value(self.music_gate_slider, int(self._fast_snapshot.get("gate_threshold", 12)))
            self._fast_snapshot = {}
            self._apply_music_audio_params()
            self._apply_music_smoothing()
            self._on_music_smoothing_changed(self.music_smoothing_slider.value())
            self._apply_music_transient()
            self._on_music_transient_changed(self.music_transient_slider.value())
            self._apply_music_limiter()
            self._on_music_limiter_changed(self.music_limiter_slider.value())
            self._apply_music_beat_gate()
            self._on_music_gate_changed(self.music_gate_slider.value())
            self._set_status("Fast mode disabled: restored previous settings.")

    def _apply_music_preset(self, name: str) -> None:
        """Load a preset into all the widgets, then apply every setting to the worker."""
        preset = self.music_presets.get(name)
        if not preset:
            return

        def set_value(widget, value):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        # Step 1: update widgets silently (signals blocked so we don't apply a half-loaded preset).
        set_value(self.music_buffer_spin, int(preset["buffer"]))
        set_value(self.music_gain_spin, float(preset["gain"]))
        set_value(self.music_floor_spin, float(preset["floor"]))
        set_value(self.music_boost_spin, float(preset["boost"]))
        set_value(self.music_silence_spin, float(preset["silence"]))
        set_value(self.music_channels_spin, int(preset["channels"]))
        set_value(self.music_smoothing_slider, int(preset["smoothing"] * 100))
        self._set_label_with_desc(
            self.music_smoothing_label, "Color smoothing", "blend speed", self.music_smoothing_slider.value()
        )
        set_value(self.music_transient_slider, int(preset["transient"] * 100))
        self._set_label_with_desc(
            self.music_transient_label, "Transient boost", "kick pop", self.music_transient_slider.value()
        )
        set_value(self.music_bass_slider, int(preset.get("bass", 1.0) * 100))
        set_value(self.music_mid_slider, int(preset.get("mid", 1.0) * 100))
        set_value(self.music_high_slider, int(preset.get("high", 1.0) * 100))
        self._set_label_with_desc(
            self.music_bass_label, "Bass weight", "lows", self.music_bass_slider.value()
        )
        self._set_label_with_desc(
            self.music_mid_label, "Mid weight", "vocals/gtrs", self.music_mid_slider.value()
        )
        self._set_label_with_desc(
            self.music_high_label, "High weight", "hi-hats", self.music_high_slider.value()
        )
        set_value(self.music_limiter_slider, int(preset.get("limiter", 0.3) * 100))
        self._set_label_with_desc(
            self.music_limiter_label, "Peak limiter", "compress peaks", self.music_limiter_slider.value()
        )
        self.music_gate_checkbox.blockSignals(True)
        self.music_gate_checkbox.setChecked(bool(preset.get("gate_enabled", False)))
        self.music_gate_checkbox.blockSignals(False)
        set_value(self.music_gate_slider, int(preset.get("gate_threshold", 0.12) * 100))
        self._set_label_with_desc(
            self.music_gate_label, "Gate sensitivity", "kick-only bias", self.music_gate_slider.value()
        )
        self.music_strobe_checkbox.blockSignals(True)
        self.music_strobe_checkbox.setChecked(bool(preset.get("strobe_enabled", False)))
        self.music_strobe_checkbox.blockSignals(False)
        set_value(self.music_strobe_strength_slider, int(preset.get("strobe_strength", 0.5) * 100))
        set_value(self.music_strobe_thresh_slider, int(preset.get("strobe_threshold", 0.25) * 100))
        self._set_label_with_desc(
            self.music_strobe_strength_label,
            "Strobe strength",
            "flash level",
            self.music_strobe_strength_slider.value(),
        )
        self._set_label_with_desc(
            self.music_strobe_thresh_label,
            "Strobe sensitivity",
            "trigger",
            self.music_strobe_thresh_slider.value(),
        )
        self._set_label_with_desc(
            self.music_gate_label, "Gate sensitivity", "kick-only bias", self.music_gate_slider.value()
        )
        self._set_label_with_desc(
            self.music_strobe_strength_label, "Strobe strength", "flash level", self.music_strobe_strength_slider.value()
        )
        self._set_label_with_desc(
            self.music_strobe_thresh_label, "Strobe sensitivity", "trigger", self.music_strobe_thresh_slider.value()
        )
        palette_name = preset.get("palette", "Default")
        self.music_palette_combo.blockSignals(True)
        idx = self.music_palette_combo.findText(palette_name)
        if idx >= 0:
            self.music_palette_combo.setCurrentIndex(idx)
        self.music_palette_combo.blockSignals(False)

        color_mode = preset.get("color_mode", "gradient")
        self.music_color_mode.blockSignals(True)
        self.music_color_mode.setCurrentIndex(0 if color_mode == "gradient" else 1)
        self.music_color_mode.blockSignals(False)
        self.music_vertical_checkbox.blockSignals(True)
        self.music_vertical_checkbox.setChecked(bool(preset.get("vertical_pulse", False)))
        self.music_vertical_checkbox.blockSignals(False)
        self.music_keep_lit_checkbox.blockSignals(True)
        self.music_keep_lit_checkbox.setChecked(bool(preset.get("keep_lit", False)))
        self.music_keep_lit_checkbox.blockSignals(False)
        self.music_ambient_checkbox.blockSignals(True)
        self.music_ambient_checkbox.setChecked(bool(preset.get("ambient_fill", False)))
        self.music_ambient_checkbox.blockSignals(False)

        # Step 2: apply everything once, now that all the widgets hold the new values.
        self._apply_music_audio_params()
        self._apply_music_smoothing()
        self._on_music_smoothing_changed(self.music_smoothing_slider.value())
        self._apply_music_transient()
        self._on_music_transient_changed(self.music_transient_slider.value())
        self._apply_music_bands()
        self._on_music_band_changed(0)
        self._apply_music_limiter()
        self._on_music_limiter_changed(self.music_limiter_slider.value())
        self._apply_music_beat_gate()
        self._on_music_gate_changed(self.music_gate_slider.value())
        self._apply_music_strobe()
        self._on_music_strobe_changed(self.music_strobe_strength_slider.value())
        self._on_music_strobe_thresh_changed(self.music_strobe_thresh_slider.value())
        self._apply_music_palette()
        self._apply_music_ambient_fill()
        self._apply_music_vertical_pulse()
        self._apply_music_melody_focus()
        self._apply_music_keep_lit()
        self._apply_music_color_mode()
        if self.music_fast_checkbox.isChecked():
            # Re-apply fast mode on top of the preset, snapshotting the preset values.
            self._fast_snapshot = {}
            self._apply_music_fast_mode()
        self._set_status(f"Preset applied: {name}")

    @staticmethod
    def _set_label_with_desc(label: QtWidgets.QLabel, title: str, desc: str, value: int) -> None:
        # Rich text lets the bracketed hint be smaller and greyer than the title.
        label.setTextFormat(QtCore.Qt.RichText)
        label.setText(
            f"{title} <span style='font-size:11px; color:#cfd3dc;'>({desc})</span>: {value}%"
        )

    def _update_music_preview(self, payload: tuple) -> None:
        """Show the colour currently being sent to the lamp (connected to UiSignals.music_color)."""
        try:
            (r, g, b), brightness = payload
        except Exception:
            return
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        brightness = max(0, min(100, int(brightness)))
        self.music_color_preview.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px;"
        )
        self.music_color_label.setText(f"Live: #{r:02X}{g:02X}{b:02X} @ {brightness}%")

    def _toggle_music(self, enabled: bool) -> None:
        if enabled:
            # Music mode and screen sync both control the colour, so only one may run at a time.
            if self.screen_sync_toggle.isChecked():
                self.screen_sync_toggle.setChecked(False)
            controller = self.manager.controller()
            # Can't react without a lamp: flip the button back and tell the user why.
            if controller is None:
                self.music_toggle.setChecked(False)
                self._set_status("Connect a device before enabling reactive mode.")
                return
            self.music_toggle.setText("Disable Audio Reactive")
            self._run_async(self.music_worker.start(), "Music reactive enabled")
        else:
            self.music_toggle.setText("Enable Audio Reactive")
            self._run_async(self.music_worker.stop(), "Music reactive disabled")

    def _toggle_screen_sync(self, enabled: bool) -> None:
        if enabled:
            if self.manager.controller() is None:
                self.screen_sync_toggle.setChecked(False)
                self._set_status("Connect a device before enabling screen color sync.")
                return
            if self.music_toggle.isChecked():
                self.music_toggle.setChecked(False)
            self.screen_sync_toggle.setText("Disable Screen Color Sync")
            self.screen_sampler.start()
            self._set_status("Screen color sync enabled")
            try:
                self.executor.submit(self.manager.set_brightness(80))
            except Exception:
                pass
        else:
            self.screen_sync_toggle.setText("Enable Screen Color Sync")
            self.screen_sampler.stop()
            self._set_status("Screen color sync disabled")

    # Small handlers that forward dropdown/slider changes to the screen sampler.
    def _on_screen_mode_changed(self, idx: int) -> None:
        mode = "dominant" if idx == 0 else "standout"
        self.screen_sampler.set_mode(mode)

    def _on_screen_monitor_changed(self, idx: int) -> None:
        self.screen_sampler.set_monitor(idx if idx >= 0 else None)

    def _on_screen_region_changed(self, idx: int) -> None:
        regions = ["full", "left", "right", "top", "bottom"]
        if 0 <= idx < len(regions):
            self.screen_sampler.set_region(regions[idx])

    def _on_screen_smoothing_changed(self, value: int) -> None:
        # value 0-90 mapped to smoothing 0.0-0.9
        self.screen_sampler.set_smoothing(value / 100.0)

    def _on_screen_edge_changed(self, value: int) -> None:
        self.screen_sampler.set_edge_emphasis(value / 100.0)

    def _run_async(self, coro, success_message: str | None = None) -> None:
        """Run a coroutine on the asyncio thread and report the outcome in the status bar.

        Used by nearly every button handler so errors show up in the UI instead of vanishing.
        """
        future = self.executor.submit(coro)

        def _complete(fut) -> None:
            try:
                fut.result()
            except Exception as exc:
                self.signals.status_message.emit(str(exc))
            else:
                if success_message:
                    self.signals.status_message.emit(success_message)

        future.add_done_callback(_complete)

    def _set_status(self, text: str) -> None:
        # Message disappears after 5 seconds.
        self.status_bar.showMessage(text, 5000)

    def _update_connection_label(self, text: str) -> None:
        self.connection_label.setText(text)
        # If the lamp disconnected, turn the music toggle off so the UI matches reality.
        if "Not connected" in text:
            self.music_toggle.setChecked(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Clean shutdown: stop audio, drop the BLE connection, stop the asyncio thread.
        self.executor.submit(self.music_worker.stop())
        self.executor.submit(self.manager.disconnect())
        self.executor.stop()
        event.accept()

    def _silent_power_off(self) -> None:
        self._run_async(self.manager.power_off(), None)
