"""Central configuration knobs for SOSLamp."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CONFIG_DIR = Path.home() / ".soslamp"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MELK_PORT = 48899
DISCOVERY_BROADCAST = "HF-A11ASSISTHREAD"
DISCOVERY_TIMEOUT = 4.0
COMMAND_THROTTLE = 0.01  # seconds between LED pushes
MELK_TRANSPORT = os.getenv("LOTUSLAMP_MELK_TRANSPORT", "udp").lower()
MELK_TCP_TIMEOUT = float(os.getenv("LOTUSLAMP_MELK_TCP_TIMEOUT", "2.0"))
MELK_TCP_SUFFIX = os.getenv("LOTUSLAMP_MELK_TCP_SUFFIX", "")

AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = int(os.getenv("LOTUSLAMP_AUDIO_BUFFER_SIZE", "256"))
AUDIO_DEVICE = None  # let sounddevice decide
AUDIO_GAIN = 1.6
AUDIO_VALUE_BOOST = float(os.getenv("LOTUSLAMP_AUDIO_VALUE_BOOST", "3.0"))
AUDIO_VALUE_FLOOR = float(os.getenv("LOTUSLAMP_AUDIO_VALUE_FLOOR", "0.6"))
AUDIO_SILENCE_GATE = float(os.getenv("LOTUSLAMP_AUDIO_SILENCE_GATE", "0.001"))
MUSIC_BASE_BRIGHTNESS = int(os.getenv("LOTUSLAMP_MUSIC_BASE_BRIGHTNESS", "100"))
MUSIC_MAX_BRIGHTNESS = int(os.getenv("LOTUSLAMP_MUSIC_MAX_BRIGHTNESS", "100"))


@dataclass(slots=True)
class ColorTheme:
    background: str = "#121212"
    panel: str = "#1f1f1f"
    accent: str = "#ff6ec4"
    accent_alt: str = "#567dff"


COLOR_THEME = ColorTheme()
