# SOSLamp

A Windows desktop app for controlling **MELK / ELK-BLE / LEDBLE Bluetooth LED strips**, with a dark-themed PySide6 GUI, effect presets, and real-time **music-reactive** and **screen-color-sync** lighting.

The BLE protocol (packet layout, UUIDs, effect codes) is ported from the Rust project [b1scoito/elk-led-controller](https://github.com/b1scoito/elk-led-controller). Everything on top of it is new: the GUI, the audio analysis engine, auto-connect, and screen sync.

## Features

- **Device discovery and auto-connect** – continuous BLE scan, connects to the first compatible strip it sees
- **Basic control** – power, RGB color picker, brightness, warm/cool white balance
- **Built-in effects** – jump / crossfade / blink presets, speed slider, and a raw effect-code sender for experimenting
- **Music-reactive mode** – FFT-based analysis of your microphone *or* system audio (WASAPI loopback) drives color and brightness
  - bass / mid / high band weighting, beat detection, strobe, soft limiter, auto-gain
  - 14 one-click presets (e.g. *EDM Rave*, *Chill Vocal*, *Drum & Bass*) and 9 color palettes
  - optional "drop" detection, melody focus, and a low-latency *Fast mode*
- **Screen color sync** – Ambilight-style: samples a monitor (or half of one) and mirrors its dominant/most vibrant color to the strip
- **Windows packaging** – PyInstaller script to build a standalone `.exe`

## Tech stack

| Area | Tools |
| --- | --- |
| GUI | PySide6 (Qt 6) |
| Bluetooth LE | [`bleak`](https://github.com/hbldh/bleak) |
| Audio capture + DSP | `sounddevice`, `numpy` (FFT, spectral flux, adaptive beat tracker) |
| Concurrency | `asyncio` on a background thread, bridged to Qt via signals |
| Packaging | PyInstaller |

## Getting started

**Requirements:** Windows 10/11, Python 3.11+, a Bluetooth LE adapter, and a MELK/ELK-style LED strip.

```powershell
git clone <this-repo-url>
cd soslamp

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run from the src folder so the `soslamp` package is importable
cd src
python -m soslamp.app
```

### Tips

- Power the strip on and keep it within range. If the app can't find it, toggle Bluetooth off/on in Windows settings to reset the adapter.
- Allow desktop apps to use the microphone (Windows Settings → Privacy → Microphone) for music mode.
- In the **Audio Source** dropdown, pick a microphone, or an entry marked *(Loopback)* to react to whatever your PC is playing.
- Music mode and screen sync both control the strip's color, so only one can run at a time.

## How it works

```
                       ┌──────────────────────────────┐
  Qt GUI thread        │  SOSLampWindow (gui/)        │
  (widgets, signals)   └──────┬───────────────▲───────┘
                              │ submit()      │ Qt signals
                       ┌──────▼───────────────┴───────┐
  asyncio thread       │  BleControllerManager        │  scan, auto-connect
  (utils/async_bridge) │  BleLedController            │  9-byte packets over GATT
                       └──────▲───────────────────────┘
                              │ set_color / set_brightness
                       ┌──────┴───────────────────────┐
  audio worker thread  │  MusicReactiveWorker         │
                       │  ReactiveAudioEngine         │  capture → FFT → bands → color
                       └──────────────────────────────┘
```

Qt and `bleak` each want their own event loop, so `AsyncExecutor` runs an asyncio loop on a background thread. The GUI hands it coroutines and gets results back through Qt signals, which keeps the window responsive during BLE writes.

### Audio-reactive pipeline

1. `sounddevice` captures small buffers (default 256 samples).
2. A Hann-windowed FFT splits the signal into six frequency bands.
3. Spectral flux (frame-to-frame increase in energy) feeds an adaptive per-band beat tracker.
4. Bands, beats, palette and user settings become a list of colors.
5. Because these strips take **one color + one brightness**, the worker averages the list down and smooths it before writing it over BLE.

To reduce latency, lower the buffer size in the GUI (or `LOTUSLAMP_AUDIO_BUFFER_SIZE`), or enable **Fast mode**.

### BLE protocol summary

Every command is a 9-byte packet starting with `0x7E` and ending with `0xEF`:

| Command | Packet |
| --- | --- |
| Power | `7E 00 04 …` (model-specific on/off bytes) |
| RGB color | `7E 00 05 03 R G B 00 EF` |
| White balance | `7E 00 05 02 warm cold 00 00 EF` |
| Brightness | `7E 00 01 <0-100> 00 00 00 00 EF` |
| Effect | `7E 00 03 <code> <speed> 00 00 00 EF` |
| Schedule | `7E 00 82 <hour> <min> 00 <0 on / 1 off> <days \| 0x80 if enabled> EF` |

See [docs/MELK_PROTOCOL.md](docs/MELK_PROTOCOL.md) for the (experimental) Wi-Fi/UDP variant, and `src/soslamp/integrations/ble_led_device.py` for the BLE implementation, device UUIDs and effect table.

## Project structure

```
src/soslamp/
├── app.py                  # entry point
├── gui/main_window.py      # Qt window, panels, presets, screen sampler
├── audio/
│   ├── reactive_engine.py  # FFT / beat detection / color mapping
│   └── device_utils.py     # list microphones and loopback devices
├── integrations/
│   ├── ble_led_device.py   # BLE packets, retries, effect + schedule codes
│   ├── bluetooth_service.py# extra discovery (OS-reported connected devices)
│   └── melk_device.py      # experimental Wi-Fi (UDP/TCP) driver
├── services/
│   ├── ble_manager.py      # scanning, auto-connect, "ensure connected"
│   └── device_registry.py  # multi-device registry for the Wi-Fi driver
├── tools/                  # pcap_compare.py, send_test.py diagnostics
└── utils/                  # asyncio bridge, config / env-var settings
docs/                       # protocol notes and build guide
scripts/build_exe.ps1       # PyInstaller wrapper
```

## Configuration

Defaults live in `src/soslamp/utils/config.py`; most can be overridden with environment variables, for example:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOTUSLAMP_AUDIO_BUFFER_SIZE` | `256` | samples per audio frame (lower = snappier) |
| `LOTUSLAMP_AUDIO_VALUE_BOOST` | `3.0` | brightness boost per band |
| `LOTUSLAMP_AUDIO_SILENCE_GATE` | `0.001` | RMS below this counts as silence |
| `LOTUSLAMP_MUSIC_MAX_BRIGHTNESS` | `100` | brightness cap in music mode |

## Building a Windows executable

```powershell
pip install pyinstaller
.\scripts\build_exe.ps1
```

The app is written to `dist/SOSLamp/`. Details in [docs/BUILD_EXE.md](docs/BUILD_EXE.md).

## Known limitations and roadmap

- **No automated tests yet.** The BLE and audio paths need real hardware, so they are currently checked manually.
- **Schedules** (timed on/off stored on the strip) are implemented in the BLE layer but have no GUI panel yet.
- The Wi-Fi/UDP driver is experimental; Bluetooth is the main supported path.
- Loopback capture relies on WASAPI, so system-audio mode is Windows only.
- Planned: per-effect tuning, saved presets per room, packaged installer with auto-update.

## Project history

SOSLamp was built over 2025 and early 2026 as a personal project, but I didn't use version control while developing it. This repository was created in November 2026 from the finished code, so the commit history starts then and doesn't reflect when the work was done. The comments, README and cleanup were added at that point.

## Credits

BLE protocol and effect codes: [elk-led-controller](https://github.com/b1scoito/elk-led-controller) by b1scoito.

## License

[MIT](LICENSE)
