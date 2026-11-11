# Building a Windows .exe

These steps produce a redistributable `SOSLamp.exe` using PyInstaller. They assume Windows 10/11, Python 3.11+ (64‑bit), and that Bluetooth + microphone permissions are allowed for Python.

## Quick build

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
.\scripts\build_exe.ps1
```

Artifacts land in `dist/SOSLamp/`:

- `dist/SOSLamp/SOSLamp.exe` – launch the app
- `dist/SOSLamp/` – include this whole folder if you copy to another PC (Qt plugins live here)

## What the script does

`scripts/build_exe.ps1` wraps PyInstaller with flags tuned for this project:

- `--windowed` keeps the console hidden (GUI app)
- `--paths src` makes the `soslamp` package importable
- `--collect-all PySide6` / `sounddevice` ensure Qt plugins and PortAudio backends are bundled
- `--hidden-import sounddevice` avoids lazy-import misses on some Python builds
- `--clean --noconfirm` start from a clean build each run

## Optional tweaks

- Want a single-file exe? Add `--onefile` in the script (launch time is slower; Qt plugins extract to temp).
- Shipping with a custom icon? Append `--icon path\\to\\icon.ico`.
- If you hit WASAPI device capture errors on first run, check Windows Privacy settings → Microphone → allow desktop apps, then retry.
