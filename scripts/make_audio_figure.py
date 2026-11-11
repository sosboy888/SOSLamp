"""Regenerate docs/images/audio_pipeline.png.

Feeds a synthetic signal (kick + hi-hat) through the real ReactiveAudioEngine and
MusicReactiveWorker and plots what would be sent to the strip.

Needs matplotlib (dev-only, not in requirements.txt):
    pip install matplotlib
    python scripts/make_audio_figure.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from soslamp.audio.reactive_engine import ReactiveAudioEngine, MusicReactiveWorker

sr, buf = 44100, 256
dur = 6.0
t = np.arange(int(sr * dur)) / sr
rng = np.random.default_rng(0)
sig = 0.02 * rng.standard_normal(t.size)
# kick drum: 60 Hz decaying burst every 0.5 s
for k in np.arange(0.2, dur, 0.5):
    m = t >= k
    sig[m] += 0.8 * np.exp(-(t[m] - k) * 18) * np.sin(2 * np.pi * 60 * (t[m] - k))
# hi-hat: short noise bursts on the off-beats
for k in np.arange(0.45, dur, 0.5):
    m = t >= k
    sig[m] += 0.35 * np.exp(-(t[m] - k) * 60) * rng.standard_normal(m.sum())

eng = ReactiveAudioEngine(leds=60)
eng._samplerate = sr
worker = MusicReactiveWorker(lambda: None)
worker.engine = eng  # share engine state so smoothing/envelopes are the real ones
worker.configure_brightness(20, 100, True)  # GUI sliders: base 20%, max 100%, scale with energy
cols, bright, times = [], [], []
for i in range(0, sig.size - buf, buf):
    colors = eng._samples_to_colors(sig[i:i + buf].astype(np.float32))
    rgb, b = worker._collapse(colors)
    cols.append(np.array(rgb) / 255.0 * (b / 100.0))
    bright.append(b); times.append(i / sr)
cols = np.clip(np.array(cols), 0, 1)

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 0.8]}, facecolor="#121212")
for a in ax:
    a.set_facecolor("#1f1f1f"); a.tick_params(colors="#cfd3dc"); [s.set_color("#444") for s in a.spines.values()]
    a.yaxis.label.set_color("#cfd3dc"); a.xaxis.label.set_color("#cfd3dc"); a.title.set_color("#ffffff")
ax[0].plot(t, sig, color="#567dff", lw=0.4)
ax[0].set_ylabel("amplitude"); ax[0].set_title("Synthetic test signal: 60 Hz kick on the beat, noise hi-hat on the off-beat", loc="left", fontsize=10)
ax[1].plot(times, bright, color="#ff6ec4", lw=1.2); ax[1].set_ylabel("brightness %"); ax[1].set_ylim(-5, 105)
ax[1].set_title("Brightness sent to the strip", loc="left", fontsize=10)
ax[2].imshow(cols[None, :, :], aspect="auto", extent=[0, dur, 0, 1]); ax[2].set_yticks([]); ax[2].set_xlabel("time (s)")
ax[2].set_title("Color sent to the strip (brightness applied)", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(str(ROOT / "docs" / "images" / "audio_pipeline.png"), dpi=130, facecolor=fig.get_facecolor())
print("ok", len(cols), min(bright), max(bright))
