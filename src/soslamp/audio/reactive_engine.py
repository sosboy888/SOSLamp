"""Audio reactive RGB engine."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable, Sequence

import numpy as np
import sounddevice as sd

from soslamp.utils.config import (
    AUDIO_BUFFER_SIZE,
    AUDIO_DEVICE,
    AUDIO_GAIN,
    AUDIO_SILENCE_GATE,
    AUDIO_SAMPLE_RATE,
    AUDIO_VALUE_BOOST,
    AUDIO_VALUE_FLOOR,
    MUSIC_BASE_BRIGHTNESS,
    MUSIC_MAX_BRIGHTNESS,
)

logger = logging.getLogger(__name__)

Color = tuple[int, int, int]


class BeatTracker:
    """Simple per-band onset detector with adaptive thresholds and refractory."""

    def __init__(
        self,
        bands: int = 3,
        alpha: float = 0.08,
        thresh_k: float = 1.8,
        refractory_frames: int = 4,
        min_flux: float = 1e-4,
    ) -> None:
        self.bands = bands
        self.alpha = alpha
        self.thresh_k = thresh_k
        self.refractory_frames = max(1, refractory_frames)
        self.min_flux = min_flux
        self._mean = np.zeros(self.bands, dtype=np.float32)
        self._mean_sq = np.zeros(self.bands, dtype=np.float32)
        self._last_hit_frame = np.full(self.bands, -1e9, dtype=np.int64)

    def set_refractory_frames(self, frames: int) -> None:
        self.refractory_frames = max(1, int(frames))

    def update(self, flux: np.ndarray, frame_idx: int) -> tuple[bool, np.ndarray, float]:
        if flux.ndim != 1:
            flux = flux.reshape(-1)
        if flux.size != self.bands:
            self.bands = int(flux.size)
            self._mean = np.zeros(self.bands, dtype=np.float32)
            self._mean_sq = np.zeros(self.bands, dtype=np.float32)
            self._last_hit_frame = np.full(self.bands, -1e9, dtype=np.int64)
        flux = np.maximum(flux.astype(np.float32), 0.0)
        a = self.alpha
        self._mean = (1 - a) * self._mean + a * flux
        self._mean_sq = (1 - a) * self._mean_sq + a * flux * flux
        var = np.maximum(self._mean_sq - self._mean * self._mean, 0.0)
        std = np.sqrt(var + 1e-9)
        thresh = self._mean + self.thresh_k * std
        hits = flux > np.maximum(thresh, self.min_flux)
        recent = (frame_idx - self._last_hit_frame) < self.refractory_frames
        hits = np.logical_and(hits, np.logical_not(recent))
        if np.any(hits):
            self._last_hit_frame = np.where(hits, frame_idx, self._last_hit_frame)
        score = np.where(hits, (flux - thresh) / (std + 1e-6), 0.0)
        strength = float(np.clip(np.max(score) if score.size else 0.0, 0.0, 4.0))
        return bool(np.any(hits)), hits, strength


class ReactiveAudioEngine:
    def __init__(self, leds: int = 60) -> None:
        self.leds = leds
        self._queue: asyncio.Queue[list[Color]] = asyncio.Queue(maxsize=1)
        self._task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._callbacks: deque[Callable[[Sequence[Color]], Awaitable[None]]] = deque()
        self.device = AUDIO_DEVICE
        self.loopback = False
        self.channels = 1
        self.buffer_size = AUDIO_BUFFER_SIZE
        self.gain = AUDIO_GAIN
        self.value_floor = AUDIO_VALUE_FLOOR
        self.value_boost = AUDIO_VALUE_BOOST
        self.silence_gate = AUDIO_SILENCE_GATE
        self.color_mode: str = "gradient"  # "gradient" or "dominant"
        self._noise_floor = AUDIO_SILENCE_GATE
        self._frame_idx = 0
        self._error_handler: Callable[[str], None] | None = None
        self._channel_candidates: list[int] = [1, 2, 4, 6, 8]
        self._sample_rate_candidates: list[float] = [AUDIO_SAMPLE_RATE]
        self._forced_channels: int | None = None
        self._stop_sentinel: object = object()
        self._prev_spectrum: np.ndarray | None = None
        self._flux_smoothed: float = 0.0
        self._flux_bass_smoothed: float = 0.0
        self._transient_gain: float = 12.0  # baseline flux scaling
        self._transient_ceiling: float = 1.5  # max additive bump
        self._transient_user_gain: float = 1.0  # user-controlled scalar
        self._band_weights: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._limit_amount: float = 0.0
        self._beat_gate_enabled: bool = False
        self._beat_gate_threshold: float = 0.12
        self._palette_base = np.array([0.00, 0.12, 0.28, 0.48, 0.66, 0.88], dtype=np.float32)
        self._palette = self._palette_base.copy()
        self._strobe_enabled: bool = False
        self._strobe_strength: float = 0.0
        self._strobe_threshold: float = 0.25
        self._strobe_level: float = 0.0
        self._strobe_decay: float = 0.6
        self._flash_enabled: bool = False
        self._flash_phase: int = 0
        self._flash_rate_frames: int = 3
        self._flicker_enabled: bool = False
        self._flicker_amount: float = 0.0
        self._last_nonzero_rgb: np.ndarray | None = None
        self._ambient_fill = False
        self._ambient_hue = 0.0
        self._melody_enabled: bool = False
        self._melody_amount: float = 0.25
        self._melody_conf: float = 0.0
        self._band_env = np.zeros(6, dtype=np.float32)
        self._band_env_decay = 0.84
        self._band_env_attack = 0.6
        self._auto_gain_enabled: bool = False
        self._auto_gain_target: float = 0.18
        self._auto_gain_state: float = 1.0
        self._auto_gain_speed: float = 0.04
        self._centroid_hue_enabled: bool = False
        self._centroid_hue_amount: float = 0.15
        self._drop_enabled: bool = False
        self._drop_ratio: float = 0.35
        self._drop_min_frames: int = 10
        self._drop_cooldown_frames: int = 60
        self._drop_decay: float = 0.98
        self._drop_slow: float = 0.0
        self._drop_fast: float = 0.0
        self._drop_cooldown: int = 0
        self._drop_active_frames: int = 0
        self._drop_callback: Callable[[], None] | None = None
        self._flux_peak: float = 0.0
        self._vertical_fill_enabled: bool = False
        self._vertical_fill: float = 0.0
        self._samplerate: float = AUDIO_SAMPLE_RATE
        self._auto_gain_enabled: bool = False
        self._auto_gain_target: float = 0.18
        self._auto_gain_state: float = 1.0
        self._auto_gain_speed: float = 0.04
        self._centroid_hue_enabled: bool = False
        self._centroid_hue_amount: float = 0.15
        self._beat_tracker = BeatTracker(bands=3)

    def register_callback(self, callback: Callable[[Sequence[Color]], Awaitable[None]]) -> None:
        self._callbacks.append(callback)
        # Callback wrapper to allow simple broadcast messages.
        self._notify_drop = lambda _: None

    def configure_source(self, device: int | None, loopback: bool) -> None:
        self.device = device
        self.loopback = loopback
        self._rebuild_candidates(device, loopback)

    def _rebuild_candidates(self, device: int | None, loopback: bool) -> None:
        try:
            info = sd.query_devices(device)
            max_in = int(info.get("max_input_channels", 0) or 0)
            max_out = int(info.get("max_output_channels", 0) or 0)
            chosen = max_in if max_in > 0 else max_out
            if loopback and chosen == 0:
                chosen = max_out
            chosen = max(1, chosen)
            self.channels = self._forced_channels or min(2, chosen)
            candidates = [self.channels, max_in, max_out, 8, 6, 4, 2, 1]
            self._channel_candidates = []
            for c in candidates:
                if c and c not in self._channel_candidates:
                    self._channel_candidates.append(int(c))
            default_sr = float(info.get("default_samplerate") or AUDIO_SAMPLE_RATE)
            self._sample_rate_candidates = []
            # Prefer device-reported default first; fall back to configured sample rate.
            for sr in (default_sr, AUDIO_SAMPLE_RATE):
                if sr and sr not in self._sample_rate_candidates:
                    self._sample_rate_candidates.append(float(sr))
            dev_name = info.get("name", "?")
            logger.info(
                "Configured audio device=%s (%s) loopback=%s max_in=%s max_out=%s channels=%s candidates=%s sample_rates=%s forced=%s",
                device,
                dev_name,
                loopback,
                max_in,
                max_out,
                self.channels,
                self._channel_candidates,
                self._sample_rate_candidates,
                self._forced_channels,
            )
        except Exception:
            self.channels = self._forced_channels or 1
            self._channel_candidates = [self.channels]
            self._sample_rate_candidates = [AUDIO_SAMPLE_RATE]
            logger.info(
                "Configured audio device=%s loopback=%s channels=%s (fallback, forced=%s)",
                device,
                loopback,
                self.channels,
                self._forced_channels,
            )

    def _find_loopback_device(self) -> int | None:
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            return None
        # Prefer WASAPI render devices with max_output > 0.
        candidates: list[tuple[int, str, int]] = []
        for idx, dev in enumerate(devices):
            host_idx = dev.get("hostapi", 0)
            host_name = hostapis[host_idx]["name"] if 0 <= host_idx < len(hostapis) else ""
            max_out = int(dev.get("max_output_channels", 0) or 0)
            name = dev.get("name", "")
            if max_out > 0 and "wasapi" in host_name.lower():
                score = 0
                if self.device is not None:
                    try:
                        cur = sd.query_devices(self.device).get("name", "")
                        if cur and cur in name:
                            score -= 1
                    except Exception:
                        pass
                candidates.append((score, idx, name))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[0][1]
        logger.info("Loopback fallback selecting device index %s (%s)", chosen, candidates[0][2])
        return chosen

    def configure_params(
        self,
        buffer_size: int | None = None,
        gain: float | None = None,
        value_floor: float | None = None,
        value_boost: float | None = None,
        silence_gate: float | None = None,
        forced_channels: int | None = None,
    ) -> None:
        if buffer_size is not None:
            self.buffer_size = max(64, min(4096, int(buffer_size)))
        if gain is not None:
            self.gain = float(max(0.1, min(10.0, gain)))
        if value_floor is not None:
            self.value_floor = float(np.clip(value_floor, 0.0, 1.0))
        if value_boost is not None:
            self.value_boost = float(max(0.1, min(10.0, value_boost)))
        if silence_gate is not None:
            self.silence_gate = float(max(0.0, min(0.1, silence_gate)))
            self._noise_floor = self.silence_gate
        if forced_channels is not None:
            fc = int(max(1, min(8, forced_channels)))
            self._forced_channels = fc
            self.channels = fc
            self._channel_candidates = [fc]
            self._sample_rate_candidates = [AUDIO_SAMPLE_RATE]
            self._rebuild_candidates(self.device, self.loopback)
        self._update_beat_tracker_refractory()

    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode

    def set_error_handler(self, handler: Callable[[str], None] | None) -> None:
        self._error_handler = handler

    def set_transient_boost(self, user_gain: float, ceiling: float | None = None) -> None:
        self._transient_user_gain = float(np.clip(user_gain, 0.0, 3.0))
        if ceiling is not None:
            self._transient_ceiling = float(np.clip(ceiling, 0.0, 3.0))

    def set_drop_notifier(self, notifier: Callable[[str], None] | None) -> None:
        self._notify_drop = notifier

    def set_vertical_fill(self, enabled: bool) -> None:
        self._vertical_fill_enabled = bool(enabled)

    def set_melody_focus(self, enabled: bool, amount: float | None = None) -> None:
        self._melody_enabled = bool(enabled)
        if amount is not None:
            self._melody_amount = float(np.clip(amount, 0.0, 1.0))

    def set_band_weights(self, bass: float, mid: float, high: float) -> None:
        weights = np.array([bass, mid, high], dtype=np.float32)
        weights = np.clip(weights, 0.1, 3.0)
        self._band_weights = weights

    def set_soft_limiter(self, amount: float) -> None:
        # amount in [0,1]; 0 = off, 1 = strongest
        self._limit_amount = float(np.clip(amount, 0.0, 1.0))

    def set_beat_gate(self, enabled: bool, threshold: float | None = None) -> None:
        self._beat_gate_enabled = bool(enabled)
        if threshold is not None:
            self._beat_gate_threshold = float(np.clip(threshold, 0.01, 0.5))

    def _update_beat_tracker_refractory(self) -> None:
        frames_per_second = self._samplerate / max(self.buffer_size, 1)
        refractory = int(np.clip(0.09 * frames_per_second, 2, 24))
        self._beat_tracker.set_refractory_frames(refractory)

    def set_palette(self, hues: Sequence[float]) -> None:
        try:
            arr = np.array(list(hues), dtype=np.float32)
            if arr.size >= 3:
                self._palette_base = np.mod(arr, 1.0)
                self._palette = self._palette_base.copy()
        except Exception:
            pass

    def set_strobe(self, enabled: bool, strength: float, threshold: float | None = None) -> None:
        self._strobe_enabled = bool(enabled)
        self._strobe_strength = float(np.clip(strength, 0.0, 1.0))
        if threshold is not None:
            self._strobe_threshold = float(np.clip(threshold, 0.01, 1.0))

    def set_flash(self, enabled: bool, rate_frames: int | None = None) -> None:
        self._flash_enabled = bool(enabled)
        if rate_frames is not None:
            self._flash_rate_frames = int(max(1, min(20, rate_frames)))

    def set_auto_gain(self, enabled: bool, target: float | None = None) -> None:
        self._auto_gain_enabled = bool(enabled)
        if target is not None:
            self._auto_gain_target = float(np.clip(target, 0.05, 0.6))

    def set_centroid_hue(self, enabled: bool, amount: float | None = None) -> None:
        self._centroid_hue_enabled = bool(enabled)
        if amount is not None:
            self._centroid_hue_amount = float(np.clip(amount, 0.0, 0.5))

    def set_drop_detector(
        self,
        enabled: bool,
        ratio: float | None = None,
        min_frames: int | None = None,
        cooldown_frames: int | None = None,
        callback: Callable[[], None] | None = None,
    ) -> None:
        self._drop_enabled = bool(enabled)
        if ratio is not None:
            self._drop_ratio = float(np.clip(ratio, 0.1, 0.9))
        if min_frames is not None:
            self._drop_min_frames = int(max(1, min(200, min_frames)))
        if cooldown_frames is not None:
            self._drop_cooldown_frames = int(max(1, min(500, cooldown_frames)))
        self._drop_callback = callback

    async def start(self) -> None:
        if self._task:
            return
        self._drain_queue()
        self._stop.clear()
        self._frame_idx = 0
        self._noise_floor = self.silence_gate
        self._prev_spectrum = None
        self._flux_smoothed = 0.0
        self._flux_bass_smoothed = 0.0
        self._flux_peak = 0.0
        self._melody_conf = 0.0
        self._vertical_fill = 0.0
        loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._capture_loop(loop))
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        running = bool(self._task or self._dispatch_task)
        self._stop.set()
        if running:
            try:
                self._queue.put_nowait(self._stop_sentinel)
            except asyncio.QueueFull:
                # Clear oldest frame to make room for the sentinel.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(self._stop_sentinel)
                except Exception:
                    pass
        if self._task:
            await self._task
        if self._dispatch_task:
            await self._dispatch_task
        self._task = None
        self._dispatch_task = None

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            colors = await self._queue.get()
            if colors is self._stop_sentinel:
                break
            await asyncio.gather(*(cb(colors) for cb in self._callbacks), return_exceptions=True)

    async def _capture_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        await asyncio.to_thread(self._capture_blocking, loop)

    def _capture_blocking(self, loop: asyncio.AbstractEventLoop) -> None:
        def _extra_settings_options() -> list:
            if not self.loopback:
                return [None]
            opts = []
            try:
                opts.append(sd.WasapiSettings(loopback=True, exclusive=True))
                opts.append(sd.WasapiSettings(loopback=True, exclusive=False))
            except Exception:
                opts.append(None)
            return opts
        def _open_stream(channels: int, samplerate: float, extra):
            # For WASAPI loopback, PortAudio expects the *output* device index as the
            # input target (not a duplex tuple). Passing a tuple here caused it to
            # fall back to the default input device instead of the selected output.
            target_device = dev
            return sd.InputStream(
                channels=channels,
                samplerate=samplerate,
                blocksize=self.buffer_size,
                device=target_device,
                extra_settings=extra,
            )

        stream = None
        attempts = []
        devices_to_try = [self.device]
        if self.loopback:
            alt = self._find_loopback_device()
            if alt is not None and alt != self.device:
                devices_to_try.append(alt)
            devices_to_try.append(None)  # default output loopback

        for dev in devices_to_try:
            self._rebuild_candidates(dev, self.loopback)
            for sr in self._sample_rate_candidates:
                for ch in self._channel_candidates:
                    for extra in _extra_settings_options():
                        try:
                            attempts.append((dev, ch, sr, getattr(extra, "exclusive", None)))
                            stream = _open_stream(ch, sr, extra)
                            self.channels = ch
                            self._samplerate = float(sr)
                            self._update_beat_tracker_refractory()
                            if attempts[0] != (dev, ch, sr, getattr(extra, "exclusive", None)):
                                logger.info(
                                    "Opened audio device %s using fallback device=%s channels=%s samplerate=%s exclusive=%s",
                                    self.device,
                                    dev,
                                    ch,
                                    sr,
                                    getattr(extra, "exclusive", None),
                                )
                            break
                        except Exception as exc:
                            logger.error(
                                "Audio input stream failed for device %s (channels=%s samplerate=%s exclusive=%s): %s",
                                dev,
                                ch,
                                sr,
                                getattr(extra, "exclusive", None),
                                exc,
                            )
                            stream = None
                            continue
                    if stream:
                        break
                if stream:
                    break
            if stream:
                break
        if stream is None:
            msg = f"Audio input stream failed for device {self.device} (tried {attempts})"
            logger.error(msg)
            if self._error_handler:
                self._error_handler(msg)
            return
        with stream:
            while not self._stop.is_set():
                samples, _ = stream.read(self.buffer_size)
                np_samples = np.squeeze(samples).astype(np.float32)
                if np_samples.ndim > 1:
                    # Some devices present interleaved/stereo data; collapse to mono to keep FFT shapes aligned.
                    np_samples = np.mean(np_samples, axis=1)
                colors = self._samples_to_colors(np_samples)
                loop.call_soon_threadsafe(self._enqueue_colors, colors)

    def _enqueue_colors(self, colors: list[Color]) -> None:
        # Keep only the freshest audio frame to reduce visible latency.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(colors)
        except asyncio.QueueFull:
            pass

    def _samples_to_colors(self, samples: np.ndarray) -> list[Color]:
        if samples.size == 0:
            return [(255, 255, 255)] * self.leds
        samples = samples * np.hanning(samples.size)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
        if rms < max(self.silence_gate, self._noise_floor * 1.4):
            return [(0, 0, 0)] * self.leds
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)
        n = magnitude.shape[0]
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / max(self._samplerate, 1.0)) if samples.size else np.array([])

        def band_mean(start: int, end: int) -> float:
            s = min(end, n)
            if start >= s:
                return 0.0
            return float(magnitude[start:s].mean()) if s > start else 0.0

        bands = [
            band_mean(1, 15),    # sub bass
            band_mean(15, 35),   # bass
            band_mean(35, 80),   # low mids
            band_mean(80, 140),  # mids
            band_mean(140, 220), # highs
            band_mean(220, 320), # sparkle
        ]
        scale = float(np.max(magnitude)) if n else 0.0
        if scale <= 1e-9:
            # Silence: return black so downstream can fade out.
            return [(0, 0, 0)] * self.leds
        gain = self.gain
        if self._auto_gain_enabled and rms > 1e-6:
            target = self._auto_gain_target
            # Simple AGC: move state toward target/rms.
            desired = target / max(rms, 1e-6)
            self._auto_gain_state = (1 - self._auto_gain_speed) * self._auto_gain_state + self._auto_gain_speed * desired
            self._auto_gain_state = float(np.clip(self._auto_gain_state, 0.5, 6.0))
            gain *= self._auto_gain_state
        bands = np.clip(gain * np.array(bands) / scale, 0.0, 1.0)
        # Apply per-band peak envelope to keep short hits visible and reduce flicker.
        if self._band_env.shape[0] == bands.shape[0]:
            rising = bands > self._band_env
            self._band_env = np.where(
                rising,
                self._band_env_attack * bands + (1 - self._band_env_attack) * self._band_env,
                self._band_env_decay * self._band_env,
            )
            bands = np.clip(self._band_env, 0.0, 1.0)
        # Apply user band weights: group into bass/mid/high then normalize back.
        grouped = np.array(
            [
                bands[0:2].mean() if bands.size >= 2 else 0.0,
                bands[2:4].mean() if bands.size >= 4 else 0.0,
                bands[4:6].mean() if bands.size >= 6 else 0.0,
            ],
            dtype=np.float32,
        )
        grouped *= self._band_weights
        # Re-expand to per-band weights by mapping groups back.
        if bands.size >= 6:
            bands = np.array(
                [
                    grouped[0],
                    grouped[0],
                    grouped[1],
                    grouped[1],
                    grouped[2],
                    grouped[2],
                ],
                dtype=np.float32,
            )
        # Spectral flux (positive deltas) gives transient/beat emphasis.
        if self._prev_spectrum is None or self._prev_spectrum.shape != magnitude.shape:
            self._prev_spectrum = magnitude
        prev_spec = self._prev_spectrum
        delta = np.maximum(magnitude - prev_spec, 0.0)
        flux_raw = float(delta.mean())
        norm_delta = delta / (prev_spec + 1e-6)
        bass_end = min(60, n)
        bass_flux = float(norm_delta[1:bass_end].mean()) if n > 1 else 0.0
        self._prev_spectrum = magnitude
        # Faster smoothing keeps individual drum hits visible while still rejecting noise.
        self._flux_smoothed = 0.8 * self._flux_smoothed + 0.2 * flux_raw
        self._flux_bass_smoothed = 0.7 * self._flux_bass_smoothed + 0.3 * bass_flux
        flux_mix = 0.65 * self._flux_smoothed + 0.35 * self._flux_bass_smoothed
        if freqs.size == norm_delta.size and freqs.size > 0:
            band_masks = [
                (freqs >= 20.0) & (freqs < 150.0),      # kick
                (freqs >= 150.0) & (freqs < 1200.0),    # snare/mids
                (freqs >= 1200.0) & (freqs < 6000.0),   # hats/sparkle
            ]
            flux_bands = np.array(
                [float(norm_delta[m].mean()) if np.any(m) else 0.0 for m in band_masks],
                dtype=np.float32,
            )
        else:
            flux_bands = np.zeros(3, dtype=np.float32)
        beat_hit, band_hits, beat_strength = self._beat_tracker.update(flux_bands, self._frame_idx)
        if bands.size >= 2:
            bass_hit = np.clip(self._flux_bass_smoothed * 1.8, 0.0, 3.0)
            bands[0:2] *= 1.0 + 0.35 * bass_hit
        if bands.size >= 2 and band_hits.size >= 2:
            # Small extra emphasis on the bands that fired this frame.
            bands[0:2] *= 1.0 + 0.25 * float(band_hits[0])
            bands[2:4] *= 1.0 + 0.18 * float(band_hits[1])
        flux_scale = self._transient_gain * self._transient_user_gain
        transient_boost = 1.0 + np.clip(flux_mix * flux_scale, 0.0, self._transient_ceiling)
        if beat_hit:
            transient_boost = min(max(transient_boost, 1.0 + 0.6 * beat_strength), 1.0 + self._transient_ceiling)
        melody_weight = 0.0
        melody_bias = 0.0
        if self._melody_enabled and n > 4:
            mask = (freqs >= 180.0) & (freqs <= 2200.0)
            band = magnitude[mask]
            if band.size > 4:
                peak_idx = int(np.argmax(band))
                peak_val = float(band[peak_idx])
                mean_val = float(band.mean() + 1e-6)
                conf = peak_val / mean_val
                self._melody_conf = 0.85 * self._melody_conf + 0.15 * conf
                pitch_freq = float(freqs[mask][peak_idx])
                pitch_norm = pitch_freq / (0.5 * max(self._samplerate, 1.0) + 1e-9)
                melody_weight = float(np.clip(self._melody_conf * self._melody_amount, 0.0, 1.0))
                melody_bias = 0.35 * melody_weight * (pitch_norm - 0.5)
                if bands.size >= 4:
                    bands[2:] *= 1.0 + 0.35 * melody_weight
        if self._drop_enabled:
            # Track a decaying flux peak so we can require both energy and flux to fall.
            self._flux_peak = max(flux_mix, 0.92 * self._flux_peak + 0.08 * flux_mix)
            self._drop_slow = self._drop_decay * self._drop_slow + (1 - self._drop_decay) * rms
            self._drop_fast = 0.6 * self._drop_fast + 0.4 * rms
            energy_drop = self._drop_fast < self._drop_ratio * max(self._drop_slow, 1e-6)
            flux_drop = flux_mix < 0.45 * max(self._flux_peak, 1e-6)
            if self._drop_cooldown > 0:
                self._drop_cooldown -= 1
            elif energy_drop and flux_drop:
                self._drop_active_frames += 1
                if self._drop_active_frames >= self._drop_min_frames:
                    self._drop_cooldown = self._drop_cooldown_frames
                    self._drop_active_frames = 0
                    msg = (
                        f"Drop detected (fast={self._drop_fast:.4f}, slow={self._drop_slow:.4f}, "
                        f"flux={flux_mix:.4f}, peak={self._flux_peak:.4f})"
                    )
                    logger.info(msg)
                    if self._drop_callback:
                        try:
                            self._drop_callback()
                        except Exception:
                            pass
                    transient_boost *= 1.3
                    if self._notify_drop:
                        try:
                            self._notify_drop(msg)
                        except Exception:
                            pass
            else:
                self._drop_active_frames = 0
        gate_thresh = self._beat_gate_threshold
        if melody_weight > 0.01:
            gate_thresh *= 1.0 - 0.4 * melody_weight
        if self._beat_gate_enabled and flux_mix < gate_thresh:
            # Suppress mids/highs when beat energy is low; keep bass-driven motion.
            bands[2:] *= 0.1
            transient_boost *= 0.7
        if self._strobe_enabled:
            if flux_raw > self._strobe_threshold or beat_hit:
                self._strobe_level = max(self._strobe_level, self._strobe_strength)
            else:
                self._strobe_level *= self._strobe_decay
        else:
            self._strobe_level = 0.0
        self._frame_idx += 1
        hue_bias = 0.0
        if self._centroid_hue_enabled and magnitude.sum() > 1e-9:
            freqs = np.linspace(0.0, 1.0, n)
            centroid = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-9))
            hue_bias = self._centroid_hue_amount * (centroid - 0.5)
        hue_bias += melody_bias
        palette_hues = (self._palette + 0.02 * self._frame_idx + hue_bias) % 1.0
        fill_target = float(np.clip(bands.max() * transient_boost, 0.0, 1.0))
        self._vertical_fill = 0.72 * self._vertical_fill + 0.28 * fill_target
        if self.color_mode == "dominant":
            idx = int(np.argmax(bands))
            hue = palette_hues[idx]
            value = float(np.clip(self.value_boost * bands[idx] * transient_boost, self.value_floor, 1.0))
            if self._vertical_fill_enabled:
                value *= float(np.clip(self._vertical_fill, 0.05, 1.0))
            return [self._hsv_to_rgb(hue, 1.0, value)] * self.leds
        weights = bands / (bands.sum() + 1e-6)
        weights = np.clip(weights, 0.05, None)
        weights = weights / weights.sum()
        stops = np.concatenate([[0.0], np.cumsum(weights)])
        palette_vals = np.clip(self.value_boost * bands * transient_boost, self.value_floor, 1.0)
        pixels = np.linspace(0, 1, self.leds)
        hues = np.interp(pixels, stops, np.concatenate([palette_hues, palette_hues[-1:]]))
        vals = np.interp(pixels, stops, np.concatenate([palette_vals, palette_vals[-1:]]))
        if self._vertical_fill_enabled:
            # Fill from bottom (0.0) to top (1.0) following audio level; soft edge for smoother motion.
            progress = float(np.clip(self._vertical_fill, 0.0, 1.0))
            mask = np.clip((progress - pixels) / 0.2, 0.0, 1.0)
            vals = np.clip(vals * mask, 0.0, 1.0)
        sats = np.ones_like(hues)
        rgb = [self._hsv_to_rgb(h % 1.0, s, v) for h, s, v in zip(hues, sats, vals)]
        if self._flash_enabled:
            self._flash_phase = (self._flash_phase + 1) % max(1, self._flash_rate_frames)
            if self._flash_phase == 0:
                rgb = [self._hsv_to_rgb(h % 1.0, 1.0, v) for h, v in zip(hues, vals)]
        if self._flicker_enabled and self._flicker_amount > 0.0:
            noise = 1.0 + self._flicker_amount * (np.random.rand(self.leds) - 0.5)
            rgb = [
                tuple(int(np.clip(ch * n, 0, 255)) for ch in px)
                for px, n in zip(rgb, noise)
            ]
        if self._strobe_level > 1e-3:
            factor = 1.0 + self._strobe_level
            rgb = [tuple(int(np.clip(ch * factor, 0, 255)) for ch in px) for px in rgb]
        return rgb

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Color:
        h = h % 1.0
        i = int(h * 6)
        f = h * 6 - i
        p = int(255 * v * (1 - s))
        q = int(255 * v * (1 - f * s))
        t = int(255 * v * (1 - (1 - f) * s))
        v_i = int(255 * v)
        i %= 6
        if i == 0:
            return v_i, t, p
        if i == 1:
            return q, v_i, p
        if i == 2:
            return p, v_i, t
        if i == 3:
            return p, q, v_i
        if i == 4:
            return t, p, v_i
        return v_i, p, q


class MusicReactiveWorker:
    def __init__(self, controller_provider, leds: int = 60) -> None:
        self.controller_provider = controller_provider
        self.engine = ReactiveAudioEngine(leds=leds)
        self.engine.register_callback(self._on_colors)
        self._active = False
        self._brightness_floor = MUSIC_BASE_BRIGHTNESS
        self._brightness_ceiling = MUSIC_MAX_BRIGHTNESS
        self._level_reactive = True
        self._keep_lit = False
        self._last_brightness: int | None = None
        self._last_rgb: np.ndarray | None = None
        self._color_smoothing = 0.3  # 0=no smoothing, 0.9=very smooth
        self._observer: Callable[[Color, int], None] | None = None
        self._transient_boost = 1.2
        self.engine.set_transient_boost(self._transient_boost)
        self._band_weights = (1.0, 1.0, 1.0)
        self._limit_amount = 0.3
        self.engine.set_band_weights(*self._band_weights)
        self.engine.set_soft_limiter(self._limit_amount)
        self._beat_gate_enabled = False
        self._beat_gate_threshold = 0.12
        self.engine.set_beat_gate(self._beat_gate_enabled, self._beat_gate_threshold)
        self._palette_name = "Default"
        self.engine.set_palette(self._default_palette(self._palette_name))
        self._strobe_enabled = False
        self._strobe_strength = 0.5
        self._strobe_threshold = 0.25
        self.engine.set_strobe(self._strobe_enabled, self._strobe_strength, self._strobe_threshold)
        self._flash_enabled = False
        self._flash_rate_frames = 3
        self.engine.set_flash(self._flash_enabled, self._flash_rate_frames)
        self._auto_gain_enabled = False
        self._auto_gain_target = 0.18
        self._auto_gain_state = 1.0
        self._auto_gain_speed = 0.04
        self.engine.set_auto_gain(self._auto_gain_enabled, self._auto_gain_target)
        self._centroid_hue_enabled = False
        self._centroid_hue_amount = 0.15
        self.engine.set_centroid_hue(self._centroid_hue_enabled, self._centroid_hue_amount)
        self._melody_focus_enabled = False
        self._melody_focus_amount = 0.25
        self.engine.set_melody_focus(self._melody_focus_enabled, self._melody_focus_amount)
        self._flicker_enabled = False
        self._flicker_amount = 0.0
        self._ambient_fill = False
        self._ambient_hue = 0.0
        self._vertical_pulse = False
        self.engine.set_vertical_fill(self._vertical_pulse)
        self._last_nonzero_rgb = None
        self._band_env = np.zeros(6, dtype=np.float32)
        self._drop_enabled = False
        self._drop_ratio = 0.35
        self._drop_min_frames = 10
        self._drop_cooldown_frames = 60
        self._drop_decay = 0.98
        self._drop_slow = 0.0
        self._drop_fast = 0.0
        self._drop_cooldown = 0
        self._drop_active_frames = 0
        self._drop_callback = None
        self._notify_drop: Callable[[str], None] | None = None

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._last_brightness = None
        self._last_rgb = None
        await self.engine.start()

    async def stop(self) -> None:
        self._active = False
        await self.engine.stop()
        self._last_brightness = None
        self._last_rgb = None

    async def set_audio_source(self, device: int | None, loopback: bool) -> None:
        was_running = self._active
        if was_running:
            await self.engine.stop()
        self.engine.configure_source(device, loopback)
        if was_running:
            await self.engine.start()
            self._active = True

    async def apply_audio_params(
        self,
        buffer_size: int | None = None,
        gain: float | None = None,
        value_floor: float | None = None,
        value_boost: float | None = None,
        silence_gate: float | None = None,
        forced_channels: int | None = None,
    ) -> None:
        was_running = self._active
        if was_running:
            await self.engine.stop()
        self.engine.configure_params(
            buffer_size=buffer_size,
            gain=gain,
            value_floor=value_floor,
            value_boost=value_boost,
            silence_gate=silence_gate,
            forced_channels=forced_channels,
        )
        if was_running:
            await self.engine.start()
            self._active = True
        self._last_rgb = None
        self._last_brightness = None

    def configure_brightness(self, base: int, ceiling: int, adaptive: bool) -> None:
        base = int(np.clip(base, 1, 100))
        ceiling = int(np.clip(ceiling, base, 100))
        self._brightness_floor = base
        self._brightness_ceiling = ceiling
        self._level_reactive = adaptive

    def configure_keep_lit(self, enabled: bool) -> None:
        self._keep_lit = bool(enabled)
        if enabled and self._last_nonzero_rgb is None and self._last_rgb is not None:
            self._last_nonzero_rgb = self._last_rgb

    def configure_audio_params(
        self,
        buffer_size: int | None = None,
        gain: float | None = None,
        value_floor: float | None = None,
        value_boost: float | None = None,
        silence_gate: float | None = None,
    ) -> None:
        self.engine.configure_params(
            buffer_size=buffer_size,
            gain=gain,
            value_floor=value_floor,
            value_boost=value_boost,
            silence_gate=silence_gate,
        )

    def configure_color_mode(self, mode: str) -> None:
        self.engine.set_color_mode(mode)

    def set_observer(self, observer: Callable[[Color, int], None] | None) -> None:
        self._observer = observer

    def configure_smoothing(self, amount: float) -> None:
        # amount in [0, 1]; stored as blend weight of previous color.
        self._color_smoothing = float(np.clip(amount, 0.0, 0.95))

    def configure_transient_boost(self, amount: float) -> None:
        # amount in [0, 3]; 1.0 = default emphasis, 0 = disabled.
        self._transient_boost = float(np.clip(amount, 0.0, 3.0))
        self.engine.set_transient_boost(self._transient_boost)

    def configure_band_weights(self, bass: float, mid: float, high: float) -> None:
        self._band_weights = (float(bass), float(mid), float(high))
        self.engine.set_band_weights(*self._band_weights)

    def configure_soft_limiter(self, amount: float) -> None:
        # amount in [0,1]; 0 = off, 1 = strong limiting.
        self._limit_amount = float(np.clip(amount, 0.0, 1.0))
        self.engine.set_soft_limiter(self._limit_amount)

    def configure_beat_gate(self, enabled: bool, threshold: float | None = None) -> None:
        self._beat_gate_enabled = bool(enabled)
        if threshold is not None:
            self._beat_gate_threshold = float(np.clip(threshold, 0.01, 0.5))
        self.engine.set_beat_gate(self._beat_gate_enabled, self._beat_gate_threshold)

    def configure_palette(self, name: str) -> None:
        self._palette_name = name
        self.engine.set_palette(self._default_palette(name))

    def configure_strobe(self, enabled: bool, strength: float, threshold: float | None = None) -> None:
        self._strobe_enabled = bool(enabled)
        self._strobe_strength = float(np.clip(strength, 0.0, 1.0))
        if threshold is not None:
            self._strobe_threshold = float(np.clip(threshold, 0.01, 1.0))
        self.engine.set_strobe(self._strobe_enabled, self._strobe_strength, self._strobe_threshold)

    def configure_ambient_fill(self, enabled: bool) -> None:
        self._ambient_fill = bool(enabled)

    def configure_flash(self, enabled: bool, rate_frames: int | None = None) -> None:
        self._flash_enabled = bool(enabled)
        if rate_frames is not None:
            self._flash_rate_frames = int(max(1, min(20, rate_frames)))
        self.engine.set_flash(self._flash_enabled, self._flash_rate_frames)

    def configure_auto_gain(self, enabled: bool, target: float | None = None) -> None:
        self._auto_gain_enabled = bool(enabled)
        if target is not None:
            self._auto_gain_target = float(np.clip(target, 0.05, 0.6))
        self.engine.set_auto_gain(self._auto_gain_enabled, self._auto_gain_target)

    def configure_centroid_hue(self, enabled: bool, amount: float | None = None) -> None:
        self._centroid_hue_enabled = bool(enabled)
        if amount is not None:
            self._centroid_hue_amount = float(np.clip(amount, 0.0, 0.5))
        self.engine.set_centroid_hue(self._centroid_hue_enabled, self._centroid_hue_amount)

    def configure_melody_focus(self, enabled: bool, amount: float | None = None) -> None:
        self._melody_focus_enabled = bool(enabled)
        if amount is not None:
            self._melody_focus_amount = float(np.clip(amount, 0.0, 1.0))
        self.engine.set_melody_focus(self._melody_focus_enabled, self._melody_focus_amount)

    def configure_vertical_pulse(self, enabled: bool) -> None:
        self._vertical_pulse = bool(enabled)
        self.engine.set_vertical_fill(self._vertical_pulse)

    def configure_flicker(self, enabled: bool, amount: float | None = None) -> None:
        self._flicker_enabled = bool(enabled)
        if amount is not None:
            self._flicker_amount = float(np.clip(amount, 0.0, 0.6))

    def configure_drop_detector(
        self,
        enabled: bool,
        ratio: float | None = None,
        min_frames: int | None = None,
        cooldown_frames: int | None = None,
    ) -> None:
        self.engine.set_drop_detector(
            enabled=enabled,
            ratio=ratio,
            min_frames=min_frames,
            cooldown_frames=cooldown_frames,
            callback=self._trigger_drop_accent,
        )
        self.engine.set_drop_notifier(self._notify_ui)

    def _trigger_drop_accent(self) -> None:
        # On drop: push a transient boost to values/brightness; handled via strobe level.
        self.engine._strobe_level = max(self.engine._strobe_level, 0.6)

    def set_drop_notifier(self, notifier: Callable[[str], None]) -> None:
        self._notify_ui = notifier
    @staticmethod
    def _default_palette(name: str) -> list[float]:
        palettes = {
            "Default": [0.00, 0.12, 0.28, 0.48, 0.66, 0.88],
            "Sunset": [0.98, 0.04, 0.06, 0.10, 0.13, 0.16],
            "Neon": [0.83, 0.58, 0.33, 0.08, 0.50, 0.68],
            "Cool": [0.55, 0.62, 0.70, 0.78, 0.84, 0.92],
            "Aurora": [0.42, 0.50, 0.58, 0.70, 0.78, 0.86],
            "Candy": [0.92, 0.08, 0.16, 0.33, 0.66, 0.83],
            "Ocean": [0.55, 0.58, 0.62, 0.68, 0.74, 0.80],
            "Currents": [0.11, 0.17, 0.32, 0.47, 0.59, 0.73],
            "Slow Rush": [0.03, 0.08, 0.14, 0.58, 0.66, 0.72],
        }
        return palettes.get(name, palettes["Default"])

    async def _on_colors(self, colors: Sequence[Color]) -> None:
        if not self._active:
            return
        controller = self.controller_provider()
        if controller is None:
            return
        rgb, brightness = self._collapse(colors)
        if self._observer:
            try:
                self._observer(rgb, brightness)
            except Exception:
                pass
        try:
            await controller.set_color(*rgb)
            if brightness != self._last_brightness:
                await controller.set_brightness(brightness)
                self._last_brightness = brightness
        except Exception:
            pass

    def _collapse(self, colors: Sequence[Color]) -> tuple[Color, int]:
        arr = np.array(colors, dtype=np.float32)
        if arr.size == 0:
            return (255, 255, 255), self._brightness_floor
        avg = arr.mean(axis=0)
        if self._color_smoothing > 0 and self._last_rgb is not None:
            avg = self._color_smoothing * self._last_rgb + (1 - self._color_smoothing) * avg
        self._last_rgb = avg
        self._ambient_hue = (self._ambient_hue + 0.01) % 1.0
        rgb = tuple(int(np.clip(v, 0, 255)) for v in avg)
        if max(rgb) > 0:
            self._last_nonzero_rgb = avg
        env_peak = float(np.max(self._band_env)) if self._band_env.size else 0.0
        intensity_raw = float(np.clip(arr.max() / 255.0, 0.0, 1.0))
        intensity = float(np.power(max(intensity_raw, env_peak), 0.6))  # expand low-level dynamics
        if self._limit_amount > 0.0 and intensity > 0.6:
            # Soft limiter: compress peaks above threshold.
            over = intensity - 0.6
            intensity = 0.6 + over / (1 + self._limit_amount * 4.0)
        if intensity <= self.engine.silence_gate:
            if self._ambient_fill:
                brightness = self._brightness_floor
                rgb_arr = np.array(self._hsv_to_rgb(self._ambient_hue, 1.0, 1.0), dtype=np.float32)
                self._last_rgb = rgb_arr
                self._last_nonzero_rgb = rgb_arr
                rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb_arr)
            elif self._keep_lit:
                brightness = self._brightness_floor
                fallback = self._last_nonzero_rgb or self._last_rgb
                if fallback is None:
                    fallback = np.array([255.0, 255.0, 255.0], dtype=np.float32)
                rgb_arr = fallback
                rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb_arr)
            else:
                brightness = 0
                rgb = (0, 0, 0)
                self._last_rgb = None
        elif self._level_reactive:
            span = max(1, self._brightness_ceiling - self._brightness_floor)
            brightness = int(self._brightness_floor + intensity * span)
        else:
            brightness = self._brightness_ceiling
        if brightness > 0:
            brightness = int(np.clip(brightness, self._brightness_floor, self._brightness_ceiling))
        return rgb, brightness
