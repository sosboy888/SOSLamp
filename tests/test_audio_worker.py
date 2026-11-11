"""Tests for MusicReactiveWorker's silence handling and the engine's colour output."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soslamp.audio.reactive_engine import MusicReactiveWorker, ReactiveAudioEngine  # noqa: E402

BLACK = [(0, 0, 0)] * 60
RED = [(255, 0, 0)] * 60


def make_worker() -> MusicReactiveWorker:
    return MusicReactiveWorker(lambda: None)


class SilenceHandlingTests(unittest.TestCase):
    def test_ambient_fill_during_silence_does_not_crash(self):
        worker = make_worker()
        worker.configure_ambient_fill(True)
        rgb, brightness = worker._collapse(BLACK)
        self.assertEqual(len(rgb), 3)
        self.assertGreater(max(rgb), 0)  # a rainbow colour, not black
        self.assertEqual(brightness, worker._brightness_floor)

    def test_keep_lit_holds_last_colour(self):
        worker = make_worker()
        worker.configure_smoothing(0.0)
        worker.configure_keep_lit(True)
        worker._collapse(RED)  # remember a non-black colour
        rgb, brightness = worker._collapse(BLACK)
        self.assertEqual(rgb[0], 255)
        self.assertEqual(brightness, worker._brightness_floor)

    def test_keep_lit_without_history_falls_back_to_white(self):
        worker = make_worker()
        worker.configure_keep_lit(True)
        rgb, _ = worker._collapse(BLACK)
        self.assertEqual(rgb, (255, 255, 255))

    def test_silence_without_options_goes_dark(self):
        worker = make_worker()
        rgb, brightness = worker._collapse(BLACK)
        self.assertEqual((rgb, brightness), ((0, 0, 0), 0))


class DropDetectorTests(unittest.TestCase):
    def test_configure_drop_detector_without_gui_notifier(self):
        worker = make_worker()
        # Used to raise AttributeError (_notify_ui was only set by the GUI).
        worker.configure_drop_detector(True, ratio=0.4, min_frames=5, cooldown_frames=20)
        self.assertTrue(worker.engine._drop_enabled)

    def test_notifier_receives_drop_message(self):
        worker = make_worker()
        messages = []
        worker.set_drop_notifier(messages.append)
        worker.configure_drop_detector(True, ratio=0.9, min_frames=1, cooldown_frames=1)
        eng = worker.engine
        eng._samplerate = 44100
        loud = (0.5 * np.sin(2 * np.pi * 100 * np.arange(256) / 44100)).astype(np.float32)
        quiet = loud * 0.02
        for _ in range(80):
            eng._samples_to_colors(loud)
        for _ in range(40):
            eng._samples_to_colors(quiet)
        self.assertTrue(any("Drop detected" in m for m in messages))

    def _run(self, frames):
        worker = make_worker()
        messages = []
        worker.set_drop_notifier(messages.append)
        worker.configure_drop_detector(True, ratio=0.35, min_frames=3, cooldown_frames=5)
        for f in frames:
            worker.engine._samples_to_colors(f)
        return messages

    def test_steady_loud_audio_is_not_a_drop(self):
        t = np.arange(256) / 44100
        loud = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
        self.assertEqual(self._run([loud] * 200), [])

    def test_sustained_silence_is_not_a_drop(self):
        self.assertEqual(self._run([np.zeros(256, dtype=np.float32)] * 200), [])

    def test_sudden_quiet_after_loud_is_a_drop(self):
        t = np.arange(256) / 44100
        loud = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
        messages = self._run([loud] * 150 + [loud * 0.01] * 30)
        self.assertEqual(len(messages) >= 1, True)


class EngineTests(unittest.TestCase):
    def test_output_has_one_colour_per_led(self):
        eng = ReactiveAudioEngine(leds=30)
        tone = (0.5 * np.sin(2 * np.pi * 80 * np.arange(256) / 44100)).astype(np.float32)
        colors = eng._samples_to_colors(tone)
        self.assertEqual(len(colors), 30)
        for c in colors:
            self.assertTrue(all(0 <= ch <= 255 for ch in c))

    def test_silence_is_black(self):
        eng = ReactiveAudioEngine(leds=10)
        self.assertEqual(set(eng._samples_to_colors(np.zeros(256, dtype=np.float32))), {(0, 0, 0)})

    def test_hsv_primaries(self):
        self.assertEqual(ReactiveAudioEngine._hsv_to_rgb(0.0, 1.0, 1.0), (255, 0, 0))
        self.assertEqual(ReactiveAudioEngine._hsv_to_rgb(1 / 3, 1.0, 1.0)[1], 255)
        self.assertEqual(ReactiveAudioEngine._hsv_to_rgb(0.0, 0.0, 1.0), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
