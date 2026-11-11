"""Tests for the byte packets BleLedController builds (no Bluetooth hardware needed)."""

import asyncio
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soslamp.integrations.ble_led_device import (  # noqa: E402
    EFFECTS,
    WEEK_DAYS,
    BleLedController,
    DeviceType,
    detect_device_type,
)


def capture(name="ELK-BLE-1234"):
    """Return a controller whose _send() records packets instead of writing over BLE."""
    ctrl = BleLedController("00:00:00:00:00:00", name)
    sent: list[bytes] = []

    async def fake_send(payload: bytes) -> None:
        sent.append(payload)

    ctrl._send = fake_send
    return ctrl, sent


def run(coro):
    return asyncio.run(coro)


class PacketTests(unittest.TestCase):
    def test_every_packet_is_9_bytes_framed_by_7e_and_ef(self):
        ctrl, sent = capture()
        run(ctrl.set_color(1, 2, 3))
        run(ctrl.set_brightness(50))
        run(ctrl.set_effect(EFFECTS.blink_red, 30))
        run(ctrl.set_schedule_on(WEEK_DAYS.all, 8, 30, True))
        for p in sent:
            self.assertEqual((len(p), p[0], p[-1]), (9, 0x7E, 0xEF))

    def test_rgb_packet(self):
        ctrl, sent = capture()
        run(ctrl.set_color(255, 128, 0))
        self.assertEqual(sent[-1], bytes([0x7E, 0x00, 0x05, 0x03, 255, 128, 0, 0x00, 0xEF]))

    def test_rgb_values_are_masked_to_one_byte(self):
        ctrl, sent = capture()
        run(ctrl.set_color(256 + 7, -1, 0))
        self.assertEqual(sent[-1][4:7], bytes([7, 255, 0]))

    def test_brightness_is_clamped(self):
        ctrl, sent = capture()
        run(ctrl.set_brightness(250))
        run(ctrl.set_brightness(-5))
        self.assertEqual([sent[0][3], sent[1][3]], [100, 0])

    def test_effect_packet_includes_speed(self):
        ctrl, sent = capture()
        run(ctrl.set_effect(EFFECTS.crossfade_red_green_blue, 42))
        self.assertEqual(sent[-1], bytes([0x7E, 0x00, 0x03, 0x89, 42, 0x00, 0x00, 0x00, 0xEF]))

    def test_color_after_effect_switches_back_to_static_first(self):
        ctrl, sent = capture()
        run(ctrl.set_effect(EFFECTS.blink_red))
        run(ctrl.set_color(10, 20, 30))
        # [effect, "static mode", colour]
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[1][3], 0x01)
        self.assertEqual(sent[2][3], 0x03)

    def test_white_balance_extremes(self):
        ctrl, sent = capture()
        run(ctrl.set_color_temperature(2700))
        run(ctrl.set_color_temperature(6500))
        self.assertEqual((sent[0][4], sent[0][5]), (0, 100))
        self.assertEqual((sent[1][4], sent[1][5]), (100, 0))


class ScheduleTests(unittest.TestCase):
    def test_schedule_on_packet(self):
        ctrl, sent = capture()
        run(ctrl.set_schedule_on(WEEK_DAYS.week_days, 18, 5, True))
        # 0x1F weekdays | 0x80 enabled = 0x9F, type byte 0x00 = "on"
        self.assertEqual(sent[-1], bytes([0x7E, 0x00, 0x82, 18, 5, 0x00, 0x00, 0x9F, 0xEF]))

    def test_schedule_off_packet_and_disabled_flag(self):
        ctrl, sent = capture()
        run(ctrl.set_schedule_off(WEEK_DAYS.weekend_days, 23, 0, False))
        # weekend 0x60, no enable bit, type byte 0x01 = "off"
        self.assertEqual(sent[-1], bytes([0x7E, 0x00, 0x82, 23, 0, 0x00, 0x01, 0x60, 0xEF]))

    def test_schedule_time_is_clamped(self):
        ctrl, sent = capture()
        run(ctrl.set_schedule_on(WEEK_DAYS.all, 99, 99, True))
        self.assertEqual((sent[-1][3], sent[-1][4]), (23, 59))

    def test_day_masks_do_not_overlap_and_combine(self):
        singles = [getattr(WEEK_DAYS, d) for d in
                   ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")]
        self.assertEqual(sum(singles), WEEK_DAYS.all)
        self.assertEqual(len(set(singles)), 7)
        self.assertEqual(WEEK_DAYS.week_days | WEEK_DAYS.weekend_days, WEEK_DAYS.all)

    def test_clock_sync_packet(self):
        ctrl, sent = capture()
        run(ctrl.set_custom_time(datetime(2026, 11, 11, 13, 45, 9)))  # a Wednesday
        self.assertEqual(sent[-1], bytes([0x7E, 0x00, 0x83, 13, 45, 9, 3, 0x00, 0xEF]))


class DeviceDetectionTests(unittest.TestCase):
    def test_prefix_detection_is_case_insensitive(self):
        self.assertIs(detect_device_type("melk-abc"), DeviceType.MELK)
        self.assertIs(detect_device_type("ELK-BLEDOM"), DeviceType.ELK_BLE)
        self.assertIs(detect_device_type("LEDBLE-1"), DeviceType.LED_BLE)

    def test_unknown_and_missing_names(self):
        self.assertIs(detect_device_type("Headphones"), DeviceType.UNKNOWN)
        self.assertIs(detect_device_type(None), DeviceType.UNKNOWN)
        self.assertIs(detect_device_type(""), DeviceType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
