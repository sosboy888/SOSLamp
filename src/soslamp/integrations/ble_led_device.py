"""BLE controller for ELK/MELK LED strips (ported from elk-led-controller).

Every command is a fixed 9-byte packet that starts with 0x7E and ends with 0xEF.
The bytes in the middle change depending on the command (see docs/MELK_PROTOCOL.md).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported BLE device prefixes.

    The strips advertise names like "ELK-BLE-1234", so we match on the start of the name.
    """

    ELK_BLE = "ELK-BLE"
    LED_BLE = "LEDBLE"
    MELK = "MELK"
    ELK_BULB = "ELK-BULB"
    ELK_LAMPL = "ELK-LAMPL"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class DeviceConfig:
    """Per-model settings: which GATT characteristic to write to and the on/off packets."""

    write_uuid: str
    read_uuid: str
    turn_on_cmd: bytes
    turn_off_cmd: bytes
    min_color_temp: int
    max_color_temp: int
    command_delay: float


# Lookup table from device type to its settings. Most models share the same values;
# ELK_BLE and LED_BLE differ in UUIDs / power commands.
DEVICE_CONFIGS: dict[DeviceType, DeviceConfig] = {
    DeviceType.ELK_BLE: DeviceConfig(
        write_uuid="0000fff3-0000-1000-8000-00805f9b34fb",
        read_uuid="0000fff4-0000-1000-8000-00805f9b34fb",
        turn_on_cmd=bytes([0x7E, 0x00, 0x04, 0xF0, 0x00, 0x01, 0xFF, 0x00, 0xEF]),
        turn_off_cmd=bytes([0x7E, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEF]),
        min_color_temp=2700,
        max_color_temp=6500,
        command_delay=0.005,
    ),
    DeviceType.LED_BLE: DeviceConfig(
        write_uuid="0000ffe1-0000-1000-8000-00805f9b34fb",
        read_uuid="0000ffe2-0000-1000-8000-00805f9b34fb",
        turn_on_cmd=bytes([0x7E, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]),
        turn_off_cmd=bytes([0x7E, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEF]),
        min_color_temp=2700,
        max_color_temp=6500,
        command_delay=0.005,
    ),
    DeviceType.MELK: DeviceConfig(
        write_uuid="0000fff3-0000-1000-8000-00805f9b34fb",
        read_uuid="0000fff4-0000-1000-8000-00805f9b34fb",
        turn_on_cmd=bytes([0x7E, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]),
        turn_off_cmd=bytes([0x7E, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEF]),
        min_color_temp=2700,
        max_color_temp=6500,
        command_delay=0.005,
    ),
    DeviceType.ELK_BULB: DeviceConfig(
        write_uuid="0000fff3-0000-1000-8000-00805f9b34fb",
        read_uuid="0000fff4-0000-1000-8000-00805f9b34fb",
        turn_on_cmd=bytes([0x7E, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]),
        turn_off_cmd=bytes([0x7E, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEF]),
        min_color_temp=2700,
        max_color_temp=6500,
        command_delay=0.005,
    ),
    DeviceType.ELK_LAMPL: DeviceConfig(
        write_uuid="0000fff3-0000-1000-8000-00805f9b34fb",
        read_uuid="0000fff4-0000-1000-8000-00805f9b34fb",
        turn_on_cmd=bytes([0x7E, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]),
        turn_off_cmd=bytes([0x7E, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEF]),
        min_color_temp=2700,
        max_color_temp=6500,
        command_delay=0.005,
    ),
}


@dataclass(slots=True)
class BleDeviceInfo:
    """Result from BLE scanning."""

    address: str
    name: str
    rssi: Optional[int] = None

    def label(self) -> str:
        """Human-readable line for the device dropdown."""
        # RSSI is signal strength in dBm (closer to 0 = stronger); it can be missing.
        rssi = f"{self.rssi} dBm" if self.rssi is not None else "?"
        return f"{self.name or 'Unknown'} ({self.address}) [{rssi}]"


@dataclass(slots=True)
class Effects:
    """Names for the built-in effect codes the firmware understands."""

    jump_red_green_blue: int
    jump_red_green_blue_yellow_cyan_magenta_white: int
    crossfade_red: int
    crossfade_green: int
    crossfade_blue: int
    crossfade_yellow: int
    crossfade_cyan: int
    crossfade_magenta: int
    crossfade_white: int
    crossfade_red_green: int
    crossfade_red_blue: int
    crossfade_green_blue: int
    crossfade_red_green_blue: int
    crossfade_red_green_blue_yellow_cyan_magenta_white: int
    blink_red: int
    blink_green: int
    blink_blue: int
    blink_yellow: int
    blink_cyan: int
    blink_magenta: int
    blink_white: int
    blink_red_green_blue_yellow_cyan_magenta_white: int


# Effect code values taken from the Rust project's protocol notes.
EFFECTS = Effects(
    jump_red_green_blue=0x87,
    jump_red_green_blue_yellow_cyan_magenta_white=0x88,
    crossfade_red=0x8B,
    crossfade_green=0x8C,
    crossfade_blue=0x8D,
    crossfade_yellow=0x8E,
    crossfade_cyan=0x8F,
    crossfade_magenta=0x90,
    crossfade_white=0x91,
    crossfade_red_green=0x92,
    crossfade_red_blue=0x93,
    crossfade_green_blue=0x94,
    crossfade_red_green_blue=0x89,
    crossfade_red_green_blue_yellow_cyan_magenta_white=0x8A,
    blink_red=0x96,
    blink_green=0x97,
    blink_blue=0x98,
    blink_yellow=0x99,
    blink_cyan=0x9A,
    blink_magenta=0x9B,
    blink_white=0x9C,
    blink_red_green_blue_yellow_cyan_magenta_white=0x95,
)


@dataclass(slots=True)
class WeekDays:
    """Bit masks for the schedule day picker (one bit per day, Monday = bit 0)."""

    monday: int
    tuesday: int
    wednesday: int
    thursday: int
    friday: int
    saturday: int
    sunday: int
    all: int
    week_days: int
    weekend_days: int
    none: int


# Combine days with | e.g. monday | friday. The grouped values below are just pre-combined masks.
WEEK_DAYS = WeekDays(
    monday=0x01,
    tuesday=0x02,
    wednesday=0x04,
    thursday=0x08,
    friday=0x10,
    saturday=0x20,
    sunday=0x40,
    all=0x7F,
    week_days=0x1F,
    weekend_days=0x60,
    none=0x00,
)


class LedCommandError(RuntimeError):
    """Raised when a BLE command fails repeatedly."""


def detect_device_type(name: str | None) -> DeviceType:
    """Work out the model from the advertised name (UNKNOWN if it isn't one of ours)."""
    if not name:
        return DeviceType.UNKNOWN
    for prefix in (
        DeviceType.ELK_BLE,
        DeviceType.LED_BLE,
        DeviceType.MELK,
        DeviceType.ELK_BULB,
        DeviceType.ELK_LAMPL,
    ):
        # Upper-case so "melk-xyz" and "MELK-XYZ" both match.
        if name.upper().startswith(prefix.value):
            return prefix
    return DeviceType.UNKNOWN


async def discover_devices(timeout: float = 5.0) -> list[BleDeviceInfo]:
    """Discover compatible BLE LED controllers."""
    devices = await BleakScanner.discover(timeout=timeout)
    infos: list[BleDeviceInfo] = []
    for device in devices:
        dtype = detect_device_type(device.name or "")
        if dtype is DeviceType.UNKNOWN:
            continue
        # Older bleak versions have device.rssi; newer ones may only expose it in details.
        rssi = getattr(device, "rssi", None)
        if rssi is None:
            details = getattr(device, "details", None)
            if isinstance(details, dict):
                rssi = details.get("RSSI")
        infos.append(
            BleDeviceInfo(
                address=str(device.address),
                name=device.name or dtype.value,
                rssi=rssi,
            )
        )
    return infos


class BleLedController:
    """High-level BLE wrapper inspired by elk-led-controller.

    Each method builds the right byte packet and sends it with _send(), which
    takes care of retries and pacing.
    """

    def __init__(self, address: str, name: str | None = None) -> None:
        self.address = address
        self.name = name or address
        self.device_type = detect_device_type(name)
        if self.device_type is DeviceType.UNKNOWN:
            # Unknown name (e.g. user connected by address): guess MELK settings.
            self.device_type = DeviceType.MELK
        self.config = DEVICE_CONFIGS[self.device_type]
        self._client: BleakClient | None = None
        self._lock = asyncio.Lock()
        # Timestamp of the last write, used to space commands out (see _send).
        self._last_command = 0.0
        # Cached state so the GUI can read values back without asking the device.
        self._effect: Optional[int] = None
        self._brightness: int = 100
        self._color_temp: int = 5000
        self._effect_speed: int = 50

    async def connect(self, power_on: bool = True) -> None:
        """Open the BLE connection and (optionally) switch the strip on."""
        await self._connect_client()
        if power_on:
            await self.power_on()
        # Only the ELK models get a clock sync here (schedules need the device to know the time).
        if self.device_type in {
            DeviceType.ELK_BLE,
            DeviceType.ELK_BULB,
            DeviceType.ELK_LAMPL,
        }:
            await self.sync_time()

    async def disconnect(self) -> None:
        """Close the connection if it is open."""
        client = self._client
        if client and client.is_connected:
            await client.disconnect()

    async def power_on(self) -> None:
        await self._send(self.config.turn_on_cmd)
        await asyncio.sleep(0.05)

    async def power_off(self) -> None:
        await self._send(self.config.turn_off_cmd)
        await asyncio.sleep(0.05)

    async def set_color(self, r: int, g: int, b: int) -> None:
        # If an effect is running, switch back to static mode first or the colour is ignored.
        if self._effect is not None:
            await self._send(bytes([0x7E, 0x00, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]))
            await asyncio.sleep(0.05)
            self._effect = None
        # `& 0xFF` keeps each value inside one byte (0-255) so bytes() can't raise.
        payload = bytes([0x7E, 0x00, 0x05, 0x03, r & 0xFF, g & 0xFF, b & 0xFF, 0x00, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.005)

    async def set_brightness(self, level: int) -> None:
        # Clamp to the 0-100 range the device accepts.
        value = max(0, min(100, level))
        payload = bytes([0x7E, 0x00, 0x01, value, 0x00, 0x00, 0x00, 0x00, 0xEF])
        await self._send(payload)
        self._brightness = value

    async def set_effect(self, effect_code: int, speed: int | None = None) -> None:
        """Start a built-in effect; reuses the last speed if none is given."""
        speed_byte = self._normalize_speed(speed if speed is not None else self._effect_speed)
        payload = bytes([0x7E, 0x00, 0x03, effect_code, speed_byte, 0x00, 0x00, 0x00, 0xEF])
        await self._send(payload)
        self._effect = effect_code
        self._effect_speed = speed_byte
        await asyncio.sleep(0.05)

    async def set_effect_speed(self, value: int) -> None:
        limited = self._normalize_speed(value)
        payload = bytes([0x7E, 0x00, 0x02, limited, 0x00, 0x00, 0x00, 0x00, 0xEF])
        await self._send(payload)
        self._effect_speed = limited
        await asyncio.sleep(0.05)

    async def set_color_temperature(self, kelvin: int) -> None:
        """Set white balance. The device wants warm/cold percentages, not Kelvin."""
        config = self.config
        temp = max(config.min_color_temp, min(config.max_color_temp, kelvin))
        # Map the Kelvin range onto 0-100, then split it into warm and cold amounts.
        color_percent = int(
            (temp - config.min_color_temp)
            * 100
            / (config.max_color_temp - config.min_color_temp)
        )
        warm = color_percent
        cold = 100 - color_percent
        if self._effect is not None:
            await self._send(bytes([0x7E, 0x00, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]))
            await asyncio.sleep(0.2)
            self._effect = None
        payload = bytes([0x7E, 0x00, 0x05, 0x02, warm, cold, 0x00, 0x00, 0xEF])
        await self._send(payload)
        self._color_temp = temp
        await asyncio.sleep(0.2)

    async def set_schedule_on(self, days: int, hour: int, minute: int, enabled: bool) -> None:
        """Program the "turn on" time stored on the device itself."""
        h = max(0, min(23, hour))
        m = max(0, min(59, minute))
        # Low 7 bits = days of the week, top bit (0x80) = schedule enabled.
        flag = (days & 0x7F) | (0x80 if enabled else 0x00)
        payload = bytes([0x7E, 0x00, 0x82, h, m, 0x00, 0x00, flag, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.2)

    async def set_schedule_off(self, days: int, hour: int, minute: int, enabled: bool) -> None:
        """Same as set_schedule_on but the 0x01 byte makes it the "turn off" time."""
        h = max(0, min(23, hour))
        m = max(0, min(59, minute))
        flag = (days & 0x7F) | (0x80 if enabled else 0x00)
        payload = bytes([0x7E, 0x00, 0x82, h, m, 0x00, 0x01, flag, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.2)

    async def set_custom_time(self, dt: Optional[datetime] = None) -> None:
        """Set the strip's internal clock (needed for schedules to fire at the right time)."""
        now = dt or datetime.now()
        payload = bytes(
            [
                0x7E,
                0x00,
                0x83,
                now.hour,
                now.minute,
                now.second,
                # Weekday as 1-7. isoweekday() is already 1-7, the maths just keeps it in range.
                ((now.isoweekday() - 1) % 7) + 1,
                0x00,
                0xEF,
            ]
        )
        await self._send(payload)

    async def sync_time(self) -> None:
        """Shortcut: set the device clock to the PC's current time."""
        await self.set_custom_time()

    async def generic_command(self, cmd_id: int, sub_id: int, arg1: int, arg2: int, arg3: int) -> None:
        """Send a raw packet; handy for experimenting with unknown commands."""
        payload = bytes(
            [0x7E, 0x00, cmd_id & 0xFF, sub_id & 0xFF, arg1 & 0xFF, arg2 & 0xFF, arg3 & 0xFF, 0x00, 0xEF]
        )
        await self._send(payload)

    # Simple getters for the cached state.
    def effect_state(self) -> Optional[int]:
        return self._effect

    def brightness(self) -> int:
        return self._brightness

    def color_temperature(self) -> int:
        return self._color_temp

    async def _send(self, payload: bytes) -> None:
        """Write one packet, with a lock, a small delay between writes, and 3 retries."""
        # The lock stops two coroutines from writing at the same time.
        async with self._lock:
            # Don't send faster than the device's command_delay allows.
            wait_for = self.config.command_delay - (time.perf_counter() - self._last_command)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            for attempt in range(3):
                try:
                    # Reconnects automatically if the link dropped.
                    client = await self._connect_client()
                    await client.write_gatt_char(
                        self.config.write_uuid, payload, response=True
                    )
                    self._last_command = time.perf_counter()
                    return
                except BleakError as exc:
                    logger.warning(
                        "BLE write failed (attempt %s/3) for %s: %s",
                        attempt + 1,
                        self.name,
                        exc,
                    )
                    await asyncio.sleep(0.3)
            raise LedCommandError(f"Failed to send BLE command to {self.name}")

    async def _connect_client(self) -> BleakClient:
        """Return a connected BleakClient, scanning for the device first if needed."""
        client = self._client
        if client and client.is_connected:
            return client
        device = await BleakScanner.find_device_by_address(self.address, timeout=7.0)
        if device is None:
            raise LedCommandError(
                f"Unable to find BLE device {self.name} ({self.address}). "
                "Ensure it is powered on, paired, and within range."
            )
        self._client = BleakClient(device, timeout=10.0)
        try:
            await self._client.connect()
        except BleakError as exc:  # pragma: no cover - hardware dependent
            raise LedCommandError(f"Failed to connect to {self.name}: {exc}") from exc
        return self._client

    @staticmethod
    def _normalize_speed(value: int) -> int:
        return max(0, min(100, value))
