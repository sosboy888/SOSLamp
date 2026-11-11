"""BLE controller for ELK/MELK LED strips (ported from elk-led-controller)."""

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
    """Supported BLE device prefixes."""

    ELK_BLE = "ELK-BLE"
    LED_BLE = "LEDBLE"
    MELK = "MELK"
    ELK_BULB = "ELK-BULB"
    ELK_LAMPL = "ELK-LAMPL"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class DeviceConfig:
    write_uuid: str
    read_uuid: str
    turn_on_cmd: bytes
    turn_off_cmd: bytes
    min_color_temp: int
    max_color_temp: int
    command_delay: float


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
        rssi = f"{self.rssi} dBm" if self.rssi is not None else "?"
        return f"{self.name or 'Unknown'} ({self.address}) [{rssi}]"


@dataclass(slots=True)
class Effects:
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
    if not name:
        return DeviceType.UNKNOWN
    for prefix in (
        DeviceType.ELK_BLE,
        DeviceType.LED_BLE,
        DeviceType.MELK,
        DeviceType.ELK_BULB,
        DeviceType.ELK_LAMPL,
    ):
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
    """High-level BLE wrapper inspired by elk-led-controller."""

    def __init__(self, address: str, name: str | None = None) -> None:
        self.address = address
        self.name = name or address
        self.device_type = detect_device_type(name)
        if self.device_type is DeviceType.UNKNOWN:
            # default to MELK characteristics
            self.device_type = DeviceType.MELK
        self.config = DEVICE_CONFIGS[self.device_type]
        self._client: BleakClient | None = None
        self._lock = asyncio.Lock()
        self._last_command = 0.0
        self._effect: Optional[int] = None
        self._brightness: int = 100
        self._color_temp: int = 5000
        self._effect_speed: int = 50

    async def connect(self, power_on: bool = True) -> None:
        await self._connect_client()
        if power_on:
            await self.power_on()
        if self.device_type in {
            DeviceType.ELK_BLE,
            DeviceType.ELK_BULB,
            DeviceType.ELK_LAMPL,
        }:
            await self.sync_time()

    async def disconnect(self) -> None:
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
        if self._effect is not None:
            await self._send(bytes([0x7E, 0x00, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0xEF]))
            await asyncio.sleep(0.05)
            self._effect = None
        payload = bytes([0x7E, 0x00, 0x05, 0x03, r & 0xFF, g & 0xFF, b & 0xFF, 0x00, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.005)

    async def set_brightness(self, level: int) -> None:
        value = max(0, min(100, level))
        payload = bytes([0x7E, 0x00, 0x01, value, 0x00, 0x00, 0x00, 0x00, 0xEF])
        await self._send(payload)
        self._brightness = value

    async def set_effect(self, effect_code: int, speed: int | None = None) -> None:
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
        config = self.config
        temp = max(config.min_color_temp, min(config.max_color_temp, kelvin))
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
        h = max(0, min(23, hour))
        m = max(0, min(59, minute))
        flag = (days & 0x7F) | (0x80 if enabled else 0x00)
        payload = bytes([0x7E, 0x00, 0x82, h, m, 0x00, 0x00, flag, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.2)

    async def set_schedule_off(self, days: int, hour: int, minute: int, enabled: bool) -> None:
        h = max(0, min(23, hour))
        m = max(0, min(59, minute))
        flag = (days & 0x7F) | (0x80 if enabled else 0x00)
        payload = bytes([0x7E, 0x00, 0x82, h, m, 0x00, 0x01, flag, 0xEF])
        await self._send(payload)
        await asyncio.sleep(0.2)

    async def set_custom_time(self, dt: Optional[datetime] = None) -> None:
        now = dt or datetime.now()
        payload = bytes(
            [
                0x7E,
                0x00,
                0x83,
                now.hour,
                now.minute,
                now.second,
                ((now.isoweekday() - 1) % 7) + 1,
                0x00,
                0xEF,
            ]
        )
        await self._send(payload)

    async def sync_time(self) -> None:
        await self.set_custom_time()

    async def generic_command(self, cmd_id: int, sub_id: int, arg1: int, arg2: int, arg3: int) -> None:
        payload = bytes(
            [0x7E, 0x00, cmd_id & 0xFF, sub_id & 0xFF, arg1 & 0xFF, arg2 & 0xFF, arg3 & 0xFF, 0x00, 0xEF]
        )
        await self._send(payload)

    def effect_state(self) -> Optional[int]:
        return self._effect

    def brightness(self) -> int:
        return self._brightness

    def color_temperature(self) -> int:
        return self._color_temp

    async def _send(self, payload: bytes) -> None:
        async with self._lock:
            wait_for = self.config.command_delay - (time.perf_counter() - self._last_command)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            for attempt in range(3):
                try:
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
