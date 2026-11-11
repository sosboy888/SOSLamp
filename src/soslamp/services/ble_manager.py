"""Services that manage BLE LED controllers.

The GUI talks to BleControllerManager instead of the low-level controller, so
scanning, auto-connect and "make sure we are connected" live in one place.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from soslamp.integrations.ble_led_device import (
    DeviceType,
    BleDeviceInfo,
    BleLedController,
    EFFECTS,
    WEEK_DAYS,
    discover_devices,
    detect_device_type,
)


@dataclass
class BleManagerState:
    """Snapshot the GUI can read: what we've seen and what we're connected to."""

    devices: list[BleDeviceInfo] = field(default_factory=list)
    connected: Optional[BleDeviceInfo] = None


class BleControllerManager:
    """Tracks discovered BLE devices and the active controller."""

    def __init__(self) -> None:
        self.state = BleManagerState()
        self._controller: Optional[BleLedController] = None
        # Lock so connect/disconnect can't run at the same time and swap the controller twice.
        self._lock = asyncio.Lock()
        self._scanner: BleakScanner | None = None
        # address -> device info, filled in by the background scanner callback.
        self._device_cache: dict[str, BleDeviceInfo] = {}
        # Remember the loop so the scanner callback (other thread) can hop back onto it.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._auto_connect_enabled = True
        self._auto_connect_task: Optional[asyncio.Task] = None

    async def refresh(self, auto_connect: bool = False) -> list[BleDeviceInfo]:
        """Start scanning (once) and return the devices found so far."""
        self._auto_connect_enabled = auto_connect
        await self._ensure_scanner()
        if not self._device_cache:
            # Nothing from the live scanner yet, so do a one-off scan instead.
            devices = await discover_devices()
            for dev in devices:
                self._cache_device(dev)
        else:
            devices = list(self._device_cache.values())
        self.state.devices = devices
        if auto_connect and devices and not self._controller:
            await self._connect_first_available()
        return self.state.devices

    async def connect(self, address: str) -> BleLedController:
        """Connect to a device by address, replacing any current connection."""
        info = next((d for d in self.state.devices if d.address == address), None)
        if not info:
            # One more scan in case the list is stale.
            self.state.devices = await discover_devices()
            info = next((d for d in self.state.devices if d.address == address), None)
        if not info:
            # Fall back to a bare entry so we still attempt the BLE connection.
            info = BleDeviceInfo(address=address, name=address)
        controller = BleLedController(info.address, info.name)
        await controller.connect()
        async with self._lock:
            # Drop the old controller (if any) only after the new one connected successfully.
            if self._controller and self._controller is not controller:
                await self._controller.disconnect()
            self._controller = controller
            self.state.connected = info
        return controller

    async def disconnect(self) -> None:
        async with self._lock:
            if self._controller:
                await self._controller.disconnect()
            self._controller = None
            self.state.connected = None

    def controller(self) -> Optional[BleLedController]:
        """The active controller, or None when disconnected."""
        return self._controller

    async def ensure_controller(self) -> BleLedController:
        """Return a connected controller, connecting to the first device if needed.

        All the power/color/effect helpers below call this first, so the user can
        press a button before manually connecting and it still works.
        """
        ctrl = self._controller
        if not ctrl:
            if not self.state.devices:
                await self.refresh(auto_connect=True)
            if self.state.devices:
                await self.connect(self.state.devices[0].address)
                ctrl = self._controller
        if not ctrl:
            raise RuntimeError("Connect to a MELK/ELK device first.")
        return ctrl

    # The methods below are thin wrappers: get a controller, then forward the call.
    async def power_on(self) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.power_on()

    async def power_off(self) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.power_off()

    async def set_color(self, rgb: tuple[int, int, int]) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.set_color(*rgb)

    async def set_brightness(self, brightness: int) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.set_brightness(brightness)

    async def set_color_temp(self, kelvin: int) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.set_color_temperature(kelvin)

    async def apply_effect(self, effect_code: int, speed: Optional[int] = None) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.set_effect(effect_code, speed=speed)

    async def set_effect_speed_value(self, speed: int) -> None:
        ctrl = await self.ensure_controller()
        await ctrl.set_effect_speed(speed)

    async def set_schedule(
        self,
        when: str,
        days_mask: int,
        hour: int,
        minute: int,
        enabled: bool,
    ) -> None:
        ctrl = await self.ensure_controller()
        # "on" and "off" schedules are separate commands on the device.
        if when == "on":
            await ctrl.set_schedule_on(days_mask, hour, minute, enabled)
        else:
            await ctrl.set_schedule_off(days_mask, hour, minute, enabled)

    async def _ensure_scanner(self) -> None:
        """Create the continuous BLE scanner the first time it's needed."""
        if self._scanner:
            return
        if not self._loop:
            self._loop = asyncio.get_running_loop()
        self._scanner = BleakScanner(detection_callback=self._handle_detection)
        await self._scanner.start()

    def _handle_detection(self, device: BLEDevice, advertisement_data: AdvertisementData) -> None:
        """Called by bleak for every advertisement; keep only MELK/ELK-looking names."""
        dtype = detect_device_type(device.name or "")
        if dtype == DeviceType.UNKNOWN:
            return
        # Different bleak versions keep the signal strength in different places.
        rssi = getattr(device, "rssi", None)
        if rssi is None:
            rssi = getattr(advertisement_data, "rssi", None)
        info = BleDeviceInfo(
            address=str(device.address),
            name=device.name or dtype.value,
            rssi=rssi,
        )
        # This callback may run off the asyncio thread, so hand the work back safely.
        if self._loop:
            self._loop.call_soon_threadsafe(self._cache_and_maybe_connect, info)

    def _cache_and_maybe_connect(self, info: BleDeviceInfo) -> None:
        self._cache_device(info)
        self.state.devices = list(self._device_cache.values())
        # Auto-connect only once: not already connected and no connect attempt running.
        if (
            self._auto_connect_enabled
            and self._controller is None
            and self._auto_connect_task is None
            and self.state.devices
        ):
            address = self.state.devices[0].address
            self._auto_connect_task = asyncio.create_task(self._connect_first_available(address))

    def _cache_device(self, info: BleDeviceInfo) -> None:
        self._device_cache[info.address] = info
        self.state.devices = list(self._device_cache.values())

    async def _connect_first_available(self, preferred: Optional[str] = None) -> None:
        # Guard against an empty device list.
        try:
            address = preferred or self.state.devices[0].address
        except IndexError:
            self._auto_connect_task = None
            return
        try:
            await self.connect(address)
        finally:
            # Always clear the flag, even if connect() raised, so we can retry later.
            self._auto_connect_task = None


__all__ = [
    "BleControllerManager",
    "BleDeviceInfo",
    "EFFECTS",
    "WEEK_DAYS",
]
