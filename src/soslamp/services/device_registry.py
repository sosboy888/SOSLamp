"""Maintains discovered MELK devices and ready-to-use clients.

This is for the Wi-Fi (UDP/TCP) MELK devices. The Bluetooth path goes through
services/ble_manager.py instead.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from soslamp.integrations.melk_device import MelkDevice, MelkDeviceClient, MelkDiscoveryClient


@dataclass
class RegistryState:
    """Plain data holder for the registry."""

    devices: list[MelkDevice] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)  # mac addresses


class DeviceRegistry:
    """Keeps one client per device and lets us send commands to a selection of them."""

    def __init__(self) -> None:
        self.state = RegistryState()
        # mac address -> client, so we reuse connections instead of making new ones.
        self._clients: Dict[str, MelkDeviceClient] = {}
        self._lock = asyncio.Lock()
        self.discovery = MelkDiscoveryClient()

    async def refresh(self) -> list[MelkDevice]:
        """Re-run discovery and make sure every found device has a client."""
        devices = await self.discovery.discover()
        async with self._lock:
            self.state.devices = devices
            for dev in devices:
                if dev.mac not in self._clients:
                    self._clients[dev.mac] = MelkDeviceClient(dev)
        return devices

    def list_devices(self) -> list[MelkDevice]:
        return list(self.state.devices)

    def set_selected(self, macs: Iterable[str]) -> None:
        # dict.fromkeys removes duplicates but keeps the original order.
        self.state.selected = list(dict.fromkeys(macs))

    def selected_clients(self) -> list[MelkDeviceClient]:
        # Skip any selected MAC we don't have a client for (e.g. device disappeared).
        return [self._clients[mac] for mac in self.state.selected if mac in self._clients]

    def ensure_client(self, device: MelkDevice) -> MelkDeviceClient:
        client = self._clients.get(device.mac)
        if not client:
            client = MelkDeviceClient(device)
            self._clients[device.mac] = client
        return client

    async def broadcast_color(self, rgb: tuple[int, int, int], brightness: int = 100) -> None:
        """Send the same colour to every selected device at once."""
        tasks = [client.set_static_color(*rgb, brightness=brightness) for client in self.selected_clients()]
        # return_exceptions=True so one offline device doesn't cancel the others.
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_scene(self, scene_id: str) -> None:
        """Apply a named scene to every selected device."""
        tasks = [client.set_scene(scene_id) for client in self.selected_clients()]
        await asyncio.gather(*tasks, return_exceptions=True)
