"""Maintains discovered MELK devices and ready-to-use clients."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from soslamp.integrations.melk_device import MelkDevice, MelkDeviceClient, MelkDiscoveryClient


@dataclass
class RegistryState:
    devices: list[MelkDevice] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)  # mac addresses


class DeviceRegistry:
    def __init__(self) -> None:
        self.state = RegistryState()
        self._clients: Dict[str, MelkDeviceClient] = {}
        self._lock = asyncio.Lock()
        self.discovery = MelkDiscoveryClient()

    async def refresh(self) -> list[MelkDevice]:
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
        self.state.selected = list(dict.fromkeys(macs))

    def selected_clients(self) -> list[MelkDeviceClient]:
        return [self._clients[mac] for mac in self.state.selected if mac in self._clients]

    def ensure_client(self, device: MelkDevice) -> MelkDeviceClient:
        client = self._clients.get(device.mac)
        if not client:
            client = MelkDeviceClient(device)
            self._clients[device.mac] = client
        return client

    async def broadcast_color(self, rgb: tuple[int, int, int], brightness: int = 100) -> None:
        tasks = [client.set_static_color(*rgb, brightness=brightness) for client in self.selected_clients()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_scene(self, scene_id: str) -> None:
        tasks = [client.set_scene(scene_id) for client in self.selected_clients()]
        await asyncio.gather(*tasks, return_exceptions=True)
