"""Driver helpers for MELK Wi-Fi LED controllers.

These talk JSON over UDP (or TCP) to the controller's IP address. The Bluetooth
version lives in ble_led_device.py.
"""

from __future__ import annotations

import asyncio
import json
import socket
import logging
from dataclasses import dataclass
from typing import Iterable

from soslamp.utils.config import (
    COMMAND_THROTTLE,
    DEFAULT_MELK_PORT,
    DISCOVERY_BROADCAST,
    DISCOVERY_TIMEOUT,
    MELK_TCP_SUFFIX,
    MELK_TCP_TIMEOUT,
    MELK_TRANSPORT,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MelkDevice:
    """A Wi-Fi controller found on the local network."""

    name: str
    ip: str
    mac: str
    port: int = DEFAULT_MELK_PORT

    def label(self) -> str:
        return f"{self.name} ({self.ip})"


class MelkDiscoveryClient:
    """Discovers MELK controllers via UDP broadcast."""

    def __init__(self, port: int = DEFAULT_MELK_PORT) -> None:
        self.port = port

    async def discover(self, timeout: float = DISCOVERY_TIMEOUT) -> list[MelkDevice]:
        loop = asyncio.get_running_loop()
        devices: list[MelkDevice] = []

        # Sockets block, so the real work happens in this sync function run on a thread pool.
        def _discover_sync() -> list[MelkDevice]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Broadcasting needs SO_BROADCAST turned on.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            sock.sendto(DISCOVERY_BROADCAST.encode(), ("255.255.255.255", self.port))
            found: list[MelkDevice] = []
            try:
                # Keep reading replies until the socket times out (= no more devices answered).
                while True:
                    data, addr = sock.recvfrom(1024)
                    payload = data.decode(errors="ignore")
                    # Reply format is comma separated: name,mac,... (extra fields ignored).
                    parts = payload.split(",")
                    if len(parts) >= 3:
                        dev = MelkDevice(name=parts[0], ip=addr[0], mac=parts[1])
                        found.append(dev)
            except (socket.timeout, OSError):
                pass
            finally:
                sock.close()
            return found

        # Run the blocking function without freezing the event loop.
        devices = await loop.run_in_executor(None, _discover_sync)
        return devices


class MelkDeviceClient:
    """High-level controller for MELK devices."""

    def __init__(self, device: MelkDevice) -> None:
        self.device = device
        self._lock = asyncio.Lock()
        self.transport = MELK_TRANSPORT

    async def send_command(self, payload: dict) -> None:
        """Serialise a dict to JSON and send it over the configured transport."""
        loop = asyncio.get_running_loop()
        message = json.dumps(payload).encode()

        def _send() -> None:
            try:
                if self.transport == "tcp":
                    self._send_tcp(message)
                else:
                    self._send_udp(message)
            except OSError as exc:
                # Network errors are logged instead of raised so the UI doesn't crash.
                logger.warning("Failed to send MELK command to %s: %s", self.device.ip, exc)

        # Lock + small sleep = commands go out one at a time and not too quickly.
        async with self._lock:
            await loop.run_in_executor(None, _send)
            await asyncio.sleep(COMMAND_THROTTLE)

    def _send_udp(self, message: bytes) -> None:
        # UDP: fire-and-forget, no connection needed.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.5)
        try:
            sock.sendto(message, (self.device.ip, self.device.port))
        finally:
            sock.close()

    def _send_tcp(self, message: bytes) -> None:
        # TCP: connect, send, then close. The optional suffix is a terminator some firmwares want.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(MELK_TCP_TIMEOUT)
        try:
            sock.connect((self.device.ip, self.device.port))
            payload = message + MELK_TCP_SUFFIX.encode()
            sock.sendall(payload)
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    async def set_static_color(self, r: int, g: int, b: int, brightness: int = 100) -> None:
        payload = {
            "msg": "color",
            "data": {
                "r": r,
                "g": g,
                "b": b,
                "brightness": brightness,
            },
        }
        await self.send_command(payload)

    async def set_scene(self, scene_id: str, speed: int = 50, brightness: int = 100) -> None:
        payload = {
            "msg": "scene",
            "data": {
                "scene": scene_id,
                "speed": speed,
                "brightness": brightness,
            },
        }
        await self.send_command(payload)

    async def push_music_frame(self, colors: Iterable[tuple[int, int, int]]) -> None:
        """Send one frame of per-segment colours for music mode."""
        payload = {
            "msg": "music",
            "data": [
                {"r": r, "g": g, "b": b}
                for r, g, b in colors
            ],
        }
        await self.send_command(payload)
