"""Bluetooth discovery helpers for MELK controllers."""

from __future__ import annotations

import asyncio
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BluetoothDeviceInfo:
    name: str
    address: str
    connected: bool = False
    rssi: Optional[int] = None

    def label(self) -> str:
        status = "connected" if self.connected else "nearby"
        return f"{self.name or 'Unknown'} ({self.address}) - {status}"


class BluetoothDiscoveryService:
    """Lists MELK BLE devices, including already connected controllers."""

    def __init__(self, prefix: str = "MELK", timeout: float = 4.0) -> None:
        self.prefix = prefix
        self.timeout = timeout

    async def discover(self) -> list[BluetoothDeviceInfo]:
        bleak_task = asyncio.create_task(self._discover_bleak())
        cli_task = asyncio.to_thread(self._connected_from_cli)
        bleak_devices, cli_devices = await asyncio.gather(
            bleak_task, cli_task, return_exceptions=False
        )
        merged: dict[str, BluetoothDeviceInfo] = {}
        for device in [*bleak_devices, *cli_devices]:
            if self.prefix and device.name and not device.name.upper().startswith(self.prefix.upper()):
                continue
            merged[device.address] = device
        return list(merged.values())

    async def _discover_bleak(self) -> list[BluetoothDeviceInfo]:
        try:
            from bleak import BleakScanner
        except ImportError:
            return []
        devices = await BleakScanner.discover(timeout=self.timeout)
        infos: list[BluetoothDeviceInfo] = []
        for device in devices:
            rssi = getattr(device, "rssi", None)
            if rssi is None:
                details = getattr(device, "details", None)
                if isinstance(details, dict):
                    rssi = details.get("RSSI")
            metadata = getattr(device, "metadata", {}) or {}
            infos.append(
                BluetoothDeviceInfo(
                    name=device.name or device.address,
                    address=device.address,
                    connected=bool(metadata.get("connected")),
                    rssi=rssi,
                )
            )
        return infos

    def _connected_from_cli(self) -> list[BluetoothDeviceInfo]:
        system = platform.system().lower()
        if "linux" in system:
            return self._from_bluetoothctl()
        if "darwin" in system:
            return self._from_macos()
        if "windows" in system:
            return self._from_windows()
        return []

    def _from_bluetoothctl(self) -> list[BluetoothDeviceInfo]:
        try:
            proc = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        infos: list[BluetoothDeviceInfo] = []
        for line in proc.stdout.strip().splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) >= 3:
                _, address, name = parts
                infos.append(BluetoothDeviceInfo(name=name, address=address, connected=True))
        return infos

    def _from_macos(self) -> list[BluetoothDeviceInfo]:
        cmd = [
            "system_profiler",
            "SPBluetoothDataType",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return []
        infos: list[BluetoothDeviceInfo] = []
        current_name: Optional[str] = None
        current_addr: Optional[str] = None
        connected = False
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":") and "Bluetooth" not in line:
                current_name = line.rstrip(":")
                current_addr = None
                connected = False
            elif line.startswith("Address:"):
                current_addr = line.split("Address:", 1)[1].strip()
            elif line.startswith("Connected:"):
                connected = line.split("Connected:", 1)[1].strip().lower() == "yes"
                if connected and current_name and current_addr:
                    infos.append(BluetoothDeviceInfo(name=current_name, address=current_addr, connected=True))
        return infos

    def _from_windows(self) -> list[BluetoothDeviceInfo]:
        # Query paired devices via PowerShell; requires Windows 10+.
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-PnpDevice -Class Bluetooth -Status OK | Select-Object -Property FriendlyName,InstanceId",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return []
        infos: list[BluetoothDeviceInfo] = []
        for line in proc.stdout.splitlines():
            if not line.strip() or "FriendlyName" in line:
                continue
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            name, instance = parts
            addr = instance.split("_")[-1].replace("&", ":")
            infos.append(BluetoothDeviceInfo(name=name, address=addr, connected=True))
        return infos
