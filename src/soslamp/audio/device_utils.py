"""Helpers to inspect available audio capture devices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import sounddevice as sd


@dataclass(slots=True)
class AudioDeviceInfo:
    """Small summary of one audio device so the GUI doesn't need to touch sounddevice."""

    # index is None for the virtual "System Default Output" entry.
    index: int | None
    name: str
    hostapi: str
    supports_loopback: bool
    max_input_channels: int = 0
    max_output_channels: int = 0

    def label(self) -> str:
        """Text shown in the Audio Source dropdown."""
        suffix = " (Loopback)" if self.supports_loopback else ""
        return f"{self.name}{suffix} [{self.hostapi}]"


def list_input_devices() -> list[AudioDeviceInfo]:
    """Return every device we can record from, including loopback options."""
    # query_devices can fail if there's no audio driver, so fall back to an empty list.
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return []

    infos: list[AudioDeviceInfo] = []
    for idx, device in enumerate(devices):
        host_idx = device.get("hostapi", 0)
        host = hostapis[host_idx]["name"] if 0 <= host_idx < len(hostapis) else "Unknown"
        max_input = int(device.get("max_input_channels", 0) or 0)
        max_output = int(device.get("max_output_channels", 0) or 0)
        name = device.get("name", f"Device {idx}")
        supports_loopback = False
        include = False
        if max_input > 0:
            # Normal microphone / line-in.
            include = True
        elif "wasapi" in host.lower() and max_output > 0:
            # WASAPI lets us record an *output* device ("loopback"), which is how we
            # capture whatever the PC is playing.
            include = True
            supports_loopback = True
        if not include:
            continue
        infos.append(
            AudioDeviceInfo(
                index=idx,
                name=name,
                hostapi=host,
                supports_loopback=supports_loopback,
                max_input_channels=max_input,
                max_output_channels=max_output,
            )
        )
    # Add a convenience default output loopback option for WASAPI users.
    infos.append(
        AudioDeviceInfo(
            index=None,
            name="System Default Output",
            hostapi="Default",
            supports_loopback=True,
            max_input_channels=0,
            max_output_channels=2,
        )
    )
    return infos


__all__ = ["AudioDeviceInfo", "list_input_devices"]
