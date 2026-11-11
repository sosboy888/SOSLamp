"""Utility for comparing LotusLamp X pcap captures to SOSLamp payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scapy.all import rdpcap
except ImportError as exc:  # pragma: no cover - scapy optional
    raise SystemExit("Install scapy (pip install scapy) to parse pcap files") from exc

from soslamp.integrations.melk_device import MelkDeviceClient, MelkDevice


IGNORED_FIELDS = {"seq", "timestamp", "nonces"}


def normalize_json(payload: str) -> dict:
    data = json.loads(payload)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in IGNORED_FIELDS}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap", type=Path, help="Path to Wireshark/pcap capture")
    args = parser.parse_args()

    packets = rdpcap(str(args.pcap))
    seen: set[str] = set()
    print(f"Loaded {len(packets)} packets")
    fake_device = MelkDevice(name="pcap", ip="0.0.0.0", mac="00:00:00:00:00:00")
    client = MelkDeviceClient(fake_device)

    def compare(payload: str) -> None:
        try:
            normalized = json.dumps(normalize_json(payload), sort_keys=True)
        except json.JSONDecodeError:
            print(f"Binary payload (len={len(payload)}): {payload[:32]}…")
            return
        if normalized in seen:
            return
        seen.add(normalized)
        print("\n=== Payload ===")
        print(payload)
        print("Matches SOSLamp builder?", _matches_builder(normalized, client))

    for pkt in packets:
        raw = bytes(pkt.payload.payload.payload)
        if not raw:
            continue
        try:
            payload = raw.decode()
        except UnicodeDecodeError:
            continue
        if "msg" in payload:
            compare(payload)


def _matches_builder(payload: str, client: MelkDeviceClient) -> bool:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False
    msg = data.get("msg")
    if msg == "color":
        built = {
            "msg": "color",
            "data": {
                "r": data["data"].get("r", 0),
                "g": data["data"].get("g", 0),
                "b": data["data"].get("b", 0),
                "brightness": data["data"].get("brightness", 100),
            },
        }
    elif msg == "scene":
        built = {
            "msg": "scene",
            "data": {
                "scene": data["data"].get("scene", ""),
                "speed": data["data"].get("speed", 50),
                "brightness": data["data"].get("brightness", 100),
            },
        }
    elif msg == "music":
        built = {
            "msg": "music",
            "data": data.get("data", []),
        }
    else:
        return False
    return built == data


if __name__ == "__main__":
    main()
