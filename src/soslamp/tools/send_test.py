"""Manual test helper to send MELK commands from Windows."""

from __future__ import annotations

import argparse
import asyncio
import json

from soslamp.integrations.melk_device import MelkDevice, MelkDeviceClient


async def send_color(ip: str, mac: str, r: int, g: int, b: int, brightness: int) -> None:
    device = MelkDevice(name=mac, ip=ip, mac=mac)
    client = MelkDeviceClient(device)
    await client.set_static_color(r, g, b, brightness=brightness)


async def send_scene(ip: str, mac: str, scene: str) -> None:
    device = MelkDevice(name=mac, ip=ip, mac=mac)
    client = MelkDeviceClient(device)
    await client.set_scene(scene)


async def send_raw(ip: str, mac: str, payload: str) -> None:
    device = MelkDevice(name=mac, ip=ip, mac=mac)
    client = MelkDeviceClient(device)
    data = json.loads(payload)
    await client.send_command(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ip", help="Device IP (see LotusLamp X capture or Windows adapter)")
    parser.add_argument("--mac", default="00:00:00:00:00:00", help="Fake mac used for registry matching")
    sub = parser.add_subparsers(dest="cmd", required=True)

    color = sub.add_parser("color", help="Send a static RGB color")
    color.add_argument("--r", type=int, required=True)
    color.add_argument("--g", type=int, required=True)
    color.add_argument("--b", type=int, required=True)
    color.add_argument("--brightness", type=int, default=100)

    scene = sub.add_parser("scene", help="Send a scene id")
    scene.add_argument("--id", required=True, dest="scene_id")

    raw = sub.add_parser("raw", help="Send an arbitrary JSON payload")
    raw.add_argument("--json", required=True)

    args = parser.parse_args()
    if args.cmd == "color":
        coro = send_color(args.ip, args.mac, args.r, args.g, args.b, args.brightness)
    elif args.cmd == "scene":
        coro = send_scene(args.ip, args.mac, args.scene_id)
    else:
        coro = send_raw(args.ip, args.mac, args.json)
    asyncio.run(coro)


if __name__ == "__main__":
    main()
