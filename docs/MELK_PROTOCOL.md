# MELK UDP Protocol Cheatsheet

The bundled driver assumes a Tuya-style MELK firmware that exposes a UDP socket (default `48899`).

## Discovery

1. Broadcast `HF-A11ASSISTHREAD` to `255.255.255.255:48899`.
2. Devices respond with comma-separated strings, e.g. `ACCF23ABCDEF,ACCF23ABCDEF,HF-LPB100`.
3. Use the sender IP as the command endpoint.

If your firmware answers with JSON instead, adjust `MelkDiscoveryClient` to parse that payload.

## Static color command

```json
{
  "msg": "color",
  "data": {"r": 255, "g": 128, "b": 0, "brightness": 80}
}
```

## Scene command

Select from firmware scenes; IDs vary by vendor. Update `SOSLampWindow.effect_map` to match.

```json
{
  "msg": "scene",
  "data": {"scene": "scene_aurora", "speed": 50, "brightness": 100}
}
```

## Music / streaming command

```json
{
  "msg": "music",
  "data": [
    {"r": 32, "g": 0, "b": 128},
    {"r": 40, "g": 5, "b": 150}
  ]
}
```

Devices usually expect 30–60 pixels. Adjust `MusicReactiveWorker(leds=…)` to match your strips.

## Next steps

- Map actual scene identifiers by sniffing traffic from the official MELK mobile app using Wireshark.
- If your firmware requires TCP instead of UDP, swap the socket in `MelkDeviceClient.send_command`.
- Wrap the driver with authentication if your controllers implement passworded JSON (rare on MELK).
