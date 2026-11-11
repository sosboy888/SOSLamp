# Bridging SOSLamp with the official LotusLamp X apps

Even though we don't have vendor documentation for MELK controllers, the LotusLamp X mobile apps already speak to them. You can piggyback on that protocol to confirm the payloads SOSLamp sends.

## 1. Capture the LAN traffic LotusLamp X produces

1. Put your phone and computer on the same Wi‑Fi network.
2. Run Wireshark or `tcpdump` on the computer, start a capture on the Wi‑Fi interface.
3. Open LotusLamp X on the phone and change colors/patterns/music modes for the MELK strips.
4. Filter the capture by `udp.port == 48899` (or `udp.port == 5987`/`7000` for some Tuya clones).
5. Save representative frames to `captures/lotuslamp_x_<date>.pcapng` for later reference.

You should see the `HF-A11ASSISTHREAD` discovery frame and small JSON payloads. These match the helpers in `soslamp/integrations/melk_device.py`.

## 2. Diff payloads vs. SOSLamp

Use the provided script to replay LotusLamp X frames and compare them with our payloads:

```bash
python -m soslamp.tools.pcap_compare captures/lotuslamp_x_2024-05-02.pcapng
```

This prints every unique JSON blob found in the capture and marks whether SOSLamp knows how to build the same structure.

## 3. Update presets/scene IDs

LotusLamp X exposes friendly names (“Lotus”, “Cycle”, “Rhythm”). Map them to the string IDs in the traffic capture and update `SOSLampWindow.effect_map`. Each tuple is `UI Name -> firmware scene id`.

## 4. Derive music streaming format

Play a music reactive preset in LotusLamp X; note the frame size (bytes) and cadence. Most MELK controllers expect roughly 30–60 RGB entries, but some use HSV or include headers:

- If you see objects shaped like `{"index": 0, "r": 255, ...}`, mirror that in `MelkDeviceClient.push_music_frame`.
- If the payload is binary (not JSON), dump the hex via Wireshark and augment `push_music_frame` to build bytes instead of JSON.

## 5. Verify against real hardware

Run `python -m soslamp.app`, select devices, and try the same presets. If a device ignores commands that worked from the phone, its firmware expects an extra field (
for example `"type": 1`). Add that to the payload builder and test again.

### Tips

- For HTTPS traffic (cloud features), use `mitmproxy` or `Proxyman` with the LotusLamp X trust certificate installed. Most MELK strips still take commands over plain UDP when both phone and device share a LAN, so you can avoid the cloud entirely.
- Some controllers require a login token; LotusLamp X sends `{"msg":"login","token":"..."}` before control messages. Use the capture to determine whether that step is necessary for your strips.

## Deliverables when reverse engineering

1. The `.pcapng` file for reference.
2. Updated `patterns` dict and any new payload helpers.
3. Notes about authentication, music frame size, and speed/brightness ranges.

Commit these assets (except raw captures) so the project benefits from the decoded protocol without shipping potentially private traffic.
