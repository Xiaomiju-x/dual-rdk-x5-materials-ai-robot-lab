#!/usr/bin/env python3
"""Record algorithm overlay frames for Video 2.

The recorder is intentionally dependency-light. It writes PNG frames using only
the Python standard library, then video2_stop_capture.sh can turn them into MP4
with ffmpeg if available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "<": ["00001", "00010", "00100", "01000", "00100", "00010", "00001"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    "#": ["01010", "11111", "01010", "01010", "11111", "01010", "00000"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
}


def chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, rgb: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 1))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = (18, 22, 26)) -> None:
        self.w = width
        self.h = height
        self.buf = bytearray(bg * (width * height))

    def set_px(self, x: int, y: int, c: tuple[int, int, int]) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.buf[i : i + 3] = bytes(c)

    def rect(self, x: int, y: int, w: int, h: int, c: tuple[int, int, int], fill: bool = True) -> None:
        if fill:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(self.w, x + w), min(self.h, y + h)
            row = bytes(c) * max(0, x1 - x0)
            for yy in range(y0, y1):
                i = (yy * self.w + x0) * 3
                self.buf[i : i + len(row)] = row
        else:
            self.line(x, y, x + w - 1, y, c)
            self.line(x, y + h - 1, x + w - 1, y + h - 1, c)
            self.line(x, y, x, y + h - 1, c)
            self.line(x + w - 1, y, x + w - 1, y + h - 1, c)

    def line(self, x0: int, y0: int, x1: int, y1: int, c: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_px(x0, y0, c)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, r: int, c: tuple[int, int, int], fill: bool = True) -> None:
        if fill:
            rr = r * r
            for y in range(cy - r, cy + r + 1):
                for x in range(cx - r, cx + r + 1):
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                        self.set_px(x, y, c)
        else:
            steps = max(16, r * 8)
            last = None
            for i in range(steps + 1):
                a = 2 * math.pi * i / steps
                p = (int(cx + r * math.cos(a)), int(cy + r * math.sin(a)))
                if last:
                    self.line(last[0], last[1], p[0], p[1], c)
                last = p

    def text(self, x: int, y: int, s: str, c: tuple[int, int, int] = (230, 236, 240), scale: int = 2) -> None:
        xx = x
        for ch in s.upper():
            glyph = FONT.get(ch, FONT.get(" ", []))
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(xx + gx * scale, y + gy * scale, scale, scale, c, True)
            xx += 6 * scale

    def wrap_text(self, x: int, y: int, width_chars: int, lines: list[str], c=(210, 218, 224), scale: int = 2) -> int:
        yy = y
        for line in lines:
            words = str(line).replace("\n", " ").split()
            cur = ""
            for w in words:
                if len(cur) + len(w) + 1 > width_chars:
                    self.text(x, yy, cur, c, scale)
                    yy += 8 * scale + 4
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                self.text(x, yy, cur, c, scale)
                yy += 8 * scale + 4
        return yy


@dataclass
class State:
    start_ts: float = field(default_factory=time.time)
    map_msg: Any = None
    scan_msg: Any = None
    depth_scan_msg: Any = None
    odom_msg: Any = None
    vision_bev_msg: Any = None
    traj: list[tuple[float, float]] = field(default_factory=list)
    lab_traj_json: dict[str, Any] = field(default_factory=dict)
    safety_json: dict[str, Any] = field(default_factory=dict)
    anomaly: float | None = None
    risk: float | None = None
    vision_objects: dict[str, Any] = field(default_factory=dict)
    vision_risk: float | None = None
    ai_flybrain: dict[str, Any] = field(default_factory=dict)
    ai_vision: dict[str, Any] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def safe_json(s: str) -> dict[str, Any]:
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {"value": out}
    except Exception:
        return {"raw": str(s)[:240]}


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 2.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class AiPoller(threading.Thread):
    def __init__(self, state: State, ai_url: str, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.ai_url = ai_url.rstrip("/")
        self.stop_event = stop_event
        self.last_fly = 0.0
        self.last_vision = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()
            try:
                if now - self.last_vision > 5.5:
                    out = http_json(self.ai_url + "/api/lab_fsd_vision_bev", {"capture": True, "include_grid": True}, 3.0)
                    with self.state.lock:
                        self.state.ai_vision = out
                    self.last_vision = now
                if now - self.last_fly > 14.0:
                    payload = {
                        "formula": "Y3Al5O12",
                        "dopant": {"symbol": "Cr3+", "site": "Al", "pct": 1.0},
                        "host_hint": "YAG",
                    }
                    out = http_json(self.ai_url + "/api/flybrain_superstack", payload, 4.0)
                    with self.state.lock:
                        self.state.ai_flybrain = out
                    self.last_fly = now
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                with self.state.lock:
                    msg = f"AI poll: {str(exc)[:120]}"
                    if not self.state.errors or self.state.errors[-1] != msg:
                        self.state.errors.append(msg)
                        self.state.errors = self.state.errors[-6:]
                time.sleep(1.0)
            self.stop_event.wait(0.3)


def panel(c: Canvas, x: int, y: int, w: int, h: int, title: str) -> tuple[int, int, int, int]:
    c.rect(x, y, w, h, (28, 34, 40), True)
    c.rect(x, y, w, h, (86, 102, 116), False)
    c.rect(x, y, w, 28, (40, 50, 58), True)
    c.text(x + 10, y + 8, title, (240, 245, 248), 2)
    return x + 10, y + 40, w - 20, h - 50


def draw_bar(c: Canvas, x: int, y: int, w: int, label: str, value: float, color: tuple[int, int, int]) -> None:
    value = max(0.0, min(1.0, float(value)))
    c.text(x, y, f"{label} {value:.2f}", (210, 218, 224), 2)
    c.rect(x, y + 20, w, 12, (55, 62, 68), True)
    c.rect(x, y + 20, int(w * value), 12, color, True)


def render_grid_msg(c: Canvas, msg: Any, x: int, y: int, w: int, h: int, unknown=(55, 59, 64)) -> None:
    if msg is None:
        c.wrap_text(x + 8, y + 8, 36, ["WAITING FOR OCCUPANCY GRID"], (160, 170, 178), 2)
        return
    mw = int(msg.info.width)
    mh = int(msg.info.height)
    data = list(msg.data)
    if mw <= 0 or mh <= 0 or len(data) < mw * mh:
        c.wrap_text(x + 8, y + 8, 36, ["BAD GRID SHAPE"], (230, 170, 90), 2)
        return
    for yy in range(h):
        sy = mh - 1 - int(yy * mh / h)
        for xx in range(w):
            sx = int(xx * mw / w)
            v = data[sy * mw + sx]
            if v < 0:
                col = unknown
            elif v > 65:
                col = (28, 32, 34)
            elif v > 25:
                col = (130, 142, 148)
            else:
                col = (230, 234, 232)
            c.set_px(x + xx, y + yy, col)


def render_scan(c: Canvas, msg: Any, x: int, y: int, w: int, h: int, color: tuple[int, int, int], max_range=4.5) -> int:
    if msg is None:
        return 0
    ranges = list(getattr(msg, "ranges", []) or [])
    if not ranges:
        return 0
    amin = float(getattr(msg, "angle_min", -math.pi))
    inc = float(getattr(msg, "angle_increment", 0.0) or 0.0)
    rmin = float(getattr(msg, "range_min", 0.02) or 0.02)
    rmax = min(float(getattr(msg, "range_max", max_range) or max_range), max_range)
    cx = x + w // 2
    cy = y + int(h * 0.72)
    scale = min(w, h) * 0.42 / max_range
    count = 0
    for i, r in enumerate(ranges[:: max(1, len(ranges) // 720)]):
        idx = i * max(1, len(ranges) // 720)
        try:
            rr = float(r)
        except Exception:
            continue
        if not (rmin <= rr <= rmax):
            continue
        a = amin + inc * idx
        px = int(cx + math.sin(a) * rr * scale)
        py = int(cy - math.cos(a) * rr * scale)
        c.circle(px, py, 2, color, True)
        count += 1
    c.circle(cx, cy, 6, (230, 80, 80), True)
    c.line(cx, cy, cx, cy - int(0.6 * scale), (250, 250, 250))
    return count


def render_slam(c: Canvas, state: State, x: int, y: int, w: int, h: int) -> None:
    render_grid_msg(c, state.map_msg, x, y, w, h)
    msg = state.map_msg
    if msg is not None and state.traj:
        try:
            mw = int(msg.info.width)
            mh = int(msg.info.height)
            res = float(msg.info.resolution)
            ox = float(msg.info.origin.position.x)
            oy = float(msg.info.origin.position.y)
            pts = []
            for tx, ty in state.traj[-400:]:
                gx = int((tx - ox) / res)
                gy = int((ty - oy) / res)
                px = x + int(gx * w / max(1, mw))
                py = y + h - int(gy * h / max(1, mh))
                pts.append((px, py))
            for a, b in zip(pts, pts[1:]):
                c.line(a[0], a[1], b[0], b[1], (230, 55, 55))
            if pts:
                c.circle(pts[-1][0], pts[-1][1], 6, (255, 60, 60), True)
        except Exception:
            pass
    c.text(x + 8, y + h - 24, f"MAP+LIDAR TRAJ PTS {len(state.traj)}", (30, 210, 230), 2)


def render_lab_fsd(c: Canvas, state: State, x: int, y: int, w: int, h: int) -> None:
    diag = state.lab_traj_json or {}
    safety = state.safety_json or {}
    risk = state.risk
    if risk is None:
        risk = float(diag.get("risk") or 0.0)
    conf = float(diag.get("shadow_confidence") or safety.get("confidence") or 0.0)
    anomaly = 0.0 if state.anomaly is None else float(state.anomaly)
    draw_bar(c, x, y, w - 20, "RISK", risk, (230, 85, 75))
    draw_bar(c, x, y + 48, w - 20, "SHADOW CONF", conf, (70, 190, 120))
    draw_bar(c, x, y + 96, w - 20, "BPU ANOMALY", anomaly, (250, 170, 60))
    lines = [
        "MODE " + str(diag.get("mode") or "WAITING"),
        "AUTH " + str(safety.get("authority") or safety.get("shadow_policy") or "NAV2/SAFETY"),
        "ASSIST " + str(safety.get("assist_allowed", False)),
        "TOPIC /LAB_FSD/SAFETY_GATE",
        "TOPIC /LAB_FSD/TRAJECTORY_SCORES",
    ]
    c.wrap_text(x, y + 150, 44, lines, (220, 226, 232), 2)


def render_ai(c: Canvas, state: State, x: int, y: int, w: int, h: int) -> None:
    out = state.ai_flybrain or {}
    fb = out.get("flybrain") if isinstance(out.get("flybrain"), dict) else {}
    verdict = str(fb.get("verdict") or "POLLING")
    confidence = float(fb.get("confidence") or 0.0)
    draw_bar(c, x, y, w - 20, "FLY-MB CONF", confidence, (80, 180, 245))
    lines = [
        "AI BRAIN DASHBOARD 8888",
        "MATERIAL Y3AL5O12:CR3+",
        "FLYHASH + KENYON SPARSE CODE",
        "MBON VERDICT " + verdict,
        "MODEL " + str(fb.get("model") or "LOCAL/BPU STACK"),
        "TRACE " + str(out.get("trace_id") or "-")[:22],
    ]
    profile = fb.get("connectome_profile") if isinstance(fb.get("connectome_profile"), dict) else {}
    if profile:
        lines.append("CONNECTOME " + str(profile.get("source") or "FLYWIRE/HEMIBRAIN")[:28])
    c.wrap_text(x, y + 48, 45, lines, (220, 226, 232), 2)


def render_vision(c: Canvas, state: State, x: int, y: int, w: int, h: int) -> None:
    out = state.ai_vision if state.ai_vision else {}
    grid_msg = state.vision_bev_msg
    grid_area_h = int(h * 0.58)
    grid_w = min(w - 10, grid_area_h)
    if out.get("grid"):
        fake = type("Grid", (), {})()
        fake.info = type("Info", (), {})()
        n = int(out.get("grid_size") or 48)
        fake.info.width = n
        fake.info.height = n
        fake.data = out.get("grid", [])
        render_grid_msg(c, fake, x, y, grid_w, grid_area_h, unknown=(20, 40, 48))
    elif grid_msg is not None:
        render_grid_msg(c, grid_msg, x, y, grid_w, grid_area_h, unknown=(20, 40, 48))
    else:
        c.rect(x, y, grid_w, grid_area_h, (18, 28, 34), True)
        c.rect(x, y, grid_w, grid_area_h, (48, 68, 78), False)
        c.text(x + 12, y + 18, "WAITING GRID", (135, 165, 178), 2)
    risk = state.vision_risk
    if risk is None:
        risk = float(out.get("risk_score") or 0.0)
    obj = state.vision_objects if state.vision_objects else out
    objects = obj.get("objects") if isinstance(obj.get("objects"), list) else []
    camera = out.get("camera") if isinstance(out.get("camera"), dict) else obj.get("camera", {})
    lines = [
        f"4K VISION-BEV RISK {risk:.2f}",
        "OBJECTS " + str(len(objects)),
        "CAMERA " + str(camera.get("shape") or camera.get("mode") or "IMX415"),
    ]
    for item in objects[:3]:
        if isinstance(item, dict):
            lines.append(str(item.get("label") or item.get("name") or item)[:34])
        else:
            lines.append(str(item)[:34])
    c.wrap_text(x + grid_w + 12, y + 4, 22, lines, (220, 226, 232), 2)


def render_depth(c: Canvas, state: State, x: int, y: int, w: int, h: int) -> None:
    c.rect(x, y, w, h, (20, 25, 30), True)
    for r in (1, 2, 3, 4):
        rr = int(min(w, h) * 0.42 * r / 4.5)
        c.circle(x + w // 2, y + int(h * 0.72), rr, (48, 58, 66), False)
    n1 = render_scan(c, state.scan_msg, x, y, w, h, (35, 210, 230))
    n2 = render_scan(c, state.depth_scan_msg, x, y, w, h, (250, 170, 55))
    c.text(x + 8, y + 8, f"LD14 {n1} PTS  DEPTH {n2} PTS", (230, 236, 240), 2)
    c.text(x + 8, y + 32, "CYAN=LIDAR  ORANGE=DEPTH CAMERA", (170, 185, 195), 2)


def render_timeline(c: Canvas, state: State, x: int, y: int, w: int, h: int, elapsed: float) -> None:
    lines = [
        "VIDEO2 SAFETY-OPERATOR DRIVE",
        "OPERATION: W FORWARD, A/D SMALL TURN 3-5S, W FORWARD",
        "LAB-FSD SHADOW DOES NOT DRIVE CHASSIS",
        "NAV STACK ONLINE: LIDAR + DEPTH + ODOM + BEV",
        "TOPIC COUNTS " + " ".join(f"{k}:{v}" for k, v in sorted(state.topic_counts.items())[:6]),
    ]
    c.wrap_text(x, y, 36, lines, (220, 226, 232), 2)
    c.rect(x, y + h - 42, w - 20, 12, (55, 62, 68), True)
    c.rect(x, y + h - 42, int((w - 20) * min(elapsed / 90.0, 1.0)), 12, (80, 200, 120), True)
    if state.errors:
        c.wrap_text(x, y + h - 26, 44, ["WARN " + state.errors[-1]], (240, 170, 80), 1)


def render_frame(state: State, frame_idx: int) -> Canvas:
    with state.lock:
        snapshot = State(
            start_ts=state.start_ts,
            map_msg=state.map_msg,
            scan_msg=state.scan_msg,
            depth_scan_msg=state.depth_scan_msg,
            odom_msg=state.odom_msg,
            vision_bev_msg=state.vision_bev_msg,
            traj=list(state.traj),
            lab_traj_json=dict(state.lab_traj_json),
            safety_json=dict(state.safety_json),
            anomaly=state.anomaly,
            risk=state.risk,
            vision_objects=dict(state.vision_objects),
            vision_risk=state.vision_risk,
            ai_flybrain=dict(state.ai_flybrain),
            ai_vision=dict(state.ai_vision),
            topic_counts=dict(state.topic_counts),
            errors=list(state.errors),
        )
    elapsed = time.time() - state.start_ts
    c = Canvas(1600, 900, (15, 18, 22))
    c.rect(0, 0, 1600, 54, (22, 29, 36), True)
    c.text(24, 18, f"VIDEO2 MULTI-SENSOR AI STACK  T+{elapsed:05.1f}s  FRAME {frame_idx:05d}", (245, 248, 250), 2)
    slots = [
        (20, 74, 500, 380, "SLAM MAP + TRAJECTORY"),
        (550, 74, 500, 380, "LIDAR + DEPTH SCAN"),
        (1080, 74, 500, 380, "LAB-FSD SHADOW"),
        (20, 490, 500, 380, "AI BRAIN FLY-MB"),
        (550, 490, 500, 380, "4K VISION-BEV"),
        (1080, 490, 500, 380, "RUN TIMELINE"),
    ]
    areas = [panel(c, *slot) for slot in slots]
    render_slam(c, snapshot, *areas[0])
    render_depth(c, snapshot, *areas[1])
    render_lab_fsd(c, snapshot, *areas[2])
    render_ai(c, snapshot, *areas[3])
    render_vision(c, snapshot, *areas[4])
    render_timeline(c, snapshot, *areas[5], elapsed)
    return c


def make_dummy_state() -> State:
    s = State()
    grid = type("Grid", (), {})()
    grid.info = type("Info", (), {})()
    grid.info.width = 120
    grid.info.height = 90
    grid.info.resolution = 0.05
    grid.info.origin = type("Origin", (), {})()
    grid.info.origin.position = type("Pos", (), {})()
    grid.info.origin.position.x = -3.0
    grid.info.origin.position.y = -2.25
    data = []
    for y in range(90):
        for x in range(120):
            occ = 0
            if x < 4 or y < 4 or x > 115 or y > 85 or (40 < x < 45 and 20 < y < 70):
                occ = 100
            elif (x + y) % 17 == 0:
                occ = -1
            data.append(occ)
    grid.data = data
    s.map_msg = grid
    s.traj = [(i * 0.015, 0.12 * math.sin(i / 15.0)) for i in range(150)]
    scan = type("Scan", (), {})()
    scan.angle_min = -math.pi
    scan.angle_increment = 2 * math.pi / 360
    scan.range_min = 0.05
    scan.range_max = 6.0
    scan.ranges = [2.2 + 0.4 * math.sin(i / 20.0) for i in range(360)]
    s.scan_msg = scan
    s.depth_scan_msg = scan
    s.lab_traj_json = {"mode": "lab_fsd_v2_probabilistic_shadow", "risk": 0.42, "shadow_confidence": 0.71}
    s.safety_json = {"authority": "nav2_mppi", "assist_allowed": False, "shadow_policy": "observe_only"}
    s.anomaly = 0.31
    s.risk = 0.42
    s.vision_risk = 0.66
    s.vision_objects = {"objects": [{"label": "plastic bottle"}, {"label": "lab shelf"}], "camera": {"shape": [2160, 3840]}}
    s.ai_flybrain = {"ok": True, "trace_id": "demo", "flybrain": {"verdict": "GO", "confidence": 0.73, "model": "FlyHash+MBON"}}
    s.ai_vision = {"ok": True, "risk_score": 0.66, "object_count": 2, "camera": {"shape": [2160, 3840]}}
    return s


def write_manifest(out_dir: Path, state: State, frame_count: int, fps: float, status: str) -> None:
    with state.lock:
        manifest = {
            "status": status,
            "started_at": state.start_ts,
            "ended_at": time.time(),
            "duration_s": round(time.time() - state.start_ts, 3),
            "frame_count": frame_count,
            "fps": fps,
            "topic_counts": state.topic_counts,
            "errors": state.errors[-20:],
            "outputs": {
                "grid_frames": "frames_grid/frame_%05d.png",
                "grid_video": "video2_data_grid.mp4",
                "panel_videos": [
                    "video2_slam_lidar.mp4",
                    "video2_depth_scan.mp4",
                    "video2_lab_fsd_shadow.mp4",
                    "video2_ai_brain.mp4",
                    "video2_vision_bev.mp4",
                ],
            },
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_self_test(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    frames = out_dir / "frames_grid"
    frames.mkdir(parents=True, exist_ok=True)
    state = make_dummy_state()
    for i in range(max(1, int(args.self_test_frames))):
        c = render_frame(state, i)
        write_png(frames / f"frame_{i:05d}.png", c.w, c.h, c.buf)
        time.sleep(0.02)
    write_manifest(out_dir, state, max(1, int(args.self_test_frames)), float(args.fps), "self_test")
    print(f"SELF_TEST_OK {out_dir}")
    return 0


def run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32, String
    from rclpy.node import Node

    out_dir = Path(args.out).expanduser().resolve()
    frames = out_dir / "frames_grid"
    frames.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file else out_dir / "STOP"
    fps = max(0.5, float(args.fps))
    state = State()
    stop_event = threading.Event()

    class RecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("video2_overlay_recorder")
            volatile_grid_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            transient_grid_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(OccupancyGrid, "/map", self.on_map, volatile_grid_qos)
            self.create_subscription(OccupancyGrid, "/map", self.on_map, transient_grid_qos)
            self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
            self.create_subscription(LaserScan, "/scan_depth", self.on_depth_scan, 10)
            self.create_subscription(Odometry, "/odom", self.on_odom, 10)
            self.create_subscription(OccupancyGrid, "/lab_fsd/vision_bev", self.on_vision_bev, volatile_grid_qos)
            self.create_subscription(OccupancyGrid, "/lab_fsd/vision_bev", self.on_vision_bev, transient_grid_qos)
            self.create_subscription(String, "/lab_fsd/trajectory_scores", self.on_traj, 10)
            self.create_subscription(String, "/lab_fsd/safety_gate", self.on_safety, 10)
            self.create_subscription(Float32, "/lab_fsd/anomaly_score", self.on_anomaly, 10)
            self.create_subscription(Float32, "/lab_fsd/risk", self.on_risk, 10)
            self.create_subscription(String, "/lab_fsd/vision_objects", self.on_vision_objects, 10)
            self.create_subscription(Float32, "/lab_fsd/vision_risk", self.on_vision_risk, 10)

        def bump(self, name: str) -> None:
            state.topic_counts[name] = state.topic_counts.get(name, 0) + 1

        def on_map(self, msg: Any) -> None:
            with state.lock:
                state.map_msg = msg
                self.bump("/map")

        def on_scan(self, msg: Any) -> None:
            with state.lock:
                state.scan_msg = msg
                self.bump("/scan")

        def on_depth_scan(self, msg: Any) -> None:
            with state.lock:
                state.depth_scan_msg = msg
                self.bump("/scan_depth")

        def on_odom(self, msg: Any) -> None:
            with state.lock:
                state.odom_msg = msg
                p = msg.pose.pose.position
                state.traj.append((float(p.x), float(p.y)))
                state.traj = state.traj[-1200:]
                self.bump("/odom")

        def on_vision_bev(self, msg: Any) -> None:
            with state.lock:
                state.vision_bev_msg = msg
                self.bump("/lab_fsd/vision_bev")

        def on_traj(self, msg: Any) -> None:
            with state.lock:
                state.lab_traj_json = safe_json(msg.data)
                self.bump("/lab_fsd/trajectory_scores")

        def on_safety(self, msg: Any) -> None:
            with state.lock:
                state.safety_json = safe_json(msg.data)
                self.bump("/lab_fsd/safety_gate")

        def on_anomaly(self, msg: Any) -> None:
            with state.lock:
                state.anomaly = float(msg.data)
                self.bump("/lab_fsd/anomaly_score")

        def on_risk(self, msg: Any) -> None:
            with state.lock:
                state.risk = float(msg.data)
                self.bump("/lab_fsd/risk")

        def on_vision_objects(self, msg: Any) -> None:
            with state.lock:
                state.vision_objects = safe_json(msg.data)
                self.bump("/lab_fsd/vision_objects")

        def on_vision_risk(self, msg: Any) -> None:
            with state.lock:
                state.vision_risk = float(msg.data)
                self.bump("/lab_fsd/vision_risk")

    def handle_signal(signum: int, frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    rclpy.init()
    node = RecorderNode()
    poller = AiPoller(state, args.ai_url, stop_event)
    poller.start()
    frame_idx = 0
    next_frame = time.monotonic()
    max_seconds = float(args.max_seconds or 0.0)
    try:
        while not stop_event.is_set():
            if stop_file.exists():
                break
            if max_seconds > 0 and (time.time() - state.start_ts) >= max_seconds:
                break
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now >= next_frame:
                c = render_frame(state, frame_idx)
                write_png(frames / f"frame_{frame_idx:05d}.png", c.w, c.h, c.buf)
                frame_idx += 1
                next_frame += 1.0 / fps
                if now - next_frame > 1.0:
                    next_frame = now + 1.0 / fps
    finally:
        stop_event.set()
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()
    write_manifest(out_dir, state, frame_idx, fps, "recorded")
    print(f"VIDEO2_RECORD_DONE {out_dir} frames={frame_idx}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--ai-url", default="http://192.0.2.103:8888")
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-frames", type=int, default=6)
    args = ap.parse_args()
    if args.self_test:
        return run_self_test(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
