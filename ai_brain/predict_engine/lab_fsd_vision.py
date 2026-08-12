"""AI-brain 4K tower-vision BEV for Lab-FSD shadow planning.

The module is deliberately low-frequency and safety-bounded. It reuses the
AI-brain IMX415 camera as a laboratory tower camera when available, but it does
not replace the car-side LiDAR/depth/IMU navigation stack.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - PC/dev fallback
    cv2 = None

try:
    import shared_locks  # type: ignore
except Exception:  # pragma: no cover - module is present on X5
    shared_locks = None


CAMERA_OWNER = "lab_fsd_vision_bev"
DEFAULT_CAMERA = "/dev/video0"


@dataclass(frozen=True)
class VisionBevConfig:
    grid_size: int = 48
    resolution_m: float = 0.10
    stale_after_s: float = 3.5
    frame_id: str = "base_footprint"


DEFAULT_CONFIG = VisionBevConfig()

_HELD_CAMERA_LOCK = False
_LAST_RESULT: dict[str, Any] | None = None


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _camera_holder() -> dict[str, Any]:
    if shared_locks is None:
        return {}
    try:
        return shared_locks.camera_holder() or {}
    except Exception:
        return {}


def camera_mode_status() -> dict[str, Any]:
    holder = _camera_holder()
    return {
        "ok": True,
        "mode": "LAB_FSD_VISION_BEV" if _HELD_CAMERA_LOCK else "IDLE",
        "held_by_this_process": _HELD_CAMERA_LOCK,
        "external_holder": holder,
        "policy": "snapshot lock only; XRD/PL and Vision-BEV must not hold IMX415 concurrently",
    }


def acquire_camera_mode() -> dict[str, Any]:
    global _HELD_CAMERA_LOCK
    if shared_locks is None:
        _HELD_CAMERA_LOCK = True
        return {"ok": True, "mode": "LAB_FSD_VISION_BEV", "lock": "stub-no-shared_locks"}
    ok, info = shared_locks.acquire_camera_lock(CAMERA_OWNER)
    if ok:
        _HELD_CAMERA_LOCK = True
        return {"ok": True, "mode": "LAB_FSD_VISION_BEV", "lock": info}
    return {
        "ok": False,
        "reason": "busy",
        "holder": info.get("holder_name", "unknown"),
        "holder_pid": info.get("holder_pid"),
    }


def release_camera_mode() -> dict[str, Any]:
    global _HELD_CAMERA_LOCK
    if shared_locks is not None:
        try:
            shared_locks.release_camera_lock()
        except Exception:
            pass
    _HELD_CAMERA_LOCK = False
    return {"ok": True, "mode": "IDLE"}


def _decode_b64_image(image_b64: str):
    if cv2 is None:
        return None
    try:
        raw = base64.b64decode(image_b64.split(",")[-1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _capture_frame(camera: str = DEFAULT_CAMERA, warmup: int = 4):
    if cv2 is None:
        return None, "cv2 unavailable"
    cap = None
    try:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            for dev in (0, 8, 1, 4):
                cap = cv2.VideoCapture(dev)
                if cap.isOpened():
                    break
        if cap is None or not cap.isOpened():
            return None, "open camera failed"
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(max(1, warmup)):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, "read frame failed"
        return frame, ""
    except Exception as exc:
        return None, str(exc)[:160]
    finally:
        if cap is not None:
            cap.release()


def _default_lab_objects() -> list[dict[str, Any]]:
    # Fixture priors are intentionally conservative. They provide a semantic
    # far-field layer before live calibration/images are available.
    return [
        {"label": "furnace_no_go", "x_m": 2.0, "y_m": -0.85, "w_m": 0.70, "h_m": 0.70, "risk": 95, "confidence": 0.72, "source": "fixture_prior"},
        {"label": "workbench_edge", "x_m": 1.25, "y_m": 1.05, "w_m": 0.95, "h_m": 0.45, "risk": 68, "confidence": 0.64, "source": "fixture_prior"},
        {"label": "sample_rack_zone", "x_m": 1.75, "y_m": 0.35, "w_m": 0.55, "h_m": 0.40, "risk": 58, "confidence": 0.60, "source": "fixture_prior"},
    ]


def _objects_from_image(frame: np.ndarray) -> list[dict[str, Any]]:
    if cv2 is None or frame is None:
        return []
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (640, max(1, int(640 * h / max(w, 1)))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    area_total = float(small.shape[0] * small.shape[1])
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < area_total * 0.004 or area > area_total * 0.25:
            continue
        candidates.append((area, (x, y, bw, bh)))
    candidates.sort(reverse=True)

    objects: list[dict[str, Any]] = []
    sh, sw = small.shape[:2]
    for i, (_, (x, y, bw, bh)) in enumerate(candidates[:6]):
        cx = (x + bw * 0.5) / max(sw, 1)
        cy = (y + bh * 0.5) / max(sh, 1)
        # Simple tower-camera perspective prior: lower image rows are nearer.
        x_m = _clip(0.45 + (1.0 - cy) * 2.6, -2.2, 3.2)
        y_m = _clip((0.5 - cx) * 2.6, -2.2, 2.2)
        risk = int(_clip(52 + 26 * (1.0 - cy) + 18 * min(1.0, bw * bh / area_total * 30.0), 35, 92))
        objects.append({
            "label": f"vision_obstacle_{i+1}",
            "x_m": round(x_m, 3),
            "y_m": round(y_m, 3),
            "w_m": round(_clip(bw / max(sw, 1) * 2.0, 0.18, 0.9), 3),
            "h_m": round(_clip(bh / max(sh, 1) * 1.5, 0.18, 0.8), 3),
            "risk": risk,
            "confidence": 0.48,
            "source": "tower_image_edges",
        })
    return objects


def _rasterize(objects: list[dict[str, Any]], cfg: VisionBevConfig) -> np.ndarray:
    n = int(cfg.grid_size)
    c = n // 2
    res = float(cfg.resolution_m)
    grid = np.zeros((n, n), dtype=np.int16)
    for obj in objects:
        x = float(obj.get("x_m", 0.0))
        y = float(obj.get("y_m", 0.0))
        w = max(0.10, float(obj.get("w_m", 0.35)))
        h = max(0.10, float(obj.get("h_m", 0.35)))
        risk = int(_clip(float(obj.get("risk", 60)), 0, 100))
        col0 = int(round(c + (x - h * 0.5) / res))
        col1 = int(round(c + (x + h * 0.5) / res))
        row0 = int(round(c - (y + w * 0.5) / res))
        row1 = int(round(c - (y - w * 0.5) / res))
        r0, r1 = sorted((max(0, row0), min(n - 1, row1)))
        c0, c1 = sorted((max(0, col0), min(n - 1, col1)))
        if r0 <= r1 and c0 <= c1:
            grid[r0:r1 + 1, c0:c1 + 1] = np.maximum(grid[r0:r1 + 1, c0:c1 + 1], risk)
    rr = max(1, int(round(0.28 / res)))
    grid[c - rr:c + rr + 1, c - rr:c + rr + 1] = 0
    return grid


def build_vision_bev(
    *,
    capture: bool = False,
    image_b64: str = "",
    objects: list[dict[str, Any]] | None = None,
    include_grid: bool = True,
    config: VisionBevConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    global _LAST_RESULT
    t0 = time.perf_counter()
    frame = None
    frame_used = False
    capture_error = ""
    lock_info: dict[str, Any] | None = None
    requested_frame_source = (
        "image_b64" if image_b64 else "camera_capture" if capture else "none"
    )

    if image_b64:
        frame = _decode_b64_image(image_b64)
        if frame is None:
            capture_error = "image_b64 decode failed"
    elif capture:
        lock = acquire_camera_mode()
        lock_info = lock
        if lock.get("ok"):
            try:
                frame, capture_error = _capture_frame()
            except Exception as exc:
                capture_error = f"capture failed: {str(exc)[:120]}"
            finally:
                release_camera_mode()
        else:
            capture_error = "camera busy"

    obj_list = list(objects or [])
    image_objects: list[dict[str, Any]] = []
    if frame is not None:
        try:
            image_objects = _objects_from_image(frame)
            obj_list.extend(image_objects)
            frame_used = True
        except Exception as exc:
            capture_error = f"image analysis failed: {str(exc)[:120]}"
    if not obj_list:
        obj_list = _default_lab_objects()

    object_sources = sorted({
        str(item.get("source") or "unknown")
        for item in obj_list
        if isinstance(item, dict)
    })
    prior_sources = {"fixture_prior", "static_prior", "map_prior"}
    if frame_used:
        provenance_state = "live_camera"
        provenance_source = requested_frame_source
    elif object_sources and all(source in prior_sources for source in object_sources):
        provenance_state = "fixture_prior"
        provenance_source = "fixture_prior"
    elif object_sources:
        provenance_state = "cached_camera"
        provenance_source = "provided_objects"
    else:
        provenance_state = "unknown"
        provenance_source = "unknown"

    used_frame_source = requested_frame_source if frame_used else "none"
    server_ts = time.time()
    provenance = {
        "state": provenance_state,
        "source": provenance_source,
        "image_supplied": frame_used,
        "image_used": frame_used,
        "capture_requested": bool(capture),
        "image_b64_requested": bool(image_b64),
        "frame_source": used_frame_source,
        "object_sources": object_sources,
        "server_ts": server_ts,
    }

    grid = _rasterize(obj_list, config)
    risk = float(grid.max()) / 100.0 if grid.size else 0.0
    result = {
        "ok": True,
        "model": "Lab-FSD Vision-BEV tower snapshot",
        "mode": "camera_to_bev_shadow_semantics",
        "ts": server_ts,
        "stale_after_s": config.stale_after_s,
        "frame_id": config.frame_id,
        "grid_size": config.grid_size,
        "resolution_m": config.resolution_m,
        "risk_score": round(risk, 4),
        "objects": obj_list,
        "object_count": len(obj_list),
        "image_object_count": len(image_objects),
        "camera": {
            "capture_requested": bool(capture),
            "image_supplied": frame_used,
            "image_b64_requested": bool(image_b64),
            "image_used": frame_used,
            "source": used_frame_source,
            "requested_source": requested_frame_source,
            "lock": lock_info,
            "error": capture_error,
            "shape": list(frame.shape[:2]) if frame is not None else None,
        },
        "provenance": provenance,
        "calibration": {
            "type": "fixture_prior_or_simple_perspective",
            "boundary": "Semantic far-field hint only; LiDAR/depth/Nav2 remain authoritative.",
        },
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
    if include_grid:
        result["grid"] = grid.astype(int).reshape(-1).tolist()
    _LAST_RESULT = result
    return result


def last_vision_bev() -> dict[str, Any]:
    if _LAST_RESULT is None:
        return build_vision_bev(capture=False)
    return _LAST_RESULT


__all__ = [
    "build_vision_bev",
    "last_vision_bev",
    "camera_mode_status",
    "acquire_camera_mode",
    "release_camera_mode",
]
