"""Pure-Python core for the Lab-FSD BEV shadow planner.

The ROS2 node imports this file, and tests can import it without rclpy.
Coordinate convention:
  x: forward from robot, y: left from robot, origin at robot base.
  BEV grid center is the robot; columns grow forward, rows grow to the right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class BevConfig:
    grid_size: int = 48
    resolution_m: float = 0.10
    max_range_m: float = 4.5
    inflation_cells: int = 2
    unknown_value: int = -1


@dataclass(frozen=True)
class TrajectoryCandidate:
    omega: float
    risk: float
    clearance: float
    goal_alignment: float
    score: float
    points_xy: tuple[tuple[float, float], ...]


def classify_vision_bev_provenance(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Classify whether a semantic BEV came from a live frame or a prior."""
    data = payload if isinstance(payload, dict) else {}
    explicit = data.get("provenance")
    if isinstance(explicit, dict) and explicit.get("state") in {
        "live_camera", "cached_camera", "fixture_prior", "unknown"
    }:
        out = dict(explicit)
        out.setdefault("image_supplied", out.get("state") == "live_camera")
        out.setdefault("object_sources", [])
        return out

    camera = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    objects = data.get("objects") if isinstance(data.get("objects"), list) else []
    object_sources = sorted({
        str(item.get("source") or "unknown")
        for item in objects
        if isinstance(item, dict)
    })
    image_supplied = bool(camera.get("image_supplied"))
    prior_sources = {"fixture_prior", "static_prior", "map_prior"}
    if image_supplied:
        state = "live_camera"
    elif object_sources and all(source in prior_sources for source in object_sources):
        state = "fixture_prior"
    elif object_sources and any(source not in prior_sources for source in object_sources):
        state = "cached_camera"
    else:
        state = "unknown"
    return {
        "state": state,
        "image_supplied": image_supplied,
        "object_sources": object_sources,
        "capture_requested": bool(camera.get("capture_requested")),
        "server_ts": data.get("ts"),
    }


def apply_vision_bev_provenance(
    transport_state: dict[str, Any],
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Refine ROS transport freshness with camera-source provenance."""
    out = dict(transport_state or {})
    prov = classify_vision_bev_provenance({"provenance": provenance or {}})
    out["transport_state"] = str(out.get("state") or "offline")
    out["provenance"] = prov
    out["semantic_only"] = True
    if out["transport_state"] in {"disabled", "offline", "stale"}:
        return out

    state = prov.get("state")
    if state == "live_camera":
        out["state"] = "live"
        out["fresh"] = True
        out["usable"] = True
    elif state == "cached_camera":
        out["state"] = "cached"
        out["fresh"] = False
        out["usable"] = False
    elif state == "fixture_prior":
        out["state"] = "fallback"
        out["fresh"] = False
        out["usable"] = False
    else:
        out["state"] = "unverified"
        out["fresh"] = False
        out["usable"] = False
    return out


def scan_to_points(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    max_range: float,
) -> np.ndarray:
    """Convert LaserScan arrays to Nx2 points in robot frame."""
    pts = []
    upper = min(float(range_max) if range_max > 0 else max_range, max_range)
    lower = max(float(range_min), 0.02)
    for i, r in enumerate(ranges):
        try:
            rr = float(r)
        except Exception:
            continue
        if not math.isfinite(rr) or rr < lower or rr > upper:
            continue
        a = angle_min + i * angle_increment
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def points_to_bev(points_xy: np.ndarray, config: BevConfig = BevConfig()) -> np.ndarray:
    """Rasterize points to an occupancy grid in [0, 100], with -1 unknown."""
    n = int(config.grid_size)
    bev = np.zeros((n, n), dtype=np.int16)
    c = n // 2
    res = float(config.resolution_m)

    for x, y in points_xy:
        if abs(x) > config.max_range_m or abs(y) > config.max_range_m:
            continue
        col = int(round(c + x / res))
        row = int(round(c - y / res))
        if row < 0 or row >= n or col < 0 or col >= n:
            continue
        r0 = max(0, row - config.inflation_cells)
        r1 = min(n, row + config.inflation_cells + 1)
        c0 = max(0, col - config.inflation_cells)
        c1 = min(n, col + config.inflation_cells + 1)
        bev[r0:r1, c0:c1] = np.maximum(bev[r0:r1, c0:c1], 70)
        bev[row, col] = 100

    # Robot footprint is known free.
    rr = max(1, int(round(0.28 / res)))
    bev[c - rr:c + rr + 1, c - rr:c + rr + 1] = 0
    return bev.astype(np.int16)


def merge_bev(*grids: np.ndarray) -> np.ndarray:
    valid = [g for g in grids if g is not None and g.size > 0]
    if not valid:
        return np.zeros((48, 48), dtype=np.int16)
    out = valid[0].astype(np.int16).copy()
    for g in valid[1:]:
        if g.shape == out.shape:
            out = np.maximum(out, g.astype(np.int16))
    return out


def _dilate_grid(grid: np.ndarray, radius: int = 1) -> np.ndarray:
    """Small max-filter implemented without scipy for X5 portability."""
    r = max(0, int(radius))
    if r <= 0:
        return np.asarray(grid, dtype=np.int16).copy()
    src = np.asarray(grid, dtype=np.int16)
    out = src.copy()
    h, w = src.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, dy)
            y1 = min(h, h + dy)
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], src[sy0:sy1, sx0:sx1])
    return out


def forecast_future_occupancy(
    history: Iterable[np.ndarray],
    horizons: int = 3,
    decay: float = 0.72,
    growth_gain: float = 0.65,
) -> list[np.ndarray]:
    """Forecast short-horizon occupancy from recent BEV grids.

    This is a deployment-friendly world-model approximation: the learned BPU
    variant can replace these tensor ops with Conv+ReLU heads while keeping the
    same ROS topics and safety boundary.
    """
    frames = [np.clip(np.asarray(g, dtype=np.float32), 0.0, 100.0) for g in history if g is not None and g.size > 0]
    if not frames:
        return []
    latest = frames[-1]
    memory = np.zeros_like(latest)
    for age, frame in enumerate(reversed(frames)):
        memory = np.maximum(memory, frame * (float(decay) ** age))
    trend = np.maximum(0.0, latest - frames[-2]) if len(frames) >= 2 else np.zeros_like(latest)

    outs: list[np.ndarray] = []
    base = 0.78 * latest + 0.22 * memory
    for i in range(max(1, int(horizons))):
        step = i + 1
        expanded = _dilate_grid(base + trend * growth_gain * step, radius=min(step, 2)).astype(np.float32)
        risk_margin = min(18.0, 4.0 * step)
        future = np.clip(expanded + risk_margin * (expanded > 15.0), 0.0, 100.0)
        outs.append(future.astype(np.int16))
    return outs


def _grid_value(bev: np.ndarray, x: float, y: float, resolution_m: float) -> float:
    n = bev.shape[0]
    c = n // 2
    col = int(round(c + x / resolution_m))
    row = int(round(c - y / resolution_m))
    if row < 0 or row >= n or col < 0 or col >= n:
        return 100.0
    return float(max(0, bev[row, col]))


def _simulate_arc(vx: float, omega: float, horizon_s: float, dt_s: float) -> list[tuple[float, float, float]]:
    x = y = th = 0.0
    pts: list[tuple[float, float, float]] = []
    t = 0.0
    while t < horizon_s + 1e-6:
        x += vx * math.cos(th) * dt_s
        y += vx * math.sin(th) * dt_s
        th += omega * dt_s
        pts.append((x, y, th))
        t += dt_s
    return pts


def score_candidate_trajectories(
    bev: np.ndarray,
    goal_xy: tuple[float, float] | None = None,
    resolution_m: float = 0.10,
    vx: float = 0.18,
    horizon_s: float = 2.4,
    dt_s: float = 0.20,
    omegas: Iterable[float] | None = None,
) -> dict:
    """Score local arcs. Lower risk and better goal alignment win."""
    if omegas is None:
        omegas = (-0.9, -0.65, -0.4, -0.2, 0.0, 0.2, 0.4, 0.65, 0.9)
    if goal_xy is None:
        goal_xy = (1.5, 0.0)
    gx, gy = goal_xy
    gnorm = max(math.hypot(gx, gy), 1e-6)

    candidates: list[TrajectoryCandidate] = []
    for om in omegas:
        pts = _simulate_arc(vx, float(om), horizon_s, dt_s)
        vals = [_grid_value(bev, x, y, resolution_m) for x, y, _ in pts]
        risk = max(vals) / 100.0
        clearance = 1.0 - (sum(vals) / max(len(vals), 1)) / 100.0
        ex, ey, _ = pts[-1]
        enorm = max(math.hypot(ex, ey), 1e-6)
        goal_alignment = (ex * gx + ey * gy) / (enorm * gnorm)
        goal_alignment = max(-1.0, min(1.0, goal_alignment))
        score = 0.55 * (1.0 - risk) + 0.25 * clearance + 0.20 * ((goal_alignment + 1.0) * 0.5)
        candidates.append(TrajectoryCandidate(
            omega=float(om),
            risk=float(risk),
            clearance=float(clearance),
            goal_alignment=float(goal_alignment),
            score=float(score),
            points_xy=tuple((float(x), float(y)) for x, y, _ in pts),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    return {
        "best": {
            "omega": best.omega,
            "risk": round(best.risk, 4),
            "clearance": round(best.clearance, 4),
            "goal_alignment": round(best.goal_alignment, 4),
            "score": round(best.score, 4),
            "points_xy": best.points_xy,
        },
        "candidates": [
            {
                "omega": c.omega,
                "risk": round(c.risk, 4),
                "clearance": round(c.clearance, 4),
                "goal_alignment": round(c.goal_alignment, 4),
                "score": round(c.score, 4),
            }
            for c in candidates
        ],
        "shadow_confidence": round(max(0.0, min(1.0, best.score)), 4),
        "mode": "shadow_only",
    }


def _softmax(values: list[float], temperature: float = 0.12) -> list[float]:
    temp = max(float(temperature), 1e-4)
    xs = np.asarray(values, dtype=np.float64) / temp
    xs = xs - float(xs.max())
    ex = np.exp(xs)
    denom = float(ex.sum())
    if denom <= 0:
        return [1.0 / max(len(values), 1)] * len(values)
    return [float(v) for v in (ex / denom)]


def score_candidate_trajectories_v2(
    bev: np.ndarray,
    future_bevs: Iterable[np.ndarray] | None = None,
    goal_xy: tuple[float, float] | None = None,
    resolution_m: float = 0.10,
    vx: float = 0.18,
    horizon_s: float = 2.4,
    dt_s: float = 0.20,
    omegas: Iterable[float] | None = None,
) -> dict:
    """Planning-oriented, probabilistic shadow policy over local arcs."""
    if omegas is None:
        omegas = (-0.9, -0.65, -0.4, -0.2, 0.0, 0.2, 0.4, 0.65, 0.9)
    if goal_xy is None:
        goal_xy = (1.5, 0.0)
    futures = [np.asarray(g, dtype=np.int16) for g in (future_bevs or []) if g is not None and g.size > 0]
    gx, gy = goal_xy
    gnorm = max(math.hypot(gx, gy), 1e-6)

    rows = []
    raw_scores = []
    for token_id, om in enumerate(omegas):
        pts3 = _simulate_arc(vx, float(om), horizon_s, dt_s)
        vals = []
        future_vals = []
        for i, (x, y, _) in enumerate(pts3):
            vals.append(_grid_value(bev, x, y, resolution_m))
            if futures:
                idx = min(len(futures) - 1, int(i * len(futures) / max(len(pts3), 1)))
                future_vals.append(_grid_value(futures[idx], x, y, resolution_m))
        now_risk = max(vals) / 100.0
        future_risk = (max(future_vals) / 100.0) if future_vals else now_risk
        mean_risk = (sum(vals) + sum(future_vals)) / max(len(vals) + len(future_vals), 1) / 100.0
        clearance = 1.0 - mean_risk
        ex, ey, _ = pts3[-1]
        enorm = max(math.hypot(ex, ey), 1e-6)
        goal_alignment = (ex * gx + ey * gy) / (enorm * gnorm)
        goal_alignment = max(-1.0, min(1.0, goal_alignment))
        smoothness = 1.0 - min(1.0, abs(float(om)) / 1.1)
        score = (
            0.42 * (1.0 - max(now_risk, future_risk))
            + 0.22 * clearance
            + 0.22 * ((goal_alignment + 1.0) * 0.5)
            + 0.14 * smoothness
        )
        raw_scores.append(float(score))
        rows.append({
            "token_id": token_id,
            "omega": float(om),
            "risk": float(max(now_risk, future_risk)),
            "now_risk": float(now_risk),
            "future_risk": float(future_risk),
            "clearance": float(clearance),
            "goal_alignment": float(goal_alignment),
            "smoothness": float(smoothness),
            "score": float(score),
            "points_xy": tuple((float(x), float(y)) for x, y, _ in pts3),
        })

    probs = _softmax(raw_scores)
    for row, prob in zip(rows, probs):
        row["probability"] = prob
    rows.sort(key=lambda c: c["score"], reverse=True)
    best = rows[0]
    n = max(len(rows), 1)
    entropy = -sum(float(p) * math.log(max(float(p), 1e-9)) for p in probs) / math.log(n)
    confidence = max(0.0, min(1.0, float(best["probability"]) * 0.65 + (1.0 - entropy) * 0.35))
    return {
        "best": {
            "token_id": int(best["token_id"]),
            "omega": round(float(best["omega"]), 4),
            "risk": round(float(best["risk"]), 4),
            "now_risk": round(float(best["now_risk"]), 4),
            "future_risk": round(float(best["future_risk"]), 4),
            "clearance": round(float(best["clearance"]), 4),
            "goal_alignment": round(float(best["goal_alignment"]), 4),
            "smoothness": round(float(best["smoothness"]), 4),
            "score": round(float(best["score"]), 4),
            "probability": round(float(best["probability"]), 4),
            "points_xy": best["points_xy"],
        },
        "candidates": [
            {
                "token_id": int(c["token_id"]),
                "omega": round(float(c["omega"]), 4),
                "risk": round(float(c["risk"]), 4),
                "now_risk": round(float(c["now_risk"]), 4),
                "future_risk": round(float(c["future_risk"]), 4),
                "clearance": round(float(c["clearance"]), 4),
                "goal_alignment": round(float(c["goal_alignment"]), 4),
                "smoothness": round(float(c["smoothness"]), 4),
                "score": round(float(c["score"]), 4),
                "probability": round(float(c["probability"]), 4),
            }
            for c in rows
        ],
        "policy": {
            "vocabulary": "arc_omega_tokens",
            "entropy": round(float(entropy), 4),
            "confidence": round(float(confidence), 4),
            "probabilities": [round(float(p), 4) for p in probs],
        },
        "shadow_confidence": round(float(confidence), 4),
        "mode": "lab_fsd_v2_probabilistic_shadow",
    }


def safety_gate_decision(
    score: dict,
    vision_diag: dict | None = None,
    anomaly_diag: dict | None = None,
    min_confidence: float = 0.12,
    max_risk: float = 0.82,
) -> dict:
    """Return a conservative explainable gate for shadow-policy proposals."""
    best = score.get("best", {}) if isinstance(score, dict) else {}
    policy = score.get("policy", {}) if isinstance(score, dict) else {}
    reasons = []
    conf_raw = policy.get("confidence", score.get("shadow_confidence", 0.0))
    risk_raw = best.get("risk", 1.0)
    conf = float(0.0 if conf_raw is None else conf_raw)
    risk = float(1.0 if risk_raw is None else risk_raw)
    if conf < min_confidence:
        reasons.append("low_policy_confidence")
    if risk > max_risk:
        reasons.append("trajectory_risk_high")
    if vision_diag and not vision_diag.get("used", False):
        reasons.append("vision_bev_unavailable")
    if anomaly_diag and anomaly_diag.get("ok") and anomaly_diag.get("level") == "high":
        reasons.append("bev_anomaly_high")
    assist = len(reasons) == 0
    return {
        "authority": "nav2_mppi",
        "shadow_policy": "assist_candidate" if assist else "observe_only",
        "assist_allowed": bool(assist),
        "confidence": round(conf, 4),
        "risk": round(risk, 4),
        "reasons": reasons,
        "cmd_vel_authority": False,
    }


def fuse_policy_with_bpu_prior(
    score: dict,
    occ_risk_diag: dict | None,
    bpu_weight: float = 0.25,
) -> dict:
    """Fuse CPU arc-token scores with an optional BPU 9-token prior.

    This is diagnostic by design. The fused prior may be logged, displayed, and
    used for offline disagreement analysis, but it does not grant /cmd_vel
    authority. Nav2/F407/safety operator remain the execution owner.
    """

    policy = score.get("policy", {}) if isinstance(score, dict) else {}
    candidates = score.get("candidates", []) if isinstance(score, dict) else []
    token_count = max(
        9,
        len(policy.get("probabilities") or []),
        max((int(c.get("token_id", -1)) + 1 for c in candidates if isinstance(c, dict)), default=0),
    )

    cpu_probs = [0.0] * token_count
    raw_cpu = policy.get("probabilities")
    if raw_cpu:
        for idx, value in enumerate(raw_cpu[:token_count]):
            cpu_probs[idx] = max(0.0, float(value))
    else:
        for cand in candidates:
            try:
                cpu_probs[int(cand.get("token_id", 0))] = max(0.0, float(cand.get("probability", 0.0)))
            except Exception:
                continue

    def _norm(vals: list[float]) -> list[float]:
        total = sum(max(0.0, float(v)) for v in vals)
        if total <= 1e-9:
            return [1.0 / max(len(vals), 1)] * len(vals)
        return [max(0.0, float(v)) / total for v in vals]

    cpu_probs = _norm(cpu_probs)
    cpu_best = int(max(range(len(cpu_probs)), key=lambda i: cpu_probs[i])) if cpu_probs else 0

    bpu_probs_raw = []
    if isinstance(occ_risk_diag, dict) and occ_risk_diag.get("used") and occ_risk_diag.get("probs"):
        bpu_probs_raw = [max(0.0, float(v)) for v in occ_risk_diag.get("probs", [])[:token_count]]
    if len(bpu_probs_raw) < token_count:
        bpu_probs_raw.extend([0.0] * (token_count - len(bpu_probs_raw)))
    bpu_used = bool(bpu_probs_raw and sum(bpu_probs_raw) > 1e-9)
    bpu_probs = _norm(bpu_probs_raw) if bpu_used else [0.0] * token_count
    bpu_best = int(max(range(len(bpu_probs)), key=lambda i: bpu_probs[i])) if bpu_used else None

    weight = max(0.0, min(0.65, float(bpu_weight))) if bpu_used else 0.0
    fused = _norm([(1.0 - weight) * c + weight * b for c, b in zip(cpu_probs, bpu_probs)])
    fused_best = int(max(range(len(fused)), key=lambda i: fused[i])) if fused else 0
    ordered = sorted(fused, reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    entropy = -sum(p * math.log(max(p, 1e-9)) for p in fused) / math.log(max(len(fused), 2))
    confidence = max(0.0, min(1.0, 0.55 * fused[fused_best] + 0.30 * margin + 0.15 * (1.0 - entropy)))
    return {
        "name": "tiny_waypoint_policy_prior",
        "source": "cpu_arc_tokens_plus_bpu_tiny_occ_risk" if bpu_used else "cpu_arc_tokens_only",
        "used_bpu": bool(bpu_used),
        "bpu_weight": round(weight, 4),
        "token_count": len(fused),
        "cpu_best_index": cpu_best,
        "bpu_best_index": bpu_best,
        "fused_best_index": fused_best,
        "agreement": bool(bpu_used and cpu_best == bpu_best),
        "confidence": round(float(confidence), 4),
        "probability_margin": round(float(margin), 4),
        "entropy": round(float(entropy), 4),
        "probabilities": [round(float(v), 4) for v in fused],
        "cmd_vel_authority": False,
        "shadow_only": True,
    }


def bev_tensor_for_bpu(bev: np.ndarray, goal_xy: tuple[float, float] | None = None) -> np.ndarray:
    """Return a fixed 1x3xHxW tensor for an optional BPU risk network."""
    occ = np.clip(bev.astype(np.float32), 0.0, 100.0) / 100.0
    h, w = occ.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2.0) / max(w / 2.0, 1.0)
    yy = (yy - h / 2.0) / max(h / 2.0, 1.0)
    if goal_xy is None:
        goal_xy = (1.5, 0.0)
    gx = np.full_like(occ, max(-1.0, min(1.0, goal_xy[0] / 3.0)))
    gy = np.full_like(occ, max(-1.0, min(1.0, goal_xy[1] / 3.0)))
    goal_channel = 0.5 * gx + 0.5 * gy
    return np.stack([occ, xx + yy, goal_channel], axis=0)[None, ...].astype(np.float32)
