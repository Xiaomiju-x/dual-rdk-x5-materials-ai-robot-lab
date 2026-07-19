"""Public, mock-only skill recording and replay helpers.

This module deliberately has no robot imports, serial access, peer endpoints, or
actuator calls. It exists so reviewers can inspect the data shape and run the
frontend with synthetic trajectories. Production hardware code is not part of
the public repository.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path


SKILLS_DIR = Path.home() / "xrd_public_demo_skills"
VLA_DIR = Path.home() / "xrd_public_demo_episodes"
RECORD_HZ = 10.0
WAYPOINT_MIN_DEG = 2.0
_lock = threading.Lock()
_stop_flag = threading.Event()
_session = {
    "state": "idle",
    "skill": None,
    "arm": None,
    "t0": 0.0,
    "n": 0,
    "progress_pct": 0.0,
    "error": "",
    "mock": True,
}


def _safe(name: str) -> str:
    value = "".join(char for char in str(name) if char.isalnum() or char in "-_")[:40]
    return value or "unnamed"


def _save(skill: dict) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"{_safe(skill['name'])}.json"
    path.write_text(json.dumps(skill, ensure_ascii=False), encoding="utf-8")


def list_skills() -> list[dict]:
    rows = []
    if not SKILLS_DIR.is_dir():
        return rows
    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                key: item.get(key)
                for key in ("name", "arm", "hz", "created_at", "n_frames", "note")
            })
        except (OSError, ValueError, TypeError):
            continue
    return rows


def load_skill(name: str) -> dict | None:
    path = SKILLS_DIR / f"{_safe(name)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def delete_skill(name: str) -> bool:
    path = SKILLS_DIR / f"{_safe(name)}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def session_status() -> dict:
    with _lock:
        return dict(_session)


def _require_mock(mock: bool) -> dict | None:
    if mock:
        return None
    return {
        "ok": False,
        "error": "Public build is mock-only and cannot access physical robots.",
        "boundary": "no_public_actuation",
    }


def start_record(name: str, arm: str, mock: bool = True, real_mod=None) -> dict:
    denied = _require_mock(mock)
    if denied:
        return denied
    with _lock:
        if _session["state"] != "idle":
            return {"ok": False, "error": f"busy: {_session['state']}"}
        _session.update(
            state="recording",
            skill=name,
            arm=arm,
            t0=time.time(),
            n=0,
            progress_pct=0.0,
            error="",
            mock=True,
        )
    _stop_flag.clear()
    threading.Thread(
        target=_record_worker,
        args=(name, arm),
        daemon=True,
        name="public-synthetic-skill-record",
    ).start()
    return {"ok": True, "source": "synthetic"}


def stop_record() -> dict:
    if _session["state"] != "recording":
        return {"ok": False, "error": "not recording"}
    _stop_flag.set()
    return {"ok": True}


def _record_worker(name: str, arm: str) -> None:
    frames: list[list[float]] = []
    grippers: list[float] = []
    period = 1.0 / RECORD_HZ
    base = [0.0, 8.0, -127.0, 40.0, 0.0, 45.0]
    try:
        step = 0
        while not _stop_flag.is_set() and step < int(6.0 * RECORD_HZ):
            t = step * period
            frames.append([
                round(value + 12.0 * math.sin(t * 1.5 + index), 2)
                for index, value in enumerate(base)
            ])
            grippers.append(50.0)
            with _lock:
                _session["n"] = len(frames)
            time.sleep(period)
            step += 1
        if len(frames) < 3:
            raise RuntimeError("too few synthetic frames")
        _save({
            "name": name,
            "arm": arm,
            "hz": RECORD_HZ,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_frames": len(frames),
            "frames": frames,
            "gripper": grippers,
            "note": "synthetic public mock; not hardware telemetry",
        })
        with _lock:
            _session.update(state="idle", error="")
    except Exception as exc:
        with _lock:
            _session.update(state="idle", error=f"{type(exc).__name__}: {exc}")


def waypoints_of(skill: dict) -> list[list[float]]:
    frames = skill.get("frames") or []
    if not frames:
        return []
    waypoints = [frames[0]]
    for frame in frames[1:]:
        if max(abs(frame[index] - waypoints[-1][index]) for index in range(6)) > WAYPOINT_MIN_DEG:
            waypoints.append(frame)
    if waypoints[-1] != frames[-1]:
        waypoints.append(frames[-1])
    return waypoints


def start_replay(
    name: str,
    speed: int,
    mock: bool = True,
    real_mod=None,
    get_joints=None,
    interlock_mod=None,
) -> dict:
    denied = _require_mock(mock)
    if denied:
        return denied
    skill = load_skill(name)
    if not skill:
        return {"ok": False, "error": f"skill not found: {name}"}
    with _lock:
        if _session["state"] != "idle":
            return {"ok": False, "error": f"busy: {_session['state']}"}
        _session.update(
            state="replaying",
            skill=name,
            arm=skill.get("arm"),
            t0=time.time(),
            n=0,
            progress_pct=0.0,
            error="",
            mock=True,
        )
    _stop_flag.clear()
    waypoints = waypoints_of(skill)
    threading.Thread(
        target=_replay_worker,
        args=(waypoints,),
        daemon=True,
        name="public-synthetic-skill-replay",
    ).start()
    return {"ok": True, "n_waypoints": len(waypoints), "source": "synthetic"}


def stop_replay() -> dict:
    if _session["state"] != "replaying":
        return {"ok": False, "error": "not replaying"}
    _stop_flag.set()
    return {"ok": True}


def _replay_worker(waypoints: list[list[float]]) -> None:
    try:
        for index, _waypoint in enumerate(waypoints):
            if _stop_flag.is_set():
                break
            time.sleep(0.1)
            with _lock:
                _session["n"] = index + 1
                _session["progress_pct"] = round(100.0 * (index + 1) / max(1, len(waypoints)), 1)
        with _lock:
            _session.update(state="idle", error="" if not _stop_flag.is_set() else "stopped")
    except Exception as exc:
        with _lock:
            _session.update(state="idle", error=f"{type(exc).__name__}: {exc}")


def export_lerobot(name: str) -> dict:
    skill = load_skill(name)
    if not skill:
        return {"ok": False, "error": f"skill not found: {name}"}
    episode = VLA_DIR / f"skill_{_safe(name)}_{int(time.time())}"
    episode.mkdir(parents=True, exist_ok=True)
    frames = skill.get("frames") or []
    hz = float(skill.get("hz") or RECORD_HZ)
    (episode / "meta.json").write_text(json.dumps({
        "task": f"synthetic replay skill: {name}",
        "source": "public_mock",
        "arm": skill.get("arm"),
        "hz": hz,
        "n_frames": len(frames),
        "camera": False,
        "note": "synthetic public mock; not hardware telemetry",
    }, ensure_ascii=False), encoding="utf-8")
    with (episode / "states.jsonl").open("w", encoding="utf-8") as states, \
            (episode / "actions.jsonl").open("w", encoding="utf-8") as actions:
        for index, frame in enumerate(frames):
            timestamp = round(index / hz, 3)
            target = frames[min(index + 1, len(frames) - 1)]
            states.write(json.dumps({"t": timestamp, "joints": frame, "gripper": 0}) + "\n")
            actions.write(json.dumps({"t": timestamp, "target_joints": target, "gripper": 0}) + "\n")
    return {"ok": True, "path": str(episode), "n_frames": len(frames), "source": "synthetic"}
