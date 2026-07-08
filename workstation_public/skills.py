"""skills — 技能库: 拖动示教录关节轨迹 → 命名/参数化/回放 (第 4 期 #1, 2026-06-12).

录制 (真机, 仅 LOCAL_ARM):
    release_all_servos() 进自由拖动 → 10Hz 采 get_angles → stop 后 power_on 回刚体.
    ⚠ 录制线程独占串口锁 — 录制中 real.py 的 20Hz poller 拿不到锁会饿死缓存,
    所以录制用同一把 real._serial_lock 且每帧间隙释放 (poller 能插队).

回放: 帧序列降采样成 waypoint (相邻帧最大关节差 > 2° 才保留) → send_angles 逐点.
    每个 waypoint 下发前过 interlock.check_move ghost 预演 (另一臂数据可用时).

存储: ~/skills/<name>.json
    {name, arm, hz, created_at, n_frames, frames: [[6]...], gripper: [pct...], note}

lerobot 导出: 跟 workstation/vla/record_episodes.py 原始格式对齐 —
    ~/vla_episodes/skill_<name>_<ts>/{meta.json, states.jsonl, actions.jsonl}
    (无相机帧 — 拖动示教时不抓图; convert_to_lerobot.py 支持 state-only episode
    需加 --no-camera, 真采数据还是用 record_episodes.py 走全观测流.)

mock 模式: record 合成一条正弦摆动轨迹 (note 标 synthetic), replay 只走进度模拟 —
    UI/镜像可演示, 不假装是真机数据.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

SKILLS_DIR = Path.home() / "skills"
VLA_DIR = Path.home() / "vla_episodes"
RECORD_HZ = 10.0
MAX_RECORD_S = 120.0
WAYPOINT_MIN_DEG = 2.0     # 回放降采样: 相邻保留帧的最大关节差阈值

_lock = threading.Lock()
_session = {"state": "idle",       # idle / recording / replaying
            "skill": None, "arm": None, "t0": 0.0, "n": 0,
            "progress_pct": 0.0, "error": "", "mock": False}
_stop_flag = threading.Event()


# ---------------------------------------------------------------- 存储
def list_skills() -> list[dict]:
    out = []
    if SKILLS_DIR.is_dir():
        for f in sorted(SKILLS_DIR.glob("*.json")):
            try:
                d = json.load(f.open(encoding="utf-8"))
                out.append({k: d.get(k) for k in
                            ("name", "arm", "hz", "created_at", "n_frames", "note")})
            except Exception:
                continue
    return out


def load_skill(name: str) -> dict | None:
    f = SKILLS_DIR / f"{_safe(name)}.json"
    if not f.exists():
        return None
    try:
        return json.load(f.open(encoding="utf-8"))
    except Exception:
        return None


def delete_skill(name: str) -> bool:
    f = SKILLS_DIR / f"{_safe(name)}.json"
    if f.exists():
        f.unlink()
        return True
    return False


def _safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_一二三四五六七八九十"
                   or "一" <= c <= "鿿")[:40] or "unnamed"


def _save(skill: dict) -> None:
    SKILLS_DIR.mkdir(exist_ok=True)
    (SKILLS_DIR / f"{_safe(skill['name'])}.json").write_text(
        json.dumps(skill, ensure_ascii=False), encoding="utf-8")


def session_status() -> dict:
    with _lock:
        return dict(_session)


# ---------------------------------------------------------------- 录制
def start_record(name: str, arm: str, mock: bool, real_mod=None) -> dict:
    with _lock:
        if _session["state"] != "idle":
            return {"ok": False, "error": f"忙: {_session['state']}"}
        _session.update(state="recording", skill=name, arm=arm,
                        t0=time.time(), n=0, progress_pct=0.0, error="", mock=mock)
    _stop_flag.clear()
    threading.Thread(target=_record_worker, args=(name, arm, mock, real_mod),
                     daemon=True, name="skill-record").start()
    return {"ok": True}


def stop_record() -> dict:
    if _session["state"] != "recording":
        return {"ok": False, "error": "没有在录"}
    _stop_flag.set()
    return {"ok": True}


def _record_worker(name: str, arm: str, mock: bool, real_mod) -> None:
    frames: list[list[float]] = []
    grippers: list[float] = []
    period = 1.0 / RECORD_HZ
    try:
        if mock:
            # 合成轨迹: HOME 附近正弦摆 (诚实标注 synthetic)
            base = [0.0, 8.0, -127.0, 40.0, 0.0, 45.0]
            t = 0.0
            while not _stop_flag.is_set() and t < 6.0:
                frames.append([round(b + 12.0 * math.sin(t * 1.5 + i), 2)
                               for i, b in enumerate(base)])
                grippers.append(50.0)
                with _lock:
                    _session["n"] = len(frames)
                time.sleep(period)
                t += period
            note = "synthetic (mock 模式合成, 非真机示教)"
        else:
            # 真机自由拖动示教
            with real_mod._serial_lock:
                mc = real_mod._arm()
                mc.release_all_servos()
            t0 = time.time()
            while not _stop_flag.is_set() and time.time() - t0 < MAX_RECORD_S:
                tick = time.time()
                try:
                    with real_mod._serial_lock:
                        a = real_mod._arm().get_angles()
                    if a and len(a) >= 6:
                        frames.append([round(float(x), 2) for x in a[:6]])
                        grippers.append(real_mod._gripper_pct.get(arm, 50.0))
                        with _lock:
                            _session["n"] = len(frames)
                except Exception:
                    pass
                time.sleep(max(0.0, period - (time.time() - tick)))
            with real_mod._serial_lock:
                try:
                    real_mod._arm().power_on()   # 回刚体
                except Exception:
                    pass
            note = "拖动示教"
        if len(frames) < 3:
            raise RuntimeError(f"只采到 {len(frames)} 帧, 不存")
        _save({"name": name, "arm": arm, "hz": RECORD_HZ,
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_frames": len(frames), "frames": frames,
               "gripper": grippers, "note": note})
        with _lock:
            _session.update(state="idle", error="")
    except Exception as e:
        with _lock:
            _session.update(state="idle", error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 回放
def waypoints_of(skill: dict) -> list[list[float]]:
    """帧 → waypoint 降采样 (最大关节差 > WAYPOINT_MIN_DEG 才保留)."""
    frames = skill.get("frames") or []
    if not frames:
        return []
    wps = [frames[0]]
    for f in frames[1:]:
        if max(abs(f[i] - wps[-1][i]) for i in range(6)) > WAYPOINT_MIN_DEG:
            wps.append(f)
    if wps[-1] != frames[-1]:
        wps.append(frames[-1])
    return wps


def start_replay(name: str, speed: int, mock: bool, real_mod=None,
                 get_joints=None, interlock_mod=None) -> dict:
    skill = load_skill(name)
    if not skill:
        return {"ok": False, "error": f"技能 {name} 不存在"}
    with _lock:
        if _session["state"] != "idle":
            return {"ok": False, "error": f"忙: {_session['state']}"}
        _session.update(state="replaying", skill=name, arm=skill["arm"],
                        t0=time.time(), n=0, progress_pct=0.0, error="", mock=mock)
    _stop_flag.clear()
    threading.Thread(target=_replay_worker,
                     args=(skill, max(5, min(80, int(speed))), mock, real_mod,
                           get_joints, interlock_mod),
                     daemon=True, name="skill-replay").start()
    return {"ok": True, "n_waypoints": len(waypoints_of(skill))}


def stop_replay() -> dict:
    if _session["state"] != "replaying":
        return {"ok": False, "error": "没有在放"}
    _stop_flag.set()
    return {"ok": True}


def _replay_worker(skill: dict, speed: int, mock: bool, real_mod,
                   get_joints, interlock_mod) -> None:
    wps = waypoints_of(skill)
    arm = skill["arm"]
    other = "arm02" if arm == "arm01" else "arm01"
    try:
        for i, wp in enumerate(wps):
            if _stop_flag.is_set():
                break
            # ghost 预演互锁
            if interlock_mod and get_joints:
                oj = get_joints(other)
                ck = interlock_mod.check_move(
                    wp, oj.get("angles") if oj.get("online") else None,
                    moving_is_arm1=(arm == "arm01"))
                if not ck["ok"]:
                    raise RuntimeError(
                        f"互锁拒绝 waypoint {i}: 预测最小距 {ck['min_dist_mm']}mm "
                        f"< {ck['clearance_mm']}mm")
            if mock:
                time.sleep(0.25)
            else:
                if arm == real_mod.LOCAL_ARM:
                    with real_mod._serial_lock:
                        real_mod._arm().send_angles([float(x) for x in wp], speed)
                else:
                    real_mod._http_json(f"{real_mod.PEER_BASE}/move", timeout=3.0,
                                        data={"angles": wp, "speed": speed})
                time.sleep(0.45)
            with _lock:
                _session["n"] = i + 1
                _session["progress_pct"] = round(100.0 * (i + 1) / max(1, len(wps)), 1)
        with _lock:
            _session.update(state="idle", error="" if not _stop_flag.is_set() else "用户停止")
    except Exception as e:
        with _lock:
            _session.update(state="idle", error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- lerobot 导出
def export_lerobot(name: str) -> dict:
    """导出为 record_episodes.py 同款原始 episode (state-only, 无相机帧)."""
    skill = load_skill(name)
    if not skill:
        return {"ok": False, "error": f"技能 {name} 不存在"}
    ep = VLA_DIR / f"skill_{_safe(name)}_{int(time.time())}"
    ep.mkdir(parents=True, exist_ok=True)
    hz = skill.get("hz", RECORD_HZ)
    frames = skill["frames"]
    grippers = skill.get("gripper") or [50.0] * len(frames)
    (ep / "meta.json").write_text(json.dumps({
        "task": f"replay taught skill: {name}",
        "source": "skill_teach_drag", "arm": skill["arm"], "hz": hz,
        "n_frames": len(frames), "camera": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": skill.get("note", ""),
    }, ensure_ascii=False), encoding="utf-8")
    with open(ep / "states.jsonl", "w", encoding="utf-8") as sf, \
         open(ep / "actions.jsonl", "w", encoding="utf-8") as af:
        for i, fr in enumerate(frames):
            t = round(i / hz, 3)
            g = 1 if grippers[i] > 50 else 0
            sf.write(json.dumps({"t": t, "joints": fr, "gripper": g}) + "\n")
            # 动作流 = 下一帧目标 (示教数据的标准 next-state-as-action)
            nxt = frames[min(i + 1, len(frames) - 1)]
            af.write(json.dumps({"t": t, "target_joints": nxt, "gripper": g}) + "\n")
    return {"ok": True, "path": str(ep), "n_frames": len(frames)}
