"""shared_locks.py — 跨进程独占锁: 4 条线共享同一支 IMX415 相机 + M260C 麦克风.

策略: first-come-first-served + 非阻塞 fcntl.flock.
拿不到就立即返回 (holder_pid, holder_name) 给调用方做提示.

X5 上 4 个 Flask 进程 (xrd_vision/xrd_num/spec_vision/spec_num) 用同一份 lock 文件
(/tmp/imx415_camera.lock, /tmp/m260c_mic.lock).
"""
from __future__ import annotations

import json
import os
from typing import Optional

try:
    import fcntl  # POSIX only; X5 是 Linux, OK
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


CAMERA_LOCK_PATH = "/tmp/imx415_camera.lock"
MIC_LOCK_PATH = "/tmp/m260c_mic.lock"


class _DeviceLock:
    """fcntl.flock 包装. 持有 fd 直到 release()."""

    def __init__(self, lock_path: str, owner_name: str):
        self.lock_path = lock_path
        self.owner_name = owner_name
        self._fd: Optional[int] = None

    def acquire(self) -> tuple[bool, dict]:
        """返回 (成功?, 信息 dict). 失败时 dict 含 {holder_pid, holder_name}."""
        if not _HAS_FCNTL:
            # 非 Linux 平台 (PC dev), 假装拿到, 不真锁
            self._fd = -1
            return True, {"platform": "no-fcntl-stub"}

        try:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            return False, {"error": "open lock failed"}

        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            holder = self._read_holder()
            os.close(self._fd)
            self._fd = None
            return False, {"holder_pid": holder.get("pid"),
                           "holder_name": holder.get("name", "unknown")}

        # 拿到锁: 写入持有者信息
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.ftruncate(self._fd, 0)
            payload = json.dumps({"pid": os.getpid(), "name": self.owner_name})
            os.write(self._fd, payload.encode("utf-8"))
        except OSError:
            pass
        return True, {"acquired_by": self.owner_name, "pid": os.getpid()}

    def release(self):
        if self._fd is None or self._fd < 0:
            self._fd = None
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def held(self) -> bool:
        return self._fd is not None

    def _read_holder(self) -> dict:
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                return json.loads(txt)
        except (OSError, json.JSONDecodeError):
            pass
        return {}


# 单进程内每条线只持有一个 camera + 一个 mic 锁实例
_camera_lock: Optional[_DeviceLock] = None
_mic_lock: Optional[_DeviceLock] = None


def acquire_camera_lock(owner_name: str) -> tuple[bool, dict]:
    """成功返回 (True, info), 失败 (False, {holder_pid, holder_name})."""
    global _camera_lock
    if _camera_lock is not None and _camera_lock.held():
        return True, {"already_held": True, "by": _camera_lock.owner_name}
    _camera_lock = _DeviceLock(CAMERA_LOCK_PATH, owner_name)
    ok, info = _camera_lock.acquire()
    if not ok:
        _camera_lock = None
    return ok, info


def release_camera_lock():
    global _camera_lock
    if _camera_lock is not None:
        _camera_lock.release()
        _camera_lock = None


def acquire_mic_lock(owner_name: str) -> tuple[bool, dict]:
    global _mic_lock
    if _mic_lock is not None and _mic_lock.held():
        return True, {"already_held": True, "by": _mic_lock.owner_name}
    _mic_lock = _DeviceLock(MIC_LOCK_PATH, owner_name)
    ok, info = _mic_lock.acquire()
    if not ok:
        _mic_lock = None
    return ok, info


def release_mic_lock():
    global _mic_lock
    if _mic_lock is not None:
        _mic_lock.release()
        _mic_lock = None


def camera_holder() -> dict:
    """查 /tmp/imx415_camera.lock 内容, 返回 {pid, name} (即使本进程没持锁)"""
    try:
        with open(CAMERA_LOCK_PATH, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if txt:
            return json.loads(txt)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def mic_holder() -> dict:
    try:
        with open(MIC_LOCK_PATH, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if txt:
            return json.loads(txt)
    except (OSError, json.JSONDecodeError):
        pass
    return {}
