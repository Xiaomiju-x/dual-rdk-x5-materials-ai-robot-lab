"""mission — 行为树任务编排引擎 (第 3 期 #1, 2026-06-11).

树 JSON 结构 (递归):
    {"type": "sequence|fallback|retry|repeat", "children": [...], "params": {...}}
    {"type": "goto|forward|spin|twist|wait|speak|photo|vlm|read_furnace|pickup_flow|detect_wait",
     "params": {...}}

组合节点:
    sequence    子节点依次执行, 任一失败即失败
    fallback    依次尝试, 任一成功即成功
    retry       params.times (默认 2): 重试唯一子节点
    repeat      params.times (默认 2): 重复唯一子节点 (全部成功才成功)

叶节点 → bridge_state.send_command 一一对应; detect_wait 是本地轮询桥的
detections 缓存 (params: label, timeout_s).

控制: pause/resume/abort (asyncio.Event), 状态机 idle→running→(paused)→
done/failed/aborted. 全程状态快照可查 (/api/missions/status), 事件落
~/blackbox/bb-*.jsonl (与桥同文件, kind='mission').

持久化: ~/cockpit_missions.json {missions: [{mid, name, tree, created_at}]}
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from bridge_state import bridge_state

MISSIONS_PATH = Path.home() / 'cockpit_missions.json'
BLACKBOX_DIR = Path.home() / 'blackbox'

COMPOSITE = {'sequence', 'fallback', 'retry', 'repeat'}
LEAF_TIMEOUT = {'goto': 120, 'forward': 40, 'spin': 40, 'twist': 10, 'wait': 35,
                'speak': 8, 'photo': 15, 'vlm': 150, 'read_furnace': 8,
                'pickup_flow': 140}


def _bb_event(data: dict) -> None:
    try:
        BLACKBOX_DIR.mkdir(exist_ok=True)
        p = BLACKBOX_DIR / f'bb-{time.strftime("%Y%m%d")}.jsonl'
        with open(p, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': round(time.time(), 2), 'k': 'mission',
                                **data}, ensure_ascii=False) + '\n')
    except OSError:
        pass


# ============================================================ 存储
def load_missions() -> dict:
    if MISSIONS_PATH.exists():
        try:
            return json.load(MISSIONS_PATH.open(encoding='utf-8'))
        except Exception:
            pass
    return {'missions': []}


def save_missions(data: dict) -> None:
    MISSIONS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                             encoding='utf-8')


def validate_tree(node: Any, depth: int = 0) -> Optional[str]:
    if depth > 8:
        return '树太深 (>8 层)'
    if not isinstance(node, dict) or 'type' not in node:
        return '节点必须是 {type, ...}'
    t = node['type']
    if t in COMPOSITE:
        ch = node.get('children') or []
        if not ch:
            return f'{t} 需要 children'
        if t in ('retry', 'repeat') and len(ch) != 1:
            return f'{t} 只能有 1 个子节点'
        for c in ch:
            err = validate_tree(c, depth + 1)
            if err:
                return err
        return None
    if t not in LEAF_TIMEOUT and t != 'detect_wait':
        return f'未知节点类型 {t}'
    return None


# ============================================================ 执行器
class MissionRunner:
    def __init__(self) -> None:
        self.state = 'idle'         # idle/running/paused/done/failed/aborted
        self.mid: Optional[str] = None
        self.name = ''
        self.tree: Optional[dict] = None
        self.node_states: dict[str, dict] = {}   # path -> {status, result, t0, t1}
        self.log: list[dict] = []
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self._pause_evt = asyncio.Event()
        self._abort = False
        self._task: Optional[asyncio.Task] = None

    # -------------- 控制 --------------
    def snapshot(self) -> dict:
        return {'state': self.state, 'mid': self.mid, 'name': self.name,
                'node_states': self.node_states, 'log': self.log[-40:],
                'started_at': self.started_at, 'ended_at': self.ended_at}

    async def start(self, mid: str, name: str, tree: dict) -> dict:
        if self.state in ('running', 'paused'):
            return {'ok': False, 'error': f'已有任务在跑 ({self.name}), 先中止'}
        err = validate_tree(tree)
        if err:
            return {'ok': False, 'error': err}
        self.state, self.mid, self.name, self.tree = 'running', mid, name, tree
        self.node_states, self.log = {}, []
        self.started_at, self.ended_at = time.time(), None
        self._abort = False
        self._pause_evt.set()
        self._task = asyncio.create_task(self._run_root())
        _bb_event({'event': 'start', 'mid': mid, 'name': name})
        return {'ok': True, 'mid': mid}

    def pause(self) -> dict:
        if self.state != 'running':
            return {'ok': False, 'error': f'当前 {self.state}, 无法暂停'}
        self._pause_evt.clear()
        self.state = 'paused'
        self._log('⏸ 已暂停 (当前叶节点跑完后挂起)')
        return {'ok': True}

    def resume(self) -> dict:
        if self.state != 'paused':
            return {'ok': False, 'error': f'当前 {self.state}, 无法恢复'}
        self._pause_evt.set()
        self.state = 'running'
        self._log('▶ 已恢复')
        return {'ok': True}

    async def abort(self) -> dict:
        if self.state not in ('running', 'paused'):
            return {'ok': False, 'error': f'当前 {self.state}'}
        self._abort = True
        self._pause_evt.set()
        estop_result = await bridge_state.send_command('estop', timeout=8)
        self._log('任务已中止，急停保持锁存，需人工显式清除', estop_result=estop_result)
        estop_ok = bool(isinstance(estop_result, dict) and estop_result.get('ok'))
        return {
            'ok': estop_ok,
            'estop': estop_result,
            'error': '' if estop_ok else '任务已中止，但急停确认失败；保持人工安全接管',
        }

    def _log(self, msg: str, **kw) -> None:
        self.log.append({'t': round(time.time(), 1), 'msg': msg, **kw})
        _bb_event({'event': 'log', 'mid': self.mid, 'msg': msg})

    # -------------- 树执行 --------------
    async def _run_root(self) -> None:
        try:
            ok = await self._run(self.tree, 'r')
            if self._abort:
                self.state = 'aborted'
            else:
                self.state = 'done' if ok else 'failed'
        except Exception as e:
            self.state = 'failed'
            self._log(f'引擎异常: {type(e).__name__}: {e}')
        self.ended_at = time.time()
        _bb_event({'event': 'end', 'mid': self.mid, 'state': self.state,
                   'elapsed_s': round(self.ended_at - (self.started_at or 0), 1)})

    async def _gate(self) -> bool:
        """暂停/中止闸门. 返回 False = 该中止."""
        await self._pause_evt.wait()
        return not self._abort

    def _mark(self, path: str, status: str, result: dict | None = None) -> None:
        st = self.node_states.setdefault(path, {})
        st['status'] = status
        if status == 'running':
            st['t0'] = round(time.time(), 1)
        else:
            st['t1'] = round(time.time(), 1)
        if result is not None:
            st['result'] = {k: v for k, v in result.items() if k != 'image_b64'}

    async def _run(self, node: dict, path: str) -> bool:
        if not await self._gate():
            self._mark(path, 'aborted')
            return False
        t = node['type']
        params = node.get('params') or {}
        self._mark(path, 'running')

        if t == 'sequence':
            for i, c in enumerate(node.get('children') or []):
                if not await self._run(c, f'{path}.{i}'):
                    self._mark(path, 'failed' if not self._abort else 'aborted')
                    return False
            self._mark(path, 'done')
            return True

        if t == 'fallback':
            for i, c in enumerate(node.get('children') or []):
                if await self._run(c, f'{path}.{i}'):
                    self._mark(path, 'done')
                    return True
                if self._abort:
                    break
            self._mark(path, 'failed' if not self._abort else 'aborted')
            return False

        if t in ('retry', 'repeat'):
            times = max(1, min(10, int(params.get('times', 2))))
            child = (node.get('children') or [None])[0]
            for k in range(times):
                ok = await self._run(child, f'{path}.{k}')
                if self._abort:
                    self._mark(path, 'aborted')
                    return False
                if t == 'retry' and ok:
                    self._mark(path, 'done')
                    return True
                if t == 'repeat' and not ok:
                    self._mark(path, 'failed')
                    return False
            ok = (t == 'repeat')
            self._mark(path, 'done' if ok else 'failed')
            return ok

        # ---------------- 叶节点 ----------------
        if t == 'detect_wait':
            label = str(params.get('label', '')).lower()
            timeout = min(120, float(params.get('timeout_s', 20)))
            t0 = time.time()
            while time.time() - t0 < timeout:
                if not await self._gate():
                    self._mark(path, 'aborted')
                    return False
                dets = bridge_state.frag.get('detections') or []
                hit = next((d for d in dets
                            if label in str(d.get('label', '')).lower()), None)
                if hit:
                    self._mark(path, 'done', {'ok': True, 'hit': hit})
                    self._log(f'👁 detect_wait 命中 {hit.get("label")}')
                    return True
                await asyncio.sleep(0.5)
            self._mark(path, 'failed', {'ok': False, 'error': f'{timeout}s 内未检到 {label}'})
            return False

        timeout = LEAF_TIMEOUT.get(t, 30) + float(params.get('timeout_s', 0))
        self._log(f'▶ {t} {json.dumps(params, ensure_ascii=False)[:80]}')
        res = await bridge_state.send_command(t, params, timeout=timeout)
        ok = bool(res.get('ok'))
        self._mark(path, 'done' if ok else 'failed', res)
        if not ok:
            self._log(f'✗ {t} 失败: {res.get("error", "")[:120]}')
        return ok


runner = MissionRunner()


def new_mid() -> str:
    return f'm{uuid.uuid4().hex[:8]}'
