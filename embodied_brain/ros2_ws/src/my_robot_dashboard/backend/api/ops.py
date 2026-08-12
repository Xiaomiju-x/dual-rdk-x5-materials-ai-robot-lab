"""api/ops — 第 3 期业务路由: 任务编排 + 语义地标 + 安全层 + 黑匣子 (2026-06-11).

(问车 chat 在 api/chatcar.py, 单独成文件因为带 SSE 流式)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Query

from bridge_state import bridge_state
from mission import load_missions, new_mid, runner, save_missions, validate_tree

router = APIRouter(prefix='/api', tags=['ops'])

LANDMARKS_PATH = Path.home() / 'cockpit_landmarks.json'
BLACKBOX_DIR = Path.home() / 'blackbox'


# ============================================================ 任务编排
@router.get('/missions')
async def missions_list() -> dict:
    return {'ok': True, **load_missions()}


@router.post('/missions')
async def missions_save(body: dict) -> dict:
    tree = body.get('tree')
    err = validate_tree(tree)
    if err:
        return {'ok': False, 'error': err}
    data = load_missions()
    mid = body.get('mid') or new_mid()
    entry = {'mid': mid, 'name': (body.get('name') or '未命名任务')[:50],
             'tree': tree, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    data['missions'] = [m for m in data['missions'] if m['mid'] != mid] + [entry]
    save_missions(data)
    return {'ok': True, 'mid': mid}


@router.delete('/missions/{mid}')
async def missions_delete(mid: str) -> dict:
    data = load_missions()
    n0 = len(data['missions'])
    data['missions'] = [m for m in data['missions'] if m['mid'] != mid]
    save_missions(data)
    return {'ok': len(data['missions']) < n0}


@router.post('/missions/{mid}/run')
async def missions_run(mid: str) -> dict:
    m = next((x for x in load_missions()['missions'] if x['mid'] == mid), None)
    if not m:
        return {'ok': False, 'error': 'mission 不存在'}
    if not bridge_state.alive:
        return {'ok': False, 'error': 'cockpit_bridge 离线, 无法执行'}
    return await runner.start(mid, m['name'], m['tree'])


@router.post('/missions/run_adhoc')
async def missions_run_adhoc(body: dict) -> dict:
    """不落库直接跑 (设计器里的 ▶ 试跑)."""
    if not bridge_state.alive:
        return {'ok': False, 'error': 'cockpit_bridge 离线, 无法执行'}
    return await runner.start(f'adhoc-{uuid.uuid4().hex[:6]}',
                              body.get('name') or '试跑', body.get('tree') or {})


@router.get('/missions/status')
async def missions_status() -> dict:
    return {'ok': True, **runner.snapshot()}


@router.post('/missions/pause')
async def missions_pause() -> dict:
    return runner.pause()


@router.post('/missions/resume')
async def missions_resume() -> dict:
    return runner.resume()


@router.post('/missions/abort')
async def missions_abort() -> dict:
    return await runner.abort()


# ============================================================ 语义地标
def _load_lm() -> dict:
    if LANDMARKS_PATH.exists():
        try:
            return json.load(LANDMARKS_PATH.open(encoding='utf-8'))
        except Exception:
            pass
    return {'landmarks': []}


@router.get('/landmarks')
async def landmarks_list() -> dict:
    return {'ok': True, **_load_lm()}


@router.post('/landmarks')
async def landmarks_add(body: dict) -> dict:
    """{name, x?, y?} — 不带 x/y 时取车当前位姿 ("记住这里是 X")."""
    name = (body.get('name') or '').strip()[:30]
    if not name:
        return {'ok': False, 'error': 'name 必填'}
    x, y = body.get('x'), body.get('y')
    source = 'map_click'
    if x is None or y is None:
        pose = bridge_state.frag.get('pose')
        if not pose:
            return {'ok': False, 'error': '无车位姿且未给坐标'}
        x, y, source = pose['x'], pose['y'], 'robot_pose'
    data = _load_lm()
    data['landmarks'] = [l for l in data['landmarks'] if l['name'] != name]
    data['landmarks'].append({'name': name, 'x': round(float(x), 3),
                              'y': round(float(y), 3), 'source': source,
                              'created_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    LANDMARKS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                              encoding='utf-8')
    return {'ok': True, 'landmarks': data['landmarks']}


@router.delete('/landmarks/{name}')
async def landmarks_del(name: str) -> dict:
    data = _load_lm()
    n0 = len(data['landmarks'])
    data['landmarks'] = [l for l in data['landmarks'] if l['name'] != name]
    LANDMARKS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                              encoding='utf-8')
    return {'ok': len(data['landmarks']) < n0}


@router.post('/landmarks/{name}/goto')
async def landmarks_goto(name: str, body: dict | None = None) -> dict:
    """按名导航. mode=direct (真低速直线, 默认) / dispatch (stub 编排)."""
    lm = next((l for l in _load_lm()['landmarks'] if l['name'] == name), None)
    if not lm:
        return {'ok': False, 'error': f'地标 {name} 不存在'}
    mode = (body or {}).get('mode', 'direct')
    args = {'backend': mode, 'x': lm['x'], 'y': lm['y'], 'location': name}
    return await bridge_state.send_command('goto', args, timeout=130)


# ============================================================ WorkCockpit pickup flow
@router.post('/pickup_flow')
async def pickup_flow(body: dict | None = None) -> dict:
    """Trigger bridge-side pickup_flow without creating a persisted mission."""
    args = body or {}
    timeout = min(180.0, max(10.0, float(args.get('timeout_s', 90)) + 30.0))
    return await bridge_state.send_command('pickup_flow', args, timeout=timeout)


# ============================================================ 安全层
@router.get('/safety')
async def safety_get() -> dict:
    return {'ok': True, 'alive': bridge_state.alive,
            'estop': bool(bridge_state.frag.get('estop')),
            'safety': bridge_state.frag.get('safety')}


@router.post('/safety')
async def safety_set(body: dict) -> dict:
    return await bridge_state.send_command('set_safety', body, timeout=10)


@router.post('/safety/estop')
async def safety_estop() -> dict:
    return await bridge_state.send_command('estop', timeout=8)


@router.post('/safety/clear_estop')
async def safety_clear() -> dict:
    return await bridge_state.send_command('clear_estop', timeout=8)


# ============================================================ 黑匣子
@router.get('/blackbox/days')
async def blackbox_days() -> dict:
    days = []
    if BLACKBOX_DIR.is_dir():
        for f in sorted(BLACKBOX_DIR.glob('bb-*.jsonl')):
            days.append({'day': f.stem[3:], 'size_kb': round(f.stat().st_size / 1024, 1)})
    return {'ok': True, 'days': days}


@router.get('/blackbox/window')
async def blackbox_window(day: str = Query(...),
                          t_from: float = Query(0), t_to: float = Query(0),
                          max_points: int = Query(1200, le=5000)) -> dict:
    """day=YYYYMMDD; t_from/t_to 为 epoch 秒 (0 = 不限). 遥测降采样, 事件全保."""
    f = BLACKBOX_DIR / f'bb-{day}.jsonl'
    if not f.exists():
        return {'ok': False, 'error': f'无 {day} 记录'}
    tel: list[dict] = []
    events: list[dict] = []
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get('t', 0)
            if t_from and t < t_from:
                continue
            if t_to and t > t_to:
                continue
            (tel if r.get('k') == 'tel' else events).append(r)
    if len(tel) > max_points:
        step = len(tel) / max_points
        tel = [tel[int(i * step)] for i in range(max_points)]
    return {'ok': True, 'day': day, 'n_tel': len(tel), 'n_events': len(events),
            'telemetry': tel, 'events': events[-400:]}
