"""api/chatcar — "问车"对话 (第 3 期 #3, 2026-06-11).

NavCockpit 聊天面板 → 车载本地 LLM (:9101 快档 0.5B / :9100 深档 1.7B) 工具调用
循环 (agent_loop.py 的 Web 化), 工具走 bridge_state 命令通道 — 断网全本地.

SSE 事件: phase / tool_call / tool_result / delta / done / error
运动类工具 (nav_goto) 默认禁用, 请求带 allow_motion=true 才放行.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from bridge_state import bridge_state

router = APIRouter(prefix='/api', tags=['chat'])

FAST_URL = 'http://127.0.0.1:9101/v1/chat/completions'
DEEP_URL = 'http://127.0.0.1:9100/v1/chat/completions'
SOP_DIR = Path.home() / 'car_llm' / 'sop_corpus'
ALARM_LOG = Path.home() / 'alarm_history.jsonl'
LANDMARKS_PATH = Path.home() / 'cockpit_landmarks.json'

_JOBS: dict[str, dict] = {}

SYSTEM = (
    '你是荧光粉实验室巡检机器人 (车载 RDK X5) 的语言中枢。回复专业简洁 (≤4 句), 中文。'
    '需要执行动作或查数据时, 输出且只输出:\n'
    '<tool_call>\n{"name": "工具名", "arguments": {...}}\n</tool_call>\n'
    '等待工具结果后再给最终结论。可用工具:\n'
    '- get_pose(): 读 SLAM 位姿/资源/SLAM 状态\n'
    '- get_alarms(limit): 查最近报警\n'
    '- read_furnace(): 读烧结炉数显 PV/SV\n'
    '- capture_photo(): 拍照 + BPU 检测 (用户会看到照片)\n'
    '- vlm_ask(prompt): 拍照问视觉模型 (英文 prompt, 约 30-60 秒, 慎用)\n'
    '- nav_goto(location): 导航到命名地标 (需用户授权运动)\n'
    '- search_sop(query): 检索实验室安全规程 SOP\n'
    '工具结果里的数字要原样引用, 不要编造。'
)


# ============================================================ 工具实现
async def _tool_get_pose(args: dict) -> str:
    f = bridge_state.frag
    if not bridge_state.alive:
        return '桥离线, 无实时数据'
    p, s = f.get('pose'), f.get('sys')
    out = []
    if p:
        out.append(f"位姿 ({p['frame']}): x={p['x']} y={p['y']} yaw={p['yaw']}rad")
    if s:
        out.append(f"CPU {s['cpu_pct']}% RAM {s['ram_used_gb']}/{s['ram_total_gb']}GB "
                   f"温度 {s['cpu_temp_c']}°C SLAM {'运行中' if s['slam_active'] else '停'} "
                   f"累计行驶 {s['distance_m']}m")
    out.append(f"estop={'置位' if f.get('estop') else '未置位'}")
    return '; '.join(out) or '无数据'


async def _tool_get_alarms(args: dict) -> str:
    n = min(10, int(args.get('limit', 5)))
    live = list(bridge_state.alarms)[-n:]
    if live:
        return '\n'.join(f"[{a['severity']}] {a['title']}: {a['detail'][:80]}" for a in live)
    if ALARM_LOG.exists():
        lines = ALARM_LOG.read_text(encoding='utf-8').splitlines()[-n:]
        return '\n'.join(lines) or '报警日志为空'
    return '本次会话无报警, 历史日志为空'


async def _tool_read_furnace(args: dict) -> str:
    r = await bridge_state.send_command('read_furnace', timeout=10)
    if r.get('ok'):
        return f"炉温 PV={r['pv']}°C SV={r['sv']}°C"
    return f"读不到: {r.get('error')}"


async def _tool_capture_photo(args: dict) -> str:
    r = await bridge_state.send_command('photo', timeout=15)
    if r.get('ok'):
        dets = r.get('detections') or []
        ds = ', '.join(f"{d['label']}({d['conf']})" for d in dets[:6] if d.get('label'))
        return f"已拍照 (用户可见)。BPU 检测: {ds or '无目标'}"
    return f"拍照失败: {r.get('error')}"


async def _tool_vlm_ask(args: dict) -> str:
    r = await bridge_state.send_command(
        'vlm', {'prompt': args.get('prompt', 'Describe the scene'),
                'timeout_s': 120}, timeout=140)
    if r.get('ok'):
        return f"VLM 回答 ({r.get('latency_s')}s): {r.get('answer')}"
    return f"VLM 失败: {r.get('error')}"


async def _tool_nav_goto(args: dict, allow_motion: bool = False) -> str:
    if not allow_motion:
        return '运动未授权 — 用户需在聊天面板勾选"允许运动"后重试'
    name = str(args.get('location', '')).strip()
    try:
        lms = json.load(LANDMARKS_PATH.open(encoding='utf-8'))['landmarks']
    except Exception:
        lms = []
    lm = next((l for l in lms if l['name'] == name), None)
    if not lm:
        return f"地标 '{name}' 不存在。已有: {', '.join(l['name'] for l in lms) or '无'}"
    r = await bridge_state.send_command(
        'goto', {'backend': 'direct', 'x': lm['x'], 'y': lm['y']}, timeout=130)
    return f"导航{'成功' if r.get('ok') else '失败'}: {json.dumps(r, ensure_ascii=False)[:200]}"


async def _tool_search_sop(args: dict) -> str:
    q = str(args.get('query', '')).strip()
    if not SOP_DIR.is_dir():
        return 'SOP 语料目录不存在'
    terms = [t for t in re.split(r'[\s,，。]+', q.lower()) if t]
    best: list[tuple[float, str, str]] = []
    for f in SOP_DIR.glob('*'):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        low = text.lower()
        score = sum(low.count(t) for t in terms)
        if score > 0:
            i = min((low.find(t) for t in terms if t in low), default=0)
            best.append((score, f.name, text[max(0, i - 40):i + 260]))
    best.sort(reverse=True)
    if not best:
        return f"SOP 里没搜到 '{q}'"
    return '\n---\n'.join(f"《{n}》…{s.strip()}…" for _, n, s in best[:2])


TOOLS = {'get_pose': _tool_get_pose, 'get_alarms': _tool_get_alarms,
         'read_furnace': _tool_read_furnace, 'capture_photo': _tool_capture_photo,
         'vlm_ask': _tool_vlm_ask, 'search_sop': _tool_search_sop}


# ============================================================ LLM
async def _llm(messages: list[dict], deep: bool) -> str:
    url = DEEP_URL if deep else FAST_URL

    def _post() -> str:
        r = requests.post(url, json={'messages': messages, 'max_tokens': 320,
                                     'temperature': 0.4},
                          timeout=180 if deep else 90)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']

    return await asyncio.to_thread(_post)


_TOOL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


# ============================================================ 路由
@router.post('/chat')
async def chat_start(body: dict) -> dict:
    q = (body.get('query') or '').strip()
    if not q:
        return {'ok': False, 'error': 'query 为空'}
    qid = uuid.uuid4().hex[:10]
    _JOBS[qid] = {'query': q, 'history': body.get('history') or [],
                  'deep': bool(body.get('deep')),
                  'allow_motion': bool(body.get('allow_motion')), 't': time.time()}
    if len(_JOBS) > 30:
        for k in sorted(_JOBS, key=lambda x: _JOBS[x]['t'])[:10]:
            _JOBS.pop(k, None)
    return {'ok': True, 'qid': qid}


@router.get('/chat/stream')
async def chat_stream(qid: str = Query(...)) -> StreamingResponse:
    job = _JOBS.get(qid)

    async def gen():
        def sse(d: dict) -> str:
            return f'data: {json.dumps(d, ensure_ascii=False)}\n\n'

        if not job:
            yield sse({'type': 'error', 'error': 'qid 不存在'})
            return
        msgs = [{'role': 'system', 'content': SYSTEM}]
        for h in job['history'][-6:]:
            if h.get('role') in ('user', 'assistant') and h.get('content'):
                msgs.append({'role': h['role'], 'content': str(h['content'])[:800]})
        msgs.append({'role': 'user', 'content': job['query']})
        model = '1.7B 深档' if job['deep'] else '0.5B 快档'
        t0 = time.time()
        try:
            for hop in range(4):     # 最多 3 次工具 + 1 次最终回答
                yield sse({'type': 'phase', 'text': f'🧠 {model} 思考中 (hop {hop + 1})…'})
                try:
                    reply = await _llm(msgs, job['deep'])
                except Exception as e:
                    yield sse({'type': 'error',
                               'error': f'本地 LLM 不可达: {type(e).__name__}: {e}'})
                    return
                m = _TOOL_RE.search(reply)
                if not m or hop == 3:
                    final = _TOOL_RE.sub('', reply).strip() or reply.strip()
                    for i in range(0, len(final), 24):   # 伪流式分块
                        yield sse({'type': 'delta', 'text': final[i:i + 24]})
                        await asyncio.sleep(0.02)
                    yield sse({'type': 'done', 'model': model,
                               'latency_ms': int((time.time() - t0) * 1000)})
                    return
                try:
                    call = json.loads(m.group(1))
                    name = call.get('name', '')
                    args = call.get('arguments') or {}
                except json.JSONDecodeError:
                    msgs.append({'role': 'assistant', 'content': reply})
                    msgs.append({'role': 'user', 'content': '工具调用 JSON 解析失败, 请直接回答。'})
                    continue
                yield sse({'type': 'tool_call', 'name': name, 'args': args})
                if name == 'nav_goto':
                    result = await _tool_nav_goto(args, job['allow_motion'])
                elif name in TOOLS:
                    result = await TOOLS[name](args)
                else:
                    result = f'未知工具 {name}'
                yield sse({'type': 'tool_result', 'name': name,
                           'result': str(result)[:600],
                           'photo': name in ('capture_photo',) and '已拍照' in str(result)})
                msgs.append({'role': 'assistant', 'content': reply})
                msgs.append({'role': 'user', 'content': f'工具 {name} 结果:\n{result}\n请基于结果回答。'})
        except Exception as e:
            yield sse({'type': 'error', 'error': f'{type(e).__name__}: {e}'})

    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})
