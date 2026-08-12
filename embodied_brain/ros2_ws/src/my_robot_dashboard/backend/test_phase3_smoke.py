"""Phase 3 后端冒烟 (PC, 无车): 模拟 cockpit_bridge 走完整上下行.

跑法: 先 uvicorn app:app --port 18890, 再 python test_phase3_smoke.py
"""
import base64
import json
import threading
import time
import zlib

import requests

B = 'http://127.0.0.1:18890'
ok_n = fail_n = 0


def check(name: str, cond: bool, detail: str = ''):
    global ok_n, fail_n
    if cond:
        ok_n += 1
        print(f'  [PASS] {name}')
    else:
        fail_n += 1
        print(f'  [FAIL] {name} {detail}')


# 1) ingest 真遥测 → alive
requests.post(f'{B}/api/bridge/ingest', json={
    'ts': time.time(),
    'pose': {'frame': 'map', 'x': 1.23, 'y': -0.5, 'yaw': 0.78},
    'vel': {'linear': 0.12, 'angular': 0.0},
    'sys': {'cpu_pct': 41.0, 'ram_used_gb': 3.1, 'ram_total_gb': 7.6,
            'cma_used_mb': 120.0, 'bpu_pct': 22.0, 'cpu_temp_c': 55.1,
            'ai_brain_reachable': True, 'slam_active': True, 'distance_m': 12.5},
    'furnace': {'pv': 1340.0, 'sv': 1350.0},
    'detections': [{'label': 'bottle', 'conf': 0.91}],
    'estop': False,
    'safety': {'fence': None, 'speed_cap': 0.2, 'fence_enabled': True},
    'scan': [[1.0, 0.0], [0.0, 1.0]],
    'alarms': [{'severity': 2, 'title': '测试报警', 'description': 'smoke', 'source': 1, 't': time.time()}],
})
st = requests.get(f'{B}/api/bridge/status').json()
check('ingest → alive', st['alive'] is True, str(st))

# 2) snapshot overlay 不破 (snapshot 仍是纯 mock 形状, WS 帧才 overlay — 这里查 bridge status 字段就够)
snap = requests.get(f'{B}/api/snapshot').json()
check('snapshot 200', 'telemetry' in snap)

# 3) 地图 RLE 往返
w, h = 8, 4
grid = bytes([0] * 16 + [100] * 8 + [255] * 8)
requests.post(f'{B}/api/bridge/map', json={
    'w': w, 'h': h, 'res': 0.05, 'ox': -0.2, 'oy': -0.1,
    'data_z64': base64.b64encode(zlib.compress(grid)).decode(),
})
mp = requests.get(f'{B}/api/map.json').json()
flat = []
for i in range(0, len(mp['rle']), 2):
    flat += [mp['rle'][i]] * mp['rle'][i + 1]
check('map RLE 解码一致', mp['ok'] and bytes(flat) == grid, str(mp)[:120])

# 4) 命令通道往返: 模拟桥长轮询 + 回执
def bridge_worker():
    cmds = requests.get(f'{B}/api/bridge/commands', params={'wait': 10}, timeout=15).json()['commands']
    for c in cmds:
        requests.post(f'{B}/api/bridge/result',
                      json={'cid': c['cid'], 'ok': True, 'echo': c['cmd']})

t = threading.Thread(target=bridge_worker)
t.start()
time.sleep(0.3)
r = requests.post(f'{B}/api/safety/estop', timeout=15).json()
t.join()
check('命令往返 estop → 桥回执', r.get('ok') is True and r.get('echo') == 'estop', str(r))

# 5) mission 保存 + 校验
r = requests.post(f'{B}/api/missions', json={
    'name': 'smoke', 'tree': {'type': 'sequence', 'children': [{'type': 'speak', 'params': {'text': 'hi'}}]},
}).json()
check('mission 保存', r.get('ok') is True, str(r))
mid = r.get('mid')
r2 = requests.post(f'{B}/api/missions', json={'name': 'bad', 'tree': {'type': 'nope'}}).json()
check('mission 非法树被拒', r2.get('ok') is False and '未知' in str(r2.get('error')), str(r2))

# 6) mission 试跑 (桥 alive, speak 叶) — 后台模拟桥接 speak 命令
def bridge_worker2():
    deadline = time.time() + 20
    while time.time() < deadline:
        cmds = requests.get(f'{B}/api/bridge/commands', params={'wait': 5}, timeout=10).json()['commands']
        for c in cmds:
            requests.post(f'{B}/api/bridge/result', json={'cid': c['cid'], 'ok': True})
            if c['cmd'] == 'speak':
                return

t2 = threading.Thread(target=bridge_worker2)
t2.start()
# 保持 alive (ingest 3s 窗口)
requests.post(f'{B}/api/bridge/ingest', json={'ts': time.time(), 'pose': None, 'vel': None,
                                              'sys': None, 'estop': False})
r = requests.post(f'{B}/api/missions/{mid}/run').json()
check('mission 启动', r.get('ok') is True, str(r))
for _ in range(20):
    time.sleep(0.5)
    s = requests.get(f'{B}/api/missions/status').json()
    if s['state'] in ('done', 'failed', 'aborted'):
        break
t2.join()
check('mission 跑完 done', s['state'] == 'done', str(s)[:200])

# 7) 地标: 有位姿后"记住这里"
requests.post(f'{B}/api/bridge/ingest', json={'ts': time.time(),
              'pose': {'frame': 'map', 'x': 2.0, 'y': 3.0, 'yaw': 0}, 'vel': None, 'sys': None, 'estop': False})
r = requests.post(f'{B}/api/landmarks', json={'name': '烧结炉前'}).json()
check('地标记住当前位置', r.get('ok') is True and r['landmarks'][0]['x'] == 2.0, str(r)[:160])
r = requests.delete(f'{B}/api/landmarks/烧结炉前').json()
check('地标删除', r.get('ok') is True, str(r))

# 8) chat 启动 (stream 需要车端 LLM, PC 上只验 qid 发放)
r = requests.post(f'{B}/api/chat', json={'query': '位姿?'}).json()
check('chat qid 发放', r.get('ok') is True and len(r.get('qid', '')) == 10, str(r))

print(f'\n==> {ok_n} PASS / {fail_n} FAIL')
raise SystemExit(1 if fail_n else 0)
