"""api/bridge — cockpit_bridge 上下行 + 地图/照片出口 (第 3 期, 2026-06-11)."""
from __future__ import annotations

import time

from fastapi import APIRouter, Query, Response

from bridge_state import bridge_state

router = APIRouter(prefix='/api', tags=['bridge'])


@router.post('/bridge/ingest')
async def bridge_ingest(body: dict) -> dict:
    bridge_state.ingest(body)
    return {'ok': True}


@router.post('/bridge/map')
async def bridge_map(body: dict) -> dict:
    bridge_state.set_map(body)
    return {'ok': True, 'etag': bridge_state.map_meta['etag']}


@router.get('/bridge/commands')
async def bridge_commands(wait: float = Query(20, le=25)) -> dict:
    cmds = await bridge_state.pull_commands(wait_s=wait)
    return {'commands': cmds}


@router.post('/bridge/result')
async def bridge_result(body: dict) -> dict:
    cid = body.pop('cid', '')
    return {'ok': bridge_state.resolve(cid, body)}


@router.get('/bridge/status')
async def bridge_status() -> dict:
    return {
        'alive': bridge_state.alive,
        'last_ingest_age_s': round(time.time() - bridge_state.last_ingest_t, 1)
                             if bridge_state.last_ingest_t else None,
        'estop': bool(bridge_state.frag.get('estop')),
        'safety': bridge_state.frag.get('safety'),
        'motion_busy': bool(bridge_state.frag.get('motion_busy')),
        'pickup_flow': bridge_state.frag.get('pickup_flow') or {
            'active': False,
            'state': 'unknown' if not bridge_state.alive else 'idle',
        },
        'provenance_mode': 'live_partial' if bridge_state.alive else 'fixture_only',
        'lab_fsd': bridge_state.frag.get('lab_fsd') if bridge_state.alive else None,
        'f407': bridge_state.frag.get('f407') if bridge_state.alive else None,
        'map_etag': (bridge_state.map_meta or {}).get('etag', 0),
        'has_photo': bridge_state.photo_jpg is not None,
    }


def _demo_map() -> dict:
    """桥离线时的演示地图 (G3): 一间程序生成的实验室房间 (外墙 + 两张台子 + 门洞).

    与 mock 遥测同一性质 — 镜像/离线演示用, 响应里 mock=True, 前端必须标"演示地图"。
    真桥一推 /map 即被真图取代, 本函数不再被走到。
    """
    w, h, res = 120, 90, 0.05          # 6m × 4.5m @ 5cm
    grid = bytearray(w * h)
    def fill(x0, y0, x1, y1, v=100):
        for r in range(y0, y1):
            for c in range(x0, x1):
                grid[r * w + c] = v
    fill(0, 0, w, 2); fill(0, h - 2, w, h)          # 上下外墙
    fill(0, 0, 2, h); fill(w - 2, 0, w, h)          # 左右外墙
    fill(w - 2, 34, w, 56, 0)                       # 右墙门洞
    fill(14, 58, 56, 70)                            # 工位台 (双臂)
    fill(80, 14, 108, 26)                           # 烧结炉台
    fill(20, 18, 34, 30)                            # 试剂柜
    rle: list[int] = []
    prev, run = grid[0], 0
    for b in grid:
        if b == prev and run < 65535:
            run += 1
        else:
            rle.extend((prev, run))
            prev, run = b, 1
    rle.extend((prev, run))
    return {'ok': True, 'mock': True, 'w': w, 'h': h, 'res': res,
            'ox': -w * res / 2, 'oy': -h * res / 2, 'etag': -2, 'rle': rle}


@router.get('/map.json')
async def map_json() -> dict:
    """OccupancyGrid: zlib+b64 原样转发, 前端 pako/手写 inflate? — 不,
    前端无 zlib 依赖, 这里解压再按行 RLE 编码 (JSON 友好且小)."""
    if not bridge_state.map_z64 or not bridge_state.map_meta:
        if bridge_state.alive:
            return {
                'ok': False,
                'mock': False,
                'unavailable': True,
                'reason': 'ROS bridge is live but no /map payload has been received.',
            }
        return _demo_map()
    import base64
    import zlib
    raw = zlib.decompress(base64.b64decode(bridge_state.map_z64))
    # RLE: [value, run] 对; -1(unknown)→255 已在桥侧转 uint8
    rle: list[int] = []
    prev, run = raw[0], 0
    for b in raw:
        if b == prev and run < 65535:
            run += 1
        else:
            rle.extend((prev, run))
            prev, run = b, 1
    rle.extend((prev, run))
    return {'ok': True, **bridge_state.map_meta, 'rle': rle}


@router.get('/photo/latest.jpg')
async def latest_photo() -> Response:
    if bridge_state.photo_jpg is None:
        return Response(status_code=404, content=b'no photo yet')
    return Response(content=bridge_state.photo_jpg, media_type='image/jpeg',
                    headers={'Cache-Control': 'no-store',
                             'X-Photo-Age-S': str(round(time.time() - bridge_state.photo_t, 1))})
