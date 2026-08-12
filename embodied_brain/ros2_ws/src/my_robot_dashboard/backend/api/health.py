from __future__ import annotations

import time

from fastapi import APIRouter

from config import settings
from ws_hub import ws_hub

router = APIRouter(prefix='/api', tags=['health'])

_START_S = time.monotonic()


@router.get('/health')
async def health() -> dict[str, object]:
    return {
        'status': 'ok',
        'uptime_s': round(time.monotonic() - _START_S, 2),
        'ws_clients': ws_hub.client_count,
        'mock_enabled': settings.mock_enabled,
        'mock_tick_hz': settings.mock_tick_hz,
        'ros2_enabled': settings.ros2_enabled,
    }
