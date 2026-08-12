from __future__ import annotations

import platform

from fastapi import APIRouter

from api.models import Snapshot
from bridge_state import bridge_state
from mock_generator import build_telemetry

router = APIRouter(prefix='/api', tags=['snapshot'])

_BUILD_INFO = {
    'app': 'navcockpit-backend',
    'version': '0.1.0',
    'python': platform.python_version(),
    'platform': platform.platform(),
}


@router.get('/snapshot', response_model=Snapshot)
async def snapshot() -> Snapshot:
    telemetry = bridge_state.overlay(build_telemetry(seq=0).model_dump())
    return Snapshot(telemetry=telemetry, build_info=_BUILD_INFO)
