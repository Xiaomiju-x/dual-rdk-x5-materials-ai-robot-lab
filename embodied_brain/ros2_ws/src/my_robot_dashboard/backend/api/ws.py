from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from bridge_state import bridge_state
from mock_generator import build_telemetry
from ws_hub import ws_hub

log = logging.getLogger('navcockpit.ws')

router = APIRouter(tags=['ws'])


@router.websocket('/ws/telemetry')
async def telemetry_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws_hub.attach(ws)
    try:
        # Send an immediate hello snapshot so the client can paint before the
        # next mock tick fires.
        hello = bridge_state.overlay(build_telemetry(seq=0).model_dump())
        await ws.send_json({'type': 'hello', 'payload': hello})
        # Idle-receive loop: accept pings and ignore everything else.
        while True:
            msg = await ws.receive_text()
            if msg == 'ping':
                await ws.send_json({'type': 'pong'})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning('ws error: %s', e)
    finally:
        await ws_hub.detach(ws)
