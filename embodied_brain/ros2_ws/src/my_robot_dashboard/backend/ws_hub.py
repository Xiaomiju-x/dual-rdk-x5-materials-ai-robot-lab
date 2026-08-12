"""WebSocket broadcast hub.

Holds the live set of connected WebSocket clients and ships each telemetry
frame to all of them. Single-process, no Redis: this dashboard is a single
backend instance bound to one X5.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

log = logging.getLogger('navcockpit.ws_hub')


class WsHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def attach(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
        log.info('client attached, total=%d', len(self._clients))

    async def detach(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info('client detached, total=%d', len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        # Snapshot the set so we can iterate without holding the lock during send.
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception as e:
                log.warning('send failed, dropping client: %s', e)
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


ws_hub = WsHub()
