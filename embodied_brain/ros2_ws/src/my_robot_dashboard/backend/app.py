"""NavCockpit backend — FastAPI entry.

Run dev:
    python -m uvicorn app:app --reload --port 8890
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import bridge, chatcar, health, ops, snapshot, ws
from config import settings
from mock_generator import mock_loop

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('navcockpit')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task: asyncio.Task[None] | None = None
    if settings.mock_enabled:
        log.info('starting mock loop at %.1f Hz', settings.mock_tick_hz)
        task = asyncio.create_task(mock_loop(), name='mock_loop')
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title='NavCockpit Backend', version='0.1.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health.router)
app.include_router(snapshot.router)
app.include_router(ws.router)
app.include_router(bridge.router)   # 第 3 期: cockpit_bridge 真数据上下行
app.include_router(ops.router)      # 第 3 期: 任务编排/地标/安全/黑匣子
app.include_router(chatcar.router)  # 第 3 期: 问车对话

# 前端静态托管 (2026-06-10): frontend/dist 由 backend 直接端, 平板/PC 开
# http://<car>:8890 即得完整 NavCockpit SPA. mount 在 routers 之后注册,
# API 路由优先匹配.
# 2026-06-12 修: StaticFiles(html=True) 对未知路径并不回落 index.html (404),
# 直链 /missions 等 history 路由会炸 — 子类捕 404 回 index.html (带 . 的
# 静态资源路径除外, 真缺文件照常 404).
_DIST = Path(__file__).resolve().parent.parent / 'frontend' / 'dist'


class _SpaStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as _HTTPExc
        try:
            return await super().get_response(path, scope)
        except _HTTPExc as e:
            if e.status_code == 404 and '.' not in path.rsplit('/', 1)[-1]:
                return await super().get_response('index.html', scope)
            raise


if _DIST.is_dir():
    app.mount('/', _SpaStaticFiles(directory=str(_DIST), html=True), name='frontend')
else:
    @app.get('/')
    async def root() -> dict[str, str]:
        return {'name': 'navcockpit-backend', 'docs': '/docs', 'ws': '/ws/telemetry'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host=settings.host, port=settings.port, log_level='warning')
