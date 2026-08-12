# NavCockpit Backend

FastAPI service that drives the Vue cockpit. Pumps a mock telemetry
packet over `/ws/telemetry` at 10 Hz until Phase 10 swaps in rclpy.

## Dev

```pwsh
# one-shot create venv + install + run
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8890
```

Hit:
- http://localhost:8890/api/health
- http://localhost:8890/api/snapshot
- http://localhost:8890/docs (Swagger UI)
- ws://localhost:8890/ws/telemetry (10 Hz packets)

Vite dev server proxies `/api` and `/ws` to 8890, so the frontend
just opens `ws://localhost:5173/ws/telemetry` and it transparently
tunnels through.

## Layout

```
backend/
  app.py              FastAPI entry + lifespan (starts mock_loop)
  config.py           pydantic-settings (NAVCOCKPIT_* env vars)
  mock_generator.py   async 10 Hz pump → ws_hub.broadcast
  ws_hub.py           connected-client set + broadcast
  api/
    models.py         pydantic wire schemas
    health.py         GET /api/health
    snapshot.py       GET /api/snapshot (one-shot)
    ws.py             WS /ws/telemetry
```

## Phase 10 — wire rclpy

Replace `mock_generator.build_telemetry()` with subscriptions to
`/odom`, `/scan`, `/furnace_reading`, etc. Keep the `TelemetryPacket`
shape stable so the Vue side is unaffected.
