#!/usr/bin/env python3
import json
import sys
import time
import urllib.request


url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:29100/api/public_status"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
deadline = time.monotonic() + duration
samples = []
failures = []

while time.monotonic() < deadline:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            response.read(64)
            status = response.status
    except Exception as exc:
        status = 0
        failures.append(type(exc).__name__)
    samples.append((time.monotonic() - started) * 1000.0)
    time.sleep(interval)

print(json.dumps({
    "url": url,
    "samples": len(samples),
    "failures": len(failures),
    "failure_types": sorted(set(failures)),
    "max_ms": round(max(samples), 1) if samples else None,
    "mean_ms": round(sum(samples) / len(samples), 1) if samples else None,
}))
