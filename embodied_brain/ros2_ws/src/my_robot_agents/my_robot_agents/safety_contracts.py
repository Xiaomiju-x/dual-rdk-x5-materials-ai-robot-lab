"""Pure safety decisions shared by ROS nodes and host-side contract tests."""

from __future__ import annotations

import math
from typing import Any, Iterable


def lab_fsd_guard_reasons(
    *,
    now: float,
    require_fresh: bool,
    stale_s: float,
    future_risk: float,
    future_risk_time: float,
    future_risk_stop: float,
    safety_gate: dict[str, Any],
    safety_gate_time: float,
    input_status: dict[str, Any],
    input_status_time: float,
    hard_reasons: Iterable[str],
) -> list[str]:
    reasons: list[str] = []
    if require_fresh:
        freshness = (
            ("safety_gate", safety_gate_time),
            ("future_risk", future_risk_time),
            ("input_status", input_status_time),
        )
        for name, stamp in freshness:
            if stamp <= 0.0:
                reasons.append(f"{name}_missing")
            elif now - stamp > stale_s:
                reasons.append(f"{name}_stale={now - stamp:.3f}s")
        if input_status_time > 0.0 and now - input_status_time <= stale_s:
            overall = str(input_status.get("overall") or "offline")
            if overall != "live":
                reasons.append(f"input_status={overall}")

    if future_risk_time > 0.0 and now - future_risk_time <= stale_s:
        if not math.isfinite(future_risk):
            reasons.append("future_risk_invalid")
        elif future_risk >= future_risk_stop:
            reasons.append(f"future_risk={future_risk:.3f}>=stop={future_risk_stop:.3f}")

    if safety_gate_time > 0.0 and now - safety_gate_time <= stale_s:
        configured = {str(item) for item in hard_reasons}
        gate_reasons = [str(item) for item in (safety_gate.get("reasons") or [])]
        hard_hits = [reason for reason in gate_reasons if reason in configured]
        if hard_hits:
            reasons.append("safety_gate_hard_reasons=" + ",".join(hard_hits))
    return reasons
