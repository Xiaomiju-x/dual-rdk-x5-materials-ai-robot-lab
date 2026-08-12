"""Fail-closed adapter surface for target capabilities not implemented at R0."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

from rb_voe.contracts.models import Maturity

ADAPTER_RESULT_SCHEMA_VERSION: Final[str] = "xrd-rb-voe-target-adapter-result-v1"


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Deterministic refusal returned by every R0 target-only adapter call."""

    schema_version: str
    subsystem: str
    operation: str
    maturity: Maturity
    ready: bool
    reason_code: str
    fallback_allowed: bool
    fallback_used: bool
    network_touched: bool
    hardware_touched: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["maturity"] = self.maturity.value
        return value

    @property
    def device_touched(self) -> bool:
        return self.hardware_touched

    @property
    def status(self) -> str:
        return "NOT_READY"

    @property
    def network_attempted(self) -> bool:
        return self.network_touched

    @property
    def hardware_touch(self) -> bool:
        return self.hardware_touched


class TargetOnlyAdapter:
    """Interface placeholder that cannot read state, prepare, or execute.

    The optional fallback argument exists only so callers can prove a supplied
    fallback is never invoked. R0 adapters do not inspect request payloads.
    """

    subsystem: str = "unbound"
    capability_schema_version: str = "unbound"
    maturity: Maturity = Maturity.TARGET_ONLY

    __slots__ = ()

    def status(self) -> AdapterResult:
        return self._not_ready("STATUS")

    def request(
        self,
        operation: str,
        request: Mapping[str, Any] | None = None,
        *,
        fallback: Callable[..., Any] | None = None,
    ) -> AdapterResult:
        del request, fallback
        normalized = operation.upper() if isinstance(operation, str) and operation else "INVALID_OPERATION"
        return self._not_ready(normalized)

    def capability_manifest(self, *, now_ms: int | None = None) -> AdapterResult:
        del now_ms
        return self._not_ready("CAPABILITY_MANIFEST")

    def get_capability_manifest(self, *, now_ms: int | None = None) -> AdapterResult:
        return self.capability_manifest(now_ms=now_ms)

    def read_state(self, *, now_ms: int | None = None) -> AdapterResult:
        del now_ms
        return self._not_ready("READ_STATE")

    def create_challenge(self, request: Mapping[str, Any] | None = None) -> AdapterResult:
        del request
        return self._not_ready("CREATE_CHALLENGE")

    def prepare(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        fallback: Callable[..., Any] | None = None,
    ) -> AdapterResult:
        del request, fallback
        return self._not_ready("PREPARE")

    def execute(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        fallback: Callable[..., Any] | None = None,
    ) -> AdapterResult:
        del request, fallback
        return self._not_ready("EXECUTE")

    def _not_ready(self, operation: str) -> AdapterResult:
        return AdapterResult(
            schema_version=ADAPTER_RESULT_SCHEMA_VERSION,
            subsystem=self.subsystem,
            operation=operation,
            maturity=Maturity.TARGET_ONLY,
            ready=False,
            reason_code="NOT_READY",
            fallback_allowed=False,
            fallback_used=False,
            network_touched=False,
            hardware_touched=False,
        )


AdapterRefusal = AdapterResult

__all__ = [
    "ADAPTER_RESULT_SCHEMA_VERSION",
    "AdapterRefusal",
    "AdapterResult",
    "TargetOnlyAdapter",
]
