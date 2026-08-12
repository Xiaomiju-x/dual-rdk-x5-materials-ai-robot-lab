"""R0 target-only adapter for holder loading and XRD/PL assay stations."""

from rb_voe.adapters.base import TargetOnlyAdapter


class AssayStationAdapter(TargetOnlyAdapter):
    subsystem = "assay_station"
    capability_schema_version = "xrd-rb-voe-assay-station-capability-v1"


AssayStationTargetAdapter = AssayStationAdapter

__all__ = ["AssayStationAdapter", "AssayStationTargetAdapter"]
