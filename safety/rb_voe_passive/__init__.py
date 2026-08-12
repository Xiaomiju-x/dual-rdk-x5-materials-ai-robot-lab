"""Isolated, offline RB-VoE passive one-shot auditing.

v1 is a candidate-only compatibility API in explicit security HOLD. New
authenticated claims use path-confined v2 only; neither API is deployable.
"""

from rb_voe_passive.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    seal_mapping,
)
from rb_voe_passive.contracts import (
    BUNDLE_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    seal_bundle,
    seal_observation,
    seal_profile,
)
from rb_voe_passive.contracts_v2 import (
    BUNDLE_SCHEMA_VERSION_V2,
    REPORT_SCHEMA_VERSION_V2,
    TRUST_POLICY_SCHEMA_VERSION,
    seal_observation_v2,
    seal_profile_v2,
    seal_receipt_v2,
    seal_tensor_contract_v2,
    seal_trust_policy_v2,
    sign_bundle_v2,
    tensor_values_sha256,
)
from rb_voe_passive.oneshot import (
    AUDIT_HOLD,
    AUDIT_INVALID,
    AUDIT_PASS,
    V1_DEPLOYMENT_STATE,
    V1_SECURITY_STATUS,
    OneShotResult,
    run_passive_oneshot,
)
from rb_voe_passive.oneshot_v2 import OneShotResultV2, run_passive_oneshot_v2

__all__ = [
    "AUDIT_HOLD",
    "AUDIT_INVALID",
    "AUDIT_PASS",
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION_V2",
    "OneShotResult",
    "OneShotResultV2",
    "REPORT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION_V2",
    "TRUST_POLICY_SCHEMA_VERSION",
    "V1_DEPLOYMENT_STATE",
    "V1_SECURITY_STATUS",
    "canonical_json_bytes",
    "canonical_sha256",
    "run_passive_oneshot",
    "run_passive_oneshot_v2",
    "seal_bundle",
    "seal_mapping",
    "seal_observation",
    "seal_observation_v2",
    "seal_profile",
    "seal_profile_v2",
    "seal_receipt_v2",
    "seal_tensor_contract_v2",
    "seal_trust_policy_v2",
    "sign_bundle_v2",
    "tensor_values_sha256",
]
