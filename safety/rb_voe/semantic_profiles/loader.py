"""Package-resource-only loader for frozen semantic profiles."""

from __future__ import annotations

from importlib import resources
from types import MappingProxyType
from typing import Final

from rb_voe.contracts.canonical import is_sha256
from rb_voe.semantic_profiles.models import (
    PROFILE_IDS,
    SemanticProfile,
    SemanticProfileError,
    _decode_profile_document,
)

_PROFILE_RESOURCES: Final = MappingProxyType(
    {
        "ai_x5.v1": "ai_x5.v1.json",
        "embodied_x5.v1": "embodied_x5.v1.json",
        "dual_arm.v1": "dual_arm.v1.json",
        "assay_station.v1": "assay_station.v1.json",
    }
)
_PROFILE_SUBSYSTEMS: Final[frozenset[str]] = frozenset(
    profile_id.removesuffix(".v1") for profile_id in PROFILE_IDS
)
_MAX_PROFILE_BYTES: Final[int] = 64 * 1024


def available_profile_ids() -> tuple[str, ...]:
    """Return the complete package-local profile registry."""
    return PROFILE_IDS


def _normalize_profile_id(profile_id: str, version: str | None) -> str:
    if not isinstance(profile_id, str) or not profile_id:
        raise SemanticProfileError("profile_id must be a non-empty string")
    if version is None:
        candidate = f"{profile_id}.v1" if profile_id in _PROFILE_SUBSYSTEMS else profile_id
    else:
        if not isinstance(version, str) or profile_id not in _PROFILE_SUBSYSTEMS or version != "v1":
            raise SemanticProfileError("only registered package profile versions may be loaded")
        candidate = f"{profile_id}.{version}"
    if candidate not in _PROFILE_RESOURCES:
        raise SemanticProfileError(f"unregistered package semantic profile: {candidate!r}")
    return candidate


def _read_profile_bytes(profile_id: str) -> bytes:
    resource_name = _PROFILE_RESOURCES[profile_id]
    try:
        raw = resources.files("rb_voe.semantic_profiles").joinpath(resource_name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise SemanticProfileError(f"bundled semantic profile is unavailable: {profile_id}") from exc
    if not raw or len(raw) > _MAX_PROFILE_BYTES:
        raise SemanticProfileError("bundled semantic profile has an invalid size")
    return raw


def load_profile(
    profile_id: str,
    version: str | None = None,
    *,
    expected_sha256: str | None = None,
) -> SemanticProfile:
    """Load one allowlisted package resource; filesystem paths are never accepted."""
    normalized = _normalize_profile_id(profile_id, version)
    profile = _decode_profile_document(
        _read_profile_bytes(normalized),
        expected_profile_id=normalized,
    )
    if expected_sha256 is not None:
        if not is_sha256(expected_sha256):
            raise SemanticProfileError("expected_sha256 must be a lowercase SHA-256 digest")
        if profile.profile_sha256 != expected_sha256:
            raise SemanticProfileError("semantic profile digest differs from the caller pin")
    return profile


def load_all_profiles() -> MappingProxyType:
    """Load all four frozen profiles into an immutable mapping."""
    return MappingProxyType({profile_id: load_profile(profile_id) for profile_id in PROFILE_IDS})


__all__ = ["available_profile_ids", "load_all_profiles", "load_profile"]
