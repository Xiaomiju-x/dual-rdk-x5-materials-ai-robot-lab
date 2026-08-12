"""Read-only sealing of existing X5-RB-VoE evidence artifacts.

The module inventories bytes. It never imports, executes, decodes, or uploads an
artifact. Historical evidence may support an offline replay claim, but this
module deliberately cannot issue or upgrade a physical execution qualification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive

_MANIFEST_SCHEMA = "xrd-rb-voe-historical-artifact-manifest-v1"
_CHALLENGE_SCHEMA = "xrd-rb-voe-hidden-challenge-manifest-v1"
_READ_CHUNK_BYTES = 1024 * 1024


class HistoricalArtifactType(str, Enum):
    """Frozen evidence categories; categories do not imply authority."""

    XRD_RAW = "XRD_RAW"
    PL_CSV = "PL_CSV"
    FOUR_LINE_RESULT = "FOUR_LINE_RESULT"
    MCAP = "MCAP"
    MPPI = "MPPI"
    F407 = "F407"
    ACCEPTANCE = "ACCEPTANCE"
    ARM_HISTORY = "ARM_HISTORY"
    DUAL_ARM_SIM = "DUAL_ARM_SIM"


class HistoricalCaptureMode(str, Enum):
    """Caller-declared origin; the declaration is retained, not trusted as a permit."""

    PHYSICAL_MEASUREMENT = "PHYSICAL_MEASUREMENT"
    PHYSICAL_EXECUTION = "PHYSICAL_EXECUTION"
    DERIVED_FROM_PHYSICAL = "DERIVED_FROM_PHYSICAL"
    HISTORICAL_UNVERIFIED = "HISTORICAL_UNVERIFIED"
    SIMULATION = "SIMULATION"


class ChallengePartition(str, Enum):
    REFERENCE = "REFERENCE"
    CHALLENGE = "CHALLENGE"


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Explicit provenance needed to interpret, but never execute, an artifact."""

    provenance_id: str
    source_system: str
    capture_mode: HistoricalCaptureMode
    lineage_sha256: str | None = None
    normalization_spec_sha256: str | None = None
    probability_semantics: str | None = None
    backend_provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_label("provenance_id", self.provenance_id)
        _require_label("source_system", self.source_system)
        if not isinstance(self.capture_mode, HistoricalCaptureMode):
            object.__setattr__(self, "capture_mode", HistoricalCaptureMode(self.capture_mode))
        for name in (
            "lineage_sha256",
            "normalization_spec_sha256",
            "backend_provenance_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(name, value)
        if self.probability_semantics is not None:
            _require_label("probability_semantics", self.probability_semantics, maximum=256)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


@dataclass(frozen=True, slots=True)
class HistoricalArtifactDeclaration:
    """An explicit local path paired with its type and provenance declaration."""

    path: str | Path
    artifact_type: HistoricalArtifactType
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path):
            raise ValueError("artifact path is required")
        if not isinstance(self.artifact_type, HistoricalArtifactType):
            object.__setattr__(self, "artifact_type", HistoricalArtifactType(self.artifact_type))
        if not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("provenance must be an ArtifactProvenance declaration")
        if (
            self.artifact_type is HistoricalArtifactType.DUAL_ARM_SIM
            and self.provenance.capture_mode is not HistoricalCaptureMode.SIMULATION
        ):
            raise ValueError("DUAL_ARM_SIM requires SIMULATION provenance")


@dataclass(frozen=True, slots=True)
class SealedHistoricalArtifact:
    relative_id: str
    artifact_type: HistoricalArtifactType
    byte_count: int
    sha256: str
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        _require_relative_id(self.relative_id)
        if not isinstance(self.artifact_type, HistoricalArtifactType):
            object.__setattr__(self, "artifact_type", HistoricalArtifactType(self.artifact_type))
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise TypeError("artifact byte_count must be an integer")
        if self.byte_count < 0:
            raise ValueError("artifact byte_count cannot be negative")
        require_sha256("artifact sha256", self.sha256)
        if not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("sealed artifact provenance has the wrong type")
        if (
            self.artifact_type is HistoricalArtifactType.DUAL_ARM_SIM
            and self.provenance.capture_mode is not HistoricalCaptureMode.SIMULATION
        ):
            raise ValueError("DUAL_ARM_SIM requires SIMULATION provenance")

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


@dataclass(frozen=True, slots=True)
class HistoricalReplayManifest:
    schema_version: str
    collection_id: str
    artifacts: tuple[SealedHistoricalArtifact, ...]
    limitation_codes: tuple[str, ...]
    physical_truth_available: bool
    execution_qualification_allowed: bool = False
    network_allowed: bool = False
    input_write_allowed: bool = False
    content_execution_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != _MANIFEST_SCHEMA:
            raise ValueError("unsupported historical manifest schema")
        _require_label("collection_id", self.collection_id)
        if not self.artifacts:
            raise ValueError("historical manifest requires at least one artifact")
        ordered = tuple(sorted(self.artifacts, key=lambda item: item.relative_id))
        if len({item.relative_id.casefold() for item in ordered}) != len(ordered):
            raise ValueError("historical manifest has duplicate relative identifiers")
        if len({item.sha256 for item in ordered}) != len(ordered):
            raise ValueError("historical manifest has duplicate artifact content")
        object.__setattr__(self, "artifacts", ordered)

        expected_limitations, expected_truth = _derive_semantics(ordered)
        supplied_limitations = tuple(sorted(set(self.limitation_codes)))
        if supplied_limitations != expected_limitations:
            raise ValueError("historical manifest limitation codes do not match artifacts")
        object.__setattr__(self, "limitation_codes", supplied_limitations)
        if not isinstance(self.physical_truth_available, bool):
            raise TypeError("physical_truth_available must be a boolean")
        if self.physical_truth_available is not expected_truth:
            raise ValueError("historical manifest physical-truth claim does not match artifacts")
        for name in (
            "execution_qualification_allowed",
            "network_allowed",
            "input_write_allowed",
            "content_execution_allowed",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
            if value:
                raise ValueError(f"historical replay forbids {name}")

    def unsigned_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))

    @property
    def root_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["root_sha256"] = self.root_sha256
        return payload


@dataclass(frozen=True, slots=True)
class HiddenChallengeMember:
    opaque_artifact_id: str
    partition: ChallengePartition

    def __post_init__(self) -> None:
        require_sha256("opaque_artifact_id", self.opaque_artifact_id)
        if not isinstance(self.partition, ChallengePartition):
            object.__setattr__(self, "partition", ChallengePartition(self.partition))

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


@dataclass(frozen=True, slots=True)
class HiddenChallengeManifest:
    schema_version: str
    source_manifest_sha256: str
    split_key_commitment_sha256: str
    challenge_count: int
    reference_count: int
    members: tuple[HiddenChallengeMember, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _CHALLENGE_SCHEMA:
            raise ValueError("unsupported hidden challenge schema")
        require_sha256("source_manifest_sha256", self.source_manifest_sha256)
        require_sha256("split_key_commitment_sha256", self.split_key_commitment_sha256)
        for name in ("challenge_count", "reference_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError("hidden split requires non-empty challenge and reference sets")
        ordered = tuple(sorted(self.members, key=lambda item: item.opaque_artifact_id))
        if len({item.opaque_artifact_id for item in ordered}) != len(ordered):
            raise ValueError("hidden challenge members must be unique")
        counts = {
            partition: sum(item.partition is partition for item in ordered)
            for partition in ChallengePartition
        }
        if counts[ChallengePartition.CHALLENGE] != self.challenge_count:
            raise ValueError("hidden challenge count does not match members")
        if counts[ChallengePartition.REFERENCE] != self.reference_count:
            raise ValueError("hidden reference count does not match members")
        object.__setattr__(self, "members", ordered)

    def unsigned_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))

    @property
    def root_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["root_sha256"] = self.root_sha256
        return payload


def build_historical_manifest(
    artifact_root: str | Path,
    declarations: tuple[HistoricalArtifactDeclaration, ...] | list[HistoricalArtifactDeclaration],
    *,
    collection_id: str,
    request_execution_qualification: bool = False,
) -> HistoricalReplayManifest:
    """Hash explicit files without decoding or modifying any artifact."""
    if not isinstance(request_execution_qualification, bool):
        raise TypeError("request_execution_qualification must be a boolean")
    if request_execution_qualification:
        raise ValueError("historical replay cannot grant execution qualification")
    root = _resolve_root(artifact_root)
    if not isinstance(declarations, (tuple, list)) or not declarations:
        raise ValueError("at least one historical artifact declaration is required")

    sealed: list[SealedHistoricalArtifact] = []
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, HistoricalArtifactDeclaration):
            raise TypeError("declarations must contain HistoricalArtifactDeclaration values")
        path, relative_id = _resolve_artifact(root, declaration.path)
        path_key = str(path).casefold()
        if path_key in seen_paths:
            raise ValueError("duplicate historical artifact path")
        seen_paths.add(path_key)
        byte_count, digest = _snapshot_file(path)
        if digest in seen_digests:
            raise ValueError("duplicate historical artifact content")
        seen_digests.add(digest)
        sealed.append(
            SealedHistoricalArtifact(
                relative_id=relative_id,
                artifact_type=declaration.artifact_type,
                byte_count=byte_count,
                sha256=digest,
                provenance=declaration.provenance,
            )
        )

    limitations, physical_truth = _derive_semantics(tuple(sealed))
    return HistoricalReplayManifest(
        schema_version=_MANIFEST_SCHEMA,
        collection_id=collection_id,
        artifacts=tuple(sealed),
        limitation_codes=limitations,
        physical_truth_available=physical_truth,
    )


def parse_historical_manifest(
    payload: dict[str, Any],
    *,
    expected_root_sha256: str,
) -> HistoricalReplayManifest:
    """Parse only a strict manifest whose root matches an independent pin."""
    require_sha256("expected_root_sha256", expected_root_sha256)
    expected_fields = {
        "schema_version",
        "collection_id",
        "artifacts",
        "limitation_codes",
        "physical_truth_available",
        "execution_qualification_allowed",
        "network_allowed",
        "input_write_allowed",
        "content_execution_allowed",
        "root_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("historical manifest fields do not match v1")
    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise TypeError("historical manifest artifacts must be a list")
    artifacts = tuple(_parse_artifact(item) for item in raw_artifacts)
    raw_limitations = payload["limitation_codes"]
    if not isinstance(raw_limitations, list) or not all(isinstance(item, str) for item in raw_limitations):
        raise TypeError("historical limitation codes must be a string list")
    manifest = HistoricalReplayManifest(
        schema_version=payload["schema_version"],
        collection_id=payload["collection_id"],
        artifacts=artifacts,
        limitation_codes=tuple(raw_limitations),
        physical_truth_available=payload["physical_truth_available"],
        execution_qualification_allowed=payload["execution_qualification_allowed"],
        network_allowed=payload["network_allowed"],
        input_write_allowed=payload["input_write_allowed"],
        content_execution_allowed=payload["content_execution_allowed"],
    )
    if payload["root_sha256"] != manifest.root_sha256:
        raise ValueError("historical manifest embedded root digest mismatch")
    if manifest.root_sha256 != expected_root_sha256:
        raise ValueError("historical manifest does not match the external pinned root")
    return manifest


def load_historical_manifest(
    path: str | Path,
    *,
    expected_root_sha256: str,
) -> HistoricalReplayManifest:
    """Read JSON as data only; no artifact referenced by it is imported or executed."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"historical manifest is missing: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError("historical manifest path must be a regular file")
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise TypeError("historical manifest root must be an object")
    return parse_historical_manifest(payload, expected_root_sha256=expected_root_sha256)


def verify_historical_artifacts(
    manifest: HistoricalReplayManifest,
    artifact_root: str | Path,
) -> None:
    """Re-hash the frozen relative inventory and reject missing or changed bytes."""
    if not isinstance(manifest, HistoricalReplayManifest):
        raise TypeError("manifest must be a HistoricalReplayManifest")
    root = _resolve_root(artifact_root)
    for artifact in manifest.artifacts:
        path, relative_id = _resolve_artifact(root, artifact.relative_id)
        if relative_id != artifact.relative_id:
            raise ValueError("historical artifact relative identifier changed")
        byte_count, digest = _snapshot_file(path)
        if byte_count != artifact.byte_count or digest != artifact.sha256:
            raise ValueError(f"historical artifact tamper detected: {artifact.relative_id}")


def build_hidden_challenge_manifest(
    manifest: HistoricalReplayManifest,
    *,
    split_key_sha256: str,
    challenge_count: int,
) -> HiddenChallengeManifest:
    """Create a deterministic keyed split containing no paths or provenance text."""
    if not isinstance(manifest, HistoricalReplayManifest):
        raise TypeError("manifest must be a HistoricalReplayManifest")
    require_sha256("split_key_sha256", split_key_sha256)
    if isinstance(challenge_count, bool) or not isinstance(challenge_count, int):
        raise TypeError("challenge_count must be an integer")
    if challenge_count < 1 or challenge_count >= len(manifest.artifacts):
        raise ValueError("challenge_count must leave non-empty challenge and reference sets")

    key = bytes.fromhex(split_key_sha256)
    ranked: list[tuple[bytes, SealedHistoricalArtifact]] = []
    for artifact in manifest.artifacts:
        score = hmac.new(
            key,
            b"xrd-rb-voe/hidden-split/score/v1\0" + bytes.fromhex(artifact.sha256),
            hashlib.sha256,
        ).digest()
        ranked.append((score, artifact))
    ranked.sort(key=lambda item: (item[0], item[1].sha256))
    challenge_digests = {artifact.sha256 for _, artifact in ranked[:challenge_count]}

    members: list[HiddenChallengeMember] = []
    for artifact in manifest.artifacts:
        opaque_id = hmac.new(
            key,
            b"xrd-rb-voe/hidden-split/id/v1\0" + bytes.fromhex(artifact.sha256),
            hashlib.sha256,
        ).hexdigest()
        partition = (
            ChallengePartition.CHALLENGE
            if artifact.sha256 in challenge_digests
            else ChallengePartition.REFERENCE
        )
        members.append(HiddenChallengeMember(opaque_artifact_id=opaque_id, partition=partition))

    commitment = hashlib.sha256(b"xrd-rb-voe/hidden-split/key-commitment/v1\0" + key).hexdigest()
    return HiddenChallengeManifest(
        schema_version=_CHALLENGE_SCHEMA,
        source_manifest_sha256=manifest.root_sha256,
        split_key_commitment_sha256=commitment,
        challenge_count=challenge_count,
        reference_count=len(manifest.artifacts) - challenge_count,
        members=tuple(members),
    )


def _derive_semantics(
    artifacts: tuple[SealedHistoricalArtifact, ...],
) -> tuple[tuple[str, ...], bool]:
    limitations = {"HISTORICAL_REPLAY_NO_EXECUTION_QUALIFICATION"}
    physical_truth = False
    for artifact in artifacts:
        provenance = artifact.provenance
        if artifact.artifact_type is HistoricalArtifactType.FOUR_LINE_RESULT:
            if provenance.normalization_spec_sha256 is None:
                limitations.add("FOUR_LINE_NORMALIZATION_SEMANTICS_MISSING")
            if provenance.probability_semantics is None:
                limitations.add("FOUR_LINE_PROBABILITY_SEMANTICS_MISSING")
            if provenance.backend_provenance_sha256 is None:
                limitations.add("FOUR_LINE_BACKEND_PROVENANCE_MISSING")
        elif artifact.artifact_type is HistoricalArtifactType.MPPI:
            limitations.add("MPPI_PROPOSAL_NOT_PHYSICAL_EXECUTION_TRUTH")
        elif artifact.artifact_type is HistoricalArtifactType.ARM_HISTORY:
            limitations.add("ARM_HISTORY_NOT_GRINDING_QUALIFICATION")
        elif artifact.artifact_type is HistoricalArtifactType.DUAL_ARM_SIM:
            limitations.add("DUAL_ARM_SIM_NOT_PHYSICAL_TRUTH")
            limitations.add("DUAL_ARM_SIM_NOT_GRINDING_QUALIFICATION")

        if _artifact_has_physical_truth(artifact):
            physical_truth = True
    if not physical_truth:
        limitations.add("PHYSICAL_TRUTH_UNAVAILABLE")
    return tuple(sorted(limitations)), physical_truth


def _artifact_has_physical_truth(artifact: SealedHistoricalArtifact) -> bool:
    if artifact.provenance.lineage_sha256 is None:
        return False
    measurement = {
        HistoricalArtifactType.XRD_RAW,
        HistoricalArtifactType.PL_CSV,
    }
    execution = {
        HistoricalArtifactType.MCAP,
        HistoricalArtifactType.F407,
        HistoricalArtifactType.ACCEPTANCE,
    }
    return (
        artifact.artifact_type in measurement
        and artifact.provenance.capture_mode is HistoricalCaptureMode.PHYSICAL_MEASUREMENT
    ) or (
        artifact.artifact_type in execution
        and artifact.provenance.capture_mode is HistoricalCaptureMode.PHYSICAL_EXECUTION
    )


def _parse_artifact(payload: object) -> SealedHistoricalArtifact:
    expected_fields = {"relative_id", "artifact_type", "byte_count", "sha256", "provenance"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("historical artifact fields do not match v1")
    raw_provenance = payload["provenance"]
    provenance_fields = {
        "provenance_id",
        "source_system",
        "capture_mode",
        "lineage_sha256",
        "normalization_spec_sha256",
        "probability_semantics",
        "backend_provenance_sha256",
    }
    if not isinstance(raw_provenance, dict) or set(raw_provenance) != provenance_fields:
        raise ValueError("historical provenance fields do not match v1")
    provenance = ArtifactProvenance(
        provenance_id=raw_provenance["provenance_id"],
        source_system=raw_provenance["source_system"],
        capture_mode=HistoricalCaptureMode(raw_provenance["capture_mode"]),
        lineage_sha256=raw_provenance["lineage_sha256"],
        normalization_spec_sha256=raw_provenance["normalization_spec_sha256"],
        probability_semantics=raw_provenance["probability_semantics"],
        backend_provenance_sha256=raw_provenance["backend_provenance_sha256"],
    )
    return SealedHistoricalArtifact(
        relative_id=payload["relative_id"],
        artifact_type=HistoricalArtifactType(payload["artifact_type"]),
        byte_count=payload["byte_count"],
        sha256=payload["sha256"],
        provenance=provenance,
    )


def _resolve_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"artifact root is missing: {root}")
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    return root.resolve(strict=True)


def _resolve_artifact(root: Path, declared_path: str | Path) -> tuple[Path, str]:
    path = Path(declared_path)
    candidate = path if path.is_absolute() else root / path
    if not candidate.exists():
        raise FileNotFoundError(f"historical artifact is missing: {candidate}")
    if candidate.is_symlink():
        raise ValueError("historical artifact symlinks are forbidden")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("historical artifact path must be a regular file")
    try:
        relative_id = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("historical artifact must stay within artifact_root") from error
    _require_relative_id(relative_id)
    return resolved, relative_id


def _snapshot_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    after = path.stat()
    before_state = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_state = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_state != after_state:
        raise ValueError("historical artifact changed while it was being hashed")
    return after.st_size, digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"historical manifest contains duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _require_relative_id(value: object) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("relative_id must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_id must not be absolute or traverse parents")
    if path.as_posix() != value:
        raise ValueError("relative_id is not in canonical POSIX form")


def _require_label(name: str, value: object, *, maximum: int = 128) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} is not a valid manifest label")


__all__ = [
    "ArtifactProvenance",
    "ChallengePartition",
    "HiddenChallengeManifest",
    "HiddenChallengeMember",
    "HistoricalArtifactDeclaration",
    "HistoricalArtifactType",
    "HistoricalCaptureMode",
    "HistoricalReplayManifest",
    "SealedHistoricalArtifact",
    "build_hidden_challenge_manifest",
    "build_historical_manifest",
    "load_historical_manifest",
    "parse_historical_manifest",
    "verify_historical_artifacts",
]
