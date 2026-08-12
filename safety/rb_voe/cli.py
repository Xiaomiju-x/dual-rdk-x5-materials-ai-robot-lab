"""Offline-only R0/R1 command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rb_voe import __version__
from rb_voe.audit import LedgerVerificationError, diagnose_ledger, verify_r1_semantic_ledger
from rb_voe.contracts.validation import (
    ContractValidationError,
    available_schema_versions,
    validate_contract,
)
from rb_voe.demo import DemoVerificationError, run_demo
from rb_voe.release import verify_manifest_payload, verify_release_bundle_artifacts
from rb_voe.shadow import (
    ManifestPayloadConnector,
    ShadowCoordinator,
    ShadowMode,
    ShadowRunBinding,
    ShadowStatus,
)
from rb_voe.shadow.coordinator import SCHEMA_BY_SUBSYSTEM


def _emit(payload: dict, *, stream=None) -> None:
    target = stream or sys.stdout
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=target)


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _cmd_info(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "x5-rb-voe",
            "version": __version__,
            "release_stage": "R1_INTEGRATION_PREPARED_R2_LIVE_NOT_RUN",
            "hardware_authority": False,
            "network_used": False,
            "available_commands": [
                "audit-ledger",
                "audit-r1-ledger",
                "demo",
                "info",
                "schemas",
                "shadow-preflight",
                "validate-contract",
                "verify-manifest",
                "verify-release-bundle",
            ],
        }
    )
    return 0


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    try:
        payload = _read_json_object(args.path)
        ok, reason = verify_manifest_payload(payload)
    except ValueError as exc:
        _emit(
            {"ok": False, "reason_code": "MANIFEST_READ_ERROR", "detail": str(exc)},
            stream=sys.stderr,
        )
        return 2
    result = {"ok": ok, "reason_code": reason, "path": args.path.as_posix()}
    _emit(result, stream=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


def _cmd_schemas(_: argparse.Namespace) -> int:
    _emit(
        {
            "ok": True,
            "schema_versions": list(available_schema_versions()),
            "hardware_authority": False,
            "network_used": False,
        }
    )
    return 0


def _cmd_validate_contract(args: argparse.Namespace) -> int:
    try:
        payload = _read_json_object(args.path)
        record = validate_contract(payload, now_ms=args.now_ms)
    except (ValueError, ContractValidationError) as exc:
        _emit(
            {"ok": False, "reason_code": "CONTRACT_INVALID", "detail": str(exc)},
            stream=sys.stderr,
        )
        return 1
    _emit(
        {
            "ok": True,
            "reason_code": "PASS",
            "schema_version": record.schema_version,
            "content_sha256": record.content_sha256,
        }
    )
    return 0


def _cmd_audit_ledger(args: argparse.Namespace) -> int:
    result = diagnose_ledger(args.path)
    _emit(result, stream=sys.stdout if result["ok"] else sys.stderr)
    return 0 if result["ok"] else 1


def _cmd_audit_r1_ledger(args: argparse.Namespace) -> int:
    try:
        report = verify_r1_semantic_ledger(args.path)
    except (LedgerVerificationError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, LedgerVerificationError) else "R1_LEDGER_INVALID"
        _emit({"ok": False, "reason_code": code, "detail": str(exc)}, stream=sys.stderr)
        return 1
    _emit({"ok": True, "reason_code": "PASS", **report.to_dict()})
    return 0


def _cmd_verify_release_bundle(args: argparse.Namespace) -> int:
    try:
        release_payload = _read_json_object(args.release_manifest)
        terminal_payload = _read_json_object(args.terminal_manifest)
        pin_payload = _read_json_object(args.external_pin)
        registry_inventory = _read_json_object(args.registry_inventory)
        policy_inventory = _read_json_object(args.policy_inventory)
        environment_inventory = _read_json_object(args.environment_inventory)
    except ValueError as exc:
        _emit(
            {"ok": False, "reason_code": "RELEASE_BUNDLE_READ_ERROR", "detail": str(exc)},
            stream=sys.stderr,
        )
        return 2
    ok, reason = verify_release_bundle_artifacts(
        release_payload=release_payload,
        terminal_payload=terminal_payload,
        external_pin_payload=pin_payload,
        ledger_path=args.ledger,
        project_root=args.project_root,
        actual_registry_sha256=registry_inventory,
        actual_policy_config_sha256=policy_inventory,
        actual_environment_sha256=environment_inventory,
    )
    _emit({"ok": ok, "reason_code": reason}, stream=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    try:
        result = run_demo(args.output_dir, external_pin_path=args.external_pin)
    except (DemoVerificationError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, DemoVerificationError) else "DEMO_FAILED"
        _emit(
            {"ok": False, "reason_code": code, "detail": str(exc)},
            stream=sys.stderr,
        )
        return 1
    pin_verification = result["external_pin_verification"]
    if not pin_verification["verified"] and args.bootstrap_unpinned:
        candidate = result["candidate_release_root"]
        _emit(
            {
                "ok": False,
                "reason_code": "UNPINNED_CANDIDATE",
                "detail": "candidate root was minted but is not an independently retained pin",
                "output_dir": args.output_dir.as_posix(),
                "candidate_release_root_sha256": candidate["root_sha256"],
                "authority": result["authority"],
            },
            stream=sys.stderr,
        )
        return 3
    if not pin_verification["verified"]:
        _emit(
            {
                "ok": False,
                "reason_code": pin_verification["reason_code"],
                "detail": "supply an independently retained --external-pin, or use "
                "--bootstrap-unpinned only to mint a non-passing candidate",
            },
            stream=sys.stderr,
        )
        return 1
    summary = {
        "ok": True,
        "reason_code": "PASS",
        "output_dir": args.output_dir.as_posix(),
        "root_option_id": result["policy_plan"]["root_option_id"],
        "policy_plan_sha256": result["policy_plan_sha256"],
        "episode_sha256": result["episode_sha256"],
        "root_scenario_id": result["episode"]["root_scenario_id"],
        "observed_options": [observation["option_id"] for observation in result["episode"]["observations"]],
        "ledger_record_count": result["ledger_verification"]["record_count"],
        "ledger_terminal_sha256": result["ledger_verification"]["terminal_record_sha256"],
        "golden_vector_verified": result["golden_vector_verified"],
        "external_pin_verification": pin_verification,
        "authority": result["authority"],
    }
    _emit(result if args.full else summary)
    return 0


def _cmd_shadow_preflight(args: argparse.Namespace) -> int:
    if args.mode != ShadowMode.OFFLINE_REPLAY.value:
        _emit(
            {
                "ok": False,
                "reason_code": "STATIC_MANIFEST_LIVE_FORBIDDEN",
                "detail": "shadow-preflight accepts captured manifests only in OFFLINE_REPLAY mode",
            },
            stream=sys.stderr,
        )
        return 2
    paths = {
        "ai_x5": args.ai_manifest,
        "embodied_x5": args.embodied_manifest,
        "dual_arm": args.dual_arm_manifest,
        "assay_station": args.assay_manifest,
    }
    try:
        connectors = {
            subsystem: ManifestPayloadConnector(
                subsystem=subsystem,
                capability_schema_version=SCHEMA_BY_SUBSYSTEM[subsystem],
                payload=_read_json_object(path),
            )
            for subsystem, path in paths.items()
        }
        binding = ShadowRunBinding(
            run_id=args.run_id,
            release_id=args.release_id,
            evaluated_at_ms=args.now_ms,
            mode=ShadowMode(args.mode),
        )
    except (OSError, TypeError, ValueError) as exc:
        _emit(
            {"ok": False, "reason_code": "SHADOW_INPUT_INVALID", "detail": str(exc)},
            stream=sys.stderr,
        )
        return 2
    report = ShadowCoordinator(connectors).evaluate(binding)
    payload = {
        "ok": report.status is ShadowStatus.SHADOW_READY,
        "reason_code": report.status.value,
        "report_sha256": report.content_sha256,
        **report.to_dict(),
    }
    _emit(payload, stream=sys.stdout if payload["ok"] else sys.stderr)
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rb_voe",
        description="Offline R0/R1 tooling for the X5-RB-VoE evidence-action compiler.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="show maturity and authority boundaries")
    info_parser.set_defaults(handler=_cmd_info)

    verify_parser = subparsers.add_parser(
        "verify-manifest",
        help="verify a content-addressed release or terminal manifest",
    )
    verify_parser.add_argument("path", type=Path)
    verify_parser.set_defaults(handler=_cmd_verify_manifest)

    schemas_parser = subparsers.add_parser("schemas", help="list frozen R0 schema versions")
    schemas_parser.set_defaults(handler=_cmd_schemas)

    contract_parser = subparsers.add_parser(
        "validate-contract",
        help="validate one frozen contract without contacting a device",
    )
    contract_parser.add_argument("path", type=Path)
    contract_parser.add_argument("--now-ms", type=int)
    contract_parser.set_defaults(handler=_cmd_validate_contract)

    ledger_parser = subparsers.add_parser(
        "audit-ledger",
        help="verify a canonical audit ledger and terminal anchor",
    )
    ledger_parser.add_argument("path", type=Path)
    ledger_parser.set_defaults(handler=_cmd_audit_ledger)

    semantic_ledger_parser = subparsers.add_parser(
        "audit-r1-ledger",
        help="verify the strict CASE -> PLAN -> OBSERVATION+ -> TERMINAL semantics",
    )
    semantic_ledger_parser.add_argument("path", type=Path)
    semantic_ledger_parser.set_defaults(handler=_cmd_audit_r1_ledger)

    shadow_parser = subparsers.add_parser(
        "shadow-preflight",
        help="validate four captured capability manifests in offline replay mode",
    )
    shadow_parser.add_argument("--ai-manifest", type=Path, required=True)
    shadow_parser.add_argument("--embodied-manifest", type=Path, required=True)
    shadow_parser.add_argument("--dual-arm-manifest", type=Path, required=True)
    shadow_parser.add_argument("--assay-manifest", type=Path, required=True)
    shadow_parser.add_argument("--release-id", required=True)
    shadow_parser.add_argument("--run-id", required=True)
    shadow_parser.add_argument("--now-ms", type=int, required=True)
    shadow_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ShadowMode],
        default=ShadowMode.OFFLINE_REPLAY.value,
    )
    shadow_parser.set_defaults(handler=_cmd_shadow_preflight)

    bundle_parser = subparsers.add_parser(
        "verify-release-bundle",
        help="verify release and terminal manifests against an externally retained root",
    )
    bundle_parser.add_argument("release_manifest", type=Path)
    bundle_parser.add_argument("terminal_manifest", type=Path)
    bundle_parser.add_argument("external_pin", type=Path)
    bundle_parser.add_argument("--ledger", type=Path, required=True)
    bundle_parser.add_argument("--project-root", type=Path, required=True)
    bundle_parser.add_argument("--registry-inventory", type=Path, required=True)
    bundle_parser.add_argument("--policy-inventory", type=Path, required=True)
    bundle_parser.add_argument("--environment-inventory", type=Path, required=True)
    bundle_parser.set_defaults(handler=_cmd_verify_release_bundle)

    demo_parser = subparsers.add_parser(
        "demo",
        help="run the deterministic SIMULATED_COUNTERFACTUAL R1 golden demo",
    )
    demo_parser.add_argument("--output-dir", type=Path, required=True)
    pin_group = demo_parser.add_mutually_exclusive_group()
    pin_group.add_argument("--external-pin", type=Path)
    pin_group.add_argument("--bootstrap-unpinned", action="store_true")
    demo_parser.add_argument("--full", action="store_true")
    demo_parser.set_defaults(handler=_cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
