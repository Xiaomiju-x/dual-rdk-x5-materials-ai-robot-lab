"""Command-line entry point for authenticated PASSIVE_ONESHOT v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from rb_voe_passive.errors import EvidenceError, PathPolicyError, TrustPolicyError
from rb_voe_passive.oneshot_v2 import (
    AUDIT_INVALID,
    EXIT_CODES_V2,
    run_passive_oneshot_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_rb_voe_passive_oneshot",
        description=(
            "Authenticate and audit one signed offline v2 bundle under an explicit "
            "local trust policy."
        ),
    )
    parser.add_argument(
        "--trust-policy",
        required=True,
        help="absolute path to an explicit rb_voe_passive_trust_policy.v2 JSON",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="absolute path to a direct-child passive_bundle.v2 JSON in the policy inbox",
    )
    parser.add_argument(
        "--evidence-root",
        required=True,
        help="absolute path that must exactly match the policy evidence root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_passive_oneshot_v2(
            trust_policy_path=args.trust_policy,
            bundle_path=args.bundle,
            evidence_root=args.evidence_root,
        )
    except (TrustPolicyError, PathPolicyError, EvidenceError) as exc:
        payload = {
            "status": AUDIT_INVALID,
            "report_path": None,
            "report_sha256": None,
            "error_code": exc.code,
            "message": exc.message,
        }
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return EXIT_CODES_V2[AUDIT_INVALID]

    payload = {
        "status": result.status,
        "report_path": str(result.report_path),
        "report_sha256": result.report["report_sha256"],
    }
    if result.status == AUDIT_INVALID:
        finding = result.report["findings"][0]
        payload["error_code"] = finding["code"]
        payload["message"] = finding["message"]
    stream = sys.stdout if result.status != AUDIT_INVALID else sys.stderr
    stream.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
