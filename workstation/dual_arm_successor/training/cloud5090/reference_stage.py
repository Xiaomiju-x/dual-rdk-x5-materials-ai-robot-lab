#!/usr/bin/env python3
"""Record heavyweight model references without downloads or execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from cloud_common import TRUTH, load_yaml, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_yaml(Path(args.config).resolve())
    write_json(
        Path(args.out).resolve(),
        {
            "schema_version": "xrd-heavy-policy-reference-v1",
            "created_at": utc_now(),
            "status": "REFERENCE_ONLY_DRY_RUN",
            "executed": False,
            "downloaded": False,
            "truthfulness": TRUTH,
            "references": config["reference_only"],
            "claims": {
                "xr0_controls_xrd_arms": False,
                "openvla_oft_controls_xrd_arms": False,
                "xr_u0_runs_on_x5": False,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
