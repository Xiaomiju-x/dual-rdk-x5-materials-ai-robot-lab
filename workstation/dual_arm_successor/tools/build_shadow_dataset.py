"""Build a canonical read-only shadow dataset from an explicit evidence copy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workstation.dual_arm_successor.adapters import (  # noqa: E402
    DatasetAdapterError,
    build_shadow_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read frozen finals JSON evidence and write a stage-only canonical shadow dataset. "
            "This command does not access cameras, robots, serial devices, GPIO, ROS, or networks."
        )
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Existing copied evidence directory containing result.json and visual gate JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing directories are refused.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_shadow_dataset(args.evidence_dir, args.output_dir)
    except DatasetAdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
