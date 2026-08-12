#!/usr/bin/env python3
"""Run the isolated TinyOccFlow v5 evaluation/ablation harness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from embodied_brain.finals_successor.x5_tribev_flow.dataset import (
    SPLIT_NAMES,
    build_episode_refs,
    split_episode_refs,
)
from embodied_brain.finals_successor.x5_tribev_flow.evaluation import (
    TorchCheckpointPredictor,
    default_artifact_hashes,
    evaluate_offline,
    load_conformal_metadata,
    write_deterministic_json,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TinyOccFlow offline with scenario metrics, diagnostic "
            "baselines, fixed ablations, subsets, and optional conformal coverage."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument(
        "--split-seed",
        type=int,
        help=(
            "session split seed; default reads the checkpoint sibling "
            "training_report.json and refuses to guess if it is unavailable"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration-metadata", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    sibling_report = (
        args.checkpoint.expanduser().resolve().parent / "training_report.json"
    )
    training_report = (
        json.loads(sibling_report.read_text(encoding="utf-8"))
        if sibling_report.is_file()
        else None
    )
    if args.split_seed is None:
        if not isinstance(training_report, dict) or "seed" not in training_report:
            raise RuntimeError(
                "--split-seed is required when training_report.json is unavailable"
            )
        split_seed = int(training_report["seed"])
        split_seed_source = "checkpoint_sibling_training_report"
    else:
        split_seed = int(args.split_seed)
        split_seed_source = "explicit_cli"
    all_refs = build_episode_refs(args.dataset_root)
    splits = split_episode_refs(all_refs, seed=split_seed)
    refs = splits[args.split]
    if not refs:
        raise RuntimeError(
            f"{args.split} split is empty; generate more independent sessions"
        )

    calibration_path = args.calibration_metadata
    if calibration_path is None:
        calibration_path = sibling_report if sibling_report.is_file() else None
    conformal_metadata, calibration_artifact = load_conformal_metadata(
        calibration_path
    )

    predictor = TorchCheckpointPredictor(args.checkpoint, device=args.device)
    artifact_hashes = default_artifact_hashes(
        refs,
        checkpoint_path=args.checkpoint,
        calibration_artifact=calibration_artifact,
    )
    report = evaluate_offline(
        refs,
        predictor,
        empirical_prior_refs=splits["train"],
        conformal_metadata=conformal_metadata,
        artifact_hashes=artifact_hashes,
        predictor_name="TinyOccFlowStudent-v5-checkpoint",
    )
    report["split_seed"] = split_seed
    report["split_seed_source"] = split_seed_source
    receipt = write_deterministic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": receipt,
                "episodes": report["episode_count"],
                "split": args.split,
                "source_summary": report["source_summary"],
                "claim_boundary": report["claim_boundary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
