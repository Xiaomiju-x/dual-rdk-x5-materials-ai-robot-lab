from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    EVIDENCE_ROOT,
    ROOT as COMMON_ROOT,
    sha256_file,
    utc_now,
    write_json,
)
from icmat_foundry.finals_post50_passive_rnd_20260803.training.train_xrd_peakjoint import (
    DATASET,
    XRDPeakJoint,
    build_degraded,
    infer,
    peak_targets,
    retrieval_metrics,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target_bool = target.astype(bool).reshape(-1)
    probability = probability.reshape(-1)
    prediction = probability >= 0.5
    tp = int(np.sum(prediction & target_bool))
    fp = int(np.sum(prediction & ~target_bool))
    fn = int(np.sum(~prediction & target_bool))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "fixed_threshold": 0.5,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "average_precision": float(average_precision_score(target_bool.astype(np.uint8), probability)),
        "positive_bins": int(np.sum(target_bool)),
        "predicted_positive_bins": int(np.sum(prediction)),
    }


def main() -> int:
    family_path = EVIDENCE_ROOT / "RND-XRD-01/family_receipt.v1.json"
    family = load_json(family_path)
    selection_lock = family.get("selection_lock", {})
    if selection_lock.get("fit_count_before_test") != 32 or selection_lock.get("post_test_retuning_allowed") is not False:
        raise RuntimeError("XRD selection lock is missing or not frozen")
    with np.load(DATASET, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}
    clean = bank["spectra"].astype(np.float32)
    test_clean = clean[bank["split_codes"] == 2]
    test_degraded = build_degraded(test_clean, "locked-test")
    exact_peaks = peak_targets(test_clean) >= 0.99
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    variants = {}
    for variant_id, export in sorted(family["exports"].items()):
        checkpoint_record = export["artifacts"]["checkpoint"]
        checkpoint_path = COMMON_ROOT / checkpoint_record["path"]
        if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
            raise RuntimeError(f"checkpoint digest mismatch: {variant_id}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["metadata"]["configuration"]
        model = XRDPeakJoint(int(config["width"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        _, peak_probability, query_embedding = infer(model, test_degraded, device)
        _, _, gallery_embedding = infer(model, test_clean, device)
        variants[variant_id] = {
            "direct_peak_head_fixed_0_5": binary_metrics(exact_peaks, peak_probability),
            "phase_embedding_retrieval": retrieval_metrics(query_embedding, gallery_embedding),
            "all_outputs_finite": bool(np.isfinite(peak_probability).all() and np.isfinite(query_embedding).all()),
            "configuration": config,
        }
    payload = {
        "schema": "x5_icmat_foundry.post50_xrd_aux_head_diagnostic.v1",
        "created_at": utc_now(),
        "family_id": "RND-XRD-01",
        "source_family_receipt": {
            "path": str(family_path.relative_to(COMMON_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(family_path),
        },
        "evaluation_policy": "Supplemental locked-test diagnostic after configuration freeze; fixed 0.5 threshold; no tuning or promotion-gate rewrite.",
        "variants": variants,
        "test_retuning_performed": False,
        "fit_count_added": 0,
        "candidate_class_changed": False,
        "network_used": False,
        "x5_contacted": False,
        "deployed": False,
        "claim_boundary": "Direct peak-head and embedding diagnostics on deterministic synthetic degradation of theoretical JARVIS XRD; not experimental-instrument evidence.",
    }
    output = EVIDENCE_ROOT / "RND-XRD-01/aux_head_diagnostic.v1.json"
    write_json(output, payload, seal=True)
    print(json.dumps({"path": str(output), "variants": variants}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
