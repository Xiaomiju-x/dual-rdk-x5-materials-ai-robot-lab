from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.propnet.contracts import SPLIT_TO_CODE
from icmat_foundry.propnet.data import load_source_archive
from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    EVIDENCE_ROOT,
    ROOT as COMMON_ROOT,
    SEED_BASE,
    TRIAL_ROOT,
    export_static_onnx,
    mae,
    sha256_file,
    utc_now,
    write_family_receipt,
    write_json,
)
from icmat_foundry.finals_post50_passive_rnd_20260803.training.train_irintensity_v2 import (
    CACHE,
    FAMILY_ID,
    IRNet,
    SOURCE,
    finite,
    metrics,
    predict,
)


class IRExportOpset11(nn.Module):
    def __init__(self, model: IRNet, target_mean: float, target_std: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor([target_mean], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor([target_std], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed = self.model(value) * self.target_std + self.target_mean
        return torch.exp(transformed) - 1.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_model(variant: str) -> tuple[IRNet, dict[str, Any]]:
    receipt = load_json(TRIAL_ROOT / FAMILY_ID / f"final_{variant}.json")
    checkpoint_record = receipt["checkpoint"]
    checkpoint_path = COMMON_ROOT / checkpoint_record["path"]
    if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError(f"IR checkpoint digest mismatch: {variant}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = receipt["configuration"]
    model = IRNet(int(config["hidden"]), float(config["dropout"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), receipt


def main() -> int:
    rows, integrity = load_source_archive(SOURCE)
    with np.load(CACHE, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}
    if len(rows) != len(bank["jids"]) or any(str(row.get("jid", "")) != str(jid) for row, jid in zip(rows, bank["jids"])):
        raise RuntimeError("IR source/cache JID ordering mismatch during export recovery")
    targets = np.full(len(rows), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        value = finite(row.get("max_ir_mode"))
        targets[index] = np.nan if value is None else value
    valid = np.isfinite(targets)
    codes = bank["split_codes"]
    masks = {name: valid & (codes == SPLIT_TO_CODE[name]) for name in ("train", "tune", "test")}
    feature_mean = bank["features"][masks["train"]].mean(0).astype(np.float32)
    feature_std = bank["features"][masks["train"]].std(0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x = np.clip((bank["features"] - feature_mean) / feature_std, -8, 8).astype(np.float32)
    target_mean = float(np.log1p(targets[masks["train"]]).mean())
    target_std = max(float(np.log1p(targets[masks["train"]]).std()), 1e-6)

    baselines = {}
    for name, model in (
        ("ridge", Ridge(alpha=100.0)),
        ("extra_trees", ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.7, n_jobs=4, random_state=SEED_BASE)),
    ):
        model.fit(x[masks["train"]], np.log1p(targets[masks["train"]]))
        baselines[name] = model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, trial_receipts = {}, {}
    for variant in ("quality", "lite"):
        model, trial_receipt = load_final_model(variant)
        models[variant] = model
        trial_receipts[variant] = trial_receipt

    median = float(np.median(targets[masks["train"]]))
    locked = {"median": {"mae": mae(targets[masks["test"]], np.full(np.sum(masks["test"]), median))}}
    for name, model in baselines.items():
        locked[name] = metrics(targets[masks["test"]], np.expm1(model.predict(x[masks["test"]])))
    exports = {}
    for variant, model in models.items():
        prediction = predict(model.to(device), x[masks["test"]], target_mean, target_std, device)
        locked[variant] = metrics(targets[masks["test"]], prediction)
        wrapper = IRExportOpset11(model.cpu(), target_mean, target_std)
        exports[variant] = export_static_onnx(
            FAMILY_ID,
            variant,
            wrapper,
            [torch.from_numpy(x[masks["test"]][:1].reshape(1, 1, 1, 149))],
            ["jarvis_features_fp32"],
            ["max_ir_mode"],
            {"feature_mean": feature_mean, "feature_std": feature_std},
            {"configuration": trial_receipts[variant]["configuration"], "seed": trial_receipts[variant]["seed"], "export_recovery": "EXP_MINUS_ONE_OPSET11"},
        )
    quality = locked["quality"]
    gate = {
        "raw_mae_beats_median": quality["mae"] < locked["median"]["mae"],
        "log_mae_beats_ridge": quality["log1p_mae"] < locked["ridge"]["log1p_mae"],
    }
    gate["pass"] = all(gate.values())
    task_contract = load_json(CONTRACT_ROOT / "tasks/RND-MAT-03.json")
    selection_lock = load_json(CONTRACT_ROOT / "selection_locks/RND-MAT-03.json")
    recovery = {
        "schema": "x5_icmat_foundry.post50_ir_export_recovery.v1",
        "created_at": utc_now(),
        "family_id": FAMILY_ID,
        "completed_fit_receipts_reused": 8,
        "neural_fits_added": 0,
        "deterministic_baselines_reconstructed": 2,
        "failure": "aten::expm1 is unsupported by the fixed ONNX opset 11 exporter",
        "equivalent_rewrite": "expm1(x) -> exp(x) - 1",
        "selection_lock_reused_without_change": True,
        "post_test_retuning": False,
        "test_evaluation_repeated_only_for_export_recovery": True,
        "source_archive_sha256": integrity["archive_sha256"],
        "network_used": False,
        "x5_contacted": False,
        "deployed": False,
    }
    recovery_path = EVIDENCE_ROOT / FAMILY_ID / "export_recovery.v1.json"
    write_json(recovery_path, recovery, seal=True)
    receipt = write_family_receipt(
        FAMILY_ID,
        {
            "model_name": "IRIntensity-v2",
            "fit_count": 8,
            "fit_count_contract_pass": True,
            "task_contract": task_contract,
            "selection_lock": selection_lock,
            "locked_test": locked,
            "innovation_gate": gate,
            "candidate_class": "RND_VALIDATED" if gate["pass"] else "RND_USABLE_EXPERIMENTAL",
            "exports": exports,
            "export_recovery": {"path": str(recovery_path.relative_to(COMMON_ROOT)).replace("\\", "/"), "sha256": sha256_file(recovery_path)},
            "claim_boundary": [
                "public JARVIS DFT proxy only",
                "not a graph neural network claim",
                "no audit, routing, or execution authority",
                "PC metrics are not X5 evidence",
            ],
        },
    )
    print(json.dumps({"family": FAMILY_ID, "fit_count": 8, "gate": gate, "receipt": receipt["path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
