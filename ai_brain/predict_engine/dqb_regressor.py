"""dqb_regressor.py — P0-1: Cr3+ Dq/B continuous regressor.

Architecture:
    formula_descriptor (24d, shared with TSPredictor) → MLP 256 → 128 → 2
    Output heads:
        raw_Dq → sigmoid scale to [1200, 2400] cm^-1
        raw_B  → sigmoid scale to [550, 800]   cm^-1
    λ_em_pred derived via 10Dq = E_4T2, Stokes shift 1800 cm^-1 fallback.

Benchmark:
    Birmingham/UCSB 2025 Chem.Mater. "Machine-Learning-Guided Discovery of
    Cr3+ NIR Phosphors" trains a Gaussian-process regressor on 100+ Cr3+ hosts
    mapping composition → (Dq/B) ratio. We replicate with PyTorch MLP on our
    39-sample corpus (22 lit + 17 empirical) and expose as /api/predict_dqb.

Usage (public):
    from predict_engine.dqb_regressor import predict_dqb
    predict_dqb("Y3Al5O12", "Al", 1.0)
    # → {"Dq_cm1": 1638, "B_cm1": 651, "Dq_over_B": 2.52, ...}
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from predict_engine.ts_torch import formula_descriptor   # 24d shared encoder

SEED = 42
DQ_LO, DQ_HI = 1200.0, 2400.0
B_LO, B_HI = 550.0, 800.0
STOKES_DEFAULT_CM1 = 1800.0    # Henderson & Imbusch 1989; see dqb_prepare_data.py
CKPT_PATH = ROOT / "predict_engine" / "dqb_regressor.pt"


class DqBRegressor(nn.Module):
    """24d formula descriptor → (Dq_cm1, B_cm1)."""

    def __init__(self, in_dim: int = 24, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden1, hidden2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(x)                  # [B, 2]
        Dq = DQ_LO + torch.sigmoid(raw[..., 0]) * (DQ_HI - DQ_LO)
        B  = B_LO  + torch.sigmoid(raw[..., 1]) * (B_HI - B_LO)
        return {"Dq_cm1": Dq, "B_cm1": B, "Dq_over_B": Dq / (B + 1e-6)}


# ---------- training ----------
def _load_corpus():
    data_path = ROOT / "predictions" / "dqb_train_data.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run `python tools/dqb_prepare_data.py` first.")
    return json.load(data_path.open(encoding="utf-8"))


def train(epochs: int = 3, lr: float = 5e-3, bs: int = 8,
          test_frac: float = 0.15, verbose: bool = True) -> Dict:
    """Train on 39-sample Cr3+ Dq/B corpus. Seed=42 fixed."""
    torch.manual_seed(SEED); random.seed(SEED)

    corpus = _load_corpus()
    samples = corpus["samples"]
    random.shuffle(samples)
    n_test = max(4, int(len(samples) * test_frac))
    test, train_ = samples[:n_test], samples[n_test:]
    if verbose:
        print(f"[dqb_regressor] train={len(train_)} test={len(test)} (epochs={epochs}, lr={lr})")

    def _to_tensors(rows):
        X = torch.stack([formula_descriptor(s["formula"], s.get("site") or "Al",
                                            float(s.get("pct") or 1.0)) for s in rows])
        y_Dq = torch.tensor([s["Dq_cm1"] for s in rows], dtype=torch.float32)
        y_B  = torch.tensor([s["B_cm1"]  for s in rows], dtype=torch.float32)
        return X, y_Dq, y_B

    X_tr, yDq_tr, yB_tr = _to_tensors(train_)
    X_te, yDq_te, yB_te = _to_tensors(test)

    model = DqBRegressor()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(train_)//bs))

    # Normalize loss: (Dq_pred - y)/300, (B_pred - y)/60
    SCALE_DQ, SCALE_B = 300.0, 60.0

    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i + bs]
            xb = X_tr[idx]; ydq = yDq_tr[idx]; yb = yB_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss_dq = ((out["Dq_cm1"] - ydq) / SCALE_DQ).pow(2).mean()
            loss_b  = ((out["B_cm1"] - yb)   / SCALE_B).pow(2).mean()
            loss = loss_dq + 0.5 * loss_b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / max(1, len(X_tr))

        # eval
        model.eval()
        with torch.no_grad():
            out_te = model(X_te)
            mae_dq_test = (out_te["Dq_cm1"] - yDq_te).abs().mean().item()
            mae_b_test  = (out_te["B_cm1"]  - yB_te).abs().mean().item()
            out_tr = model(X_tr)
            mae_dq_train = (out_tr["Dq_cm1"] - yDq_tr).abs().mean().item()
            mae_b_train  = (out_tr["B_cm1"]  - yB_tr).abs().mean().item()
        history.append({
            "epoch": ep + 1, "loss": round(avg_loss, 5),
            "train_mae_Dq_cm1": round(mae_dq_train, 1),
            "train_mae_B_cm1":  round(mae_b_train, 1),
            "test_mae_Dq_cm1":  round(mae_dq_test, 1),
            "test_mae_B_cm1":   round(mae_b_test, 1),
        })
        if verbose:
            print(f"  ep {ep+1} loss={avg_loss:.4f}  "
                  f"train_MAE Dq={mae_dq_train:.1f}/B={mae_b_train:.1f}  "
                  f"test_MAE Dq={mae_dq_test:.1f}/B={mae_b_test:.1f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "seed": SEED, "epochs": epochs, "lr": lr,
        "n_train": len(train_), "n_test": len(test),
        "test_mae_Dq_cm1": history[-1]["test_mae_Dq_cm1"],
        "test_mae_B_cm1":  history[-1]["test_mae_B_cm1"],
        "history": history,
        "arch": {"in_dim": 24, "hidden1": 256, "hidden2": 128,
                 "Dq_range": [DQ_LO, DQ_HI], "B_range": [B_LO, B_HI]},
    }
    torch.save(ckpt, CKPT_PATH)
    if verbose:
        print(f"[write] {CKPT_PATH}")
    return {"ckpt_path": str(CKPT_PATH), "history": history, **{k: v for k, v in ckpt.items() if k != "model_state_dict"}}


# ---------- inference ----------
_MODEL_CACHE = {"m": None}


def _load_model():
    if _MODEL_CACHE["m"] is not None:
        return _MODEL_CACHE["m"]
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"{CKPT_PATH} missing. Run `python predict_engine/dqb_regressor.py --train`.")
    m = DqBRegressor()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    _MODEL_CACHE["m"] = m
    return m


def _host_family_hint(dq_over_b: float, Dq: float) -> str:
    """Map Dq/B → host_family heuristic (matches ts_inverse.py)."""
    if dq_over_b > 2.6:
        return "Strong-field (spinel/perovskite, e.g. ZnGa2O4, MgAl2O4, LaAlO3)"
    if dq_over_b > 2.35:
        return "Intermediate-field (garnet/olivine, e.g. YAG, GGG, GAGG)"
    if dq_over_b > 2.05:
        return "Weak-intermediate (gallate/indate, e.g. LaGaO3, La3Ga5SiO14)"
    return "Weak-field (scandate/fluoride, e.g. CaSc2O4, Ca3Sc2Si3O12)"


def _reference_papers(source_match: str) -> list[str]:
    """Return relevant paper DOIs from cft_params for the closest family match."""
    cft_path = ROOT / "predict_engine" / "cft_params.json"
    try:
        d = json.load(cft_path.open(encoding="utf-8"))
    except Exception:
        return []
    refs = []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        doi = v.get("source_doi")
        if doi and source_match.lower() in (v.get("notes", "") + " " + k).lower():
            refs.append(f"{k} ({doi})")
    return refs[:3]


def predict_dqb(formula: str, site: str = "Al", pct: float = 1.0,
                stokes_cm1: float = STOKES_DEFAULT_CM1) -> Dict:
    """Single-formula Dq/B prediction with λ_em back-projection."""
    model = _load_model()
    feat = formula_descriptor(formula, site, pct).unsqueeze(0)
    with torch.no_grad():
        out = model(feat)
    Dq = float(out["Dq_cm1"].item())
    B = float(out["B_cm1"].item())
    dqb = Dq / B
    # λ_em back-projection: 10Dq = E_4T2, λ_em = 1e7/(E_4T2 − Stokes)
    E_4T2 = 10.0 * Dq
    E_em = E_4T2 - stokes_cm1
    lam_em = 1.0e7 / max(E_em, 1.0)

    # Family hint + papers
    if dqb > 2.6:
        hint_key = "strong"; paper_lookup = "spinel"
    elif dqb > 2.35:
        hint_key = "inter"; paper_lookup = "garnet"
    elif dqb > 2.05:
        hint_key = "weak_inter"; paper_lookup = "gallate"
    else:
        hint_key = "weak"; paper_lookup = "scandate"
    return {
        "formula": formula,
        "dopant": {"element": "Cr3+", "site": site, "pct": pct},
        "Dq_cm1": round(Dq, 1),
        "B_cm1": round(B, 1),
        "Dq_over_B": round(dqb, 3),
        "lambda_em_predicted_nm": round(lam_em, 1),
        "E_4T2_cm1": round(E_4T2, 1),
        "stokes_cm1_used": stokes_cm1,
        "host_family_hint": _host_family_hint(dqb, Dq),
        "reference_papers": _reference_papers(paper_lookup),
        "method": "MLP 24d->256->128->2, trained on 39-sample Cr3+ Dq/B corpus",
        "hint_key": hint_key,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Train regressor + save ckpt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--formula", type=str, default="Y3Al5O12")
    ap.add_argument("--site", type=str, default="Al")
    args = ap.parse_args()
    if args.train:
        res = train(epochs=args.epochs)
        print(json.dumps({"final": res["history"][-1]}, indent=2))
    else:
        print(json.dumps(predict_dqb(args.formula, args.site, 1.0), indent=2))
