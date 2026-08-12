"""Bayesian Active Learning for NIR 荧光粉实验候选推荐.

P0-2: Given 67 行 labeled ground truth (formula, dopant → λ_em), 用 sklearn
GaussianProcessRegressor (RBF + WhiteKernel) 在 24d formula_descriptor 空间上
拟合 surrogate, 对 unlabeled 250 个 MatterGen/MatterSim 候选预测 (mu, sigma),
再用 Expected Improvement / UCB acquisition 选 top-5, 并用 k-means(4)
做覆盖性 diversity 过滤.

输出: {top5: [...], gp_summary: {...}, all_scored: [...(可选截断)]}

规范:
- 随机种子 42
- 禁 mock 数据
- 只用 sklearn (不手写 GP)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# sklearn deps
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# ---- repo root path resolution (同 dashboard.py 规则) ----
_HERE = Path(__file__).resolve().parent
_REPO_CANDS = [_HERE.parent, Path("/home/rdk Path.cwd()]
REPO_ROOT = next((p for p in _REPO_CANDS if (p / "exp_ground_truth").exists()), _HERE.parent)

for _p in _REPO_CANDS:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RANDOM_SEED = 42
N_CLUSTERS = 4
TOP_K = 5
# 目标 λ_em: NIR-II 窗口中心 (900 nm 是 bio-imaging 典型需求, 也超越大多数 67 样本的 750-850 区间 → 鼓励 explore)
TARGET_LAMBDA_NM = 900.0


def _import_descriptor():
    """Lazy import — ts_torch 需要 torch, 仅在真调用 AL 时拉."""
    from predict_engine.ts_torch import formula_descriptor
    return formula_descriptor


# ---------- 数据加载 ----------

def load_labeled_pl(observed_csv: Path | None = None,
                   actuals_csv: Path | None = None) -> list[dict]:
    """读 observed_pl.csv + actuals.csv. 优先 actual_lambda_em_nm (实测), 次 lambda_em_nm (文献).
    返回 [{formula, dopant_element, dopant_site, dopant_pct, lambda_em_nm, source_row}, ...]
    """
    rows: list[dict] = []
    if observed_csv is None:
        observed_csv = REPO_ROOT / "exp_ground_truth" / "observed_pl.csv"
    if actuals_csv is None:
        actuals_csv = REPO_ROOT / "predictions" / "actuals.csv"

    # observed_pl.csv (67 行)
    if observed_csv.exists():
        with open(observed_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                formula = (r.get("formula") or "").strip()
                if not formula:
                    continue
                # 实测优先, 否则文献
                em_s = (r.get("actual_lambda_em_nm") or "").strip()
                source_tag = "actual"
                if not em_s:
                    em_s = (r.get("lambda_em_nm") or "").strip()
                    source_tag = "literature"
                if not em_s:
                    continue
                try:
                    em = float(em_s)
                except ValueError:
                    continue
                dop_el = (r.get("dopant_element") or "Cr").strip() or "Cr"
                dop_site = (r.get("dopant_site") or "Al").strip() or "Al"
                try:
                    dop_pct = float(r.get("dopant_pct") or 1.0)
                except ValueError:
                    dop_pct = 1.0
                rows.append({
                    "formula": formula,
                    "dopant_element": dop_el,
                    "dopant_site": dop_site,
                    "dopant_pct": dop_pct,
                    "lambda_em_nm": em,
                    "source": f"observed_pl.csv/{source_tag}",
                })

    # actuals.csv (M2.3 实测回填; X5 上 dopant 是 JSON blob, 需解析)
    if actuals_csv and actuals_csv.exists():
        try:
            with open(actuals_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    formula = (r.get("formula") or "").strip()
                    em_s = (r.get("actual_lambda_em_nm") or "").strip()
                    if not formula or not em_s:
                        continue
                    try:
                        em = float(em_s)
                    except ValueError:
                        continue
                    # dopant 列可能是 JSON {"element":"Cr","site":"Ga","pct":1.0} 或扁平列
                    dop_el = "Cr"
                    dop_site = "Al"
                    dop_pct = 1.0
                    dop_raw = (r.get("dopant") or "").strip()
                    if dop_raw.startswith("{"):
                        try:
                            dj = json.loads(dop_raw)
                            dop_el = (dj.get("element") or "Cr").replace("3+", "").replace("2+", "")
                            dop_site = dj.get("site") or "Al"
                            dop_pct = float(dj.get("pct") or 1.0)
                        except Exception:
                            pass
                    else:
                        dop_el = (r.get("dopant_element") or "Cr").strip() or "Cr"
                        dop_site = (r.get("dopant_site") or "Al").strip() or "Al"
                        try:
                            dop_pct = float(r.get("dopant_pct") or 1.0)
                        except ValueError:
                            dop_pct = 1.0
                    rows.append({
                        "formula": formula,
                        "dopant_element": dop_el,
                        "dopant_site": dop_site,
                        "dopant_pct": dop_pct,
                        "lambda_em_nm": em,
                        "source": "actuals.csv",
                    })
        except Exception:
            pass

    return rows


def _parse_formula_sum(formula_raw: str) -> str:
    """'O12 Ga3 Y2 Al3' → 'Y2Al3Ga3O12' (同 dashboard.api_ai_candidates)"""
    import re
    parts: dict[str, int] = {}
    for tok in re.findall(r"([A-Z][a-z]?)(\d*)", formula_raw):
        if tok[0]:
            parts[tok[0]] = int(tok[1]) if tok[1] else 1
    order = ["Y", "Gd", "Lu", "La", "Sc", "Al", "Ga", "Ca", "Sr", "Ba",
             "Mg", "Si", "Ge", "Ti", "Zn", "In", "O"]
    out = ""
    for e in order:
        if e in parts:
            n = parts[e]
            out += f"{e}{n}" if n > 1 else e
            del parts[e]
    for e, n in parts.items():
        out += f"{e}{n}" if n > 1 else e
    return out


def load_unlabeled_candidates() -> list[dict]:
    """Merge nir_v2 (31 宽) + nir_filtered (216 老) manifest.
    返回 [{formula, elements, file, source, default_dopant_site, dopant_pct=1.0}, ...]
    """
    out: list[dict] = []
    seen_formulas: set[str] = set()

    def _add_from_manifest(path: Path, src_tag: str):
        if not path.exists():
            return
        try:
            m = json.load(path.open(encoding="utf-8"))
        except Exception:
            return
        for e in m.get("entries", []):
            formula_raw = e.get("formula", "")
            if not formula_raw:
                continue
            formula = _parse_formula_sum(formula_raw)
            if formula in seen_formulas:
                continue
            seen_formulas.add(formula)
            elems = e.get("elements", [])
            octa = [x for x in ("Al", "Ga", "Sc") if x in elems]
            site = octa[0] if octa else "Al"
            out.append({
                "formula": formula,
                "formula_raw": formula_raw,
                "elements": elems,
                "file": e.get("file", ""),
                "source": src_tag,
                "default_dopant_site": site,
                "dopant_element": "Cr",
                "dopant_pct": 1.0,
                "n_atoms": e.get("n_atoms"),
            })

    _add_from_manifest(REPO_ROOT / "crystal_data_shared" / "generated" / "nir_v2" / "manifest.json",
                       "mattergen_nir_v2")
    _add_from_manifest(REPO_ROOT / "crystal_data_shared" / "generated" / "nir_filtered" / "manifest.json",
                       "mattergen_nir")

    # 排除已 labeled 的 formula (避免推荐已做过的)
    labeled_formulas = {r["formula"] for r in load_labeled_pl()}
    out = [c for c in out if c["formula"] not in labeled_formulas]
    return out


def _featurize(rows: list[dict]) -> np.ndarray:
    """[(formula, dopant_site, dopant_pct)] → (N, 24) array"""
    fd = _import_descriptor()
    feats = []
    for r in rows:
        site = r.get("dopant_site") or r.get("default_dopant_site") or "Al"
        pct = r.get("dopant_pct", 1.0)
        v = fd(r["formula"], site, float(pct)).detach().cpu().numpy()
        feats.append(v)
    return np.asarray(feats, dtype=np.float64)


# ---------- Acquisition functions ----------

def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         best_y: float, xi: float = 0.01,
                         target: float | None = None) -> np.ndarray:
    """EI for maximization 类型 (goal: 超越 best_y).

    若 target 指定, 则 improvement = -(μ-target)² 的近似: 对近 target 的候选加权
    (但仍用 μ>best_y 的标准 EI, best_y 设为 target 即得"接近 target").
    此处直接用 |λ−target| → 最小化等价, 重新定义 y = -|λ−target|, 对 y max.
    """
    sigma = np.maximum(sigma, 1e-6)
    if target is not None:
        # 转换: y_transformed = -|λ − target|; μ_t ≈ -|μ−target|, σ_t ≈ σ (近似)
        mu_t = -np.abs(mu - target)
        best_t = -np.abs(best_y - target) if best_y is not None else mu_t.max()
        imp = mu_t - best_t - xi
    else:
        imp = mu - best_y - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-5] = 0.0
    return ei


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray,
                           kappa: float = 2.0,
                           target: float | None = None) -> np.ndarray:
    """UCB: maximize μ + κσ.

    若 target 指定, 返回 -(|μ−target| − κσ) (距离越近越优, 高 σ 也鼓励 explore).
    """
    if target is not None:
        return -(np.abs(mu - target) - kappa * sigma)
    return mu + kappa * sigma


# ---------- Main AL ----------

def fit_gp_and_recommend(top_k: int = TOP_K,
                         target_lambda_nm: float | None = TARGET_LAMBDA_NM,
                         kappa: float = 2.0) -> dict[str, Any]:
    """拟合 GP, 对 unlabeled 打分, 返回 top_k.

    Steps:
    1. load 67 labeled → X_train (N,24), y_train (N,)
    2. StandardScaler X; GaussianProcessRegressor(RBF + WhiteKernel) fit
    3. load 250 unlabeled → X_test → predict mu, sigma
    4. EI + UCB 排序
    5. K-means(4) 聚类, 保证每 cluster 至少 1 个 (diversity)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    labeled = load_labeled_pl()
    if len(labeled) < 5:
        return {
            "ok": False,
            "error": f"labeled 样本不足 ({len(labeled)}), 至少需要 5 行 lambda_em_nm",
            "n_labeled": len(labeled),
        }

    X_train = _featurize(labeled)
    y_train = np.array([r["lambda_em_nm"] for r in labeled], dtype=np.float64)

    unlabeled = load_unlabeled_candidates()
    if not unlabeled:
        return {"ok": False, "error": "无 unlabeled 候选可供推荐"}
    X_test = _featurize(unlabeled)

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Kernel: RBF (length_scale init 1.0, 可自动拟合) + WhiteKernel (noise)
    # 初始 noise_level=10²=100 (nm²) 对应 σ~10nm 先验, 合理
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=100.0, noise_level_bounds=(1.0, 1e4)))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=RANDOM_SEED,
        alpha=1e-6,
    )
    gp.fit(X_train_s, y_train)

    mu, sigma = gp.predict(X_test_s, return_std=True)
    mu = np.asarray(mu).ravel()
    sigma = np.asarray(sigma).ravel()

    # Acquisitions
    best_y = float(y_train.max())
    ei = expected_improvement(mu, sigma, best_y=best_y, target=target_lambda_nm)
    ucb = upper_confidence_bound(mu, sigma, kappa=kappa, target=target_lambda_nm)

    # K-means diversity clustering on scaled features
    n_clusters = min(N_CLUSTERS, max(2, len(unlabeled) // 10))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    clusters = km.fit_predict(X_test_s)

    # Build per-candidate scored list
    scored = []
    for i, c in enumerate(unlabeled):
        scored.append({
            **c,
            "predicted_lambda_em_nm": float(mu[i]),
            "uncertainty_nm": float(sigma[i]),
            "EI": float(ei[i]),
            "UCB": float(ucb[i]),
            "cluster": int(clusters[i]),
        })

    # Diversity-aware top_k: per cluster pick 1 highest EI, then fill by global EI
    picks: list[dict] = []
    picked_idx: set[int] = set()
    for cid in range(n_clusters):
        members = [(i, s) for i, s in enumerate(scored) if s["cluster"] == cid]
        if not members:
            continue
        members.sort(key=lambda t: t[1]["EI"], reverse=True)
        top_i, top_s = members[0]
        picks.append({**top_s, "pick_reason": f"cluster-{cid}-leader"})
        picked_idx.add(top_i)
        if len(picks) >= top_k:
            break

    if len(picks) < top_k:
        remaining = sorted([(i, s) for i, s in enumerate(scored) if i not in picked_idx],
                          key=lambda t: t[1]["EI"], reverse=True)
        for i, s in remaining:
            if len(picks) >= top_k:
                break
            picks.append({**s, "pick_reason": "global-EI-top"})
            picked_idx.add(i)

    picks.sort(key=lambda s: s["EI"], reverse=True)

    # "why pick" sentence
    for p in picks:
        lam = p["predicted_lambda_em_nm"]
        sig = p["uncertainty_nm"]
        cid = p["cluster"]
        dist = abs(lam - (target_lambda_nm or best_y))
        why_parts = []
        if target_lambda_nm is not None and dist < 40:
            why_parts.append(f"预测 λ_em={lam:.0f}nm 接近目标 {target_lambda_nm:.0f}nm (Δ={dist:.0f}nm)")
        else:
            why_parts.append(f"预测 λ_em={lam:.0f}nm")
        if sig > 40:
            why_parts.append(f"σ={sig:.0f}nm 高 → GP 信息增益大 (纯 explore)")
        elif sig > 20:
            why_parts.append(f"σ={sig:.0f}nm 中等 (explore/exploit 平衡)")
        else:
            why_parts.append(f"σ={sig:.0f}nm 低 → 模型高信心 (pure exploit)")
        why_parts.append(f"代表 cluster #{cid}")
        p["why"] = "; ".join(why_parts)

    # GP summary
    try:
        learned_kernel = str(gp.kernel_)
    except Exception:
        learned_kernel = "unknown"

    gp_summary = {
        "n_labeled": int(len(labeled)),
        "n_unlabeled": int(len(unlabeled)),
        "n_clusters": int(n_clusters),
        "feature_dim": int(X_train.shape[1]),
        "y_min": float(y_train.min()),
        "y_max": float(y_train.max()),
        "y_mean": float(y_train.mean()),
        "y_std": float(y_train.std()),
        "pred_mu_range": [float(mu.min()), float(mu.max())],
        "pred_sigma_mean_nm": float(sigma.mean()),
        "pred_sigma_max_nm": float(sigma.max()),
        "target_lambda_nm": target_lambda_nm,
        "kappa_ucb": kappa,
        "learned_kernel": learned_kernel,
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
        "random_seed": RANDOM_SEED,
    }

    # Trim candidate payload for transport
    top5_clean = []
    for p in picks[:top_k]:
        top5_clean.append({
            "formula": p["formula"],
            "dopant_element": p.get("dopant_element", "Cr"),
            "dopant_site": p.get("default_dopant_site", "Al"),
            "dopant_pct": p.get("dopant_pct", 1.0),
            "predicted_lambda_em_nm": round(p["predicted_lambda_em_nm"], 1),
            "uncertainty_nm": round(p["uncertainty_nm"], 1),
            "EI": round(p["EI"], 4),
            "UCB": round(p["UCB"], 4),
            "cluster": p["cluster"],
            "source": p.get("source"),
            "file": p.get("file"),
            "elements": p.get("elements"),
            "pick_reason": p.get("pick_reason"),
            "why": p.get("why"),
        })

    return {
        "ok": True,
        "top5": top5_clean,
        "gp_summary": gp_summary,
    }


# ---------- CLI for smoke test ----------

if __name__ == "__main__":
    import pprint
    result = fit_gp_and_recommend()
    if result.get("ok"):
        pprint.pprint(result["gp_summary"])
        print("\n==== TOP-5 Active Learning Recommendations ====")
        for i, p in enumerate(result["top5"], 1):
            print(f"\n#{i} {p['formula']}  ({p['source']})")
            print(f"   site={p['dopant_site']} Cr@{p['dopant_pct']}%")
            print(f"   λ_em = {p['predicted_lambda_em_nm']}±{p['uncertainty_nm']} nm "
                  f"| EI={p['EI']:.3f} UCB={p['UCB']:.3f} cluster=#{p['cluster']}")
            print(f"   why: {p['why']}")
    else:
        print("ERROR:", result.get("error"))
