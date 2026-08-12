"""graphrag_hop2.py — GraphRAG 2-hop 多跳推理 (P0-3).

从 kg.duckdb (729 valid triplets, 354 hosts) 构建 networkx.DiGraph:
  节点类型:
    - Host:<formula>     (绿, 寄主晶格)
    - Dopant:<ion>       (橙, 掺杂离子 e.g. Cr3+)
    - Paper:<doi|title>  (灰, 文献来源)
  边类型:
    - doped_with   (Host → Dopant, 带 λ_em/FWHM 属性)
    - emits_at     (Dopant → Host, 反向, 带 λ_em)
    - cited_by     (Host → Paper)
    - similar_host (Host ↔ Host, 元素集合 Jaccard ≥ 0.5)

find_path(source_formula, target_lambda, max_hops=2):
    从 source_formula 出发, 通过 similar_host → Host_b → doped_with → λ 范围匹配,
    返回 top-5 路径 (score = Jaccard × exp(-|Δλ|/50nm) × confidence).

使用:
    from predict_engine.graphrag_hop2 import find_path, build_graph
    paths = find_path("Sr2YAlO5", target_lambda=700.0, max_hops=2)
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import duckdb
import networkx as nx


KG_PATH = Path(__file__).parent.parent / "spectrum_knowledge_shared" / "kg.duckdb"

# 单例图缓存 (避免重复构建 ~1s)
# 全局 RLock 保护: (1) build_graph 多 worker 并发只真建一次, (2) find_path 临时 mutate
# 避免共享图污染. RLock 允许 find_path 递归调用 build_graph 无死锁.
_GRAPH_CACHE: dict[str, nx.DiGraph] = {}
_GRAPH_LOCK = threading.RLock()


# -------------------- 元素解析 (轻量, 不引 pymatgen) --------------------

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def formula_elements(formula: str) -> set[str]:
    """'Sr2YAlO5' → {'Sr','Y','Al','O'}. 容错: 移除空格/括号/加号."""
    if not formula:
        return set()
    s = re.sub(r"[()\s\+\-\·•]", "", formula.strip())
    out: set[str] = set()
    for m in _ELEM_RE.finditer(s):
        el = m.group(1)
        if el:
            out.add(el)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def lambda_proximity(lam_a: Optional[float], lam_b: Optional[float], sigma: float = 50.0) -> float:
    """exp(-|Δλ|/sigma). 如果任一为 None → 1.0 (中性)."""
    if lam_a is None or lam_b is None:
        return 1.0
    return math.exp(-abs(float(lam_a) - float(lam_b)) / sigma)


# -------------------- 图构建 --------------------

def build_graph(kg_path: Path = KG_PATH, jaccard_thresh: float = 0.5) -> nx.DiGraph:
    """读 kg.duckdb → nx.DiGraph. jaccard_thresh 越低, similar_host 边越密.

    线程安全 (double-check locking): lock-free fast path + 锁保护首次构建
    + 锁内 re-check 防止 race. 多 worker 并发只真建一次.
    """
    key = f"{kg_path}:{jaccard_thresh}"
    # Fast path: lock-free read
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    with _GRAPH_LOCK:
        # Re-check under lock
        cached = _GRAPH_CACHE.get(key)
        if cached is not None:
            return cached

        if not kg_path.exists():
            raise FileNotFoundError(f"kg.duckdb 不存在: {kg_path}")

        G = _do_build_graph(kg_path, jaccard_thresh)
        _GRAPH_CACHE[key] = G
        return G


def _do_build_graph(kg_path: Path, jaccard_thresh: float) -> nx.DiGraph:
    """构建图的纯实现, 不涉及 cache/lock. 由 build_graph 在锁内调用."""
    G = nx.DiGraph()
    con = duckdb.connect(str(kg_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT host, host_family, dopant_ion, dopant_pct, lambda_em_nm,
                   fwhm_nm, thermal_stability_pct, source_doi, source_title, confidence
            FROM triplets
            WHERE host IS NOT NULL AND dopant_ion IS NOT NULL
        """).fetchall()
    finally:
        con.close()

    # 第一阶段: Host / Dopant / Paper 节点 + doped_with / emits_at / cited_by 边
    host_elems: dict[str, set[str]] = {}
    host_lams: dict[str, list[float]] = {}

    for (host, family, dop, pct, lam, fwhm, tstab, doi, title, conf) in rows:
        if not host or not dop:
            continue

        h_id = f"Host:{host}"
        d_id = f"Dopant:{dop}"

        if h_id not in G:
            elems = formula_elements(host)
            G.add_node(h_id, kind="Host", label=host, formula=host,
                       family=family or "", elements=sorted(elems))
            host_elems[host] = elems
            host_lams[host] = []

        if lam is not None:
            host_lams[host].append(float(lam))

        if d_id not in G:
            G.add_node(d_id, kind="Dopant", label=dop, ion=dop)

        edge_attrs = {
            "kind": "doped_with",
            "lambda_em_nm": float(lam) if lam is not None else None,
            "fwhm_nm": float(fwhm) if fwhm is not None else None,
            "dopant_pct": float(pct) if pct is not None else None,
            "thermal_stability_pct": float(tstab) if tstab is not None else None,
            "confidence": float(conf) if conf is not None else 0.5,
            "source_doi": doi or "",
            "source_title": (title or "")[:120],
        }
        # 如多条三元组同 (host, dop), 保留 λ_em 更可靠的 (confidence 高)
        if G.has_edge(h_id, d_id):
            old = G[h_id][d_id]
            if (edge_attrs["confidence"] or 0) > (old.get("confidence") or 0):
                G[h_id][d_id].update(edge_attrs)
        else:
            G.add_edge(h_id, d_id, **edge_attrs)

        # 反向 emits_at (方便从目标 λ 反查)
        if not G.has_edge(d_id, h_id):
            G.add_edge(d_id, h_id, kind="emits_at",
                       lambda_em_nm=edge_attrs["lambda_em_nm"])

        # Paper 节点
        paper_key = doi.strip() if doi and doi.strip() else (title or "")[:80]
        if paper_key:
            p_id = f"Paper:{paper_key}"
            if p_id not in G:
                G.add_node(p_id, kind="Paper", label=paper_key,
                           doi=doi or "", title=(title or "")[:120])
            if not G.has_edge(h_id, p_id):
                G.add_edge(h_id, p_id, kind="cited_by")

    # 第二阶段: similar_host 边 (Jaccard ≥ thresh)
    hosts = list(host_elems.keys())
    for i, h1 in enumerate(hosts):
        e1 = host_elems[h1]
        if len(e1) < 2:
            continue
        for h2 in hosts[i + 1 :]:
            e2 = host_elems[h2]
            if len(e2) < 2:
                continue
            j = jaccard(e1, e2)
            if j >= jaccard_thresh:
                G.add_edge(f"Host:{h1}", f"Host:{h2}", kind="similar_host", jaccard=j)
                G.add_edge(f"Host:{h2}", f"Host:{h1}", kind="similar_host", jaccard=j)

    return G


# -------------------- Path finding --------------------

@dataclass
class PathResult:
    nodes: list[dict] = field(default_factory=list)  # [{id, kind, label, ...}]
    edges: list[dict] = field(default_factory=list)  # [{source, target, kind, ...}]
    score: float = 0.0
    hops: int = 0
    explanation: str = ""
    target_host: str = ""
    target_dopant: str = ""
    target_lambda_em: Optional[float] = None
    target_doi: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_label"] = self.explanation
        return d


def find_path(source_formula: str,
              target_lambda: Optional[float] = None,
              max_hops: int = 2,
              top_k: int = 5,
              jaccard_thresh: float = 0.3) -> list[PathResult]:
    """
    从 source_formula 出发, 经 similar_host (1 hop) → target host 的 doped_with 边 (1 hop),
    返回 top-k 子图路径.

    score = jaccard(source, intermediate) × λ_proximity(target_λ, actual_λ) × edge.confidence
    """
    G = build_graph(jaccard_thresh=jaccard_thresh)
    src_elems = formula_elements(source_formula)
    if not src_elems:
        return []

    # 持 _GRAPH_LOCK 保护 mutate/query/cleanup 整段, 防多 worker 污染共享图
    with _GRAPH_LOCK:
        source_id = f"Host:{source_formula}"
        # source 可能不在图中 (用户输入新化学式), 临时加入
        injected = False
        if source_id not in G:
            G.add_node(source_id, kind="Host", label=source_formula,
                       formula=source_formula, family="", elements=sorted(src_elems),
                       _injected=True)
            # 为 source 计算 similar_host 边 (临时)
            for n in list(G.nodes):
                if n == source_id or not n.startswith("Host:"):
                    continue
                h_elems = set(G.nodes[n].get("elements", []))
                j = jaccard(src_elems, h_elems)
                if j >= jaccard_thresh:
                    G.add_edge(source_id, n, kind="similar_host", jaccard=j, _temp=True)
            injected = True

        try:
            candidates: list[PathResult] = []

            # 枚举所有 similar_host 邻居 (1 hop)
            for nbr in G.successors(source_id):
                edge1 = G[source_id][nbr]
                if edge1.get("kind") != "similar_host":
                    continue
                if nbr == source_id:
                    continue
                j1 = edge1.get("jaccard", 0.0)
                nbr_formula = G.nodes[nbr].get("formula", nbr.replace("Host:", ""))

                # 从 nbr 出发枚举 doped_with 边 (2nd hop)
                for dop_id in G.successors(nbr):
                    edge2 = G[nbr][dop_id]
                    if edge2.get("kind") != "doped_with":
                        continue
                    lam = edge2.get("lambda_em_nm")
                    conf = edge2.get("confidence", 0.5)
                    doi = edge2.get("source_doi", "")
                    title = edge2.get("source_title", "")
                    fwhm = edge2.get("fwhm_nm")

                    # λ_proximity
                    lam_prox = lambda_proximity(target_lambda, lam, sigma=50.0)
                    # 若指定了 target_lambda 但该 edge 无 lambda → 略微惩罚
                    if target_lambda is not None and lam is None:
                        lam_prox = 0.4

                    score = j1 * lam_prox * max(conf, 0.2)

                    # 抓对应 Paper (如有)
                    paper_node = None
                    for p in G.successors(nbr):
                        if p.startswith("Paper:") and G[nbr][p].get("kind") == "cited_by":
                            pdoi = G.nodes[p].get("doi", "")
                            if doi and pdoi and pdoi.strip() == doi.strip():
                                paper_node = p
                                break
                            if not paper_node:
                                paper_node = p  # fallback 任意一篇

                    # 构造 nodes/edges 子图
                    nodes = [
                        {"id": source_id, "kind": "Host", "label": source_formula,
                         "formula": source_formula,
                         "elements": sorted(src_elems),
                         "_injected": injected},
                        {"id": nbr, "kind": "Host", "label": nbr_formula,
                         "formula": nbr_formula,
                         "elements": G.nodes[nbr].get("elements", []),
                         "family": G.nodes[nbr].get("family", "")},
                        {"id": dop_id, "kind": "Dopant",
                         "label": G.nodes[dop_id].get("ion", dop_id.replace("Dopant:", "")),
                         "ion": G.nodes[dop_id].get("ion", "")},
                    ]
                    edges = [
                        {"source": source_id, "target": nbr, "kind": "similar_host",
                         "label": f"Jaccard={j1:.2f}", "jaccard": round(j1, 3)},
                        {"source": nbr, "target": dop_id, "kind": "doped_with",
                         "label": f"λ_em={lam}nm" if lam else "doped",
                         "lambda_em_nm": lam, "fwhm_nm": fwhm,
                         "confidence": round(conf, 2),
                         "source_doi": doi, "source_title": title},
                    ]
                    if paper_node:
                        nodes.append({
                            "id": paper_node, "kind": "Paper",
                            "label": G.nodes[paper_node].get("doi") or G.nodes[paper_node].get("label", "")[:40],
                            "doi": G.nodes[paper_node].get("doi", ""),
                            "title": G.nodes[paper_node].get("title", ""),
                        })
                        edges.append({
                            "source": nbr, "target": paper_node, "kind": "cited_by",
                            "label": "cited_by",
                        })

                    # 生成可读 explanation
                    shared = sorted(src_elems & set(G.nodes[nbr].get("elements", [])))
                    pieces = [
                        f"{source_formula} →(shared {{{','.join(shared)}}}, Jaccard={j1:.2f})→ "
                        f"{nbr_formula}",
                        f"→(doped {G.nodes[dop_id].get('ion')})"
                        + (f" λ_em={lam:.0f}nm" if lam else "")
                        + (f", FWHM={fwhm:.0f}nm" if fwhm else ""),
                    ]
                    if target_lambda is not None and lam is not None:
                        pieces.append(f"|Δλ|={abs(target_lambda - lam):.0f}nm vs target {target_lambda:.0f}nm")
                    if doi:
                        pieces.append(f"DOI: {doi}")
                    elif title:
                        pieces.append(f"Paper: {title[:60]}")
                    explanation = "  ".join(pieces)

                    candidates.append(PathResult(
                        nodes=nodes, edges=edges, score=round(score, 4),
                        hops=2, explanation=explanation,
                        target_host=nbr_formula,
                        target_dopant=G.nodes[dop_id].get("ion", ""),
                        target_lambda_em=lam, target_doi=doi,
                    ))

            candidates.sort(key=lambda p: p.score, reverse=True)
            # 去重 (同 target_host + dopant 只保留最高分)
            seen: set[tuple] = set()
            uniq: list[PathResult] = []
            for p in candidates:
                k = (p.target_host, p.target_dopant)
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(p)
                if len(uniq) >= top_k:
                    break
            return uniq

        finally:
            # 清掉临时注入的 source 节点 (保持图缓存纯净)
            if injected:
                to_del = [(u, v) for u, v, a in G.edges(data=True)
                          if a.get("_temp")]
                G.remove_edges_from(to_del)
                if G.nodes[source_id].get("_injected"):
                    G.remove_node(source_id)


# -------------------- CLI 自测 --------------------

if __name__ == "__main__":
    import json, sys
    src = sys.argv[1] if len(sys.argv) > 1 else "Sr2YAlO5"
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 700.0
    print(f"[graphrag_hop2] source={src} target_λ={lam}nm")
    G = build_graph()
    print(f"  graph: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    paths = find_path(src, target_lambda=lam, max_hops=2, top_k=5)
    for i, p in enumerate(paths, 1):
        print(f"\n--- Path {i} (score={p.score}) ---")
        print(" ", p.explanation)
    if not paths:
        print("  (no paths)")
    print("\nJSON sample:")
    print(json.dumps(paths[0].to_dict() if paths else {}, indent=2, ensure_ascii=False)[:500])
