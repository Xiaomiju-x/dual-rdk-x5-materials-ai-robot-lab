"""
crystal_agent.py — 候选晶体结构 Agent (v4.1, vision 线)

两步决策:
    Step 1 generate_candidates(classification, top_k)
        → 从 ../../crystal_data_shared/candidate_pool.json 读预拉的池 (离线 fallback)
        → 若无, 实时调 tools.mp_cif_fetch.fetch_mp_candidates (需联网 + mp-api)
        → 返回 K 个 Candidate dict

    Step 2 rank_candidates(candidates, experimental_peaks) -> (best, reasoning)
        → 每个候选 pymatgen.XRDCalculator 算理论峰
        → 用加权峰位匹配打 similarity_score
        → 把 K 个候选的 (formula, 空间群, 理论峰 top-5) + 实测峰 喂 DeepSeek-R1
        → R1 输出最优选择 + 推理链

调用者: deploy_xrd_system.py 的 /api/crystal/candidates 和 /api/crystal/rank 两个端点.
"""
from __future__ import annotations

import json
import os
import re as _re
import sys
from pathlib import Path
from typing import Any

# v4.1 Round 9: 清理 R1 偶尔漏出的 DSML / invoke / parameter 工具协议标记
# (直接内嵌正则, 不强依赖 voice_backend.py, 保持 crystal_agent 可独立使用)
_DSML_RE = [
    _re.compile(r"<[\s|]*DSML[\s|]*function_calls[\s|]*>.*?<[\s|]*/[\s|]*DSML[\s|]*function_calls[\s|]*>",
                _re.DOTALL | _re.IGNORECASE),
    _re.compile(r"<[^>]*DSML[^>]*>", _re.IGNORECASE),
    _re.compile(r"<[^>]*function_calls[^>]*>", _re.IGNORECASE),
    _re.compile(r"</?\s*invoke[^>]*>", _re.IGNORECASE),
    _re.compile(r"</?\s*parameter[^>]*>", _re.IGNORECASE),
    _re.compile(r"<\|[^|>]+\|>"),
    _re.compile(r"\n{3,}"),
]


def _clean(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat in _DSML_RE[:-1]:
        out = pat.sub("", out)
    out = _DSML_RE[-1].sub("\n\n", out).strip()
    return out


_THIS = Path(__file__).resolve()
_SCRIPT_DIR = _THIS.parent                    # xrd_vision/visual_line/ 或 X5 上的 ~/xrd1/
_REPO = _THIS.parent.parent.parent             # PC 端: xrd/   (X5 上无意义, 但不会出错)
_TOOLS = _REPO / "tools"
if _TOOLS.exists():
    sys.path.insert(0, str(_TOOLS))

# candidate_pool.json 多路径查找 (命中即用):
#   1. PC 开发环境: xrd/crystal_data_shared/candidate_pool.json
#   2. X5 生产环境: 与 crystal_agent.py 同目录
#   3. X5 生产环境: crystal_data/candidate_pool.json
_POOL_CANDIDATES = [
    _REPO / "crystal_data_shared" / "candidate_pool.json",
    _SCRIPT_DIR / "candidate_pool.json",
    _SCRIPT_DIR / "crystal_data" / "candidate_pool.json",
]


def _find_pool_json() -> Path | None:
    for p in _POOL_CANDIDATES:
        if p.exists():
            return p
    return None


# ============ Step 1: 生成候选 ============
def generate_candidates(classification: str, top_k: int = 3) -> list[dict]:
    """
    返回形如:
        [{"mp_id":..., "formula":..., "spacegroup_symbol":...,
          "spacegroup_number":..., "num_sites":..., "processed_cif_path":...}, ...]
    """
    # 优先读离线池 (X5 生产环境)
    pool_path = _find_pool_json()
    if pool_path is not None:
        try:
            with open(pool_path, "r", encoding="utf-8") as f:
                pool = json.load(f)
            if classification in pool and pool[classification]:
                return pool[classification][:top_k]
        except Exception as e:
            print(f"[crystal_agent] pool read failed ({pool_path}): {e}")

    # Fallback: 实时查 MP (PC 开发环境)
    try:
        from mp_cif_fetch import fetch_mp_candidates
        cands = fetch_mp_candidates(classification, top_k=top_k)
        return [c.to_dict() for c in cands]
    except Exception as e:
        print(f"[crystal_agent] live fetch failed: {e}")
        return []


# ============ Step 2: R1 选优 ============
def _compute_theoretical_peaks(cif_path: str, top_n: int = 8) -> list[tuple[float, float]]:
    """用 pymatgen 算某个 CIF 的理论 XRD 峰 (Cu Kα), 返回 top-N 强峰.

    注意: X5 没装 pymatgen, 这里会直接返回 []. 实际使用中应优先从候选字典的
    theoretical_peaks 字段读 PC 端预计算的缓存 (见 _get_cached_peaks).
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError:
        return []
    try:
        s = Structure.from_file(cif_path)
        calc = XRDCalculator(wavelength="CuKa")
        p = calc.get_pattern(s, two_theta_range=(10, 80))
        peaks = sorted(zip(p.x, p.y), key=lambda z: -z[1])[:top_n]
        return [(round(float(x), 3), round(float(y), 2)) for x, y in peaks]
    except Exception as e:
        print(f"[crystal_agent] theoretical peaks failed for {cif_path}: {e}")
        return []


def _get_cached_peaks(candidate: dict) -> list:
    """优先读 candidate_pool.json 里 PC 端预计算的 theoretical_peaks,
    没缓存才 fallback 到 pymatgen (仅 PC 可用).
    """
    cached = candidate.get("theoretical_peaks")
    if cached:
        # 可能是 [[x,y],...] 或 [(x,y),...], 统一成 tuple 列表
        return [tuple(p) if isinstance(p, (list, tuple)) else p for p in cached]
    # Fallback: 实时算 (只有 PC 能成功)
    cif = candidate.get("processed_cif_path") or candidate.get("raw_cif_path")
    if cif:
        return _compute_theoretical_peaks(cif)
    return []


def _score_similarity(theoretical: list[tuple[float, float]],
                      experimental: list[float],
                      tol: float = 0.3) -> float:
    """
    简单加权峰位匹配打分:
        对每个实测峰, 找理论峰里 2θ 距离 <= tol 的, 匹配权重 = 理论强度
    返回归一化分数 [0, 1]
    """
    if not theoretical or not experimental:
        return 0.0
    matched_intensity = 0.0
    total_intensity = sum(i for _, i in theoretical)
    if total_intensity == 0:
        return 0.0
    for exp_x in experimental:
        best = 0.0
        for th_x, th_i in theoretical:
            if abs(th_x - exp_x) <= tol:
                if th_i > best:
                    best = th_i
        matched_intensity += best
    return round(min(1.0, matched_intensity / total_intensity), 3)


def rank_candidates(
    candidates: list[dict],
    experimental_peaks: list[float] | None = None,
    call_r1_func: Any = None,
) -> dict:
    """
    Args:
        candidates: generate_candidates 返回的列表 (dict)
        experimental_peaks: 实测峰 2θ 列表 (从 deploy_xrd_system.state 拿)
        call_r1_func: deploy_xrd_system.call_deepseek_r1 的引用 (避免循环 import)

    Returns:
        {
            "best_mp_id": str,
            "best_formula": str,
            "reasoning": str,
            "scores": {mp_id: similarity_score, ...}
        }
    """
    if not candidates:
        return {"best_mp_id": None, "best_formula": None, "reasoning": "无候选结构", "scores": {}}

    # 1. 每个候选算理论峰 (优先读 pool 缓存, X5 无 pymatgen 也能跑) + 打分
    enriched = []
    for c in candidates:
        th = _get_cached_peaks(c)
        sim = _score_similarity(th, experimental_peaks or [])
        enriched.append({**c, "theoretical_peaks": th, "similarity_score": sim})

    scores = {c["mp_id"]: c["similarity_score"] for c in enriched}

    # 2. R1 推理选最优 (如果能调通)
    best_mp_id = None
    reasoning = ""
    if call_r1_func is not None:
        try:
            prompt = _build_r1_prompt(enriched, experimental_peaks or [])
            result = call_r1_func([
                {"role": "system",
                 "content": "你是一个XRD晶体学专家。根据实测峰位和候选结构的理论峰, 选择最匹配的候选并给出简洁理由。输出必须以 'BEST: mp-xxx' 开头。"},
                {"role": "user", "content": prompt},
            ])
            content = (result.get("content") or "").strip()
            reasoning = content
            # 从 R1 输出里抽 mp_id
            for line in content.splitlines():
                if line.strip().upper().startswith("BEST:"):
                    token = line.split(":", 1)[1].strip().split()[0].strip(",.")
                    best_mp_id = token
                    break
        except Exception as e:
            print(f"[crystal_agent] R1 rank failed: {e}")

    # 3. Fallback: 按 similarity_score 排
    if not best_mp_id:
        best = max(enriched, key=lambda c: c["similarity_score"])
        best_mp_id = best["mp_id"]
        if not reasoning:
            reasoning = f"离线打分: 相似度 {best['similarity_score']:.3f} 最高"

    best_entry = next((c for c in enriched if c["mp_id"] == best_mp_id), enriched[0])
    return {
        "best_mp_id": best_mp_id,
        "best_formula": best_entry["formula"],
        "reasoning": reasoning,
        "scores": scores,
    }


def _is_authoritative(candidate: dict) -> bool:
    """判断候选是否为本地权威参考 (LOCAL_CANDIDATES 注入, 非 MP 来源).

    规则: mp_id 不以 'mp-' 开头的都是本地条目 (例如 SYGO_ref_04-019-6536, YCAS_74606).
    """
    mp_id = candidate.get("mp_id", "")
    return bool(mp_id) and not mp_id.startswith("mp-")


# 目标材料 → 结构类型先验知识 (给 R1 作为物理直觉 anchor)
TARGET_MATERIAL_HINTS = {
    "SYGO": (
        "实验室自合成的 Sr₃YGa₂O₇.₅ 新相, ICSD/MP/COD 无公开条目. "
        "空间群应为 C2 #5 单斜 (参考 Sr₆Y₂Al₄O₁₅ 同构类似物, 课题组 PI 确认). "
        "Ga³⁺ 比 Al³⁺ 稍大, 实际晶胞预计比 Al 类似物膨胀 1-2%."
    ),
    "YCAS": (
        "Y-Ca-Al-Si 石榴石 (NIR 荧光粉宿主), Cr³⁺ 掺杂在 Al 位发射近红外光. "
        "空间群必须是 Ia-3d #230 立方石榴石, a≈12.01Å. "
        "ICSD 74606 是直接对应条目. 不要选非石榴石空间群的候选."
    ),
}


def _build_r1_prompt(enriched: list[dict], exp_peaks: list[float],
                     target_material: str = "") -> str:
    lines = []
    if target_material:
        lines.append(f"# Qwen-VL 视觉判定的目标材料: **{target_material}**")
        hint = TARGET_MATERIAL_HINTS.get(target_material)
        if hint:
            lines.append(f"# 目标材料先验: {hint}")
        lines.append("")
    lines.append("## 实测 XRD 峰位 (2θ, 度):")
    if exp_peaks:
        lines.append("  " + ", ".join(f"{p:.2f}" for p in exp_peaks[:15]))
    else:
        lines.append("  (无实测峰可用 — 请基于空间群一致性 + 晶胞参数 + 先验知识判断)")
    lines.append("")
    lines.append(f"## 候选结构 ({len(enriched)} 个, 按优先级顺序):")
    for i, c in enumerate(enriched, 1):
        mark = "⭐ **本地权威参考 (PI 确认)**" if _is_authoritative(c) else "(Materials Project)"
        lines.append(f"{i}. `{c['mp_id']}`  {c['formula']}  "
                     f"{c['spacegroup_symbol']} #{c['spacegroup_number']}  {mark}")
        lines.append(f"   原子数={c['num_sites']}  预打分={c['similarity_score']:.3f}")
        th = c.get("theoretical_peaks") or []
        if th:
            tops = ", ".join(f"{x:.2f}({y:.0f})" for x, y in th[:5])
            lines.append(f"   理论 top5 峰: {tops}")
    lines.append("")
    lines.append("## 选择准则 (必须严格执行):")
    lines.append("1. 如果有 ⭐ 标记的本地权威参考, 且其空间群与目标材料先验一致, **必须选它**")
    lines.append("2. ⭐ 候选可能是同构类似物 (例如 Al 代替 Ga), 这是正常的, **不是淘汰理由**")
    lines.append("3. 只有当 ⭐ 候选的空间群和目标先验严重不一致时, 才考虑 MP 候选")
    lines.append("4. 淘汰非 ⭐ 候选时, 理由应该是'空间群与目标先验不一致'而不是'化学式不同'")
    lines.append("")
    lines.append("## 输出格式 (mp_id 必须和上面候选列表完全一致):")
    lines.append("BEST: <mp_id>")
    lines.append("推理: <≤200 字, 说明为什么选它, 以及为什么淘汰其他候选>")
    return "\n".join(lines)


# ============================================================================
# v4.1 Round 2: ReAct 晶体 Agent — 复用 deploy_xrd_system.run_agent 的成熟模式
# ============================================================================

CRYSTAL_AGENT_SYSTEM_PROMPT = (
    "你是一个专业的 AI 晶体学家, 部署在 RDK X5 嵌入式平台上, 负责从 K 个候选晶体"
    "结构里为 Qwen-VL 已识别的目标材料选择最匹配的结构参考.\n\n"
    "关键规则:\n"
    "1. **目标材料已知**: 用户 prompt 会明确告知 Qwen-VL 识别的目标材料 (SYGO/YCAS/...), "
    "你的任务是为这个已知目标找最佳结构参考, 而不是从候选里反推目标是什么.\n"
    "2. **尊重 ⭐ 本地权威参考**: 标 ⭐ 的候选是课题组 PI 已确认的权威结构参考 "
    "(例如 SYGO 的 Al 同构物, YCAS 的 ICSD 74606 Cr-garnet). 只有当实测 XRD 峰位 "
    "严重不匹配时才能推翻它; 默认情况下 ⭐ 候选就是正确答案.\n"
    "3. **诚实对待数据局限**: 如果 ⭐ 候选是类似物 (例如 Al 代替 Ga), 在推理链里要指出 "
    "'这是同构类似物, 真实版本晶胞略有膨胀', 但不因此降级它 — 同构参考依然是最佳选择.\n"
    "4. **物理直觉**: 对比候选的空间群、晶系、晶胞体积和目标材料的已知结构类型. "
    "例如目标是 YCAS (Ia-3d 石榴石), 那么 Ia-3d 候选就是正确的; 其他空间群的候选即使"
    "化学式相近也应淘汰.\n"
    "5. **数据驱动 (可选)**: 有实测峰位时可调 score_candidate_similarity; 没有时"
    "就基于空间群一致性和晶胞参数相近度排序.\n"
    "6. **工具调用节制**: 最多 2 轮工具调用, 之后必须给结论.\n"
    "7. **输出格式**: 最终答案必须以 'BEST: <mp_id>' 开头 (mp_id 要和候选列表里的"
    "完全一致), 后跟 '推理: ...', 篇幅 ≤ 200 字.\n\n"
    "可用工具:\n"
    "- compute_theoretical_xrd_for_candidate: 给某个 mp_id 算理论 XRD 峰 (pymatgen)\n"
    "- score_candidate_similarity: 把理论峰和实测峰比对, 返回相似度分 [0,1]\n"
    "- query_rag_knowledge: 从 197 篇论文库检索该结构/化学体系的文献信息\n"
    "- match_pdf_card: 标准 PDF 卡片匹配, 验证峰位"
)


CRYSTAL_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_theoretical_xrd_for_candidate",
            "description": "用 pymatgen XRDCalculator 算指定候选的理论 Cu Kα 衍射峰 (top 8)",
            "parameters": {
                "type": "object",
                "properties": {
                    "mp_id": {"type": "string",
                              "description": "候选的 Materials Project ID, 例如 mp-12345"}
                },
                "required": ["mp_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "score_candidate_similarity",
            "description": "加权峰位匹配打分: 对每个实测峰找最近的理论峰, 按理论强度加权求和. 返回 [0,1] 分数",
            "parameters": {
                "type": "object",
                "properties": {
                    "mp_id": {"type": "string", "description": "候选 mp_id"},
                    "exp_peaks": {"type": "string",
                                  "description": "实测峰 2θ 列表, 逗号分隔, 例如 '19.3,33.8,38.5,45.1'"}
                },
                "required": ["mp_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_rag_knowledge",
            "description": "从 197 篇 XRD 论文知识库里语义检索该材料或结构类型的文献",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "检索词, 例如 '石榴石 Ia-3d Cr 掺杂 NIR 发光'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_pdf_card",
            "description": "把一组峰位和 PDF 标准卡片数据库比对, 验证材料身份",
            "parameters": {
                "type": "object",
                "properties": {
                    "peak_positions": {"type": "string",
                                       "description": "峰位列表 2θ 逗号分隔"},
                    "crystal_system": {"type": "string",
                                       "description": "晶系 cubic/monoclinic/..."}
                },
                "required": ["peak_positions"]
            }
        }
    }
]


def _execute_crystal_tool(name: str, args: dict, enriched: list[dict],
                          exp_peaks: list[float],
                          external_tool_exec=None) -> str:
    """执行候选 Agent 的工具调用.

    Args:
        external_tool_exec: deploy_xrd_system._execute_agent_tool 的引用,
                            用于把 query_rag_knowledge / match_pdf_card 转交给主 Agent 的工具实现
    """
    if name == "compute_theoretical_xrd_for_candidate":
        mp_id = args.get("mp_id", "")
        cand = next((c for c in enriched if c["mp_id"] == mp_id), None)
        if cand is None:
            return f"错误: 未找到候选 {mp_id}"
        # 优先读缓存的理论峰, 没有才 fallback 到 pymatgen
        th = cand.get("theoretical_peaks") or []
        if not th:
            th = _get_cached_peaks(cand)
            cand["theoretical_peaks"] = th
        if not th:
            return f"{mp_id}: 无法计算理论峰 (CIF 缺失或 pymatgen 未安装, 且 pool 无缓存)"
        tops = ", ".join(f"{x:.2f}°({y:.0f})" for x, y in th[:8])
        return f"{mp_id} ({cand['formula']}, {cand['spacegroup_symbol']}) top-8 理论峰: {tops}"

    if name == "score_candidate_similarity":
        mp_id = args.get("mp_id", "")
        cand = next((c for c in enriched if c["mp_id"] == mp_id), None)
        if cand is None:
            return f"错误: 未找到候选 {mp_id}"
        # 解析 exp_peaks (可能来自 args 或 fallback 到调用方传入的)
        ep_str = args.get("exp_peaks", "")
        if ep_str:
            try:
                ep = [float(x) for x in ep_str.split(",") if x.strip()]
            except ValueError:
                ep = exp_peaks
        else:
            ep = exp_peaks
        th = cand.get("theoretical_peaks") or []
        if not th:
            th = _get_cached_peaks(cand)
            cand["theoretical_peaks"] = th
        score = _score_similarity(th, ep)
        cand["similarity_score"] = score
        return f"{mp_id} ({cand['formula']}) 相似度 = {score:.3f} (基于 {len(ep)} 实测峰 × {len(th)} 理论峰)"

    if name in ("query_rag_knowledge", "match_pdf_card") and external_tool_exec is not None:
        try:
            return external_tool_exec(name, args)
        except Exception as e:
            return f"{name} 调用失败: {e}"

    return f"未知工具: {name}"


def run_crystal_agent(
    candidates: list[dict],
    experimental_peaks: list[float] | None = None,
    call_r1_func: Any = None,
    state_ref: Any = None,
    external_tool_exec: Any = None,
    max_rounds: int = 2,
    target_material: str = "",
) -> dict:
    """
    ReAct 循环版本的候选结构 Agent (复用 deploy_xrd_system.run_agent 模式).

    流式写入:
        state_ref.crystal_thinking_buffer  (str, 不断追加的思考/工具日志)
        state_ref.crystal_thinking_done    (bool, 结束标志)

    Args:
        candidates:        generate_candidates 的输出 (已带 CIF 路径)
        experimental_peaks: 实测峰 2θ 度数列表 (可空)
        call_r1_func:      deploy_xrd_system.call_deepseek_r1 的引用
        state_ref:         deploy_xrd_system.state (用于写流式缓冲)
        external_tool_exec: deploy_xrd_system._execute_agent_tool 的引用
                            (用于把 query_rag_knowledge / match_pdf_card 转交)
        max_rounds:        最多几轮工具调用, 之后强制输出结论
        target_material:   Qwen-VL 识别的目标材料 (SYGO/YCAS/...), 关键参数!
                            R1 据此锁定物理先验和优先选 ⭐ 本地权威参考

    Returns:
        {best_mp_id, best_formula, reasoning, thinking, scores}
    """
    if not candidates:
        return {"best_mp_id": None, "best_formula": None,
                "reasoning": "无候选", "thinking": "", "scores": {}}

    # 先对每个候选做一次预打分 (便于 prompt 展示 + fallback)
    enriched = []
    for c in candidates:
        th = _get_cached_peaks(c)
        sim = _score_similarity(th, experimental_peaks or [])
        enriched.append({**c, "theoretical_peaks": th, "similarity_score": sim})

    scores = {c["mp_id"]: c["similarity_score"] for c in enriched}

    # 没有 R1 可用: 直接走 fallback (相似度排)
    if call_r1_func is None:
        best = max(enriched, key=lambda c: c["similarity_score"])
        return {
            "best_mp_id": best["mp_id"],
            "best_formula": best["formula"],
            "reasoning": f"离线打分: 相似度 {best['similarity_score']:.3f} 最高 (无 R1)",
            "thinking": "",
            "scores": scores,
        }

    def _write_buf(text: str):
        if state_ref is not None and hasattr(state_ref, "lock"):
            with state_ref.lock:
                state_ref.crystal_thinking_buffer = text
        else:
            print(text)

    full_thinking = ""
    if target_material:
        full_thinking += f"🎯 目标材料 (Qwen-VL 识别): {target_material}\n"
        # 在候选列表里标出哪些是 ⭐ 本地权威
        for c in enriched:
            if _is_authoritative(c):
                full_thinking += f"⭐ 本地权威参考: {c['mp_id']} ({c['formula']}, {c['spacegroup_symbol']})\n"
        full_thinking += "\n"
        _write_buf(full_thinking)
    messages = [
        {"role": "system", "content": CRYSTAL_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_r1_prompt(enriched, experimental_peaks or [], target_material)}
    ]
    full_response = ""

    for round_i in range(max_rounds + 1):
        use_tools = CRYSTAL_AGENT_TOOLS if round_i < max_rounds else None
        try:
            resp = call_r1_func(messages, tools=use_tools)
        except Exception as e:
            full_thinking += f"\n[DeepSeek-R1 调用失败: {e}]\n"
            _write_buf(full_thinking)
            break

        thinking = _clean(resp.get("reasoning_content", ""))
        if thinking:
            full_thinking += f"\n🤔 第{round_i+1}轮思考:\n{thinking}\n"
            _write_buf(full_thinking)

        tool_calls = resp.get("tool_calls", [])
        content = _clean(resp.get("content", ""))

        if not tool_calls:
            full_response = content
            if content:
                full_thinking += f"\n💡 结论:\n{content}\n"
                _write_buf(full_thinking)
            break

        # 构建 assistant 消息 + 执行工具
        assistant_msg = {"role": "assistant", "content": content or ""}
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        for tc in tool_calls:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            try:
                func_args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                func_args = {}
            full_thinking += f"🔧 调用工具: {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]})\n"
            _write_buf(full_thinking)
            result = _execute_crystal_tool(func_name, func_args, enriched,
                                           experimental_peaks or [], external_tool_exec)
            short = result[:400] + ("..." if len(result) > 400 else "")
            full_thinking += f"📋 结果: {short}\n"
            _write_buf(full_thinking)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{round_i}_{func_name}"),
                "content": result,
            })

    # 强制最终结论
    if not full_response:
        try:
            messages.append({"role": "user",
                             "content": "请立即输出最终结论, 格式: 'BEST: mp-xxxx\\n推理: ...', 不要再调用工具."})
            final = call_r1_func(messages, tools=None)
            full_response = _clean(final.get("content", ""))
            final_reasoning = _clean(final.get("reasoning_content", ""))
            if final_reasoning:
                full_thinking += f"\n🤔 最终推理:\n{final_reasoning}\n"
            if full_response:
                full_thinking += f"\n💡 结论:\n{full_response}\n"
            _write_buf(full_thinking)
        except Exception:
            pass

    # 解析 best_mp_id
    best_mp_id = None
    for line in (full_response or "").splitlines():
        if line.strip().upper().startswith("BEST:"):
            token = line.split(":", 1)[1].strip().split()[0].strip(",.")
            best_mp_id = token
            break

    # Fallback: 相似度最高
    if not best_mp_id:
        best = max(enriched, key=lambda c: c["similarity_score"])
        best_mp_id = best["mp_id"]

    best_entry = next((c for c in enriched if c["mp_id"] == best_mp_id), enriched[0])

    if state_ref is not None and hasattr(state_ref, "lock"):
        with state_ref.lock:
            state_ref.crystal_thinking_done = True

    # 更新 scores (工具调用中可能改过)
    scores = {c["mp_id"]: c["similarity_score"] for c in enriched}

    return {
        "best_mp_id": best_mp_id,
        "best_formula": best_entry["formula"],
        "reasoning": full_response or "",
        "thinking": full_thinking,
        "scores": scores,
    }
