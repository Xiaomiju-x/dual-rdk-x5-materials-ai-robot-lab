"""r1_judge.py — DeepSeek-R1 合成评审 Agent (tool-calling, 强制结构化 JSON).

调用者:
    from predict_engine.r1_judge import run_r1_judge_stream, extract_verdict

    for chunk in run_r1_judge_stream(prediction_payload):
        # chunk is dict: {"type": "thinking"|"final"|"error", "text": str, "done": bool, "verdict": dict|None}
        ...

系统 prompt 硬约束: 只能调用 submit_synthesis_verdict 工具, 禁止自由文本.
R1 的 reasoning_content 用打字机流给前端, 但最终 verdict 来自 tool_calls.arguments.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import Any, Iterator

import requests

from .cove import (
    majority_vote_verdict, needs_cove, build_cove_followup_prompt,
    summarize_self_consistency,
)


DEEPSEEK_R1_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_R1_KEY = os.environ.get("DEEPSEEK_R1_KEY", "")
DEEPSEEK_R1_MODEL = "deepseek-reasoner"


# DSML 清洗 (和 crystal_agent.py 同实现, 避免跨模块依赖)
_DSML_RE = [
    re.compile(r"<[\s|]*DSML[\s|]*function_calls[\s|]*>.*?<[\s|]*/[\s|]*DSML[\s|]*function_calls[\s|]*>",
               re.DOTALL | re.IGNORECASE),
    re.compile(r"<[^>]*DSML[^>]*>", re.IGNORECASE),
    re.compile(r"<[^>]*function_calls[^>]*>", re.IGNORECASE),
    re.compile(r"</?\s*invoke[^>]*>", re.IGNORECASE),
    re.compile(r"</?\s*parameter[^>]*>", re.IGNORECASE),
    re.compile(r"<\|[^|>]+\|>"),
    re.compile(r"\n{3,}"),
]


def _clean(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat in _DSML_RE[:-1]:
        out = pat.sub("", out)
    return _DSML_RE[-1].sub("\n\n", out).strip()


# ============ Tool schema ============
VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_synthesis_verdict",
        "description": "提交合成决策. 必须调用此工具返回最终结论, 不能用自由文本.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["GO", "REVISE", "DROP", "UNKNOWN"],
                    "description": "GO=推荐做, REVISE=改参数再做, DROP=不做, UNKNOWN=信息不足",
                },
                "confidence": {
                    "type": "number",
                    "description": "0-1, 对 verdict 的置信度",
                },
                "reasoning": {
                    "type": "string",
                    "description": "≤300字, 必须引用具体 BPU 概率数值和 1 条文献 DOI/标题",
                },
                "optimization": {
                    "type": "array",
                    "description": "优化建议列表, 每项 {action, why}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "why": {"type": "string"},
                        },
                        "required": ["action", "why"],
                    },
                },
                "suggested_sinter": {
                    "type": "object",
                    "description": "推荐烧结条件 (空气气氛)",
                    "properties": {
                        "temp_C": {"type": "number"},
                        "hours": {"type": "number"},
                    },
                },
                "predicted_lambda_em_nm_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[min, max] nm, 发射峰预测区间",
                },
                "predicted_phase_purity_prob": {
                    "type": "number",
                    "description": "0-1, 纯相概率预测",
                },
                "predicted_excitation_peaks_nm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "激发峰波长 [λ_ex1, λ_ex2] nm (4A2→4T2, 4A2→4T1a)",
                },
                "predicted_fwhm_nm": {
                    "type": "number",
                    "description": "半峰宽 FWHM (nm)",
                },
                "predicted_thermal_stability_pct": {
                    "type": "number",
                    "description": "热稳定性 I(423K)/I(298K) × 100%",
                },
                "predicted_quantum_efficiency_pct": {
                    "type": "number",
                    "description": "量子效率估算 (%)",
                },
            },
            "required": ["verdict", "confidence", "reasoning"],
        },
    },
}


SYSTEM_PROMPT = """你是材料合成评审专家, 专精 NIR 荧光粉 (garnet/perovskite 族). 输入:
- 拟合成配方 (化学式 + 掺杂)
- 4 条 BPU 线输出 (XRD MLP 主相分类 ★主力, XRD YOLO 形态校验 ☆辅助, PL MLP 掺杂识别 ★主力, PL YOLO 形态校验 ☆辅助)
- XRD 计算源标注:
    * "MACE-MPA-0" = 通用机器学习势能面 (Batatia 2024, MatBench leaderboard #1, F1=0.96)
      → 真·原子替代 + 静态结构弛豫, 可视为 DFT 级精度. 含 lattice / 形成能 / 弹性 / 声子稳定性
      → 这条比 Vegard 可信度高得多, 推理中可直接引用 "MACE-MPA-0 算出 K_VRH=XX GPa, 结构稳定"
    * "Vegard 一阶" = Shannon 1976 半径平均近似, 仅 host 同族 (Jaccard≥0.4) 时近似可用
- Top-1 XRD 类比 + Top-3 PL 实测类比
- 失败旗帜 (半径失配/电荷/浓度 etc.)
- Top-4 相关文献 chunk

判决规则 (按优先级, 找到第一条匹配就停):

**【规则 0 — 强 PL 实测锚定, 最高优先级】**
Top-1 PL analog similarity ≥ 0.85 时, **必须**以 PL 实测 xrd_result 为准, 无视 XRD analog 和 BPU xrd_num:
  - Top-1 PL xrd_result=pure → verdict=GO, confidence≥0.75 (这是你们实验室已验证过的配方或近邻)
  - Top-1 PL xrd_result=mixed → verdict=REVISE, confidence≥0.7 (同配方已证杂相, 需换元素或调参)
  - Top-1 PL xrd_result=amorphous → verdict=DROP, confidence≥0.6
  - 推理里必须引用 Top-1 类比的 λ_em / FWHM / 热稳定性 作为预测基线
  (理由: 同/近邻配方的实测结果比 BPU 虚拟特征推理 更可信. 这是研究员要的核心价值 —
   "别人做过类似的, 告诉我他们的结果")

**【规则 1 — error 级失败旗帜】**
任何 error 级 flag → verdict ≤ REVISE, 在 reasoning 指明具体 flag code

**【规则 2 — BPU 主相分类强烈反对】**
BPU xrd_num prob < 0.5 且 Top-1 PL analog similarity < 0.85 → verdict = DROP

**【规则 3 — 类比全弱】**
xrd_analog similarity < 0.4 AND Top-1 PL analog similarity < 0.5 → verdict = UNKNOWN,
建议 PC 侧扩池 pymatgen 算新 CIF

**【规则 4 — 信号一致绿】**
flags 全绿 + BPU xrd_num prob ≥ 0.7 + Top-1 PL 类比 pure → verdict = GO, confidence ≥ 0.7

**【规则 5 — 其他】**
verdict = REVISE, 指出应改的参数

输出硬约束 (请严格遵守):
- **必须调用 submit_synthesis_verdict 工具返回结构化结果**
- 如果工具调用不可用, **必须且只能** 在 content 里输出一个 ```json ... ``` 代码块, 字段同 submit_synthesis_verdict 参数
- reasoning ≤300 字, 包含至少 1 个 BPU 概率数值 + 1 个文献引用 (标题或 DOI)
- optimization 最多 3 条, 每条 action 用祈使句 (例如 "改 Ga 为 In 降低半径失配")
- suggested_sinter 必须基于类比实测 (Top-1 pl_analog 的 sinter_temp_C ±50°C)
- 禁止 Markdown 标题 / LaTeX / 解释性自由文本 (只要 reasoning 一个字符串字段内自然句子可以)

JSON schema (备用路径, R1 不调工具时必须用这个):
{
  "verdict": "GO" | "REVISE" | "DROP" | "UNKNOWN",
  "confidence": 0-1 之间 number,
  "reasoning": string,
  "optimization": [{"action": string, "why": string}, ...],
  "suggested_sinter": {"temp_C": number, "hours": number},
  "predicted_lambda_em_nm_range": [min, max],
  "predicted_excitation_peaks_nm": [λ_ex1, λ_ex2],
  "predicted_fwhm_nm": number,
  "predicted_thermal_stability_pct": number,
  "predicted_quantum_efficiency_pct": number,
  "predicted_phase_purity_prob": number
}
"""


# ============ JSON 提取 (R1 不调工具时的 fallback) ============
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BARE_RE = re.compile(r"(\{[^{}]*\"verdict\"[^{}]*\})", re.DOTALL)


def _extract_json_verdict(content: str) -> dict | None:
    """从 R1 的自由文本 content 里抠出 JSON. 兼容 ```json ... ``` 代码块和裸 JSON."""
    if not content:
        return None
    # 优先 code block
    m = _JSON_BLOCK_RE.search(content)
    if not m:
        m = _JSON_BARE_RE.search(content)
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    # 清洗 reasoning 里的 DSML 标记
    if "reasoning" in v:
        v["reasoning"] = _clean(str(v["reasoning"]))
    for opt in v.get("optimization", []) or []:
        if isinstance(opt, dict):
            for k in ("action", "why"):
                if k in opt:
                    opt[k] = _clean(str(opt[k]))
    return v


def _format_user_input(payload: dict) -> str:
    """把 predict() 返回的 partial result 压成给 R1 的 user message."""
    p = payload

    # BPU 输出摘要
    stages = p.get("stages", {})
    def _bpu_line(name, key, is_mlp=True):
        s = stages.get(key, {})
        tag = "★" if is_mlp else "☆"
        if not s.get("ok"):
            return f"{tag} {name}: FAIL ({s.get('error', 'unknown')[:60]})"
        if is_mlp:
            return f"{tag} {name}: label={s.get('label')}, prob={s.get('prob', 0):.3f}, latency={s.get('latency_ms','?')}ms"
        else:
            return f"{tag} {name}: detected={s.get('detected')}, score={s.get('score', 0):.3f}, latency={s.get('latency_ms','?')}ms"

    bpu_lines = "\n".join([
        _bpu_line("xrd_num MLP (主相分类)", "bpu_xrd_num", True),
        _bpu_line("xrd_vision YOLO (XRD 谱图形态)", "bpu_xrd_vision", False),
        _bpu_line("spec_num MLP (掺杂识别)", "bpu_pl_num", True),
        _bpu_line("spec_vision YOLO (PL 谱图形态)", "bpu_pl_vision", False),
    ])

    # 类比
    xrd_a = p.get("xrd_analog")
    xrd_line = (f"Top-1 XRD 类比: {xrd_a['formula']} ({xrd_a['host_family']}, {xrd_a['spacegroup']}), "
                f"similarity={xrd_a['similarity']}") if xrd_a else "无 XRD 类比"
    pl_analogs = p.get("pl_analogs", [])
    if pl_analogs:
        pl_lines = "\n".join([
            f"  {i+1}. {a['formula']} dopant={a['dopant']}, "
            f"sinter={a.get('sinter','?')}, xrd={a['xrd_result']}, "
            f"λ_em={a['lambda_em_nm']}nm, FWHM={a['fwhm_nm']}nm, "
            f"热稳={a.get('thermal_stability_pct')}%"
            for i, a in enumerate(pl_analogs[:3])
        ])
    else:
        pl_lines = "  (无 PL 实测类比; 需 exp_ground_truth/observed_pl.csv 填充)"

    fly = p.get("flybrain") or {}
    if fly.get("ok"):
        top_hit = (fly.get("memory_hits") or [{}])[0]
        fly_line = (
            f"Fly-MB material memory: verdict={fly.get('verdict')}, "
            f"confidence={fly.get('confidence')}, novelty={fly.get('novelty_score')}, "
            f"OOD={fly.get('ood_score')}, top_hit="
            f"{top_hit.get('formula', 'n/a')}({top_hit.get('verdict', '?')}, "
            f"sim={top_hit.get('similarity', 0)})"
        )
    else:
        fly_line = f"Fly-MB material memory: unavailable ({fly.get('error', 'not run')})"

    # Flags
    flags = p.get("flags", [])
    flag_lines = "\n".join([f"  [{f['level'].upper()}] {f['code']}: {f['message']}"
                            for f in flags]) if flags else "  (全绿)"

    # RAG
    rag = p.get("rag", [])
    rag_lines = ""
    for r in rag[:4]:
        if isinstance(r, dict):
            text = (r.get("text") or r.get("snippet") or "")[:200]
            src = r.get("source") or r.get("title") or "?"
            rag_lines += f"  - [{src}] {text}\n"
    if not rag_lines:
        rag_lines = "  (RAG 未返回)"

    # Virtual XRD (MACE-MPA-0 ML 势能面 优先, Vegard 一阶 fallback)
    veg = p.get("vegard", {})
    method = veg.get("method", "vegard_1st_order" if veg.get("applied") else "none")
    if method == "mace_mpa_0":
        # MACE 真·结构替代 + DFT-级精度 (Batatia et al. 2024, MatBench leaderboard #1)
        e_atom = veg.get("energy_per_atom_eV")
        a_ang = veg.get("lattice_a_ang")
        K_GPa = veg.get("bulk_modulus_GPa")
        ph_stable = veg.get("phonon_stable")
        n_repl = veg.get("n_replaced")
        n_sites = veg.get("n_sites_total")
        pieces = [f"XRD 计算源: MACE-MPA-0 universal ML potential (Batatia 2024, MatBench F1=0.96)"]
        pieces.append(f"  替位: {n_repl}/{n_sites} sites; lattice a={a_ang}Å; energy={e_atom} eV/atom")
        if K_GPa is not None:
            pieces.append(f"  弹性: K_VRH={K_GPa} GPa")
        if ph_stable is not None:
            pieces.append(f"  声子稳定性: {'STABLE' if ph_stable else 'UNSTABLE (虚频)'} (lowest_freq={veg.get('phonon_lowest_THz')} THz)")
        veg_line = "\n".join(pieces)
    else:
        # 老路径 Vegard 一阶半径修正 (回退兼容)
        veg_line = (f"XRD 计算源: Vegard 一阶半径修正 (Shannon 1976), applied={veg.get('applied')}, "
                    f"Δa/a={veg.get('delta_a_over_a', 0):.4f}, Jaccard={veg.get('jaccard', 0)}") if veg else "Vegard: n/a"

    pl_meta = p.get("virtual_pl_meta", {})
    pl_baseline = ""
    if pl_meta.get("applied"):
        pl_method = pl_meta.get("method", "analog_empirical")
        if pl_method == "tanabe_sugano_huang_rhys":
            # Phase 1.2: TS 硬核路径
            ex_peaks = pl_meta.get("excitation_peaks_nm") or pl_meta.get("ts_excitation_peaks_nm") or []
            ex_str = "+".join(str(int(x)) for x in ex_peaks) if ex_peaks else "?"
            t_stab = pl_meta.get("thermal_stability_pct_423K") or pl_meta.get("ts_thermal_stability_pct_423K") or "?"
            t_ea = pl_meta.get("thermal_activation_energy_eV") or pl_meta.get("ts_thermal_activation_energy_eV") or "?"
            pl_baseline = (
                f"虚拟 PL 源: Tanabe-Sugano d3 晶场对角化 + Huang-Rhys 电声耦合 "
                f"(Sugano-Tanabe-Kamimura 1970; Alkauskas 2014 NJP doi:10.1088/1367-2630/16/7/073026)\n"
                f"  host={pl_meta.get('ts_host')}, Dq={pl_meta.get('ts_Dq_cm1')} cm⁻¹, "
                f"B={pl_meta.get('ts_B_cm1')} cm⁻¹, S={pl_meta.get('ts_S_huang_rhys')} (电声耦合因子), "
                f"T={pl_meta.get('ts_T_kelvin')}K\n"
                f"  → 预测 λ_em={pl_meta.get('predicted_lambda_em_nm')} nm, "
                f"FWHM={pl_meta.get('fwhm_nm')} nm\n"
                f"  → 激发峰 λ_ex={ex_str} nm, "
                f"热稳定性 I(423K)/I(298K)={t_stab}%, 活化能 Ea={t_ea} eV\n"
                f"  (ref DOI: {pl_meta.get('ts_source_doi')})"
            )
        else:
            pl_baseline = (f"虚拟 PL 源: nearest-analog 经验斜率 ({pl_method})\n"
                           f"  基线: {pl_meta.get('baseline_analog')} λ_em={pl_meta.get('baseline_lambda_em')}nm, "
                           f"晶场修正 Δλ={pl_meta.get('shift_nm', 0)}nm → "
                           f"预测 λ_em≈{pl_meta.get('predicted_lambda_em_nm')}nm")

    # 启发式粗判 (给 R1 作参考, R1 可推翻)
    heu = p.get("heuristic_verdict", {})
    heu_line = f"启发式粗判: {heu.get('verdict')}, {heu.get('reason')}"

    return f"""## 拟合成配方
- 化学式: {p['formula']}
- 掺杂: {p['dopant'].get('symbol', '?')} (site={p['dopant'].get('site')}, pct={p['dopant'].get('pct')}%)
- 解析: {(p.get('parse') or {}).get('elements', '?')}, 电荷 net={(p.get('parse') or {}).get('charge_balance', '?')}

## 4 条 BPU 输出
{bpu_lines}

## Fly-MB 果蝇连接组启发材料记忆
{fly_line}

## 类比参考
{xrd_line}
{veg_line}

PL 实测 Top-3:
{pl_lines}

{pl_baseline}

## 失败旗帜
{flag_lines}

## RAG 文献
{rag_lines}

## 参考启发式
{heu_line}

请根据上述信息调用 submit_synthesis_verdict 工具输出结论."""


def run_r1_judge_stream(payload: dict) -> Iterator[dict]:
    """流式生成 reasoning + 最终 verdict.

    Yields:
        {"type": "thinking", "text": str}   — R1 的 reasoning_content 增量
        {"type": "verdict", "verdict": dict} — tool_call 解析出的结构化判决
        {"type": "error", "error": str}     — 失败
    每次 yield 都是完整状态 (非增量).
    """
    user_msg = _format_user_input(payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_R1_KEY}"}
    body = {
        "model": DEEPSEEK_R1_MODEL,
        "messages": messages,
        "tools": [VERDICT_TOOL],
        "tool_choice": "auto",   # deepseek-reasoner 只支持 auto, 工具调用靠 prompt 强制
        "max_tokens": 3000,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(DEEPSEEK_R1_URL, headers=headers, json=body, timeout=90)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        yield {"type": "error", "error": f"R1 调用失败: {e}", "latency_ms": round((time.perf_counter() - t0) * 1000)}
        return

    msg = data["choices"][0]["message"]
    reasoning = _clean(msg.get("reasoning_content") or "")
    # 流式把 reasoning 一口气吐 (R1 非流式 API, 前端仍可打字机假装流)
    if reasoning:
        yield {"type": "thinking", "text": reasoning}

    tool_calls = msg.get("tool_calls") or []
    verdict = None
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") == "submit_synthesis_verdict":
            try:
                args = json.loads(fn.get("arguments", "{}"))
                # 清洗文本字段
                if "reasoning" in args:
                    args["reasoning"] = _clean(args["reasoning"])
                for opt in args.get("optimization", []):
                    for k in ("action", "why"):
                        if k in opt:
                            opt[k] = _clean(opt[k])
                verdict = args
                break
            except json.JSONDecodeError as e:
                yield {"type": "error", "error": f"工具参数解析失败: {e}"}
                return

    if verdict is None:
        # R1 没走 tool path → 从 content 里抠 JSON (它常常直接贴 JSON 在 markdown code block)
        content = msg.get("content") or ""
        verdict = _extract_json_verdict(content)
        if verdict is None:
            yield {
                "type": "error",
                "error": "R1 既没调用 submit_synthesis_verdict 工具, content 也没有可解析的 JSON",
                "content_fallback": _clean(content)[:500],
            }
            return

    yield {
        "type": "verdict",
        "verdict": verdict,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }


def extract_verdict_sync(payload: dict) -> dict:
    """非流式版本, 用于测试 / CLI."""
    result = {"thinking": "", "verdict": None, "error": None}
    for chunk in run_r1_judge_stream(payload):
        if chunk["type"] == "thinking":
            result["thinking"] = chunk["text"]
        elif chunk["type"] == "verdict":
            result["verdict"] = chunk["verdict"]
            result["latency_ms"] = chunk.get("latency_ms")
        elif chunk["type"] == "error":
            result["error"] = chunk["error"]
    return result


# ============================================================================
# Phase 1.3: Self-Consistency + Chain-of-Verification
# ============================================================================

DEFAULT_TEMPS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _single_r1_call_sync(payload: dict, temperature: float = 0.5,
                          timeout_sec: float = 90.0,
                          extra_user_msg: str | None = None) -> dict:
    """单次 R1 调用, 返回 {verdict, thinking, latency_ms, error}.

    和 run_r1_judge_stream 共享 SYSTEM_PROMPT + 工具 schema + JSON 兜底, 但同步返回,
    用于 Self-Consistency 并发调用.

    Args:
        payload: 合成预测 partial result (含 xrd_analog/vegard/stages/...)
        temperature: R1 采样温度 (0.3-0.7 推荐)
        extra_user_msg: 可选追加 user 消息 (用于 CoVe 第二轮 follow-up)
    """
    user_msg = _format_user_input(payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    if extra_user_msg:
        messages.append({"role": "user", "content": extra_user_msg})

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_R1_KEY}"}
    body = {
        "model": DEEPSEEK_R1_MODEL,
        "messages": messages,
        "tools": [VERDICT_TOOL],
        "tool_choice": "auto",
        "max_tokens": 3000,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(DEEPSEEK_R1_URL, headers=headers, json=body, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"verdict": None, "thinking": "", "error": f"HTTP: {e}",
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "temperature": temperature}

    msg = data.get("choices", [{}])[0].get("message", {}) or {}
    reasoning = _clean(msg.get("reasoning_content") or "")

    tool_calls = msg.get("tool_calls") or []
    verdict = None
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") == "submit_synthesis_verdict":
            try:
                args = json.loads(fn.get("arguments", "{}"))
                if "reasoning" in args:
                    args["reasoning"] = _clean(args["reasoning"])
                for opt in args.get("optimization", []):
                    for k in ("action", "why"):
                        if k in opt:
                            opt[k] = _clean(opt[k])
                verdict = args
                break
            except json.JSONDecodeError:
                pass

    if verdict is None:
        content = msg.get("content") or ""
        verdict = _extract_json_verdict(content)

    return {
        "verdict": verdict,
        "thinking": reasoning,
        "error": None if verdict else "no_verdict_parsed",
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "temperature": temperature,
    }


def run_r1_judge_self_consistent(
    payload: dict,
    n_samples: int = 5,
    temperatures: list[float] | None = None,
    enable_cove: bool = True,
    cove_consensus_threshold: float = 0.6,
) -> dict:
    """Self-Consistency 多样本并发投票 + (分歧时) Chain-of-Verification 二轮修正.

    步骤:
      1. 并发 n_samples 次 R1 调用 (不同 temperature)
      2. 多数投票 winner verdict label
      3. 若 consensus < threshold 或置信度差异 > 0.3, 触发 CoVe
         (让 R1 看到所有样本, 生成 follow-up verdict)
      4. 返回最终 verdict dict, 带 self_consistency_meta

    Returns:
        最终 verdict (可能是 chosen sample 或 CoVe 修正版) + meta:
        {
            "verdict": "GO" | "REVISE" | ...,
            "confidence": 0-1,
            "reasoning": ...,
            ...(原 tool schema 其他字段)...
            "self_consistency_meta": {
                "n_samples": 5,
                "n_valid": 4,
                "vote_dist": {"GO": 3, "REVISE": 1},
                "consensus_strength": 0.75,
                "cove_triggered": False,
                "samples": [  # 每个样本摘要 (thinking 截断)
                    {"idx": 0, "temp": 0.3, "verdict": "GO", "conf": 0.85, "latency_ms": 15200, "thinking_preview": "..."},
                    ...
                ],
                "total_latency_ms": 17800,
            }
        }
    """
    temps = temperatures or DEFAULT_TEMPS
    if len(temps) < n_samples:
        # 不够的补重复
        temps = (temps * ((n_samples // len(temps)) + 1))[:n_samples]
    else:
        temps = temps[:n_samples]

    t0 = time.perf_counter()
    results: list[dict] = [None] * n_samples
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_samples) as pool:
        futs = {pool.submit(_single_r1_call_sync, payload, t): i for i, t in enumerate(temps)}
        for fut in concurrent.futures.as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"verdict": None, "thinking": "", "error": f"exception: {e}",
                                 "latency_ms": 0, "temperature": temps[idx]}
    total_ms = round((time.perf_counter() - t0) * 1000)

    verdicts = [r.get("verdict") for r in results]
    vote = majority_vote_verdict(verdicts)

    cove_verdict = None
    if enable_cove and needs_cove(vote, min_consensus=cove_consensus_threshold):
        try:
            followup = build_cove_followup_prompt(verdicts, _format_user_input(payload))
            cove_result = _single_r1_call_sync(payload, temperature=0.3, extra_user_msg=followup)
            cove_verdict = cove_result.get("verdict")
        except Exception:
            cove_verdict = None

    final = summarize_self_consistency(vote, cove_verdict)

    # 加 samples 明细 + 总延迟
    meta = final.get("self_consistency_meta", {})
    meta["samples"] = [
        {
            "idx": i,
            "temp": r.get("temperature"),
            "verdict": (r.get("verdict") or {}).get("verdict"),
            "conf": (r.get("verdict") or {}).get("confidence"),
            "latency_ms": r.get("latency_ms"),
            "thinking_preview": (r.get("thinking") or "")[:200],
            "error": r.get("error"),
        }
        for i, r in enumerate(results)
    ]
    meta["total_latency_ms"] = total_ms
    final["self_consistency_meta"] = meta
    return final


def run_r1_judge_self_consistent_stream(
    payload: dict,
    n_samples: int = 5,
    temperatures: list[float] | None = None,
    enable_cove: bool = True,
) -> Iterator[dict]:
    """Self-Consistency 的流式变体 — 前端看第 1 个样本的 thinking 实时进度, 后台同时跑 2-N.

    Yields:
        {"type": "thinking", "text": str, "sample_idx": 0}
        {"type": "progress", "completed": N, "total": 5}
        {"type": "verdict", "verdict": dict, "self_consistency_meta": ...}
        {"type": "error", "error": str}

    实现: 5 个 thread 并发, thread 0 的 thinking 边产生边 yield (polling), thread 1-4 静默.
    简化: 其实 thinking 只在 R1 返回时拿得到 (非真 SSE), 所以就先 yield 提示, 再跑完后
    yield 最终 verdict.
    """
    temps = temperatures or DEFAULT_TEMPS
    temps = temps[:n_samples] if len(temps) >= n_samples else (temps * ((n_samples // len(temps)) + 1))[:n_samples]

    yield {"type": "progress", "completed": 0, "total": n_samples,
           "note": f"Self-Consistency 启动 {n_samples} 个 R1 并发调用"}

    try:
        final = run_r1_judge_self_consistent(
            payload, n_samples=n_samples, temperatures=temps, enable_cove=enable_cove,
        )
    except Exception as e:
        yield {"type": "error", "error": f"Self-Consistency failed: {e}"}
        return

    meta = final.get("self_consistency_meta", {})
    # 把第一个有效样本的 thinking 吐给前端 (作为"代表性推理展示")
    samples = meta.get("samples", [])
    first_valid = next((s for s in samples if s.get("verdict")), None)
    if first_valid:
        yield {
            "type": "thinking",
            "text": first_valid.get("thinking_preview", "") + "\n\n[... 其余 4 个样本并发, 见下方投票分布]",
            "sample_idx": first_valid.get("idx"),
        }

    yield {
        "type": "progress",
        "completed": meta.get("n_valid", 0),
        "total": meta.get("n_samples", n_samples),
        "vote_dist": meta.get("vote_dist", {}),
        "consensus_strength": meta.get("consensus_strength", 0),
        "cove_triggered": meta.get("cove_triggered", False),
    }

    yield {
        "type": "verdict",
        "verdict": final,
        "latency_ms": meta.get("total_latency_ms"),
    }


def extract_verdict_self_consistent_sync(
    payload: dict, n_samples: int = 5,
) -> dict:
    """同步 SC 调用, 用于测试 / CLI. 返回和 extract_verdict_sync 兼容的 dict."""
    final = run_r1_judge_self_consistent(payload, n_samples=n_samples)
    meta = final.get("self_consistency_meta", {})
    return {
        "thinking": (meta.get("samples", [{}])[0].get("thinking_preview") or "") if meta.get("samples") else "",
        "verdict": final,
        "error": None if final.get("verdict") else "no_winner",
        "latency_ms": meta.get("total_latency_ms"),
        "self_consistency_meta": meta,
    }
