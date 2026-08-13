"""batch_parser.py — 解析 textarea 多行成结构化预测请求.

支持 3 种行格式 (容错):
  1. CSV:    `La3ZnGa3GeO12,Cr3+,Ga,0.75[,1450]`
  2. 紧凑:   `La3ZnGa3GeO12 + Cr-0.75%@Ga` 或 `La3ZnGa3GeO12 Cr@Ga 0.75`
  3. 自由:   `La3ZnGa3GeO12 Cr3+ 占位Ga 0.75%`

返回 {items: [{formula, dopant:{element,valence,site,pct,symbol}, sinter_temp_C}], errors: [...]}.
错误行不阻塞, 单独收集.
"""
from __future__ import annotations

import re

_MAX_BATCH_LINE_CHARS = 1024


def _try_parse_dopant_token(tok: str) -> dict | None:
    """`"Cr3+"` → {element, valence}.  `"Cr"` → 默认 valence=3."""
    tok = tok.strip()
    m = re.match(r"^([A-Z][a-z]?)(\d*)\+?$", tok)
    if m:
        return {"element": m.group(1),
                "valence": int(m.group(2)) if m.group(2) else 3,
                "symbol": tok}
    return None


def _try_parse_pct(tok: str) -> float | None:
    tok = tok.strip().rstrip("%")
    try:
        v = float(tok)
        # 容忍 0.01 (factor) → 1%
        if 0 < v < 0.2:
            v *= 100
        return round(v, 3)
    except ValueError:
        return None


def _is_element_symbol(value: str) -> bool:
    return len(value) in (1, 2) and value[0].isupper() and (
        len(value) == 1 or value[1].islower()
    )


def _parse_compact_line(line: str) -> tuple[str, str, str, str] | None:
    """Parse ``Formula + Dopant-pct%@site`` using bounded linear scans.

    This replaces the previous whitespace-heavy regular expression.  Every
    split/search traverses the input at most once and all tokens have an
    explicit grammar, so adversarial whitespace cannot cause regex backtracking.
    """

    if len(line) > _MAX_BATCH_LINE_CHARS:
        return None
    separator = line.find("+")
    if separator < 0:
        return None
    formula_text, compact = line[:separator].strip(), line[separator + 1:].strip()
    if not formula_text or not all(character.isalnum() or character in "_()" for character in formula_text):
        return None

    if compact.count("@") != 1:
        return None
    dopant_and_pct, site = (part.strip() for part in compact.rsplit("@", 1))
    if not _is_element_symbol(site) or dopant_and_pct.count("-") != 1:
        return None
    dopant_text, pct_text = (part.strip() for part in dopant_and_pct.split("-", 1))
    if pct_text.endswith("%"):
        pct_text = pct_text[:-1].strip()
    if not pct_text or any(character not in "0123456789." for character in pct_text):
        return None

    element_length = 2 if len(dopant_text) >= 2 and dopant_text[1].islower() else 1
    element = dopant_text[:element_length]
    charge = dopant_text[element_length:]
    if not _is_element_symbol(element):
        return None
    if charge.endswith("+"):
        charge = charge[:-1]
    if charge and not charge.isdigit():
        return None
    return formula_text, element, pct_text, site


def parse_line(line: str) -> tuple[dict | None, str | None]:
    """返回 (item, error_msg).  item=None 表示无法解析."""
    line = line or ""
    if len(line) > _MAX_BATCH_LINE_CHARS:
        return None, f"单行超过 {_MAX_BATCH_LINE_CHARS} 字符上限"
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None        # 空行/注释忽略, 不算错
    raw = line

    # ---- 格式 1: CSV ----
    if line.count(",") >= 3:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            formula = parts[0]
            dop_tok = parts[1]
            site = parts[2]
            pct = _try_parse_pct(parts[3])
            sinter = float(parts[4]) if len(parts) >= 5 and parts[4].strip() else None
            dop = _try_parse_dopant_token(dop_tok)
            if formula and dop and site and pct is not None:
                dop["site"] = site
                dop["pct"] = pct
                return ({"formula": formula, "dopant": dop,
                        "sinter_temp_C": sinter, "host_hint": None}, None)

    # ---- 格式 2 紧凑: "Formula + Dopant-pct%@site"  ----
    compact_fields = _parse_compact_line(line)
    if compact_fields:
        formula, el, pct_s, site = compact_fields
        pct = _try_parse_pct(pct_s)
        if pct is not None:
            return ({"formula": formula,
                    "dopant": {"element": el, "valence": 3, "site": site,
                              "pct": pct, "symbol": f"{el}3+"},
                    "sinter_temp_C": None, "host_hint": None}, None)

    # ---- 格式 3 自由 (空白分隔, 顺序: formula dopant site pct) ----
    tokens = re.split(r"\s+", line)
    tokens = [t.strip(",;|") for t in tokens if t.strip(",;|")]
    if len(tokens) >= 4:
        formula = tokens[0]
        # 找 dopant token (含 + 号) 或第 2 个 token
        dop = _try_parse_dopant_token(tokens[1])
        if not dop:
            return None, f"无法识别掺杂离子: {tokens[1]!r}"
        # 找 site (第 3 个或带 "占位" / "@" 前缀)
        site = None
        for t in tokens[2:]:
            t2 = t.replace("占位", "").replace("@", "").strip()
            if re.match(r"^[A-Z][a-z]?$", t2):
                site = t2
                break
        if not site:
            return None, f"无法识别替代位点 (期望大写字母元素如 Ga/Al)"
        # 找 pct (含 % 或纯数字, 通常是最后一个 token)
        pct = None
        for t in reversed(tokens):
            v = _try_parse_pct(t)
            if v is not None and 0 < v <= 20:
                pct = v
                break
        if pct is None:
            return None, f"无法识别浓度 (期望 0.01-10 之间数字, 可带 %)"
        dop["site"] = site
        dop["pct"] = pct
        return ({"formula": formula, "dopant": dop,
                "sinter_temp_C": None, "host_hint": None}, None)

    return None, f"无法识别行格式 (尝试 3 种均失败): {raw[:80]!r}"


def parse_lines(text: str, max_items: int = 20) -> dict:
    """主入口. 返回 {items, errors, n_total, n_parsed, n_skipped}."""
    items: list[dict] = []
    errors: list[dict] = []
    raw_lines = (text or "").splitlines()
    n_total = sum(1 for l in raw_lines if l.strip() and not l.strip().startswith("#"))

    for i, line in enumerate(raw_lines, 1):
        if len(items) >= max_items:
            errors.append({"line_num": i, "raw": line.strip(),
                          "error": f"超过 max_items={max_items}, 后续行忽略"})
            break
        item, err = parse_line(line)
        if item:
            item["_source_line"] = i
            item["_source_raw"] = line.strip()
            items.append(item)
        elif err:
            errors.append({"line_num": i, "raw": line.strip()[:80], "error": err})

    return {
        "items": items,
        "errors": errors,
        "n_total": n_total,
        "n_parsed": len(items),
        "n_skipped": n_total - len(items),
    }


if __name__ == "__main__":
    test = """
    # 一些测试
    La3ZnGa3GeO12,Cr3+,Ga,0.75
    Gd3InGa4O12 + Cr-0.75%@Ga
    Y3Al5O12 Cr3+ Al 1.0
    Y3ZnGa3GeO12 Cr3+ Zn 1
    invalid garbage line here
    """
    out = parse_lines(test)
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2))
