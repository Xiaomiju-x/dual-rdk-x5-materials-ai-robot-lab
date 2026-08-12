"""
从文件名 + 父目录名自动抽取标签.

主任务: 掺杂离子 (Cr / Ni / Cr+Ni / other)
次任务: 宿主材料 (NaY2Ga2InGe2O12 / Y3ZnGa3GeO12)
额外: 估计掺杂浓度 (从正则 \\d+\\.\\d+)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 主任务标签
LABEL_CR = "cr"
LABEL_NI = "ni"
LABEL_CR_NI = "cr_ni"
LABEL_OTHER = "other"

LABELS = [LABEL_CR, LABEL_NI, LABEL_CR_NI, LABEL_OTHER]
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABELS)}


@dataclass
class PLLabel:
    dopant: str                    # cr / ni / cr_ni / other
    dopant_id: int
    host: str                      # NaY2Ga2InGe2O12 / Y3ZnGa3GeO12 / unknown
    cr_conc: Optional[float] = None
    ni_conc: Optional[float] = None
    notes: list[str] = None         # codopants (Be/Sr/Zn/h3bo3/...)

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


# 正则: 捕获 "浓度 + 掺杂" 组合, 例如 "0.002cr", "0.05NI", "0.01Cr"
_DOPANT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cr|ni)", re.IGNORECASE)

# 其他可能出现的共掺杂元素 (非 Cr/Ni)
_COCAT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(be|ca|mg|sr|zn|h3bo3|li|na|k)",
    re.IGNORECASE,
)


def label_from_path(path: str | Path) -> PLLabel:
    """
    解析路径 → PLLabel.

    Args:
        path: 相对或绝对路径, 例如
              NaY2Ga2InGe2O12/0.002cr-yni-pl/0.002ni-455-em.csv
              Y3ZnGa3GeO12/CR/0.01cr+助溶剂+PL-PLE-狭缝2.5+2.5/0.01cr-445-em.csv
    """
    p = Path(path)
    parts = [x.lower() for x in p.parts]
    fname_lower = p.name.lower()

    # --- Host 识别 ---
    host = "unknown"
    for seg in p.parts:
        if "nay2ga2inge2o12" in seg.lower() or "nay2g" in seg.lower():
            host = "NaY2Ga2InGe2O12"
            break
        if "y3znga3geo12" in seg.lower() or "y3zng" in seg.lower():
            host = "Y3ZnGa3GeO12"
            break

    # --- 掺杂浓度 (只看文件名 + 最近父目录) ---
    search_text = fname_lower + " " + (p.parent.name.lower() if p.parent else "")

    cr_conc: Optional[float] = None
    ni_conc: Optional[float] = None
    for m in _DOPANT_RE.finditer(search_text):
        conc = float(m.group(1))
        dop = m.group(2).lower()
        if dop == "cr":
            cr_conc = conc if cr_conc is None else max(cr_conc, conc)
        elif dop == "ni":
            ni_conc = conc if ni_conc is None else max(ni_conc, conc)

    # --- 主任务标签 (Cr / Ni / Cr+Ni / other) ---
    # 注意: 有的文件名只有 "0.01cr" 没有浓度数字直接写 "ni", 也算 ni
    has_cr = (cr_conc is not None) or ("cr" in fname_lower and "-cr" in fname_lower)
    has_ni = (ni_conc is not None) or ("ni" in fname_lower and "-ni" in fname_lower)
    # 特殊: 文件名形如 "0.002ni-455-em" 在父目录 "0.002cr-yni-pl" 下
    # (意思是 Cr 0.002 基础上加变化的 Ni), 属于 cr_ni 共掺
    parent_lower = p.parent.name.lower() if p.parent else ""
    if "cr" in parent_lower and ("ni" in fname_lower or "ni" in parent_lower):
        has_cr = True
        has_ni = True

    if has_cr and has_ni:
        dopant = LABEL_CR_NI
    elif has_cr:
        dopant = LABEL_CR
    elif has_ni:
        dopant = LABEL_NI
    else:
        dopant = LABEL_OTHER

    # --- 共掺杂 notes ---
    notes = []
    for m in _COCAT_RE.finditer(search_text):
        notes.append(f"{m.group(1)}{m.group(2).lower()}")
    # 关键字 notes
    for kw in ("助溶剂", "缺陷", "quexian", "h3bo3", "固溶"):
        if kw in search_text:
            notes.append(kw)

    return PLLabel(
        dopant=dopant,
        dopant_id=LABEL_TO_ID[dopant],
        host=host,
        cr_conc=cr_conc,
        ni_conc=ni_conc,
        notes=notes,
    )


# ============ CLI 自测 ============
if __name__ == "__main__":
    cases = [
        "NaY2Ga2InGe2O12/0.002cr-yni-pl/0.002ni-455-em.csv",
        "NaY2Ga2InGe2O12/0.002cr-yni-pl/0.007ni-455-em.csv",
        "Y3ZnGa3GeO12/CR/0.01cr+助溶剂+PL-PLE-狭缝2.5+2.5/0.01cr-445-em.csv",
        "Y3ZnGa3GeO12/CR/0.01cr+助溶剂+PL-PLE-狭缝2.5+2.5/0.01cr-830-ex.csv",
        "Y3ZnGa3GeO12/Ni/0.05ni-455-em.csv",
        "Y3ZnGa3GeO12/cr-ni/0.01cr-0.005ni-455-em.csv",
        "NaY2Ga2InGe2O12/Xcr-pl+ple-狭缝2.5+2.5/0.05cr-445-em.csv",
        "NaY2Ga2InGe2O12/0.03cr+quexian-pl-狭缝2.5+2.5/0.03cr-0.05Be-455-pl.csv",
    ]
    for c in cases:
        lbl = label_from_path(c)
        print(f"  {c}")
        print(f"    → dopant={lbl.dopant:6s} host={lbl.host:20s} "
              f"cr={lbl.cr_conc} ni={lbl.ni_conc} notes={lbl.notes}")
