"""
Stage 1: PL CSV Parser (Fluoromax / Horiba FluorEssence 格式)

输入: .csv 或 .txt 文件 (同内容, UTF-8 或 GBK 编码)
输出: dict, 含 wavelength / counts / scan_type / meta

Header 结构 (前 ~21 行, "Key,Value," 形式):
    Labels,0,
    Type,Emission Scan,
    Comment,,
    Start,600.00,
    Stop,1650.00,
    Step,1.00,
    Fixed/Offset,455.00,
    Xaxis,Wavelength,
    Yaxis,Counts,
    ...更多仪器参数...
    (空行分隔)
    600.00,3.21088989E+2,       ← 数据区开始
    601.00,1.31500807E+1,
    ...

Header 行数不固定 (常见 17~22 行), 用 "空行 OR 首字段可转 float" 作为分界。

本轮只处理常规 single-column emission / excitation, 遇到 QY 双列或 TQ 温度
序列会返回 None 并在 meta 里标注原因。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PLSpectrum:
    """一份解析后的 PL 光谱."""
    wavelength: np.ndarray          # nm
    counts: np.ndarray              # raw counts
    scan_type: str                  # 'em' | 'ex' | 'pl' | 'unknown'
    start: float
    stop: float
    step: float
    fixed_offset: float             # em 模式: 激发 λ; ex 模式: 检测 λ
    path: str                       # 原始文件路径
    meta: dict = field(default_factory=dict)   # 全部 header 键值对
    skip_reason: Optional[str] = None  # 若为 None 即解析成功; 否则是跳过原因

    def is_valid(self) -> bool:
        return self.skip_reason is None and self.wavelength is not None

    def n_points(self) -> int:
        return 0 if self.wavelength is None else len(self.wavelength)


# ============ 核心函数 ============
_MAX_PL_FILE_BYTES = 16 * 1024 * 1024


def _decode_text(payload: bytes) -> list[str]:
    """Decode one bounded PL export with the instrument's known encodings."""

    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            return payload.decode(enc).splitlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError("无法解码 PL 光谱")


def _read_text(path: str | Path) -> list[str]:
    """Read an offline/trusted file with a strict size bound."""

    source = Path(path)
    with source.open("rb") as stream:
        payload = stream.read(_MAX_PL_FILE_BYTES + 1)
    if len(payload) > _MAX_PL_FILE_BYTES:
        raise ValueError("PL 光谱文件超过 16 MiB 上限")
    return _decode_text(payload)


def _is_data_row(row: str) -> bool:
    """判断一行是否是"两列数值"数据行."""
    if not row or row.strip() == "":
        return False
    parts = [p.strip() for p in row.split(",") if p.strip()]
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def _classify_scan_type(meta: dict, filename: str) -> str:
    """
    判定扫描类型:
    - header "Type" 字段 = "Emission Scan" → em
    - header "Type" 字段 = "Excitation Scan" → ex
    - 文件名含 '-em' → em
    - 文件名含 '-ex' → ex
    - 文件名含 '-pl' → pl (宽带 PL)
    - 其他 → unknown
    """
    t = (meta.get("Type") or "").strip().lower()
    if "emission" in t:
        return "em"
    if "excitation" in t:
        return "ex"
    name = filename.lower()
    if "-em" in name or "_em" in name:
        return "em"
    if "-ex" in name or "_ex" in name:
        return "ex"
    if "-pl" in name or "_pl" in name:
        return "pl"
    return "unknown"


def _is_qy_or_special(lines: list[str], filename: str) -> Optional[str]:
    """
    检测 QY / TQ / KONGBAI / fitted 等特殊格式, 返回跳过原因 (None 表示正常).

    QY 文件:
    - 文件名含 'KONGBAI' (参考光谱)
    - header 里有 "QY ="
    - 数据区有超过 2 列 (sample + reference)

    Fitted 文件:
    - 文件名含 'fitted'
    """
    name = filename.lower()
    if "kongbai" in name:
        return "QY 参考光谱 (KONGBAI)"
    if "fitted" in name:
        return "拟合结果文件 (fitted)"

    # header 里找 "QY"
    for ln in lines[:30]:
        if re.search(r"\bQY\s*=", ln, re.IGNORECASE):
            return "QY 测量数据 (含 QY= 字段)"

    # 数据行列数 > 2 → 双列 QY
    for ln in lines:
        if _is_data_row(ln):
            cols = [c for c in ln.split(",") if c.strip()]
            if len(cols) > 2:
                return f"多列数据 ({len(cols)} cols), 可能是 QY 双列"
            break
    return None


def _parse_pl_lines(lines: list[str], path: str) -> PLSpectrum:
    """Parse already-decoded lines without performing filesystem access."""

    source = Path(path)
    path = str(source)
    fname = source.name

    # 检查是否为 QY / 特殊格式
    special_reason = _is_qy_or_special(lines, fname)
    if special_reason:
        return PLSpectrum(
            wavelength=None, counts=None, scan_type="unknown",
            start=0, stop=0, step=0, fixed_offset=0, path=path,
            skip_reason=special_reason,
        )

    # 分离 header 和 data
    meta: dict = {}
    data_start = None
    for i, ln in enumerate(lines):
        if _is_data_row(ln):
            data_start = i
            break
        # header row: "Key,Value,..." 形式
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2 and parts[0]:
            key = parts[0]
            val = parts[1]
            meta[key] = val

    if data_start is None:
        return PLSpectrum(
            wavelength=None, counts=None, scan_type="unknown",
            start=0, stop=0, step=0, fixed_offset=0, path=path, meta=meta,
            skip_reason="未找到数据区",
        )

    # 解析数据行
    xs, ys = [], []
    for ln in lines[data_start:]:
        if not _is_data_row(ln):
            # 数据区结束 (某些文件后面还有尾部 meta)
            if ln.strip() == "":
                continue
            break
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except (ValueError, IndexError):
            continue

    if len(xs) < 10:
        return PLSpectrum(
            wavelength=None, counts=None, scan_type="unknown",
            start=0, stop=0, step=0, fixed_offset=0, path=path, meta=meta,
            skip_reason=f"数据点过少 ({len(xs)})",
        )

    wavelength = np.array(xs, dtype=np.float64)
    counts = np.array(ys, dtype=np.float64)

    # 元数据字段
    def _fget(k: str, default: float = 0.0) -> float:
        try:
            return float(meta.get(k, default))
        except (ValueError, TypeError):
            return default

    scan_type = _classify_scan_type(meta, fname)

    return PLSpectrum(
        wavelength=wavelength,
        counts=counts,
        scan_type=scan_type,
        start=_fget("Start", float(wavelength[0])),
        stop=_fget("Stop", float(wavelength[-1])),
        step=_fget("Step", 1.0),
        fixed_offset=_fget("Fixed/Offset"),
        path=path,
        meta=meta,
    )


def _read_failure(path: str, error: Exception) -> PLSpectrum:
    return PLSpectrum(
        wavelength=None,
        counts=None,
        scan_type="unknown",
        start=0,
        stop=0,
        step=0,
        fixed_offset=0,
        path=path,
        skip_reason=f"读文件失败: {error}",
    )


def parse_pl_bytes(payload: bytes, *, path: str = "uploaded-spectrum.csv") -> PLSpectrum:
    """Parse bytes supplied by a caller that already performed safe I/O."""

    if not isinstance(payload, bytes):
        return _read_failure(path, TypeError("payload must be bytes"))
    if len(payload) > _MAX_PL_FILE_BYTES:
        return _read_failure(path, ValueError("PL 光谱文件超过 16 MiB 上限"))
    try:
        lines = _decode_text(payload)
    except Exception as error:
        return _read_failure(path, error)
    return _parse_pl_lines(lines, path)


def parse_pl_csv(path: str | Path) -> PLSpectrum:
    """Parse an offline/trusted Fluoromax PL CSV path.

    HTTP handlers must read through ``read_contained_bytes`` and call
    :func:`parse_pl_bytes`, keeping request-derived path text away from this
    filesystem API.
    """

    path_text = str(Path(path))
    try:
        lines = _read_text(path)
    except Exception as error:
        return _read_failure(path_text, error)
    return _parse_pl_lines(lines, path_text)


# ============ CLI 自测 ============
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # 默认测试样品
        test = str(
            Path(__file__).resolve().parent.parent
            / "NaY2Ga2InGe2O12" / "0.002cr-yni-pl" / "0.002ni-455-em.csv"
        )
    else:
        test = sys.argv[1]
    s = parse_pl_csv(test)
    print(f"path: {s.path}")
    print(f"scan_type: {s.scan_type}")
    print(f"range: {s.start} → {s.stop} step {s.step}")
    print(f"fixed/offset: {s.fixed_offset}")
    if s.is_valid():
        print(f"points: {s.n_points()}")
        print(f"counts max/min: {s.counts.max():.2e} / {s.counts.min():.2e}")
    else:
        print(f"SKIP: {s.skip_reason}")
