"""
Stage 1: Bruker .raw文件解析
优先fabio，失败则用xylib命令行工具fallback
"""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path


def parse_with_fabio(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """用fabio解析Bruker .raw文件"""
    import fabio
    img = fabio.open(filepath)

    # fabio读Bruker RAW后，数据在img.data中
    # 头信息中包含起始角度、步长等
    header = img.header

    # 尝试从header中提取2θ信息
    # Bruker RAW格式的header key因版本不同而异
    intensity = img.data.flatten().astype(np.float64)

    # 提取2θ起始角和步长
    start_2theta = None
    step_2theta = None

    # 常见的header key
    for start_key in ['THETA_START', 'START', '2THETA_START', 'start_2theta']:
        if start_key in header:
            start_2theta = float(header[start_key])
            break

    for step_key in ['THETA_STEP', 'STEP', '2THETA_STEP', 'step_2theta']:
        if step_key in header:
            step_2theta = float(header[step_key])
            break

    if start_2theta is None or step_2theta is None:
        # 如果header中没有直接的角度信息，尝试其他方式
        # 有些版本会在 'theta_start' 和 'theta_step' 中
        for key, val in header.items():
            key_lower = key.lower()
            if 'start' in key_lower and 'theta' in key_lower:
                start_2theta = float(val)
            elif 'step' in key_lower and 'theta' in key_lower:
                step_2theta = float(val)

    if start_2theta is None or step_2theta is None:
        raise ValueError(
            f"无法从header中提取2θ信息。可用的header keys: {list(header.keys())}"
        )

    n_points = len(intensity)
    two_theta = start_2theta + np.arange(n_points) * step_2theta

    return two_theta, intensity


def parse_with_xylib(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """用xylib命令行工具解析(fallback方案)"""
    # xylib的命令行工具 xyconv 可以把各种格式转成txt
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # xyconv: xylib的命令行转换工具
        result = subprocess.run(
            ['xyconv', filepath, tmp_path],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            # 尝试指定格式
            result = subprocess.run(
                ['xyconv', '-t', 'bruker_raw', filepath, tmp_path],
                capture_output=True, text=True, timeout=30
            )

        if result.returncode != 0:
            raise RuntimeError(f"xyconv转换失败: {result.stderr}")

        # 读取转换后的txt文件
        data = np.loadtxt(tmp_path, comments='#')
        two_theta = data[:, 0]
        intensity = data[:, 1]

        return two_theta, intensity

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_raw_manual(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    手动解析Bruker RAW v2格式(最后的fallback)
    RAW v2结构: 4字节magic + header + data blocks
    """
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()

    # 检查magic number
    magic = raw_bytes[:4]

    if magic == b'RAW ':
        # RAW v1 格式
        return _parse_raw_v1(raw_bytes)
    elif magic == b'RAW2':
        # RAW v2 格式
        return _parse_raw_v2(raw_bytes)
    elif magic[:3] == b'RAW':
        # RAW v3/v4 - 结构更复杂
        raise ValueError(f"RAW v3/v4格式暂不支持手动解析，请安装xylib")
    else:
        raise ValueError(f"不是Bruker RAW格式，magic: {magic}")


def _parse_raw_v1(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """解析RAW v1 (magic: 'RAW ')"""
    import struct

    # offset 4: uint32 n_steps
    n_steps = struct.unpack_from('<I', raw_bytes, 4)[0]

    # offset 8: float32 start_2theta
    start_2theta = struct.unpack_from('<f', raw_bytes, 8)[0]

    # offset 12: float32 step_size
    step_size = struct.unpack_from('<f', raw_bytes, 12)[0]

    # 数据从偏移量156开始
    data_offset = 156
    intensity = np.frombuffer(
        raw_bytes, dtype=np.float32, count=n_steps, offset=data_offset
    ).astype(np.float64)

    two_theta = start_2theta + np.arange(n_steps) * step_size
    return two_theta, intensity


def _parse_raw_v2(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """解析RAW v2 (magic: 'RAW2')"""
    import struct

    # 通用头: offset 0-3 "RAW2", offset 4-7 n_ranges
    # 通用头 + range头共324字节，数据紧随其后
    # offset 272: float32 start_2theta
    # offset 268: float32 step_size
    start_2theta = struct.unpack_from('<f', raw_bytes, 272)[0]
    step_size = struct.unpack_from('<f', raw_bytes, 268)[0]

    data_offset = 324
    n_steps = (len(raw_bytes) - data_offset) // 4

    intensity = np.frombuffer(
        raw_bytes, dtype=np.float32, count=n_steps, offset=data_offset
    ).astype(np.float64)

    two_theta = start_2theta + np.arange(n_steps) * step_size
    return two_theta, intensity


def parse_raw_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    解析Bruker .raw文件的统一入口
    按优先级尝试: fabio → xylib → 手动解析

    Returns:
        two_theta: 2θ角度数组 (度)
        intensity: 对应强度数组
    """
    filepath = str(filepath)

    # Materials Project下载的.npz格式
    if str(filepath).endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        two_theta = data['two_theta']
        intensity = data['intensity']
        print(f"[npz] 成功读取 {Path(filepath).name}: "
              f"{len(two_theta)}个数据点, 2θ=[{two_theta[0]:.2f}, {two_theta[-1]:.2f}]")
        return two_theta, intensity

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    errors = []

    # 方法1: fabio
    try:
        two_theta, intensity = parse_with_fabio(filepath)
        print(f"[fabio] 成功解析 {Path(filepath).name}: "
              f"{len(two_theta)}个数据点, 2θ=[{two_theta[0]:.2f}, {two_theta[-1]:.2f}]")
        return two_theta, intensity
    except Exception as e:
        errors.append(f"fabio: {e}")

    # 方法2: xylib
    try:
        two_theta, intensity = parse_with_xylib(filepath)
        print(f"[xylib] 成功解析 {Path(filepath).name}: "
              f"{len(two_theta)}个数据点, 2θ=[{two_theta[0]:.2f}, {two_theta[-1]:.2f}]")
        return two_theta, intensity
    except Exception as e:
        errors.append(f"xylib: {e}")

    # 方法3: 手动解析
    try:
        two_theta, intensity = parse_raw_manual(filepath)
        print(f"[manual] 成功解析 {Path(filepath).name}: "
              f"{len(two_theta)}个数据点, 2θ=[{two_theta[0]:.2f}, {two_theta[-1]:.2f}]")
        return two_theta, intensity
    except Exception as e:
        errors.append(f"manual: {e}")

    raise RuntimeError(
        f"所有解析方法均失败 ({Path(filepath).name}):\n" +
        "\n".join(f"  - {err}" for err in errors)
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python parse_raw.py <file.raw>")
        sys.exit(1)

    two_theta, intensity = parse_raw_file(sys.argv[1])
    print(f"数据点数: {len(two_theta)}")
    print(f"2θ范围: {two_theta[0]:.4f} - {two_theta[-1]:.4f}")
    print(f"强度范围: {intensity.min():.1f} - {intensity.max():.1f}")
