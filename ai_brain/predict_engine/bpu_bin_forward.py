"""bpu_bin_forward.py — 单 bin 加载+forward+保存, 一次性脚本.

用法:
    python3 -m predict_engine.bpu_bin_forward <bin_path> <input_npy> <output_npy>

主 worker 对 lazy_bins slot 的 10 段链式, 每段 spawn 此脚本一次 (进程退出释放 CMA).
"""
from __future__ import annotations
import sys
import numpy as np


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: bpu_bin_forward <bin> <in.npy> <out.npy>")
    bin_path, in_npy, out_npy = sys.argv[1], sys.argv[2], sys.argv[3]
    hidden = np.load(in_npy)
    from hobot_dnn import pyeasy_dnn as dnn
    m = dnn.load(bin_path)[0]
    out = m.forward(hidden)
    np.save(out_npy, out[0].buffer.copy())


if __name__ == "__main__":
    main()
