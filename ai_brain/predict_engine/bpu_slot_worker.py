"""bpu_slot_worker.py — 单进程单次调用 worker, 主进程 subprocess.run 调.

解决: pyeasy_dnn 不释放 CMA, swap-load 会 CMA 碎片化 / 不够用.
进程退出后内核必回收 CMA. 每次 verdict 开一个新进程, 装 bin → forward → 输出 JSON → 退出.

用法:
    python3 -m predict_engine.bpu_slot_worker <slot_name> <prompt>
    stdout: 单行 JSON (结果)
"""
from __future__ import annotations
import json
import sys
from .bpu_slot_manager import bpu_slot_verdict


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: worker <slot> <prompt>"}))
        sys.exit(1)
    slot = sys.argv[1]
    prompt = sys.argv[2]
    try:
        res = bpu_slot_verdict(prompt, slot=slot)
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "slot": slot}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
