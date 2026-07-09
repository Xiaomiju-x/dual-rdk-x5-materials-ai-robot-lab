from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parent

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "navy": "#1F3B57",
    "blue": "#2E6FB0",
    "teal": "#1A9E9E",
    "green": "#2E9E5B",
    "amber": "#E0A020",
    "orange": "#E8702A",
    "red": "#C44E52",
    "bg": "#F4F6F9",
    "line": "#6F7F91",
}


def box(ax, xy, wh, text, color, fs=10, fc="white"):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.3,
        edgecolor=color,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#17202A", wrap=True)
    return p


def arrow(ax, a, b, color=None):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color or COLORS["line"],
            connectionstyle="arc3,rad=0.0",
        )
    )


def setup(figsize=(10.8, 6.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_architecture():
    fig, ax = setup()
    box(ax, (0.05, 0.67), (0.26, 0.21), "AI 脑 RDK X5\n材料预测 / 4K 语义\n本地模型与知识库", COLORS["blue"], 10.5, "#EEF5FF")
    box(ax, (0.37, 0.67), (0.26, 0.21), "具身脑 RDK X5\nLiDAR / 深度 / SLAM\nLab-FSD Shadow", COLORS["teal"], 10.5, "#ECFBFA")
    box(ax, (0.69, 0.67), (0.26, 0.21), "固定工位\n双 myCobot / 末端相机\n袋爪 / 瓶爪 / 快拆爪", COLORS["amber"], 10.5, "#FFF7E7")
    box(ax, (0.17, 0.36), (0.26, 0.18), "STM32F407 下位机\n电机 / 电推杆 / 电磁铁\n限位与安全时序", COLORS["orange"], 10, "#FFF1EA")
    box(ax, (0.57, 0.36), (0.26, 0.18), "公网展示与证据平台\n状态孪生 / API / 发布记录\n答辩防御与资产页", COLORS["green"], 10, "#EEF9F1")
    box(ax, (0.24, 0.09), (0.52, 0.15), "材料实验闭环\n配方设计 -> AI 预筛 -> 取放执行 -> 烧制/表征 -> 失败回流", COLORS["navy"], 10.5, "#F4F6F9")
    arrow(ax, (0.31, 0.72), (0.37, 0.72))
    arrow(ax, (0.63, 0.72), (0.69, 0.72))
    arrow(ax, (0.50, 0.67), (0.31, 0.54))
    arrow(ax, (0.82, 0.67), (0.73, 0.54))
    arrow(ax, (0.30, 0.36), (0.40, 0.24))
    arrow(ax, (0.70, 0.36), (0.58, 0.24))
    ax.text(0.5, 0.96, "双 RDK X5 异构协同总体架构", ha="center", va="center", fontsize=15, weight="bold", color=COLORS["navy"])
    save(fig, "fig_architecture_xrd")


def fig_workflow():
    fig, ax = setup((10.8, 4.8))
    labels = [
        ("研究员输入\nhost / 掺杂 / 浓度", COLORS["blue"]),
        ("AI 脑预测\n规则 + 类比 + 光谱模型", COLORS["teal"]),
        ("候选排序\nGO / REVISE / DROP", COLORS["green"]),
        ("具身执行\n移动感知 + 工位取放", COLORS["amber"]),
        ("实验回流\nXRD / PL / 失败库", COLORS["orange"]),
    ]
    xs = [0.04, 0.24, 0.44, 0.64, 0.84]
    for (txt, c), x in zip(labels, xs):
        box(ax, (x, 0.47), (0.13, 0.24), txt, c, 9.5, "#FFFFFF")
    for i in range(4):
        arrow(ax, (xs[i] + 0.13, 0.59), (xs[i + 1], 0.59))
    arrow(ax, (0.91, 0.47), (0.10, 0.35), COLORS["orange"])
    arrow(ax, (0.10, 0.35), (0.10, 0.47), COLORS["orange"])
    box(ax, (0.25, 0.12), (0.50, 0.16), "数据闭环: append-only 记录 + 哈希链 + 实测回填 + 周期复盘", COLORS["navy"], 10, "#F4F6F9")
    ax.text(0.5, 0.89, "材料合成实验闭环流程", ha="center", fontsize=15, weight="bold", color=COLORS["navy"])
    save(fig, "fig_workflow_xrd")


def fig_hardware():
    fig, ax = setup()
    items = [
        ("AI 脑", "RDK X5 8G\nIMX415 4K\nM260C 音频", 0.06, 0.63, COLORS["blue"]),
        ("具身脑", "RDK X5 8G\nLD14 LiDAR\nAstra 深度 / USB 相机", 0.38, 0.63, COLORS["teal"]),
        ("执行层", "STM32F407\n步进电机 / 电推杆\n电磁铁 / 夹爪", 0.70, 0.63, COLORS["orange"]),
        ("固定工位", "双 myCobot 280-Pi\n末端相机\nAprilTag 视觉门控", 0.21, 0.25, COLORS["amber"]),
        ("展示平台", "公网展示站\nOpenAPI / 证据链\n系统状态孪生", 0.55, 0.25, COLORS["green"]),
    ]
    for title, desc, x, y, c in items:
        box(ax, (x, y), (0.24, 0.20), f"{title}\n{desc}", c, 10, "#FFFFFF")
    arrow(ax, (0.30, 0.68), (0.38, 0.68))
    arrow(ax, (0.62, 0.68), (0.70, 0.68))
    arrow(ax, (0.82, 0.63), (0.67, 0.45))
    arrow(ax, (0.50, 0.63), (0.34, 0.45))
    arrow(ax, (0.30, 0.45), (0.55, 0.35), COLORS["green"])
    ax.text(0.5, 0.95, "原型硬件组成与协同关系", ha="center", fontsize=15, weight="bold", color=COLORS["navy"])
    save(fig, "fig_hardware_xrd")


def fig_metrics():
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    names = ["BPU 小模型\n单次推理", "Qwen2 0.5B\nBPU forward", "Lab-FSD BPU\n风险模型", "公网站\nOpenAPI paths", "全站验收\nGates"]
    values = [2, 553, 1.7, 89, 5]
    labels = ["<2 ms", "553 ms", "~1.7 ms", "89", "5/5"]
    colors = [COLORS["green"], COLORS["blue"], COLORS["teal"], COLORS["amber"], COLORS["orange"]]
    ax.bar(range(len(values)), values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("数值（对数轴，单位随指标标注）")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.22)
    for i, (v, lab) in enumerate(zip(values, labels)):
        ax.text(i, v * 1.22, lab, ha="center", va="bottom", fontsize=10.5, weight="bold")
    ax.set_title("关键工程指标快照", fontsize=15, weight="bold", color=COLORS["navy"])
    save(fig, "fig_metrics_xrd")


def main():
    fig_architecture()
    fig_workflow()
    fig_hardware()
    fig_metrics()


if __name__ == "__main__":
    main()
