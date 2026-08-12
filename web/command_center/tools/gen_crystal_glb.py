#!/usr/bin/env python3
"""H17 AR — 程序化生成 NIR 荧光粉「示意晶胞」GLB (纯标准库, 无依赖)。

诚实声明: 这是用于 AR/3D 展示的**示意结构** (schematic), 非任何具体 CIF 的精确坐标。
motif = Cr3+ 居中 + 6 配位 O 八面体 (NIR 发光中心) + 外围 Y/Al 格点 + 晶胞框线, 颜色按元素惯例。
单网格 + 顶点色 (COLOR_0), model-viewer 直接渲染。输出 static/models/phosphor_cell.glb。
"""
import json
import math
import os
import struct

OUT = os.path.join(os.path.dirname(__file__), "..", "static", "models", "phosphor_cell.glb")

# 元素惯例色 (近似 CPK / VESTA)
COL = {
    "Cr": (0.55, 0.20, 0.85),   # 紫 — 发光中心 Cr3+
    "O":  (0.90, 0.25, 0.25),   # 红 — 氧
    "Y":  (0.35, 0.85, 0.80),   # 青 — 钇
    "Al": (0.70, 0.72, 0.78),   # 灰 — 铝
    "bond": (0.62, 0.66, 0.74),
}
RAD = {"Cr": 0.34, "O": 0.30, "Y": 0.40, "Al": 0.30}

# 八面体 6 个 O 方向 (单位)
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _atoms():
    """返回 [(elem, x,y,z)] —— 示意 motif。"""
    a = [("Cr", 0.0, 0.0, 0.0)]
    d = 1.15
    for (x, y, z) in _OCT:
        a.append(("O", x * d, y * d, z * d))
    # 外围 Y / Al 立方角 (示意格点)
    c = 1.9
    corners = [(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]
    for i, (x, y, z) in enumerate(corners):
        a.append(("Y" if i % 2 == 0 else "Al", x * c, y * c, z * c))
    return a


def _sphere(cx, cy, cz, r, color, stacks=14, slices=14):
    """UV 球 → (verts[(x,y,z, nx,ny,nz, r,g,b)], tris[(i,j,k)])."""
    V, T = [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            th = 2 * math.pi * j / slices
            nx = math.sin(phi) * math.cos(th)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(th)
            V.append((cx + r * nx, cy + r * ny, cz + r * nz, nx, ny, nz,
                      color[0], color[1], color[2]))
    w = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * w + j
            b = a + w
            T.append((a, b, a + 1))
            T.append((a + 1, b, b + 1))
    return V, T


def _cylinder(p0, p1, r, color, seg=10):
    """两点间圆柱 (键) → (verts, tris)。"""
    ax = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    L = math.sqrt(sum(c * c for c in ax)) or 1e-6
    az = (ax[0] / L, ax[1] / L, ax[2] / L)
    # 任取垂直基
    up = (0, 1, 0) if abs(az[1]) < 0.9 else (1, 0, 0)
    ax1 = (az[1] * up[2] - az[2] * up[1], az[2] * up[0] - az[0] * up[2], az[0] * up[1] - az[1] * up[0])
    n1 = math.sqrt(sum(c * c for c in ax1)) or 1e-6
    ax1 = (ax1[0] / n1, ax1[1] / n1, ax1[2] / n1)
    ax2 = (az[1] * ax1[2] - az[2] * ax1[1], az[2] * ax1[0] - az[0] * ax1[2], az[0] * ax1[1] - az[1] * ax1[0])
    V, T = [], []
    for k in range(seg + 1):
        t = 2 * math.pi * k / seg
        nx = ax1[0] * math.cos(t) + ax2[0] * math.sin(t)
        ny = ax1[1] * math.cos(t) + ax2[1] * math.sin(t)
        nz = ax1[2] * math.cos(t) + ax2[2] * math.sin(t)
        V.append((p0[0] + r * nx, p0[1] + r * ny, p0[2] + r * nz, nx, ny, nz, *color))
        V.append((p1[0] + r * nx, p1[1] + r * ny, p1[2] + r * nz, nx, ny, nz, *color))
    for k in range(seg):
        a = 2 * k
        T.append((a, a + 1, a + 2))
        T.append((a + 1, a + 3, a + 2))
    return V, T


def build():
    verts, tris = [], []

    def add(V, T):
        base = len(verts)
        verts.extend(V)
        tris.extend((i + base, j + base, k + base) for (i, j, k) in T)

    atoms = _atoms()
    cr = atoms[0]
    for (el, x, y, z) in atoms:
        add(*_sphere(x, y, z, RAD[el], COL[el]))
    # Cr–O 键
    for (el, x, y, z) in atoms[1:7]:
        add(*_cylinder((cr[1], cr[2], cr[3]), (x, y, z), 0.06, COL["bond"]))

    # —— 打包 glTF buffers ——
    pos = bytearray()
    nrm = bytearray()
    col = bytearray()
    for v in verts:
        pos += struct.pack("<fff", v[0], v[1], v[2])
        nrm += struct.pack("<fff", v[3], v[4], v[5])
        col += struct.pack("<ffff", v[6], v[7], v[8], 1.0)
    idx = bytearray()
    use32 = len(verts) > 65535
    for t in tris:
        idx += struct.pack("<III" if use32 else "<HHH", *t)

    def pad4(b):
        while len(b) % 4:
            b += b"\x00"
        return b

    pos = pad4(pos); nrm = pad4(nrm); col = pad4(col); idx = pad4(idx)
    blob = pos + nrm + col + idx
    o_pos, o_nrm, o_col, o_idx = 0, len(pos), len(pos) + len(nrm), len(pos) + len(nrm) + len(col)

    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]

    gltf = {
        "asset": {"version": "2.0", "generator": "xrd gen_crystal_glb"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "phosphor_cell"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"name": "atoms", "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.1, "roughnessFactor": 0.55},
            "emissiveFactor": [0.06, 0.02, 0.10]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": o_pos, "byteLength": len(pos), "target": 34962},
            {"buffer": 0, "byteOffset": o_nrm, "byteLength": len(nrm), "target": 34962},
            {"buffer": 0, "byteOffset": o_col, "byteLength": len(col), "target": 34962},
            {"buffer": 0, "byteOffset": o_idx, "byteLength": len(idx), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3",
             "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 1, "componentType": 5126, "count": len(verts), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(verts), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125 if use32 else 5123, "count": len(tris) * 3, "type": "SCALAR"},
        ],
    }

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b"\x20"
    bin_chunk = blob
    while len(bin_chunk) % 4:
        bin_chunk += b"\x00"
    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)         # glTF magic, version, length
    out += struct.pack("<II", len(js), 0x4E4F534A)            # JSON chunk
    out += js
    out += struct.pack("<II", len(bin_chunk), 0x004E4942)     # BIN chunk
    out += bin_chunk

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(out)
    print(f"wrote {OUT}: {len(verts)} verts, {len(tris)} tris, {len(out)} bytes")


if __name__ == "__main__":
    build()
