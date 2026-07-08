<script setup lang="ts">
// SkillTrajectory3D (G4) — 技能 waypoint 的真末端轨迹 3D 预览.
// 关节角序列经真 myCobot DH 正运动学 (realFK.ts, 与 interlock.py 同一张 DH 表)
// 算出末端走过的曲线, 回放前先看一眼路径。轨迹按进度渐变着色 + 动点跑一遍。
import * as THREE from 'three'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  dense: Array<[number, number, number]>
  marks: Array<[number, number, number]>
  playing: boolean
}>()

const host = ref<HTMLDivElement | null>(null)
let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene, cam: THREE.PerspectiveCamera
let pathG: THREE.Group | null = null
let marker: THREE.Mesh | null = null
let raf = 0
let span = 0.4
let progress = 0 // 0..1 along dense path

const orbit = { r: 0.9, th: 0.9, ph: 0.95, cx: 0, cy: 0.12, cz: 0 }
const cur = { ...orbit }
let drag = false, lx = 0, ly = 0, idle = 0

function centroid(): [number, number, number] {
  if (!props.dense.length) return [0, 0.12, 0]
  let sx = 0, sy = 0, sz = 0
  for (const p of props.dense) { sx += p[0]; sy += p[1]; sz += p[2] }
  const n = props.dense.length
  return [sx / n, sy / n, sz / n]
}

function rebuild() {
  if (!scene) return
  if (pathG) { scene.remove(pathG); pathG.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.() }) }
  pathG = new THREE.Group()
  const pts = props.dense
  if (pts.length < 2) { scene.add(pathG); return }

  // 渐变折线 (进度 蓝→紫→洋红)
  const positions: number[] = []
  const colors: number[] = []
  const cA = new THREE.Color(0x2563eb), cB = new THREE.Color(0x7c3aed), cC = new THREE.Color(0xdb2777)
  const tmp = new THREE.Color()
  for (let i = 0; i < pts.length; i++) {
    positions.push(pts[i][0], pts[i][1], pts[i][2])
    const u = i / (pts.length - 1)
    if (u < 0.5) tmp.copy(cA).lerp(cB, u * 2)
    else tmp.copy(cB).lerp(cC, (u - 0.5) * 2)
    colors.push(tmp.r, tmp.g, tmp.b)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 }))
  pathG.add(line)

  // waypoint 节点球
  const wpMat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, emissive: 0x4c1d95, emissiveIntensity: 0.4 })
  for (const m of props.marks) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 10), wpMat)
    s.position.set(m[0], m[1], m[2])
    pathG.add(s)
  }
  // 起点(绿)/终点(红) 标识
  const start = new THREE.Mesh(new THREE.SphereGeometry(0.013, 14, 12),
    new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x059669, emissiveIntensity: 0.6 }))
  start.position.set(pts[0][0], pts[0][1], pts[0][2]); pathG.add(start)
  const end = new THREE.Mesh(new THREE.SphereGeometry(0.013, 14, 12),
    new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xdc2626, emissiveIntensity: 0.6 }))
  end.position.set(pts.at(-1)![0], pts.at(-1)![1], pts.at(-1)![2]); pathG.add(end)

  // base 柱 (臂基座 → 显示尺度参照)
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.025, 0.04, 16),
    new THREE.MeshStandardMaterial({ color: 0x334155 }))
  base.position.set(0, 0.02, 0); pathG.add(base)

  // 投影到地面的灰影 (XZ 平面, 帮助判读高度)
  const shadow: number[] = []
  for (const p of pts) shadow.push(p[0], 0.001, p[2])
  const sg = new THREE.BufferGeometry()
  sg.setAttribute('position', new THREE.Float32BufferAttribute(shadow, 3))
  pathG.add(new THREE.Line(sg, new THREE.LineBasicMaterial({ color: 0xcbd5e1, transparent: true, opacity: 0.7 })))

  scene.add(pathG)

  // 动点
  if (!marker) {
    marker = new THREE.Mesh(new THREE.SphereGeometry(0.016, 16, 14),
      new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0xf59e0b, emissiveIntensity: 0.9 }))
    scene.add(marker)
  }

  // 自适应取景
  const [cx, cy, cz] = centroid()
  orbit.cx = cx; orbit.cy = cy; orbit.cz = cz
  let maxd = 0.2
  for (const p of pts) maxd = Math.max(maxd, Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz))
  span = maxd
  orbit.r = maxd * 2.6 + 0.18
  progress = 0
}

function loop() {
  raf = requestAnimationFrame(loop)
  if (!renderer) return
  idle += 0.016
  if (!drag && idle > 3) orbit.th += 0.0022
  for (const k of ['r', 'th', 'ph', 'cx', 'cy', 'cz'] as const) cur[k] += (orbit[k] - cur[k]) * 0.08
  cam.position.set(
    cur.cx + cur.r * Math.sin(cur.ph) * Math.sin(cur.th),
    cur.cy + cur.r * Math.cos(cur.ph),
    cur.cz + cur.r * Math.sin(cur.ph) * Math.cos(cur.th),
  )
  cam.lookAt(cur.cx, cur.cy, cur.cz)

  // 动点沿密集路径推进
  if (marker && props.dense.length > 1) {
    if (props.playing) progress = (progress + 0.004) % 1
    const fi = progress * (props.dense.length - 1)
    const i0 = Math.floor(fi), i1 = Math.min(i0 + 1, props.dense.length - 1)
    const f = fi - i0
    const a = props.dense[i0], b = props.dense[i1]
    marker.position.set(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)
    marker.visible = true
  } else if (marker) marker.visible = false

  renderer.render(scene, cam)
}

onMounted(() => {
  const el = host.value!
  const W = el.clientWidth || 560, H = el.clientHeight || 360
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.setSize(W, H)
  el.appendChild(renderer.domElement)
  scene = new THREE.Scene()
  cam = new THREE.PerspectiveCamera(48, W / H, 0.01, 40)
  scene.add(new THREE.AmbientLight(0xffffff, 0.8))
  const key = new THREE.DirectionalLight(0xffffff, 0.7); key.position.set(2, 4, 3); scene.add(key)
  const fill = new THREE.PointLight(0x7c3aed, 0.35, 8); fill.position.set(-2, 2, -1); scene.add(fill)
  const grid = new THREE.GridHelper(1.2, 24, 0xc7d2fe, 0xe5eaf6)
  scene.add(grid)
  rebuild()
  el.addEventListener('pointerdown', (e) => { drag = true; lx = e.clientX; ly = e.clientY; idle = 0 })
  window.addEventListener('pointerup', onUp)
  window.addEventListener('pointermove', onMove)
  el.addEventListener('wheel', onWheel, { passive: false })
  loop()
})
function onUp() { drag = false }
function onMove(e: PointerEvent) {
  if (!drag) return
  orbit.th -= (e.clientX - lx) * 0.008
  orbit.ph = Math.max(0.2, Math.min(1.4, orbit.ph + (e.clientY - ly) * 0.005))
  lx = e.clientX; ly = e.clientY; idle = 0
}
function onWheel(e: WheelEvent) {
  e.preventDefault()
  orbit.r = Math.max(0.12, Math.min(6, orbit.r * (e.deltaY > 0 ? 1.08 : 0.92)))
  idle = 0
}

watch(() => props.dense, () => rebuild())

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  window.removeEventListener('pointerup', onUp)
  window.removeEventListener('pointermove', onMove)
  renderer?.dispose(); renderer = null
})
</script>

<template>
  <div ref="host" class="traj-host" />
</template>

<style scoped>
.traj-host { width: 100%; height: 100%; min-height: 320px; cursor: grab; }
.traj-host:active { cursor: grabbing; }
</style>
