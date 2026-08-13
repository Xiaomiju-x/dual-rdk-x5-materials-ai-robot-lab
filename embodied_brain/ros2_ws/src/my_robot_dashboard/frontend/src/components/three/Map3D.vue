<script setup lang="ts">
// Map3D (G3) — SLAM 占据栅格挤出 3D 墙体 + 车实时位姿 + 语义地标柱.
// 数据源与 2D LiveMap 完全同一张 /api/map.json 栅格 (真图或标注 mock 的演示图),
// 行向 RLE 合并成长条 Box 控制 draw call (86×174 真图 ~几百个 box, 无压力)。
import * as THREE from 'three'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface MapMeta { w: number; h: number; res: number; ox: number; oy: number }
interface Landmark { name: string; x: number; y: number }

const props = defineProps<{
  grid: Uint8Array | null
  meta: MapMeta | null
  pose: { x: number; y: number; yaw: number } | null
  landmarks: Landmark[]
}>()

const host = ref<HTMLDivElement | null>(null)
let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene, cam: THREE.PerspectiveCamera
let wallsG: THREE.Group | null = null
let lmG: THREE.Group | null = null
let carG: THREE.Group | null = null
let raf = 0
// 球坐标轨道相机
const orbit = { r: 7, th: 0.7, ph: 0.95, cx: 0, cz: 0 }
const cur = { r: 9, th: 0.7, ph: 0.95, cx: 0, cz: 0 }
let drag = false, lx = 0, ly = 0, idle = 0

function world2scene(x: number, y: number): [number, number] {
  // ROS 世界 (x 前 y 左, m) → three (x 右, z 前): sx=x, sz=-y
  return [x, -y]
}

function rebuildWalls() {
  if (!scene) return
  if (wallsG) {
    scene.remove(wallsG)
    wallsG.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
        object.geometry.dispose()
      }
    })
  }
  wallsG = new THREE.Group()
  const g = props.grid, m = props.meta
  if (!g || !m) { scene.add(wallsG); return }
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.2, roughness: 0.7 })
  const H = 0.45
  for (let row = 0; row < m.h; row++) {
    let runStart = -1
    for (let col = 0; col <= m.w; col++) {
      const occ = col < m.w && g[row * m.w + col] >= 65
      if (occ && runStart < 0) runStart = col
      else if (!occ && runStart >= 0) {
        const len = (col - runStart) * m.res
        const wx = m.ox + (runStart + (col - runStart) / 2) * m.res
        const wy = m.oy + (row + 0.5) * m.res
        const [sx, sz] = world2scene(wx, wy)
        const box = new THREE.Mesh(new THREE.BoxGeometry(len, H, m.res), wallMat)
        box.position.set(sx, H / 2, sz)
        box.castShadow = true
        wallsG.add(box)
        runStart = -1
      }
    }
  }
  // 地板按地图范围铺
  const mw = m.w * m.res, mh = m.h * m.res
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(mw, mh),
    new THREE.MeshStandardMaterial({ color: 0xf1f5fb, roughness: 0.95 }),
  )
  floor.rotation.x = -Math.PI / 2
  const [fcx, fcz] = world2scene(m.ox + mw / 2, m.oy + mh / 2)
  floor.position.set(fcx, -0.005, fcz)
  floor.receiveShadow = true
  wallsG.add(floor)
  const grid3 = new THREE.GridHelper(Math.max(mw, mh) + 2, Math.round(Math.max(mw, mh)) * 2, 0xc7d2fe, 0xe5eaf6)
  grid3.position.set(fcx, 0.002, fcz)
  wallsG.add(grid3)
  scene.add(wallsG)
  orbit.cx = fcx; orbit.cz = fcz
  orbit.r = Math.max(mw, mh) * 1.05 + 1.5
}

function rebuildLandmarks() {
  if (!scene) return
  if (lmG) scene.remove(lmG)
  lmG = new THREE.Group()
  const postMat = new THREE.MeshStandardMaterial({ color: 0xd97706, metalness: 0.3, roughness: 0.5 })
  const topMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xf59e0b, emissiveIntensity: 0.9 })
  for (const lm of props.landmarks) {
    const [sx, sz] = world2scene(lm.x, lm.y)
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.6, 8), postMat)
    post.position.set(sx, 0.3, sz)
    lmG.add(post)
    const top = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 10), topMat)
    top.position.set(sx, 0.64, sz)
    lmG.add(top)
  }
  scene.add(lmG)
}

function buildCar() {
  carG = new THREE.Group()
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.30, 0.10, 0.24),
    new THREE.MeshStandardMaterial({ color: 0x2563eb, metalness: 0.35, roughness: 0.45 }),
  )
  body.position.y = 0.07; body.castShadow = true
  carG.add(body)
  const lidar = new THREE.Mesh(
    new THREE.CylinderGeometry(0.035, 0.035, 0.03, 14),
    new THREE.MeshStandardMaterial({ color: 0x334155 }),
  )
  lidar.position.y = 0.15
  carG.add(lidar)
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.05, 0.12, 10),
    new THREE.MeshStandardMaterial({ color: 0x22d3ee, emissive: 0x06b6d4, emissiveIntensity: 0.6 }),
  )
  nose.rotation.z = -Math.PI / 2
  nose.position.set(0.2, 0.07, 0)
  carG.add(nose)
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.26, 28),
    new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.35, side: THREE.DoubleSide }),
  )
  halo.rotation.x = -Math.PI / 2
  halo.position.y = 0.012
  carG.add(halo)
  scene.add(carG)
}

function loop() {
  raf = requestAnimationFrame(loop)
  if (!renderer) return
  idle += 0.016
  if (props.pose && carG) {
    const [sx, sz] = world2scene(props.pose.x, props.pose.y)
    carG.position.x += (sx - carG.position.x) * 0.12
    carG.position.z += (sz - carG.position.z) * 0.12
    carG.rotation.y = props.pose.yaw
  }
  if (!drag && idle > 5) orbit.th += 0.0016
  for (const k of ['r', 'th', 'ph', 'cx', 'cz'] as const) cur[k] += (orbit[k] - cur[k]) * 0.07
  cam.position.set(
    cur.cx + cur.r * Math.sin(cur.ph) * Math.sin(cur.th),
    cur.r * Math.cos(cur.ph),
    cur.cz + cur.r * Math.sin(cur.ph) * Math.cos(cur.th),
  )
  cam.lookAt(cur.cx, 0.1, cur.cz)
  renderer.render(scene, cam)
}

onMounted(() => {
  const el = host.value!
  const W = el.clientWidth || 800, H = el.clientHeight || 460
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.setSize(W, H)
  renderer.shadowMap.enabled = true
  el.appendChild(renderer.domElement)
  scene = new THREE.Scene()
  cam = new THREE.PerspectiveCamera(46, W / H, 0.05, 80)
  scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const key = new THREE.DirectionalLight(0xffffff, 0.85)
  key.position.set(4, 8, 5); key.castShadow = true
  scene.add(key)
  const p1 = new THREE.PointLight(0x7c3aed, 0.3, 20); p1.position.set(-4, 4, 2); scene.add(p1)
  buildCar()
  rebuildWalls()
  rebuildLandmarks()
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
  orbit.ph = Math.max(0.15, Math.min(1.35, orbit.ph + (e.clientY - ly) * 0.005))
  lx = e.clientX; ly = e.clientY; idle = 0
}
function onWheel(e: WheelEvent) {
  e.preventDefault()
  orbit.r = Math.max(1.5, Math.min(30, orbit.r * (e.deltaY > 0 ? 1.08 : 0.92)))
  idle = 0
}

watch(() => props.grid, () => { rebuildWalls() })
watch(() => props.landmarks, () => { rebuildLandmarks() }, { deep: true })

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  window.removeEventListener('pointerup', onUp)
  window.removeEventListener('pointermove', onMove)
  renderer?.dispose()
  renderer = null
})
</script>

<template>
  <div ref="host" class="map3d-host" />
</template>

<style scoped>
.map3d-host { width: 100%; height: 100%; min-height: 420px; cursor: grab; }
.map3d-host:active { cursor: grabbing; }
</style>
