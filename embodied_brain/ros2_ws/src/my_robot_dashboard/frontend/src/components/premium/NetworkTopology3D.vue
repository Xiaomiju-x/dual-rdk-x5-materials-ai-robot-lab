<script setup lang="ts">
/**
 * NetworkTopology3D — animated 3D node graph showing the project's
 * cross-system topology: AI Brain X5, Embodied Brain X5, dual arms,
 * dual furnaces, cloud LLM. Edges carry flowing data particles.
 *
 * Cinematic but cheap: ~150 verts + 6 nodes + ~120 instanced particle sprites.
 * Bloom + emissive baked in. Click a node to focus (camera dolly to it).
 *
 *   <NetworkTopology3D :height="420" />
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'

interface Props {
  height?: number
  cinematic?: boolean
}
const props = withDefaults(defineProps<Props>(), { height: 420, cinematic: true })

const telemetry = useTelemetryStore()
const settings = useSettingsStore()

interface NodeDef {
  id: string
  label: string
  sub: string
  pos: [number, number, number]
  color: number
  online: () => boolean
  hint: () => string
}

interface EdgeDef {
  a: string
  b: string
  /** packets per second flowing on this edge (drives particle density) */
  rate: () => number
  /** primary flow direction color */
  color: number
}

const NODES: NodeDef[] = [
  { id: 'cloud', label: 'DeepSeek R1', sub: 'cloud reasoner',     pos: [ 0,   2.2, -3.0], color: 0xa78bfa, online: () => true, hint: () => '15-30s deep reasoning' },
  { id: 'ai',    label: 'AI Brain X5', sub: 'lab · :8888',         pos: [-3.0, 1.2,  0.0], color: 0x2563eb, online: () => true, hint: () => '9 LLM · 5 BPU slot · 53 routes' },
  { id: 'eb',    label: 'Embodied X5', sub: 'cart · :8890',        pos: [ 3.0, 1.2,  0.0], color: 0x06b6d4, online: () => telemetry.isConnected, hint: () => `${telemetry.observedHz.toFixed(1)} Hz · seq ${telemetry.packet?.heartbeat.sequence ?? 0}` },
  { id: 'arm1',  label: 'Arm 01',      sub: 'mycobot · .64',       pos: [-1.5, 0.0,  2.6], color: 0x059669, online: () => true, hint: () => 'Pi 4B · 6 joints' },
  { id: 'arm2',  label: 'Arm 02',      sub: 'mycobot · .136',      pos: [ 1.5, 0.0,  2.6], color: 0x10b981, online: () => true, hint: () => 'Pi 4B · 6 joints' },
  { id: 'f1',    label: 'Furnace 1',   sub: 'PV/SV/MV',            pos: [ 2.6, 0.0, -2.4], color: 0xd97706, online: () => (telemetry.packet?.furnaces?.[0] != null), hint: () => `PV ${(telemetry.packet?.furnaces?.[0]?.pv ?? 0).toFixed(0)}°C` },
  { id: 'f2',    label: 'Furnace 2',   sub: 'PV/SV/MV',            pos: [-2.6, 0.0, -2.4], color: 0xe11d48, online: () => (telemetry.packet?.furnaces?.[1] != null), hint: () => `PV ${(telemetry.packet?.furnaces?.[1]?.pv ?? 0).toFixed(0)}°C` },
]

const EDGES: EdgeDef[] = [
  { a: 'cloud', b: 'ai',   rate: () => 0.6, color: 0x7c3aed },
  { a: 'ai',    b: 'eb',   rate: () => 2.0, color: 0x06b6d4 },
  { a: 'ai',    b: 'arm1', rate: () => 0.8, color: 0x059669 },
  { a: 'ai',    b: 'arm2', rate: () => 0.8, color: 0x10b981 },
  { a: 'eb',    b: 'f1',   rate: () => 1.2, color: 0xd97706 },
  { a: 'eb',    b: 'f2',   rate: () => 1.0, color: 0xe11d48 },
  { a: 'eb',    b: 'arm1', rate: () => 0.4, color: 0x2563eb },
  { a: 'eb',    b: 'arm2', rate: () => 0.4, color: 0x2563eb },
]

const canvasEl = ref<HTMLCanvasElement | null>(null)
const wrapEl = ref<HTMLDivElement | null>(null)
const hovered = ref<string | null>(null)
const focused = ref<string | null>(null)

const hoveredNode = computed(() => hovered.value ? NODES.find((n) => n.id === hovered.value) : null)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let composer: EffectComposer | null = null
let bloom: UnrealBloomPass | null = null
let raf = 0
let resizeObs: ResizeObserver | null = null

const nodeMeshes = new Map<string, THREE.Mesh>()
const nodeHaloMeshes = new Map<string, THREE.Mesh>()
const edgeLines: THREE.Line[] = []

interface Particle {
  edge: EdgeDef
  t: number             // 0..1 along edge
  speed: number
  sprite: THREE.Mesh
}
const particles: Particle[] = []
const edgeBudget = new Map<EdgeDef, number>()

let raycaster: THREE.Raycaster | null = null
const pointer = new THREE.Vector2(-10, -10)

function init() {
  if (!canvasEl.value || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl.value, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h, false)
  renderer.setClearColor(0x000000, 0)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = settings.theme === 'dark' ? 1.05 : 1.2
  renderer.outputColorSpace = THREE.SRGBColorSpace

  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(46, w / h, 0.1, 100)
  camera.position.set(0, 4, 7.5)
  camera.lookAt(0, 0.5, 0)

  // soft ambient + key
  scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const key = new THREE.DirectionalLight(0xffffff, 0.75)
  key.position.set(3, 5, 4)
  scene.add(key)

  // floor ring grid for context
  const gridGroup = new THREE.Group()
  for (let i = 1; i <= 5; i++) {
    const r = 0.8 * i
    const ringGeom = new THREE.BufferGeometry()
    const verts: number[] = []
    const segs = 64
    for (let j = 0; j < segs; j++) {
      const a0 = (j / segs) * Math.PI * 2
      const a1 = ((j + 1) / segs) * Math.PI * 2
      verts.push(Math.cos(a0) * r, -0.6, Math.sin(a0) * r)
      verts.push(Math.cos(a1) * r, -0.6, Math.sin(a1) * r)
    }
    ringGeom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
    const ringMat = new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.10 - i * 0.012 })
    gridGroup.add(new THREE.LineSegments(ringGeom, ringMat))
  }
  scene.add(gridGroup)

  // build nodes
  for (const n of NODES) {
    const geom = new THREE.SphereGeometry(0.24, 32, 24)
    const mat = new THREE.MeshPhysicalMaterial({
      color: n.color,
      emissive: n.color,
      emissiveIntensity: 0.85,
      metalness: 0.5,
      roughness: 0.25,
      clearcoat: 0.6,
    })
    const mesh = new THREE.Mesh(geom, mat)
    mesh.position.set(...n.pos)
    mesh.userData.id = n.id
    scene.add(mesh)
    nodeMeshes.set(n.id, mesh)

    // halo ring
    const haloGeom = new THREE.RingGeometry(0.34, 0.40, 48)
    const haloMat = new THREE.MeshBasicMaterial({
      color: n.color, transparent: true, opacity: 0.5, side: THREE.DoubleSide,
    })
    const halo = new THREE.Mesh(haloGeom, haloMat)
    halo.position.copy(mesh.position)
    halo.rotation.x = -Math.PI / 2
    scene.add(halo)
    nodeHaloMeshes.set(n.id, halo)
  }

  // build edges
  for (const e of EDGES) {
    const aN = NODES.find((n) => n.id === e.a)!
    const bN = NODES.find((n) => n.id === e.b)!
    // curved bezier — bend slightly upward for visual interest
    const curve = makeCurve(aN.pos, bN.pos, 0.6)
    const pts = curve.getPoints(50)
    const geom = new THREE.BufferGeometry().setFromPoints(pts)
    const mat = new THREE.LineBasicMaterial({
      color: e.color, transparent: true, opacity: 0.45,
      blending: THREE.AdditiveBlending,
    })
    const line = new THREE.Line(geom, mat)
    line.userData.curve = curve
    line.userData.edge = e
    scene.add(line)
    edgeLines.push(line)
    edgeBudget.set(e, 0)
  }

  // post-fx — bloom adds the magic
  if (props.cinematic) {
    composer = new EffectComposer(renderer)
    composer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    composer.setSize(w, h)
    composer.addPass(new RenderPass(scene, camera))
    bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.6, 0.78)
    composer.addPass(bloom)
    composer.addPass(new OutputPass())
  }

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.1
  controls.minDistance = 4
  controls.maxDistance = 14
  controls.maxPolarAngle = Math.PI / 2 - 0.05
  controls.target.set(0, 0.5, 0)

  raycaster = new THREE.Raycaster()

  resizeObs = new ResizeObserver(() => onResize())
  resizeObs.observe(wrapEl.value)
  renderer.domElement.addEventListener('pointermove', onPointerMove)
  renderer.domElement.addEventListener('click', onClick)

  loop()
}

function makeCurve(a: [number, number, number], b: [number, number, number], lift: number): THREE.QuadraticBezierCurve3 {
  const va = new THREE.Vector3(...a)
  const vb = new THREE.Vector3(...b)
  const mid = va.clone().add(vb).multiplyScalar(0.5)
  mid.y += lift + Math.hypot(va.x - vb.x, va.z - vb.z) * 0.05
  return new THREE.QuadraticBezierCurve3(va, mid, vb)
}

function onResize() {
  if (!renderer || !camera || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight
  if (w === 0 || h === 0) return
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  composer?.setSize(w, h)
  bloom?.setSize(w, h)
}

function onPointerMove(e: PointerEvent) {
  if (!canvasEl.value) return
  const r = canvasEl.value.getBoundingClientRect()
  pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1
  pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1
}
function onClick() {
  if (hovered.value) {
    focused.value = focused.value === hovered.value ? null : hovered.value
  }
}

function spawnParticle(edge: EdgeDef) {
  if (!scene) return
  const geom = new THREE.SphereGeometry(0.05, 8, 6)
  const mat = new THREE.MeshBasicMaterial({
    color: edge.color, transparent: true, opacity: 0.95,
  })
  const sprite = new THREE.Mesh(geom, mat)
  scene.add(sprite)
  particles.push({ edge, t: 0, speed: 0.45 + Math.random() * 0.35, sprite })
}

function tickParticles(dt: number) {
  // emit
  for (const e of EDGES) {
    const budget = (edgeBudget.get(e) ?? 0) + e.rate() * dt
    let spawn = Math.floor(budget)
    edgeBudget.set(e, budget - spawn)
    while (spawn-- > 0) spawnParticle(e)
  }
  // advance
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i]
    p.t += p.speed * dt
    if (p.t >= 1) {
      scene?.remove(p.sprite)
      p.sprite.geometry.dispose()
      ;(p.sprite.material as THREE.Material).dispose()
      particles.splice(i, 1)
      continue
    }
    const line = edgeLines.find((l) => l.userData.edge === p.edge)
    if (!line) continue
    const curve = line.userData.curve as THREE.Curve<THREE.Vector3>
    const pos = curve.getPointAt(p.t)
    p.sprite.position.copy(pos)
    ;(p.sprite.material as THREE.MeshBasicMaterial).opacity = 0.4 + 0.6 * Math.sin(p.t * Math.PI)
  }
}

function loop() {
  raf = requestAnimationFrame(loop)
  const dt = 1 / 60

  // node pulse + halo throb
  const t = performance.now() / 1000
  for (const n of NODES) {
    const mesh = nodeMeshes.get(n.id)!
    const halo = nodeHaloMeshes.get(n.id)!
    const online = n.online()
    const isHover = hovered.value === n.id
    const isFocus = focused.value === n.id
    const baseScale = isFocus ? 1.30 : isHover ? 1.18 : 1.0
    const breathe = 1 + 0.06 * Math.sin(t * 1.5 + n.id.length)
    mesh.scale.setScalar(baseScale * breathe)
    halo.scale.setScalar(baseScale * (1 + 0.18 * Math.sin(t * 1.2 + n.id.length)))
    ;(mesh.material as THREE.MeshPhysicalMaterial).emissiveIntensity = online ? (isHover ? 1.4 : 0.85) : 0.18
    ;(halo.material as THREE.MeshBasicMaterial).opacity = online ? 0.5 : 0.10
  }

  tickParticles(dt)

  // hover ray
  if (raycaster && camera) {
    raycaster.setFromCamera(pointer, camera)
    const intersects = raycaster.intersectObjects(Array.from(nodeMeshes.values()))
    hovered.value = intersects.length > 0 ? (intersects[0].object.userData.id as string) : null
    if (canvasEl.value) canvasEl.value.style.cursor = hovered.value ? 'pointer' : 'grab'
  }

  // focus dolly
  if (focused.value && camera && controls) {
    const n = NODES.find((x) => x.id === focused.value)!
    const target = new THREE.Vector3(n.pos[0], n.pos[1] + 0.4, n.pos[2])
    controls.target.lerp(target, 0.04)
  }

  controls?.update()
  if (composer) composer.render()
  else if (renderer && scene && camera) renderer.render(scene, camera)
}

function disposeScene() {
  cancelAnimationFrame(raf)
  resizeObs?.disconnect()
  controls?.dispose()
  composer?.dispose()
  if (renderer?.domElement) {
    renderer.domElement.removeEventListener('pointermove', onPointerMove)
    renderer.domElement.removeEventListener('click', onClick)
  }
  if (scene) {
    scene.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.geometry) m.geometry.dispose()
      const mat = m.material as THREE.Material | THREE.Material[] | undefined
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose())
      else if (mat) mat.dispose()
    })
  }
  renderer?.dispose()
  renderer = null
  scene = null
  camera = null
  controls = null
  composer = null
  bloom = null
  nodeMeshes.clear()
  nodeHaloMeshes.clear()
  edgeLines.length = 0
  particles.length = 0
  edgeBudget.clear()
}

onMounted(init)
onBeforeUnmount(disposeScene)
watch(() => settings.theme, (theme) => {
  if (renderer) renderer.toneMappingExposure = theme === 'dark' ? 1.05 : 1.2
})
</script>

<template>
  <div ref="wrapEl" class="topo-wrap" :style="{ height: `${props.height}px` } as any">
    <canvas ref="canvasEl"></canvas>
    <!-- legend top-left -->
    <div class="topo-legend">
      <span class="section-label">Network · {{ NODES.length }} nodes · {{ EDGES.length }} edges</span>
      <div class="legend-row mono">click node to focus · drag to orbit · {{ particles.length }} packets in flight</div>
    </div>
    <!-- floating card following hover -->
    <Transition name="hud">
      <div v-if="hoveredNode" class="topo-card glass">
        <div class="card-label">{{ hoveredNode.label }}</div>
        <div class="card-sub">{{ hoveredNode.sub }}</div>
        <div class="card-hint mono">{{ hoveredNode.hint() }}</div>
        <div class="card-state">
          <span class="dot" :class="hoveredNode.online() ? 'dot-ok' : 'dot-idle'"></span>
          <span>{{ hoveredNode.online() ? 'online' : 'idle' }}</span>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.topo-wrap {
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: inherit;
  background:
    radial-gradient(ellipse at 50% 100%, rgba(37, 99, 235, 0.08), transparent 60%),
    linear-gradient(180deg, var(--bg-elevated), var(--bg-base));
}
canvas { display: block; width: 100% !important; height: 100% !important; }

.topo-legend {
  position: absolute; top: 12px; left: 12px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
}
[data-theme='dark'] .topo-legend { background: rgba(20, 25, 38, 0.65); }
.legend-row { font-size: 0.66rem; color: var(--ink-muted); margin-top: 2px; }

.topo-card {
  position: absolute;
  top: 12px; right: 12px;
  width: 220px;
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 4px;
  border-radius: 14px;
  pointer-events: none;
}
.card-label { font-weight: 600; font-size: 0.92rem; color: var(--ink-primary); }
.card-sub { font-size: 0.7rem; color: var(--ink-tertiary); }
.card-hint { font-size: 0.7rem; color: var(--ink-secondary); margin-top: 4px; }
.card-state { display: flex; align-items: center; gap: 6px; font-size: 0.7rem; color: var(--ink-tertiary); margin-top: 4px; }

.hud-enter-from, .hud-leave-to { opacity: 0; transform: translateY(-6px); }
.hud-enter-active, .hud-leave-active { transition: opacity 0.18s var(--ease-out-quint), transform 0.18s var(--ease-out-quint); }
</style>
