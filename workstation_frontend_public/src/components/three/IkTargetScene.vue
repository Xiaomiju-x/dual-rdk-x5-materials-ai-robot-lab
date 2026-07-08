<script setup lang="ts">
// Single-arm IK target scene — operator drags a TCP ball, DLS IK solver runs,
// and the arm articulates to the closest reachable joint configuration. Emits
// the solved 6-DoF angles back to the parent so the RingSliders update in lock.
//
// Intentional simple OrbitControls + dblclick recentre; pointer drag on the
// target sphere is captured before OrbitControls (we disable orbit during
// active drag).
import { onBeforeUnmount, onMounted, ref, watch, computed } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { buildArm, type ArmBundle } from './buildArm'
import { fk, maxReach, BASE_Y } from './kinematics'
import { solveIK } from './ikSolver'
import type { ArmId } from '@/types/telemetry'

interface Props {
  arm: ArmId
  angles: number[]      // initial seed (deg, 6)
  gripper: number
}
const props = defineProps<Props>()
const emit = defineEmits<{
  (e: 'solved', angles: number[]): void
  (e: 'target', xyz: { x: number; y: number; z: number }): void
}>()

const canvasEl = ref<HTMLCanvasElement | null>(null)
const wrapEl = ref<HTMLDivElement | null>(null)
const reachable = ref(true)
const errorMm = ref(0)
const targetPos = ref<{ x: number; y: number; z: number }>({ x: 0, y: 0.32, z: 0.12 })
const dragging = ref(false)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let arm: ArmBundle | null = null
let pmrem: THREE.PMREMGenerator | null = null
let envRT: THREE.WebGLRenderTarget | null = null
let target: THREE.Mesh | null = null
let raf = 0
let resizeObs: ResizeObserver | null = null
let currentAngles: number[] = props.angles.slice()

const accent = computed(() => (props.arm === 'arm01' ? '#d97706' : '#2563eb'))

function makeTargetSphere(color: number): THREE.Mesh {
  const geo = new THREE.SphereGeometry(0.02, 24, 16)
  const mat = new THREE.MeshStandardMaterial({
    color, emissive: color, emissiveIntensity: 0.6, transparent: true, opacity: 0.85,
  })
  const m = new THREE.Mesh(geo, mat)
  m.userData.dispose = () => { geo.dispose(); mat.dispose() }
  m.renderOrder = 10
  return m
}

function init() {
  if (!canvasEl.value || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl.value, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h, false)
  renderer.setClearColor(0x000000, 0)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.82
  renderer.shadowMap.enabled = false

  scene = new THREE.Scene()
  pmrem = new THREE.PMREMGenerator(renderer)
  pmrem.compileEquirectangularShader()
  const room = new RoomEnvironment()
  envRT = pmrem.fromScene(room, 0.04)
  scene.environment = envRT.texture
  ;(scene as any).environmentIntensity = 0.32

  camera = new THREE.PerspectiveCamera(40, w / h, 0.05, 100)
  camera.position.set(0.45, 0.4, 0.55)
  camera.lookAt(0, 0.25, 0)

  // ambient + key light
  scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const key = new THREE.DirectionalLight(0xffffff, 1.0)
  key.position.set(2, 3, 2); scene.add(key)
  const fill = new THREE.DirectionalLight(0xa5b4fc, 0.3)
  fill.position.set(-2, 1, -1); scene.add(fill)

  // small floor disc
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(0.6, 48),
    new THREE.MeshPhysicalMaterial({ color: 0xf4f7fb, roughness: 0.5, metalness: 0.05, clearcoat: 0.25 }),
  )
  floor.rotation.x = -Math.PI / 2; floor.position.y = -0.001
  scene.add(floor)

  // arm
  arm = buildArm(props.arm === 'arm01' ? 'amber' : 'blue')
  scene.add(arm.group)

  // target sphere
  target = makeTargetSphere(props.arm === 'arm01' ? 0xf59e0b : 0x60a5fa)
  // seed target at current tip
  arm.setAngles(currentAngles, props.gripper)
  arm.group.updateMatrixWorld(true)
  const t0 = arm.tipWorld()
  target.position.set(t0.x, t0.y, t0.z)
  targetPos.value = { x: t0.x, y: t0.y, z: t0.z }
  scene.add(target)

  // a translucent reach sphere outline (always on in IK view to hint volume)
  const reachGeo = new THREE.SphereGeometry(maxReach(), 24, 16)
  const reachMat = new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.04, wireframe: true })
  const reachMesh = new THREE.Mesh(reachGeo, reachMat)
  reachMesh.position.y = BASE_Y; scene.add(reachMesh)

  // orbit
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true; controls.dampingFactor = 0.08
  controls.minDistance = 0.3; controls.maxDistance = 1.6
  controls.maxPolarAngle = Math.PI / 2 - 0.02
  controls.target.set(0, 0.22, 0)

  // drag setup
  canvasEl.value.addEventListener('pointerdown', onPointerDown)
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)

  resizeObs = new ResizeObserver(onResize)
  resizeObs.observe(wrapEl.value)
  loop()
}

const dragPlane = new THREE.Plane()
const dragOffset = new THREE.Vector3()
const ray = new THREE.Raycaster()
const ndc = new THREE.Vector2()

function pointerToNDC(ev: PointerEvent) {
  if (!canvasEl.value) return
  const r = canvasEl.value.getBoundingClientRect()
  ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1
  ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1
}

function onPointerDown(ev: PointerEvent) {
  if (!camera || !target || !controls) return
  pointerToNDC(ev)
  ray.setFromCamera(ndc, camera)
  const hit = ray.intersectObject(target, false)
  if (hit.length === 0) return
  // start drag: build a plane perpendicular to camera through target
  const camDir = new THREE.Vector3()
  camera.getWorldDirection(camDir)
  dragPlane.setFromNormalAndCoplanarPoint(camDir, target.position)
  dragOffset.copy(hit[0].point).sub(target.position)
  dragging.value = true
  controls.enabled = false
  ;(target.material as THREE.MeshStandardMaterial).opacity = 1
  ev.preventDefault()
}

function onPointerMove(ev: PointerEvent) {
  if (!dragging.value || !camera || !target) return
  pointerToNDC(ev)
  ray.setFromCamera(ndc, camera)
  const hit = new THREE.Vector3()
  if (!ray.ray.intersectPlane(dragPlane, hit)) return
  hit.sub(dragOffset)
  // clamp to a sensible reach box around the arm
  hit.x = Math.max(-0.42, Math.min(0.42, hit.x))
  hit.y = Math.max(BASE_Y - 0.05, Math.min(BASE_Y + 0.45, hit.y))
  hit.z = Math.max(-0.42, Math.min(0.42, hit.z))
  target.position.copy(hit)
  targetPos.value = { x: hit.x, y: hit.y, z: hit.z }
  emit('target', targetPos.value)

  // solve IK
  const result = solveIK(hit, currentAngles)
  currentAngles = result.angles
  reachable.value = result.reachable
  errorMm.value = Math.round(result.finalErrorM * 1000)
  ;(target.material as THREE.MeshStandardMaterial).color.setHex(result.reachable
    ? (props.arm === 'arm01' ? 0xf59e0b : 0x60a5fa)
    : 0xef4444)
  emit('solved', result.angles.slice())
}

function onPointerUp() {
  if (!dragging.value) return
  dragging.value = false
  if (controls) controls.enabled = true
  if (target) (target.material as THREE.MeshStandardMaterial).opacity = 0.85
}

function onResize() {
  if (!renderer || !camera || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight
  if (!w || !h) return
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}

function loop() {
  raf = requestAnimationFrame(loop)
  if (arm) arm.setAngles(currentAngles, props.gripper)
  if (controls) controls.update()
  if (renderer && scene && camera) renderer.render(scene, camera)
}

function recentre() {
  if (!arm || !target) return
  // place target at current FK tip
  arm.group.updateMatrixWorld(true)
  const t = arm.tipWorld()
  target.position.set(t.x, t.y, t.z)
  targetPos.value = { x: t.x, y: t.y, z: t.z }
  reachable.value = true; errorMm.value = 0
  ;(target.material as THREE.MeshStandardMaterial).color.setHex(props.arm === 'arm01' ? 0xf59e0b : 0x60a5fa)
}

function disposeScene() {
  cancelAnimationFrame(raf)
  canvasEl.value?.removeEventListener('pointerdown', onPointerDown)
  window.removeEventListener('pointermove', onPointerMove)
  window.removeEventListener('pointerup', onPointerUp)
  resizeObs?.disconnect()
  controls?.dispose()
  arm?.dispose()
  envRT?.dispose(); pmrem?.dispose()
  if (target) (target.userData.dispose as () => void)?.()
  if (scene) scene.traverse((obj) => {
    const m = obj as THREE.Mesh
    if (m.geometry) m.geometry.dispose?.()
    const mat = m.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose()
  })
  renderer?.dispose()
  renderer = scene = camera = null; controls = null; arm = null; target = null
  envRT = null; pmrem = null
}

onMounted(init)
onBeforeUnmount(disposeScene)

// keep arm visual in sync when parent angles change (e.g. user moves sliders)
watch(() => props.angles, (a) => {
  currentAngles = a.slice()
  if (!dragging.value) {
    // also re-park target to new tip so the next drag starts from current pose
    const tip = fk(currentAngles)
    if (target) target.position.set(tip.x, tip.y, tip.z)
    targetPos.value = { x: tip.x, y: tip.y, z: tip.z }
  }
}, { deep: true })
</script>

<template>
  <div ref="wrapEl" class="ik-scene">
    <canvas ref="canvasEl"></canvas>
    <div class="ik-hud">
      <div class="row">
        <span class="label">IK 目标</span>
        <span class="mono" :style="{ color: reachable ? accent : '#ef4444' }">
          {{ reachable ? 'reachable' : `${errorMm}mm out` }}
        </span>
      </div>
      <div class="mono coord">
        x {{ targetPos.x.toFixed(3) }} · y {{ targetPos.y.toFixed(3) }} · z {{ targetPos.z.toFixed(3) }}
      </div>
      <button class="recentre" @click="recentre">↺ 同步到当前 tip</button>
    </div>
    <div class="ik-hint">拖拽彩球 → 实时 IK · 球红表示超出可达域</div>
  </div>
</template>

<style scoped>
.ik-scene {
  position: relative; width: 100%; height: 100%; min-height: 280px; overflow: hidden; border-radius: inherit;
  background:
    radial-gradient(ellipse at 50% 0%, rgba(37, 99, 235, 0.05) 0%, transparent 60%),
    linear-gradient(180deg, var(--bg-card) 0%, var(--bg-base) 100%);
}
canvas { display: block; width: 100% !important; height: 100% !important; cursor: grab; }
.ik-scene:has(canvas:active) canvas { cursor: grabbing; }
.ik-hud {
  position: absolute; top: 10px; left: 10px; right: 10px;
  display: flex; flex-direction: column; gap: 4px;
  padding: 8px 12px; border-radius: 10px;
  background: rgba(255, 255, 255, 0.65); backdrop-filter: blur(14px) saturate(160%);
  border: 1px solid var(--line-divider);
  pointer-events: auto;
}
[data-theme='dark'] .ik-hud { background: rgba(17,21,31,.6); border-color: rgba(148,163,184,.18); }
.row { display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; }
.label { color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
.coord { font-size: 0.66rem; color: var(--ink-tertiary); }
.recentre {
  align-self: flex-end; padding: 4px 10px; border-radius: 8px; border: 1px solid var(--line-border);
  background: rgba(255,255,255,.7); color: var(--ink-secondary); font-size: 0.7rem; font-weight: 600;
  cursor: pointer; transition: all .15s;
}
.recentre:hover { color: var(--accent-blue); border-color: rgba(37,99,235,.3); }
[data-theme='dark'] .recentre { background: rgba(17,21,31,.6); border-color: rgba(148,163,184,.2); }
.ik-hint {
  position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
  font-size: 0.66rem; color: var(--ink-muted); pointer-events: none;
  padding: 3px 10px; border-radius: 999px;
  background: rgba(255,255,255,.5); backdrop-filter: blur(8px);
}
[data-theme='dark'] .ik-hint { background: rgba(17,21,31,.5); }
</style>
