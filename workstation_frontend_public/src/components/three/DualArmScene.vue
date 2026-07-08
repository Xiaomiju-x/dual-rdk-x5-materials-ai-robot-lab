<script setup lang="ts">
// DualArmScene PRO+ — the hero of the workstation cockpit. Workshop world
// (carbon-weave bench + AprilTag floor markers + reagent rack + handover
// hotzone) wraps two procedural myCobot 280-Pi arms. TCP ribbons trace
// recent tip motion; collision halos pulse when the two tips approach;
// dual-click focuses the camera on one arm.
import { onBeforeUnmount, onMounted, ref, watch, computed } from 'vue'
import * as THREE from 'three'
import gsap from 'gsap'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'
import { buildStage } from './buildScene'
import { buildArm, type ArmBundle } from './buildArm'
import { benchTexture, aprilTagTexture, RibbonTrail, buildReagentRack, buildHandoverZone } from './workshop'

interface Props {
  embed?: boolean
  focus?: 'arm01' | 'arm02' | 'both'
  showReach?: boolean
  showWorld?: boolean
}
const props = withDefaults(defineProps<Props>(), {
  embed: false, focus: 'both', showReach: false, showWorld: true,
})

const telemetry = useTelemetryStore()
const settings = useSettingsStore()
const canvasEl = ref<HTMLCanvasElement | null>(null)
const wrapEl = ref<HTMLDivElement | null>(null)
const fps = ref(0)
const verb = ref<'IDLE' | 'MOVING' | 'PICKING' | 'HANDOVER' | 'ERROR'>('IDLE')
const proximityCm = ref(99)

// reactive controls used in template & for camera tween
const focusState = ref<'arm01' | 'arm02' | 'both'>(props.focus)
const reachVisibleState = ref<boolean>(props.showReach)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let stage: ReturnType<typeof buildStage> | null = null
let armA: ArmBundle | null = null
let armB: ArmBundle | null = null
let composer: EffectComposer | null = null
let bloomPass: UnrealBloomPass | null = null
let pmrem: THREE.PMREMGenerator | null = null
let envRT: THREE.WebGLRenderTarget | null = null
let trailA: RibbonTrail | null = null
let trailB: RibbonTrail | null = null
let bench: THREE.Mesh | null = null
let benchTex: THREE.CanvasTexture | null = null
let tagA: THREE.Mesh | null = null
let tagB: THREE.Mesh | null = null
let rack: ReturnType<typeof buildReagentRack> | null = null
let hotzone: ReturnType<typeof buildHandoverZone> | null = null
let proxRingA: THREE.Mesh | null = null
let proxRingB: THREE.Mesh | null = null
let raf = 0
let resizeObs: ResizeObserver | null = null
let lastTs = performance.now()
let frameAcc = 0
let frameSamples = 0

const j = (n: number) => (n >= 0 ? '+' : '') + n.toFixed(0)

function labelSprite(text: string, color: string): THREE.Sprite {
  const cvs = document.createElement('canvas')
  cvs.width = 256; cvs.height = 64
  const ctx = cvs.getContext('2d')!
  ctx.font = 'bold 34px Inter, system-ui, sans-serif'
  ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, 128, 32)
  const tex = new THREE.CanvasTexture(cvs); tex.anisotropy = 4
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
  const sp = new THREE.Sprite(mat); sp.scale.set(0.22, 0.055, 1)
  return sp
}

function makeProximityRing(color: number): THREE.Mesh {
  const geo = new THREE.RingGeometry(0.04, 0.052, 32)
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
  })
  const m = new THREE.Mesh(geo, mat)
  m.rotation.x = -Math.PI / 2
  m.userData.dispose = () => { geo.dispose(); mat.dispose() }
  return m
}

function focusCamera(target: 'arm01' | 'arm02' | 'both') {
  if (!camera || !controls) return
  const baseY = 0.18
  const presets = {
    arm01: { pos: new THREE.Vector3(-0.45, 0.32, 0.4), target: new THREE.Vector3(-0.2, baseY, 0) },
    arm02: { pos: new THREE.Vector3(0.45, 0.32, 0.4), target: new THREE.Vector3(0.2, baseY, 0) },
    both:  { pos: new THREE.Vector3(0.55, 0.5, 0.62), target: new THREE.Vector3(0, baseY, 0) },
  }
  const preset = presets[target]
  gsap.to(camera.position, { x: preset.pos.x, y: preset.pos.y, z: preset.pos.z, duration: 0.7, ease: 'power3.inOut' })
  gsap.to(controls.target, { x: preset.target.x, y: preset.target.y, z: preset.target.z, duration: 0.7, ease: 'power3.inOut',
    onUpdate: () => controls?.update() })
  focusState.value = target
}

function onCanvasDoubleClick(ev: MouseEvent) {
  if (!canvasEl.value || !camera || !armA || !armB) return
  const rect = canvasEl.value.getBoundingClientRect()
  const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1
  const y = -((ev.clientY - rect.top) / rect.height) * 2 + 1
  const ray = new THREE.Raycaster()
  ray.setFromCamera(new THREE.Vector2(x, y), camera)
  // approximate hit-test using two cylinder helpers around each arm
  const da = ray.ray.distanceToPoint(new THREE.Vector3(-0.2, 0.2, 0))
  const db = ray.ray.distanceToPoint(new THREE.Vector3(0.2, 0.2, 0))
  if (focusState.value !== 'both') { focusCamera('both'); return }
  if (da < db && da < 0.18) focusCamera('arm01')
  else if (db < 0.18) focusCamera('arm02')
}

function init() {
  if (!canvasEl.value || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl.value, antialias: true, alpha: true, powerPreference: 'high-performance' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h, false)
  renderer.setClearColor(0x000000, 0)
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.82

  scene = new THREE.Scene()

  pmrem = new THREE.PMREMGenerator(renderer)
  pmrem.compileEquirectangularShader()
  const room = new RoomEnvironment()
  envRT = pmrem.fromScene(room, 0.04)
  scene.environment = envRT.texture
  ;(scene as any).environmentIntensity = 0.35

  camera = new THREE.PerspectiveCamera(props.embed ? 40 : 46, w / h, 0.05, 100)
  camera.position.set(0.55, 0.5, 0.62)
  camera.lookAt(0, 0.18, 0)

  stage = buildStage(settings.theme)
  scene.add(stage.group)

  // ---------- workshop world ----------
  if (props.showWorld) {
    benchTex = benchTexture(settings.theme)
    bench = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 0.012, 0.55),
      new THREE.MeshPhysicalMaterial({ map: benchTex, metalness: 0.25, roughness: 0.6, clearcoat: 0.2 }),
    )
    bench.position.y = 0.005
    bench.receiveShadow = true
    scene.add(bench)

    // AprilTag floor markers
    const tagTexA = aprilTagTexture(0)
    const tagTexB = aprilTagTexture(3)
    const tagMatA = new THREE.MeshBasicMaterial({ map: tagTexA, side: THREE.DoubleSide })
    const tagMatB = new THREE.MeshBasicMaterial({ map: tagTexB, side: THREE.DoubleSide })
    tagA = new THREE.Mesh(new THREE.PlaneGeometry(0.06, 0.06), tagMatA)
    tagB = new THREE.Mesh(new THREE.PlaneGeometry(0.06, 0.06), tagMatB)
    tagA.rotation.x = -Math.PI / 2; tagA.position.set(-0.32, 0.0125, 0.18)
    tagB.rotation.x = -Math.PI / 2; tagB.position.set(0.32, 0.0125, -0.18)
    scene.add(tagA); scene.add(tagB)

    // reagent rack at the back of the bench
    rack = buildReagentRack()
    rack.group.position.set(0, 0.012, -0.22)
    scene.add(rack.group)

    // handover hot zone in the centre
    hotzone = buildHandoverZone()
    hotzone.mesh.position.y = 0.013
    scene.add(hotzone.mesh)
  }

  // ---------- arms ----------
  armA = buildArm('amber')
  armA.group.position.set(-0.2, props.showWorld ? 0.011 : 0, 0)
  armA.setReachVisible(reachVisibleState.value)
  scene.add(armA.group)
  const lblA = labelSprite('arm01', '#d97706')
  lblA.position.set(-0.2, 0.46, 0); scene.add(lblA)

  armB = buildArm('blue')
  armB.group.position.set(0.2, props.showWorld ? 0.011 : 0, 0)
  armB.setReachVisible(reachVisibleState.value)
  scene.add(armB.group)
  const lblB = labelSprite('arm02', '#2563eb')
  lblB.position.set(0.2, 0.46, 0); scene.add(lblB)

  // proximity halos that sit around each arm tip
  proxRingA = makeProximityRing(0xef4444); scene.add(proxRingA)
  proxRingB = makeProximityRing(0xef4444); scene.add(proxRingB)

  // ribbon trails (replace point trails)
  trailA = new RibbonTrail(0xf59e0b, 80); scene.add(trailA.mesh)
  trailB = new RibbonTrail(0x60a5fa, 80); scene.add(trailB.mesh)

  if (!props.embed) {
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 0.3
    controls.maxDistance = 2.0
    controls.maxPolarAngle = Math.PI / 2 - 0.02
    controls.target.set(0, 0.18, 0)
  }

  // post-processing
  composer = new EffectComposer(renderer)
  composer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  composer.setSize(w, h)
  composer.addPass(new RenderPass(scene, camera))
  bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.22, 0.4, 0.88)
  composer.addPass(bloomPass)
  composer.addPass(new OutputPass())

  resizeObs = new ResizeObserver(() => onResize())
  resizeObs.observe(wrapEl.value)

  // dbl-click focus
  canvasEl.value.addEventListener('dblclick', onCanvasDoubleClick)

  // initial focus if not "both"
  if (focusState.value !== 'both') focusCamera(focusState.value)

  loop()
}

function onResize() {
  if (!renderer || !camera || !wrapEl.value || !composer) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight
  if (!w || !h) return
  renderer.setSize(w, h, false)
  composer.setSize(w, h)
  if (bloomPass) bloomPass.setSize(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}

function inferVerb(velA: number, velB: number, dist: number, gripA: number, gripB: number) {
  // 1) close + both moderate vel + gripper change → HANDOVER
  if (dist < 0.18 && velA > 4 && velB > 4) return 'HANDOVER' as const
  // 2) gripper closing on either arm → PICKING
  if ((gripA > 70 || gripB > 70) && (velA > 1 || velB > 1)) return 'PICKING' as const
  // 3) any arm moving → MOVING
  if (velA > 2 || velB > 2) return 'MOVING' as const
  return 'IDLE' as const
}

let lastAnglesA = telemetry.arm01.angles.slice()
let lastAnglesB = telemetry.arm02.angles.slice()
function loop() {
  raf = requestAnimationFrame(loop)
  const now = performance.now()
  const dt = (now - lastTs) / 1000
  lastTs = now
  frameAcc += dt; frameSamples += 1
  if (frameAcc >= 0.5) { fps.value = Math.round((frameSamples / frameAcc) * 10) / 10; frameAcc = 0; frameSamples = 0 }

  const t = now / 1000
  stage?.setGridFade(t)
  hotzone?.update(t)

  const aA = telemetry.arm01.angles, aB = telemetry.arm02.angles
  let velA = 0, velB = 0
  for (let i = 0; i < 6; i++) {
    velA += Math.abs(aA[i] - lastAnglesA[i]) / dt
    velB += Math.abs(aB[i] - lastAnglesB[i]) / dt
  }
  velA /= 6; velB /= 6
  lastAnglesA = aA.slice(); lastAnglesB = aB.slice()

  if (armA) armA.setAngles(aA, telemetry.arm01.gripper)
  if (armB) armB.setAngles(aB, telemetry.arm02.gripper)

  // trails
  if (armA && trailA) { armA.group.updateMatrixWorld(true); trailA.push(armA.tipWorld()) }
  if (armB && trailB) { armB.group.updateMatrixWorld(true); trailB.push(armB.tipWorld()) }

  // proximity halo + verb
  if (armA && armB && proxRingA && proxRingB) {
    const tA = armA.tipWorld(); const tB = armB.tipWorld()
    const d = tA.distanceTo(tB)
    proximityCm.value = Math.round(d * 100)
    const HALO_NEAR = 0.18
    const HALO_FAR = 0.35
    const intensity = Math.max(0, Math.min(1, 1 - (d - HALO_NEAR) / (HALO_FAR - HALO_NEAR)))
    const pulse = 0.5 + 0.5 * Math.sin(t * 6)
    const op = intensity * (0.4 + 0.4 * pulse)
    ;(proxRingA.material as THREE.MeshBasicMaterial).opacity = op
    ;(proxRingB.material as THREE.MeshBasicMaterial).opacity = op
    proxRingA.position.set(tA.x, tA.y - 0.02, tA.z)
    proxRingB.position.set(tB.x, tB.y - 0.02, tB.z)
    const scl = 1 + 0.5 * pulse * intensity
    proxRingA.scale.setScalar(scl); proxRingB.scale.setScalar(scl)

    const v = inferVerb(velA, velB, d, telemetry.arm01.gripper, telemetry.arm02.gripper)
    if (v !== verb.value) verb.value = v
    armA.setVerb(v === 'IDLE' ? 'idle' : 'moving')
    armB.setVerb(v === 'IDLE' ? 'idle' : 'moving')
  }

  if (props.embed && camera) {
    const r = 0.72
    camera.position.x = r * Math.cos(t * 0.16)
    camera.position.z = r * Math.sin(t * 0.16)
    camera.position.y = 0.46 + 0.06 * Math.sin(t * 0.3)
    camera.lookAt(0, 0.18, 0)
  } else if (controls) {
    controls.update()
  }
  if (composer) composer.render()
}

function disposeScene() {
  cancelAnimationFrame(raf)
  resizeObs?.disconnect()
  controls?.dispose()
  canvasEl.value?.removeEventListener('dblclick', onCanvasDoubleClick)
  armA?.dispose(); armB?.dispose()
  trailA?.dispose(); trailB?.dispose()
  rack?.dispose(); hotzone?.dispose()
  composer?.dispose()
  envRT?.dispose()
  pmrem?.dispose()
  benchTex?.dispose()
  if (proxRingA) (proxRingA.userData.dispose as () => void)?.()
  if (proxRingB) (proxRingB.userData.dispose as () => void)?.()
  if (bench) { (bench.geometry as THREE.BufferGeometry).dispose(); (bench.material as THREE.Material).dispose() }
  if (tagA) { (tagA.geometry as THREE.BufferGeometry).dispose(); ((tagA.material as THREE.MeshBasicMaterial).map as THREE.Texture).dispose(); (tagA.material as THREE.Material).dispose() }
  if (tagB) { (tagB.geometry as THREE.BufferGeometry).dispose(); ((tagB.material as THREE.MeshBasicMaterial).map as THREE.Texture).dispose(); (tagB.material as THREE.Material).dispose() }
  if (scene) scene.traverse((obj) => {
    const m = obj as THREE.Mesh
    if (m.geometry) m.geometry.dispose?.()
    const mat = m.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose()
  })
  renderer?.dispose()
  renderer = scene = camera = null; controls = null; stage = null; armA = armB = null
  composer = null; bloomPass = null; trailA = trailB = null
  bench = null; tagA = tagB = null; rack = null; hotzone = null; benchTex = null
  proxRingA = proxRingB = null; envRT = null; pmrem = null
}

onMounted(init)
onBeforeUnmount(disposeScene)
watch(() => props.embed, () => { disposeScene(); init() })
watch(() => settings.theme, (t) => stage?.applyTheme(t))
watch(() => props.showReach, (v) => {
  reachVisibleState.value = v
  armA?.setReachVisible(v); armB?.setReachVisible(v)
})
watch(reachVisibleState, (v) => {
  armA?.setReachVisible(v); armB?.setReachVisible(v)
})

const verbColor = computed(() => {
  switch (verb.value) {
    case 'HANDOVER': return '#10b981'
    case 'PICKING':  return '#f59e0b'
    case 'MOVING':   return '#2563eb'
    case 'ERROR':    return '#ef4444'
    default:         return '#94a3b8'
  }
})

function toggleReach() {
  reachVisibleState.value = !reachVisibleState.value
}
function setFocusManual(f: 'arm01' | 'arm02' | 'both') { focusCamera(f) }
</script>

<template>
  <div ref="wrapEl" class="arm-scene">
    <canvas ref="canvasEl"></canvas>
    <div class="hud-overlay">
      <!-- TL: fps + status -->
      <div class="hud-corner hud-tl">
        <div class="hud-line section-label">Workstation Stage · PRO+</div>
        <div class="hud-line mono">{{ fps.toFixed(1) }} fps · HDR + ribbon · 6 DoF ×2</div>
      </div>

      <!-- TC: big verb -->
      <div class="hud-verb" :style="{ color: verbColor }">
        <span class="verb-dot" :style="{ background: verbColor, boxShadow: `0 0 14px ${verbColor}` }"></span>
        <span class="verb-text">{{ verb }}</span>
        <span class="verb-meta mono">tip ↔ tip {{ proximityCm }} cm</span>
      </div>

      <!-- TR: arm01 angles -->
      <div class="hud-corner hud-tr">
        <div class="hud-line section-label" style="color:var(--accent-amber)">arm01 角度</div>
        <div class="hud-line mono">{{ telemetry.arm01.angles.map(j).join(' ') }}</div>
      </div>

      <!-- BR: arm02 angles -->
      <div class="hud-corner hud-br">
        <div class="hud-line section-label" style="color:var(--accent-blue)">arm02 角度</div>
        <div class="hud-line mono">{{ telemetry.arm02.angles.map(j).join(' ') }}</div>
      </div>

      <!-- BL: gripper + controls -->
      <div class="hud-corner hud-bl">
        <div class="hud-line section-label">夹爪 · 操作</div>
        <div class="hud-line mono">a01 {{ telemetry.arm01.gripper.toFixed(0) }}% · a02 {{ telemetry.arm02.gripper.toFixed(0) }}%</div>
        <div class="hud-controls">
          <button class="hud-pill" :class="{ active: focusState === 'arm01' }" @click="setFocusManual('arm01')">amber</button>
          <button class="hud-pill" :class="{ active: focusState === 'both' }"  @click="setFocusManual('both')">both</button>
          <button class="hud-pill" :class="{ active: focusState === 'arm02' }" @click="setFocusManual('arm02')">blue</button>
          <button class="hud-pill" :class="{ active: reachVisibleState }" @click="toggleReach">reach</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.arm-scene {
  position: relative; width: 100%; height: 100%; overflow: hidden; border-radius: inherit;
  background:
    radial-gradient(ellipse at 50% 100%, rgba(217, 119, 6, 0.07) 0%, transparent 60%),
    radial-gradient(ellipse at 50% 0%, rgba(37, 99, 235, 0.06) 0%, transparent 50%),
    linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-base) 100%);
}
canvas { display: block; width: 100% !important; height: 100% !important; }
.hud-overlay { position: absolute; inset: 0; pointer-events: none; z-index: 2; }
.hud-corner {
  position: absolute; padding: 8px 12px; background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(16px) saturate(160%); -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid var(--line-divider); border-radius: 10px; display: flex; flex-direction: column; gap: 2px;
}
[data-theme='dark'] .hud-corner { background: rgba(17,21,31,.55); border-color: rgba(148,163,184,.18); }
.hud-tl { top: 12px; left: 12px; }
.hud-tr { top: 12px; right: 12px; align-items: flex-end; }
.hud-bl { bottom: 12px; left: 12px; pointer-events: auto; }
.hud-br { bottom: 12px; right: 12px; align-items: flex-end; }
.hud-line { font-size: 0.7rem; color: var(--ink-secondary); white-space: nowrap; }
.hud-line.section-label { color: var(--ink-muted); }

.hud-verb {
  position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  display: flex; align-items: center; gap: 10px;
  padding: 6px 14px; border-radius: 999px;
  background: rgba(255,255,255,.6); backdrop-filter: blur(18px) saturate(180%); -webkit-backdrop-filter: blur(18px) saturate(180%);
  border: 1px solid var(--line-divider);
  font-weight: 800; letter-spacing: 0.12em; font-size: 0.86rem;
}
[data-theme='dark'] .hud-verb { background: rgba(17,21,31,.55); border-color: rgba(148,163,184,.18); }
.verb-dot { width: 10px; height: 10px; border-radius: 50%; }
.verb-text { letter-spacing: 0.18em; }
.verb-meta { color: var(--ink-tertiary); font-weight: 600; font-size: 0.7rem; letter-spacing: 0.04em; }

.hud-controls { display: flex; gap: 4px; margin-top: 4px; }
.hud-pill {
  background: rgba(255,255,255,.7); border: 1px solid var(--line-border);
  border-radius: 999px; padding: 3px 9px; font-size: 0.66rem; font-weight: 600;
  color: var(--ink-tertiary); cursor: pointer; transition: all .15s;
}
.hud-pill:hover { color: var(--accent-blue); border-color: rgba(37,99,235,.3); }
.hud-pill.active { background: rgba(37,99,235,.12); color: var(--accent-blue); border-color: rgba(37,99,235,.35); }
[data-theme='dark'] .hud-pill { background: rgba(17,21,31,.6); border-color: rgba(148,163,184,.2); }
</style>
