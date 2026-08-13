<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'
import { buildRobot } from './buildRobot'
import { buildStage } from './buildScene'
import { buildLidarSweep } from './buildLidar'
import { buildEnvironment, type EnvironmentBundle } from './buildEnvironment'
import { buildPostFX, type PostFXBundle } from './buildPostFX'
import { buildParticleField, type ParticleFieldBundle } from './buildParticleField'
import { buildContactShadow, type ContactShadowBundle } from './buildContactShadow'
import { buildTrail, type TrailBundle } from './buildTrail'
import { createHoloMaterial } from './holoMaterial'
import { buildVolumetricCone, type VolumetricBundle } from './buildVolumetric'
import { buildStarfield, type SkyboxBundle } from './buildSkybox'
import { PRESETS, PRESET_KEYS, flyToPreset } from './cameraPresets'

interface Props {
  /** if true: hide OrbitControls, lock camera, smaller fov — used inside the cockpit hero */
  embed?: boolean
  /** static camera mode = 'orbit-auto' | 'follow' | 'free' */
  mode?: 'orbit-auto' | 'follow' | 'free'
  /** enable bloom + particle + trail (premium cinematic) */
  cinematic?: boolean
  /** holographic Fresnel robot mode */
  holoMode?: boolean
  /** request real HDR from polyhaven (defaults true if cinematic) */
  hdr?: boolean
  /** add screen-space god ray cone + skybox + accept 1-5 preset shortcuts */
  cinemaPlus?: boolean
  /** photo mode — hide HUD, pause particles, slow auto-orbit */
  photoMode?: boolean
}
const props = withDefaults(defineProps<Props>(), {
  embed: false,
  mode: 'orbit-auto',
  cinematic: true,
  holoMode: false,
  hdr: true,
  cinemaPlus: false,
  photoMode: false,
})

const emit = defineEmits<{ (e: 'preset-active', name: string): void }>()

const telemetry = useTelemetryStore()
const settings = useSettingsStore()
const canvasEl = ref<HTMLCanvasElement | null>(null)
const wrapEl = ref<HTMLDivElement | null>(null)
const fps = ref(0)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let robot: ReturnType<typeof buildRobot> | null = null
let stage: ReturnType<typeof buildStage> | null = null
let lidar: ReturnType<typeof buildLidarSweep> | null = null
let env: EnvironmentBundle | null = null
let fx: PostFXBundle | null = null
let particles: ParticleFieldBundle | null = null
let shadow: ContactShadowBundle | null = null
let trail: TrailBundle | null = null
let volumetric: VolumetricBundle | null = null
let skybox: SkyboxBundle | null = null
let cancelFly: (() => void) | null = null
const activePreset = ref<string | null>(null)

let originalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>()
let holoMaterials: THREE.ShaderMaterial[] = []

let raf = 0
let resizeObs: ResizeObserver | null = null
let lastFrameTs = performance.now()
let frameSamples = 0
let frameAcc = 0
let idleTimer = 0

const cinematicActive = computed(() => props.cinematic && settings.cinematic !== false && !settings.reduceMotion)

function init() {
  if (!canvasEl.value || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight

  renderer = new THREE.WebGLRenderer({
    canvas: canvasEl.value,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h, false)
  renderer.setClearColor(0x000000, 0)
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = settings.theme === 'dark' ? 1.0 : 1.15
  renderer.outputColorSpace = THREE.SRGBColorSpace

  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(props.embed ? 38 : 45, w / h, 0.05, 100)
  camera.position.set(2.4, 2.0, 2.4)
  camera.lookAt(0, 0.3, 0)

  // Environment — premium PMREM. Adds tasteful reflections without flat plastic look.
  env = buildEnvironment(renderer)
  scene.environment = env.envTexture
  if (props.hdr && cinematicActive.value) env.upgradeToHdr()

  stage = buildStage(settings.theme)
  scene.add(stage.group)

  robot = buildRobot()
  scene.add(robot)

  // Soft contact shadow under robot
  shadow = buildContactShadow(0.55, settings.theme === 'dark' ? 0.32 : 0.45)
  robot.add(shadow.group)

  // Trail (cinematic only)
  if (cinematicActive.value) {
    trail = buildTrail({ maxPoints: 80, color: 0x22d3ee })
    scene.add(trail.group)
  }

  // Particle field (cinematic only)
  if (cinematicActive.value) {
    particles = buildParticleField({
      count: props.embed ? 600 : 1200,
      radius: 4.5,
      colorA: 0x2563eb,
      colorB: 0x06b6d4,
    })
    scene.add(particles.group)
  }

  lidar = buildLidarSweep()
  robot.add(lidar.group)

  // Cinema++: volumetric god rays + starfield skybox
  if (props.cinemaPlus && cinematicActive.value) {
    volumetric = buildVolumetricCone({ color: 0xffefd8, height: 5.5, radius: 1.8, intensity: 0.35 })
    volumetric.group.position.set(3, 4.5, 2.5)  // matches keyLight
    volumetric.group.rotation.x = Math.PI  // flip so opening faces stage center
    volumetric.group.lookAt(0, 0.3, 0)
    scene.add(volumetric.group)

    skybox = buildStarfield({ themeTone: settings.theme === 'dark' ? 'dark' : 'light' })
    scene.add(skybox.mesh)
  }

  if (!props.embed) {
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1.2
    controls.maxDistance = 6
    controls.maxPolarAngle = Math.PI / 2 - 0.05
    controls.target.set(0, 0.3, 0)
    controls.addEventListener('start', () => { idleTimer = 0 })
  }

  // PostFX bloom — premium glow on emissive surfaces
  if (cinematicActive.value) {
    fx = buildPostFX(renderer, scene, camera, w, h)
  }

  resizeObs = new ResizeObserver(() => onResize())
  resizeObs.observe(wrapEl.value)

  // 1-5 preset keys, only meaningful in cinemaPlus mode w/ free controls
  if (props.cinemaPlus && !props.embed) {
    window.addEventListener('keydown', onPresetKey)
  }

  applyHoloMode(props.holoMode)
  loop()
}

function onPresetKey(e: KeyboardEvent) {
  if (e.target && (e.target as HTMLElement).tagName === 'INPUT') return
  const preset = PRESET_KEYS[e.key]
  if (!preset || !camera) return
  applyPreset(preset)
}

function applyPreset(name: keyof typeof PRESETS) {
  if (!camera) return
  cancelFly?.()
  cancelFly = flyToPreset(camera, controls, PRESETS[name], 900)
  activePreset.value = name
  emit('preset-active', name)
  idleTimer = 0  // reset auto-dolly grace
}
defineExpose({ applyPreset })

function onResize() {
  if (!renderer || !camera || !wrapEl.value) return
  const w = wrapEl.value.clientWidth
  const h = wrapEl.value.clientHeight
  if (w === 0 || h === 0) return
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  fx?.setSize(w, h)
}

function applyHoloMode(holo: boolean) {
  if (!robot) return
  if (holo) {
    // swap each mesh in robot tree to hologram material
    robot.traverse((obj) => {
      const m = obj as THREE.Mesh
      if (!m.isMesh) return
      if (!originalMaterials.has(m)) originalMaterials.set(m, m.material)
      const holoMat = createHoloMaterial({ color: 0x22d3ee })
      holoMaterials.push(holoMat)
      m.material = holoMat
      m.castShadow = false
    })
  } else {
    // restore
    robot.traverse((obj) => {
      const m = obj as THREE.Mesh
      if (!m.isMesh) return
      const orig = originalMaterials.get(m)
      if (orig) m.material = orig
      m.castShadow = true
    })
    holoMaterials.forEach((mat) => mat.dispose())
    holoMaterials = []
  }
}

function loop() {
  raf = requestAnimationFrame(loop)
  const now = performance.now()
  const dt = (now - lastFrameTs) / 1000
  lastFrameTs = now
  frameAcc += dt
  frameSamples += 1
  idleTimer += dt
  if (frameAcc >= 0.5) {
    fps.value = Math.round((frameSamples / frameAcc) * 10) / 10
    frameAcc = 0
    frameSamples = 0
  }

  const t = now / 1000
  if (stage) stage.setGridFade(t)
  if (lidar) lidar.update(t)
  if (particles && !props.photoMode) particles.update(t)
  if (volumetric) volumetric.update(t)
  if (holoMaterials.length) holoMaterials.forEach((m) => { m.uniforms.uTime.value = t })

  // drive robot pose from telemetry store
  if (robot && telemetry.packet) {
    const p = telemetry.packet.pose
    robot.position.x = p.x
    robot.position.z = p.y // ROS x→x, y→z mapping (front: +x, left: +y → +z in three)
    robot.rotation.y = -p.yaw // ROS yaw CCW → three.js needs negative
    // spin wheels proportional to linear velocity
    const wheelOmega = telemetry.packet.velocity.linear / 0.08 // r=0.08m
    for (const w of robot.userData.wheels) w.rotation.x -= wheelOmega * dt
    // feed trail
    if (trail) {
      trail.push(robot.position)
      trail.update()
    }
  }

  // auto-orbit camera in embed (cockpit hero) mode, or after idle in free mode
  if (props.embed && camera) {
    const radius = 3.0
    const cy = 1.5 + 0.15 * Math.sin(t * 0.3)
    camera.position.x = radius * Math.cos(t * 0.18)
    camera.position.z = radius * Math.sin(t * 0.18)
    camera.position.y = cy
    camera.lookAt(0, 0.3, 0)
  } else if (controls) {
    // cinematic auto-dolly after 8s idle (only when cinematic mode)
    if (cinematicActive.value && idleTimer > 8 && !props.embed) {
      const radius = controls.getDistance()
      const cy = camera!.position.y
      const a = t * 0.08
      camera!.position.x = radius * Math.cos(a)
      camera!.position.z = radius * Math.sin(a)
      camera!.position.y = cy
      camera!.lookAt(controls.target)
    } else {
      controls.update()
    }
  }

  if (fx && cinematicActive.value) {
    fx.composer.render()
  } else if (renderer && scene && camera) {
    renderer.render(scene, camera)
  }
}

function disposeScene(): void {
  cancelAnimationFrame(raf)
  resizeObs?.disconnect()
  if (props.cinemaPlus && !props.embed) window.removeEventListener('keydown', onPresetKey)
  cancelFly?.()
  controls?.dispose()
  particles?.dispose()
  shadow?.dispose()
  trail?.dispose()
  volumetric?.dispose()
  skybox?.dispose()
  fx?.dispose()
  env?.dispose()
  if (scene) {
    scene.traverse((obj) => {
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry?.dispose()
      const mat = (obj as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
      else if (mat) mat.dispose()
    })
  }
  renderer?.dispose()
  renderer = null
  scene = null
  camera = null
  controls = null
  robot = null
  stage = null
  lidar = null
  env = null
  fx = null
  particles = null
  shadow = null
  trail = null
  volumetric = null
  skybox = null
  cancelFly = null
  originalMaterials = new Map()
  holoMaterials = []
}

onMounted(() => {
  init()
})

onBeforeUnmount(() => {
  disposeScene()
})

watch(() => props.embed, () => {
  disposeScene()
  init()
})

watch(() => settings.theme, (t) => {
  stage?.applyTheme(t)
  if (renderer) renderer.toneMappingExposure = t === 'dark' ? 1.0 : 1.15
  if (shadow) shadow.setStrength(t === 'dark' ? 0.32 : 0.45)
})

watch(() => props.holoMode, (v) => applyHoloMode(v))

watch(cinematicActive, () => {
  // toggling cinematic requires a re-init (composer/particles add/remove)
  disposeScene()
  init()
})
</script>

<template>
  <div ref="wrapEl" class="slam-scene">
    <canvas ref="canvasEl"></canvas>
    <div class="hud-overlay">
      <slot name="hud">
        <div class="hud-corner hud-tl">
          <div class="hud-line section-label">SLAM Stage</div>
          <div class="hud-line mono">{{ fps.toFixed(1) }} fps · webgl2{{ cinematicActive ? ' · bloom' : '' }}</div>
        </div>
        <div v-if="telemetry.packet" class="hud-corner hud-tr">
          <div class="hud-line section-label">Pose</div>
          <div class="hud-line mono">
            x {{ telemetry.packet.pose.x.toFixed(2) }} ·
            y {{ telemetry.packet.pose.y.toFixed(2) }} ·
            θ {{ ((telemetry.packet.pose.yaw * 180) / Math.PI).toFixed(0) }}°
          </div>
        </div>
        <div v-if="telemetry.packet" class="hud-corner hud-bl">
          <div class="hud-line section-label">LD14</div>
          <div class="hud-line mono">270° · 10 Hz · 666 pts</div>
        </div>
        <div v-if="telemetry.packet" class="hud-corner hud-br">
          <div class="hud-line section-label">Velocity</div>
          <div class="hud-line mono">{{ telemetry.packet.velocity.linear.toFixed(2) }} m/s · ω {{ telemetry.packet.velocity.angular.toFixed(2) }}</div>
        </div>
      </slot>
    </div>
  </div>
</template>

<style scoped>
.slam-scene {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: inherit;
  background:
    radial-gradient(ellipse at 50% 100%, rgba(37, 99, 235, 0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 50% 0%,   rgba(8, 145, 178, 0.05) 0%, transparent 50%),
    linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-base) 100%);
}
canvas {
  display: block;
  width: 100% !important;
  height: 100% !important;
}
.hud-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2;
}
.hud-corner {
  position: absolute;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.hud-tl { top: 12px; left: 12px; }
.hud-tr { top: 12px; right: 12px; align-items: flex-end; }
.hud-bl { bottom: 12px; left: 12px; }
.hud-br { bottom: 12px; right: 12px; align-items: flex-end; }
.hud-line { font-size: 0.7rem; color: var(--ink-secondary); white-space: nowrap; }
.hud-line.section-label { color: var(--ink-muted); }
</style>
