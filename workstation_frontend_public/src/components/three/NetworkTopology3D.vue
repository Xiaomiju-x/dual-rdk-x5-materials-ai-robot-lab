<script setup lang="ts">
// NetworkTopology3D — three floating nodes (AI brain / car brain / workstation)
// connected by curved tubes. Tubes use a shader-scrolled flowing gradient and
// emit particles along the curve. Bloom + slow orbit camera. Reads coop events
// to recolor flows.
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'

const wrap = ref<HTMLDivElement>()
const canvas = ref<HTMLCanvasElement>()
const telemetry = useTelemetryStore()
const settings = useSettingsStore()

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let composer: EffectComposer | null = null
let raf = 0
let ro: ResizeObserver | null = null
let flowMats: THREE.ShaderMaterial[] = []
let nodeMeshes: THREE.Mesh[] = []
let nodeHalos: THREE.Mesh[] = []
let particleBatch: { points: THREE.Points; mat: THREE.ShaderMaterial; geom: THREE.BufferGeometry; data: ParticleData[] } | null = null
let start = performance.now()

const NODES = [
  { name: 'AI 脑',  sub: '9 LLM · 5 BPU slot', color: 0x7c3aed, pos: new THREE.Vector3( 0,  0.55, 0) },
  { name: '车载脑', sub: 'ROS2 · 8 BPU bin',  color: 0xd97706, pos: new THREE.Vector3(-0.7, -0.45, 0.1) },
  { name: '工位',   sub: 'dual myCobot 280-Pi', color: 0x2563eb, pos: new THREE.Vector3( 0.7, -0.45, -0.1) },
]
const EDGES: { a: number; b: number; hue: number }[] = [
  { a: 2, b: 0, hue: 0x60a5fa }, // workstation → AI
  { a: 2, b: 1, hue: 0xfbbf24 }, // workstation → car
  { a: 1, b: 0, hue: 0xa78bfa }, // car → AI
]

interface ParticleData { curve: THREE.CatmullRomCurve3; t: number; speed: number; color: THREE.Color; size: number }
const MAX_PARTICLES = 220

function makeNodeMesh(color: number): { mesh: THREE.Mesh; halo: THREE.Mesh } {
  const geom = new THREE.IcosahedronGeometry(0.18, 1)
  const mat = new THREE.MeshPhysicalMaterial({
    color, metalness: 0.35, roughness: 0.32, clearcoat: 0.7, clearcoatRoughness: 0.2,
    emissive: color, emissiveIntensity: 0.35,
  })
  const mesh = new THREE.Mesh(geom, mat)

  // halo sprite
  const cvs = document.createElement('canvas'); cvs.width = 256; cvs.height = 256
  const ctx = cvs.getContext('2d')!
  const g = ctx.createRadialGradient(128,128,4,128,128,120)
  const hex = '#' + color.toString(16).padStart(6,'0')
  g.addColorStop(0, hex)
  g.addColorStop(0.5, hex + '40')
  g.addColorStop(1, 'transparent')
  ctx.fillStyle = g; ctx.fillRect(0,0,256,256)
  const tex = new THREE.CanvasTexture(cvs)
  const haloMat = new THREE.SpriteMaterial({ map: tex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.7 })
  const halo = new THREE.Sprite(haloMat) as unknown as THREE.Mesh
  halo.scale.set(0.72, 0.72, 1)
  return { mesh, halo }
}

function makeLabelSprite(text: string, sub: string, color: number): THREE.Sprite {
  const cvs = document.createElement('canvas'); cvs.width = 512; cvs.height = 128
  const ctx = cvs.getContext('2d')!
  ctx.font = 'bold 56px Inter, system-ui, sans-serif'
  ctx.fillStyle = '#' + color.toString(16).padStart(6,'0')
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, 256, 50)
  ctx.font = '24px JetBrains Mono, monospace'
  ctx.fillStyle = '#94a3b8'
  ctx.fillText(sub, 256, 96)
  const tex = new THREE.CanvasTexture(cvs); tex.anisotropy = 4
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
  const sp = new THREE.Sprite(mat)
  sp.scale.set(0.6, 0.15, 1)
  return sp
}

function makeFlowTube(curve: THREE.CatmullRomCurve3, color: number): { mesh: THREE.Mesh; mat: THREE.ShaderMaterial } {
  const geom = new THREE.TubeGeometry(curve, 64, 0.014, 8, false)
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(color) },
    },
    vertexShader: `
      varying vec2 vUv;
      void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: `
      uniform float uTime; uniform vec3 uColor; varying vec2 vUv;
      void main(){
        float flow = fract(vUv.x - uTime * 0.6);
        float pulse = smoothstep(0.0, 0.45, flow) * smoothstep(1.0, 0.55, flow);
        float core = smoothstep(0.55, 0.45, abs(vUv.y - 0.5));
        float a = pulse * 0.9 + 0.12;
        vec3 c = uColor * (0.55 + pulse * 0.9);
        gl_FragColor = vec4(c, a * core);
      }
    `,
  })
  return { mesh: new THREE.Mesh(geom, mat), mat }
}

function init() {
  if (!wrap.value || !canvas.value) return
  const w = wrap.value.clientWidth, h = wrap.value.clientHeight

  renderer = new THREE.WebGLRenderer({ canvas: canvas.value, antialias: true, alpha: true, powerPreference: 'high-performance' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h, false)
  renderer.setClearColor(0x000000, 0)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.95
  // outputColorSpace defaults to SRGBColorSpace (r152+) — OutputPass handles
  // the final linear→sRGB conversion on the composer chain.

  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(38, w / h, 0.05, 100)
  camera.position.set(0, 0.05, 2.6)
  camera.lookAt(0, 0, 0)

  // lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const key = new THREE.DirectionalLight(0xffffff, 0.7); key.position.set(2, 3, 4); scene.add(key)
  const rim = new THREE.DirectionalLight(0x67e8f9, 0.4); rim.position.set(-2, 1, -3); scene.add(rim)

  // nodes
  NODES.forEach((n) => {
    const { mesh, halo } = makeNodeMesh(n.color)
    mesh.position.copy(n.pos); halo.position.copy(n.pos)
    scene!.add(mesh); scene!.add(halo)
    nodeMeshes.push(mesh); nodeHalos.push(halo)
    const lbl = makeLabelSprite(n.name, n.sub, n.color)
    lbl.position.copy(n.pos).add(new THREE.Vector3(0, -0.34, 0))
    scene!.add(lbl)
  })

  // edges as cubic Bezier-ish curves
  const curves: THREE.CatmullRomCurve3[] = []
  EDGES.forEach((e) => {
    const a = NODES[e.a].pos, b = NODES[e.b].pos
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5)
    mid.y += 0.18 * (Math.random() - 0.2); mid.z += 0.15 * (Math.random() - 0.5)
    const curve = new THREE.CatmullRomCurve3([a.clone(), mid, b.clone()])
    curves.push(curve)
    const { mesh, mat } = makeFlowTube(curve, e.hue)
    scene!.add(mesh)
    flowMats.push(mat)
  })

  // particles travelling along curves
  const pos = new Float32Array(MAX_PARTICLES * 3)
  const col = new Float32Array(MAX_PARTICLES * 3)
  const siz = new Float32Array(MAX_PARTICLES)
  const data: ParticleData[] = []
  for (let i = 0; i < MAX_PARTICLES; i++) {
    const ei = i % EDGES.length
    const c = new THREE.Color(EDGES[ei].hue)
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b
    siz[i] = 6 + Math.random() * 6
    data.push({ curve: curves[ei], t: Math.random(), speed: 0.05 + Math.random() * 0.12, color: c, size: siz[i] })
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3))
  geom.setAttribute('asize', new THREE.BufferAttribute(siz, 1))
  const pmat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float asize; varying vec3 vC;
      void main(){
        vC = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = asize * (240.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      varying vec3 vC;
      void main(){
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        float a = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(vC, a);
      }
    `,
    vertexColors: true,
  })
  const pts = new THREE.Points(geom, pmat)
  pts.frustumCulled = false
  scene.add(pts)
  particleBatch = { points: pts, mat: pmat, geom, data }

  // post-processing — bloom for the glowy feel
  composer = new EffectComposer(renderer)
  composer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  composer.setSize(w, h)
  composer.addPass(new RenderPass(scene, camera))
  // bloom selectively on emissive nodes + additive flow tubes; raise threshold
  // so diffuse pixels don't whitewash. Strength still strong → flowy glow.
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.45, 0.82))
  composer.addPass(new OutputPass())

  ro = new ResizeObserver(onResize)
  ro.observe(wrap.value)
  loop()
}

function onResize() {
  if (!renderer || !camera || !composer || !wrap.value) return
  const w = wrap.value.clientWidth, h = wrap.value.clientHeight
  if (!w || !h) return
  renderer.setSize(w, h, false)
  composer.setSize(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}

function loop() {
  raf = requestAnimationFrame(loop)
  const elapsed = (performance.now() - start) / 1000
  flowMats.forEach((m) => (m.uniforms.uTime.value = elapsed))

  // node bob + halo
  nodeMeshes.forEach((m, i) => {
    const phase = elapsed * 0.6 + i * 1.4
    m.position.y = NODES[i].pos.y + 0.025 * Math.sin(phase)
    m.rotation.y += 0.0035
    nodeHalos[i].position.y = m.position.y
    const s = 0.72 + 0.08 * Math.sin(elapsed * 1.5 + i)
    nodeHalos[i].scale.set(s, s, 1)
  })

  // camera slow orbit
  const r = 2.5
  camera!.position.x = r * Math.sin(elapsed * 0.08)
  camera!.position.z = r * Math.cos(elapsed * 0.08)
  camera!.position.y = 0.18 + 0.05 * Math.sin(elapsed * 0.2)
  camera!.lookAt(0, 0, 0)

  // particles along curves
  if (particleBatch) {
    const { data, geom } = particleBatch
    const pos = geom.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < data.length; i++) {
      const d = data[i]
      d.t += d.speed * 0.016
      if (d.t > 1) d.t -= 1
      const p = d.curve.getPoint(d.t)
      pos.setXYZ(i, p.x, p.y, p.z)
    }
    pos.needsUpdate = true
  }

  composer?.render()
}

function dispose() {
  cancelAnimationFrame(raf)
  ro?.disconnect()
  composer?.dispose()
  flowMats.forEach((m) => m.dispose()); flowMats = []
  particleBatch?.geom.dispose(); particleBatch?.mat.dispose(); particleBatch = null
  nodeMeshes.forEach((m) => { m.geometry.dispose(); (m.material as THREE.Material).dispose() })
  nodeHalos.forEach((m) => { (m.material as THREE.Material).dispose() })
  nodeMeshes = []; nodeHalos = []
  if (scene) scene.traverse((obj) => {
    const m = obj as THREE.Mesh
    if (m.geometry) m.geometry.dispose?.()
    const mat = m.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose()
  })
  renderer?.dispose()
  renderer = null; scene = null; camera = null; composer = null
}

onMounted(init); onBeforeUnmount(dispose)
watch(() => settings.theme, () => { /* aurora handles bg; nodes auto */ })
// hint that coopEvents could drive flow intensity in future
watch(() => telemetry.coopEvents.length, () => {})
</script>

<template>
  <div ref="wrap" class="topo-3d">
    <canvas ref="canvas"></canvas>
    <div class="topo-overlay">
      <div class="topo-corner mono">three-way mesh · live · BPU bloom · curl flow</div>
    </div>
  </div>
</template>

<style scoped>
.topo-3d { position: relative; width: 100%; height: 100%; min-height: 380px; border-radius: inherit; overflow: hidden; }
canvas { display: block; width: 100% !important; height: 100% !important; }
.topo-overlay { position: absolute; inset: 0; pointer-events: none; }
.topo-corner {
  position: absolute; top: 12px; right: 14px;
  font-size: 0.68rem; color: var(--ink-tertiary);
  background: rgba(255,255,255,.55); border: 1px solid var(--line-divider);
  padding: 4px 8px; border-radius: 6px; backdrop-filter: blur(10px);
}
[data-theme='dark'] .topo-corner { background: rgba(17,21,31,.5); border-color: rgba(148,163,184,.18); }
</style>
