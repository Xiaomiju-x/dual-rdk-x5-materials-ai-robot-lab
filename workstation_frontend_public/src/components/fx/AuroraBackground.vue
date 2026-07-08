<script setup lang="ts">
// 全屏 WebGL aurora 渐变背景 — GPU 60fps, 跟随主题, 减动效自动降级.
// 蓝青基调 (与现有 accent 协同), 多频 simplex-style 噪声驱动色团流动.
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as THREE from 'three'
import { useSettingsStore } from '@/stores/settings'

const props = defineProps<{ intensity?: number }>()
const host = ref<HTMLDivElement>()
const settings = useSettingsStore()

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.OrthographicCamera | null = null
let mesh: THREE.Mesh | null = null
let mat: THREE.ShaderMaterial | null = null
let raf = 0
let ro: ResizeObserver | null = null
const mouse = { x: 0.5, y: 0.5 }
const start = performance.now()

const FRAG = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform vec2 uRes;
  uniform vec2 uMouse;
  uniform float uIntensity;
  uniform vec3 uA; uniform vec3 uB; uniform vec3 uC; uniform vec3 uD;

  // hash + value noise + fbm (cheap, fragment-shader-friendly)
  float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
  float noise(vec2 p){
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.-2.*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), u.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);
  }
  float fbm(vec2 p){
    float v=0., a=.5;
    for(int i=0;i<5;i++){ v += a*noise(p); p*=2.05; a*=.5; }
    return v;
  }

  void main(){
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = (uv - 0.5);
    p.x *= uRes.x / uRes.y;
    float t = uTime * 0.06;

    // three flowing blobs
    float n1 = fbm(p*1.6 + vec2(t, t*.6));
    float n2 = fbm(p*2.4 - vec2(t*.7, -t*.4) + 3.7);
    float n3 = fbm(p*0.9 + vec2(-t*.5, t*1.1) - 1.3);

    vec3 col = uA;
    col = mix(col, uB, smoothstep(.20, .85, n1));
    col = mix(col, uC, smoothstep(.20, .85, n2) * 0.85);
    col = mix(col, uD, smoothstep(.30, .90, n3) * 0.55);

    // mouse spotlight glow (very soft)
    vec2 mp = uMouse - 0.5; mp.x *= uRes.x / uRes.y;
    float d = length(p - mp);
    float glow = exp(-d*2.3) * 0.20 * uIntensity;
    col += vec3(0.35, 0.55, 0.95) * glow;

    // vignette
    float vg = smoothstep(1.15, 0.35, length(uv-0.5));
    col *= mix(0.86, 1.0, vg);

    // film grain
    float g = (hash(uv*uRes + t*60.0) - 0.5) * 0.020;
    col += g;

    gl_FragColor = vec4(col, 1.0);
  }
`

const VERT = /* glsl */ `
  void main(){ gl_Position = vec4(position, 1.0); }
`

// theme-driven palettes
function paletteFor(theme: 'light' | 'dark'): [number[], number[], number[], number[]] {
  if (theme === 'dark') {
    return [
      [0.040, 0.055, 0.090], // base deep
      [0.060, 0.110, 0.190], // blue
      [0.040, 0.180, 0.230], // teal
      [0.180, 0.110, 0.290], // violet hint
    ]
  }
  // light — keep airy, low chroma, but with subtle blue/teal/violet motion
  return [
    [0.960, 0.972, 0.988], // near white
    [0.870, 0.910, 0.985], // soft blue
    [0.840, 0.945, 0.975], // soft teal
    [0.945, 0.910, 0.985], // soft lavender
  ]
}

function applyPalette() {
  if (!mat) return
  const [a, b, c, d] = paletteFor(settings.theme === 'dark' ? 'dark' : 'light')
  ;(mat.uniforms.uA.value as THREE.Vector3).set(a[0], a[1], a[2])
  ;(mat.uniforms.uB.value as THREE.Vector3).set(b[0], b[1], b[2])
  ;(mat.uniforms.uC.value as THREE.Vector3).set(c[0], c[1], c[2])
  ;(mat.uniforms.uD.value as THREE.Vector3).set(d[0], d[1], d[2])
}

function resize() {
  if (!host.value || !renderer || !mat) return
  const w = host.value.clientWidth
  const h = host.value.clientHeight
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
  renderer.setPixelRatio(dpr)
  renderer.setSize(w, h, false)
  ;(mat.uniforms.uRes.value as THREE.Vector2).set(w * dpr, h * dpr)
}

function tick() {
  if (!renderer || !scene || !camera || !mat) return
  const elapsed = (performance.now() - start) / 1000
  ;(mat.uniforms.uTime.value as number) // tsc keep
  mat.uniforms.uTime.value = elapsed
  ;(mat.uniforms.uMouse.value as THREE.Vector2).set(mouse.x, mouse.y)
  renderer.render(scene, camera)
  raf = requestAnimationFrame(tick)
}

function onMove(e: MouseEvent) {
  if (!host.value) return
  const r = host.value.getBoundingClientRect()
  mouse.x = (e.clientX - r.left) / r.width
  mouse.y = 1 - (e.clientY - r.top) / r.height
}

onMounted(() => {
  if (!host.value) return
  const w = host.value.clientWidth, h = host.value.clientHeight
  renderer = new THREE.WebGLRenderer({ alpha: false, antialias: false, powerPreference: 'low-power' })
  renderer.setClearColor(0xf6f8fb, 1)
  host.value.appendChild(renderer.domElement)
  scene = new THREE.Scene()
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
  mat = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uTime: { value: 0 },
      uRes: { value: new THREE.Vector2(w, h) },
      uMouse: { value: new THREE.Vector2(0.5, 0.5) },
      uIntensity: { value: props.intensity ?? 1.0 },
      uA: { value: new THREE.Vector3() },
      uB: { value: new THREE.Vector3() },
      uC: { value: new THREE.Vector3() },
      uD: { value: new THREE.Vector3() },
    },
  })
  applyPalette()
  mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat)
  scene.add(mesh)
  resize()
  ro = new ResizeObserver(resize)
  ro.observe(host.value)
  window.addEventListener('pointermove', onMove, { passive: true })
  if (!settings.reduceMotion) {
    raf = requestAnimationFrame(tick)
  } else {
    renderer.render(scene, camera)
  }
})

watch(() => settings.theme, () => applyPalette())
watch(() => settings.reduceMotion, (rm) => {
  if (rm && raf) { cancelAnimationFrame(raf); raf = 0 }
  else if (!rm && !raf) raf = requestAnimationFrame(tick)
})

onBeforeUnmount(() => {
  if (raf) cancelAnimationFrame(raf)
  if (ro) ro.disconnect()
  window.removeEventListener('pointermove', onMove)
  mesh?.geometry.dispose()
  mat?.dispose()
  renderer?.dispose()
  renderer?.domElement.remove()
})
</script>

<template>
  <div ref="host" class="aurora-bg" aria-hidden="true"></div>
</template>

<style scoped>
.aurora-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  overflow: hidden;
}
.aurora-bg :deep(canvas) { display: block; width: 100% !important; height: 100% !important; }
</style>
