<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Camera } from '@/types/telemetry'

interface Props {
  camera: Camera
  /** what overlays to draw */
  mode?: 'yolo' | 'apriltag' | 'depth' | 'none'
  /** main label color theme */
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  /** detection callback so parent log can listen */
  onDetect?: (det: Detection) => void
}

export interface Detection {
  id: string
  camera_id: string
  cls: string          // 'bottle' | 'person' | 'apriltag' | ...
  conf: number
  bbox: [number, number, number, number]  // [x, y, w, h] normalised 0..1
  tag_id?: number      // for apriltag
  at_ms: number
}

const props = withDefaults(defineProps<Props>(), {
  mode: 'yolo',
  accent: 'blue',
  onDetect: undefined,
})

const canvasRef = ref<HTMLCanvasElement | null>(null)
const overlayRef = ref<HTMLCanvasElement | null>(null)

const W = 480
const H = 270

const ACCENT_RGB: Record<string, [number, number, number]> = {
  blue:    [37, 99, 235],
  teal:    [8, 145, 178],
  emerald: [5, 150, 105],
  violet:  [124, 58, 237],
  amber:   [217, 119, 6],
  rose:    [225, 29, 72],
}

const fps = computed(() => props.camera.fps)
const res = computed(() => `${props.camera.width}×${props.camera.height}`)
const detCount = ref(0)

// ---------- procedural "scene" ----------
// Simulate a lab-bench view: shelf + bottles. We don't render real video, but
// the canvas evolves smoothly so the overlay boxes "track" something visible.

interface SceneObject {
  id: string
  cls: 'bottle' | 'person' | 'apriltag'
  x: number    // centre, normalised 0..1
  y: number
  w: number
  h: number
  vx: number
  vy: number
  hue: number
  tag_id?: number
}

let objs: SceneObject[] = []

function seedObjects() {
  if (props.mode === 'apriltag') {
    objs = [
      { id: 'tag-0', cls: 'apriltag', x: 0.30, y: 0.45, w: 0.16, h: 0.20, vx: 0.0001, vy: 0.00005, hue: 200, tag_id: 0 },
      { id: 'tag-3', cls: 'apriltag', x: 0.68, y: 0.55, w: 0.14, h: 0.18, vx: -0.0001, vy: 0.00005, hue: 280, tag_id: 3 },
    ]
  } else if (props.mode === 'depth') {
    objs = [
      { id: 'obs-1', cls: 'person',  x: 0.55, y: 0.60, w: 0.22, h: 0.55, vx: -0.0001, vy: 0,       hue: 30  },
      { id: 'obs-2', cls: 'bottle',  x: 0.25, y: 0.70, w: 0.10, h: 0.18, vx: 0,        vy: 0.0001,  hue: 120 },
    ]
  } else {
    // yolo: lift-cam looking down at bottles
    objs = [
      { id: 'sygo-1', cls: 'bottle', x: 0.35, y: 0.50, w: 0.13, h: 0.28, vx: 0.0001, vy: -0.00005, hue: 200 },
      { id: 'sygo-2', cls: 'bottle', x: 0.62, y: 0.48, w: 0.12, h: 0.26, vx: -0.0001, vy: 0.00005,  hue: 280 },
      { id: 'sygo-3', cls: 'bottle', x: 0.50, y: 0.78, w: 0.10, h: 0.18, vx: 0,        vy: 0,        hue: 160 },
    ]
  }
}

function step(dtMs: number) {
  for (const o of objs) {
    o.x += o.vx * dtMs
    o.y += o.vy * dtMs
    if (o.x < 0.1 || o.x > 0.9) o.vx = -o.vx
    if (o.y < 0.1 || o.y > 0.9) o.vy = -o.vy
  }
}

// ---------- canvas scene draw ----------
function drawScene(ctx: CanvasRenderingContext2D, t: number) {
  // background gradient — feels like a lab bench under sodium light
  const g = ctx.createLinearGradient(0, 0, 0, H)
  if (props.mode === 'depth') {
    g.addColorStop(0, '#0a1322')
    g.addColorStop(1, '#16284a')
  } else if (props.mode === 'apriltag') {
    g.addColorStop(0, '#1c1330')
    g.addColorStop(1, '#0f0a1c')
  } else {
    g.addColorStop(0, '#1a2030')
    g.addColorStop(1, '#0e131d')
  }
  ctx.fillStyle = g
  ctx.fillRect(0, 0, W, H)

  // shelf line (lift-cam style horizon)
  if (props.mode === 'yolo' || props.mode === 'apriltag') {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    for (let y = 40; y < H; y += 32) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
    }
  }

  // depth view: radial pseudo-depth rings
  if (props.mode === 'depth') {
    for (let r = 30; r < 300; r += 24) {
      ctx.strokeStyle = `rgba(120, 200, 255, ${0.04 + 0.02 * Math.sin(t / 600 + r / 50)})`
      ctx.beginPath()
      ctx.arc(W / 2, H, r, Math.PI, 2 * Math.PI)
      ctx.stroke()
    }
  }

  // objects
  for (const o of objs) {
    const cx = o.x * W
    const cy = o.y * H
    const ow = o.w * W
    const oh = o.h * H

    if (o.cls === 'bottle') {
      // bottle body
      const grad = ctx.createLinearGradient(cx - ow / 2, cy, cx + ow / 2, cy)
      grad.addColorStop(0, `hsl(${o.hue}, 60%, 35%)`)
      grad.addColorStop(0.5, `hsl(${o.hue}, 80%, 55%)`)
      grad.addColorStop(1, `hsl(${o.hue}, 60%, 30%)`)
      ctx.fillStyle = grad
      ctx.fillRect(cx - ow / 2, cy - oh / 2, ow, oh)
      // bottle cap
      ctx.fillStyle = `hsl(${o.hue}, 30%, 22%)`
      ctx.fillRect(cx - ow / 4, cy - oh / 2 - 6, ow / 2, 6)
      // glint
      ctx.fillStyle = 'rgba(255,255,255,0.18)'
      ctx.fillRect(cx - ow / 4, cy - oh / 2 + 4, 2, oh - 8)
    } else if (o.cls === 'apriltag') {
      // 4x4 black-white grid (fake apriltag)
      const cell = ow / 6
      ctx.fillStyle = '#fff'
      ctx.fillRect(cx - ow / 2, cy - oh / 2, ow, oh)
      ctx.fillStyle = '#000'
      ctx.fillRect(cx - ow / 2 + cell, cy - oh / 2 + cell, ow - 2 * cell, oh - 2 * cell)
      // checker pattern in inner
      const inner = ow - 2 * cell
      const seed = (o.tag_id ?? 0) * 7 + 11
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          if (((i * 4 + j + seed) % 3) === 0) {
            ctx.fillStyle = '#fff'
            ctx.fillRect(cx - ow / 2 + cell + (inner / 4) * j, cy - oh / 2 + cell + (inner / 4) * i, inner / 4, inner / 4)
          }
        }
      }
    } else if (o.cls === 'person') {
      // soft silhouette
      ctx.fillStyle = `hsl(${o.hue}, 60%, 40%)`
      ctx.beginPath()
      ctx.ellipse(cx, cy - oh / 4, ow / 4, ow / 4, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillRect(cx - ow / 2, cy - oh / 2 + ow / 2, ow, oh - ow / 2)
    }
  }

  // scan-line + grain pass
  ctx.fillStyle = 'rgba(255,255,255,0.025)'
  for (let y = ((t / 20) | 0) % 4; y < H; y += 4) {
    ctx.fillRect(0, y, W, 1)
  }
  // very subtle moving vignette
  const vgrad = ctx.createRadialGradient(W / 2, H / 2, 80, W / 2, H / 2, 320)
  vgrad.addColorStop(0, 'rgba(0,0,0,0)')
  vgrad.addColorStop(1, 'rgba(0,0,0,0.55)')
  ctx.fillStyle = vgrad
  ctx.fillRect(0, 0, W, H)
}

// ---------- overlay (bbox + label + tag axis) ----------
function drawOverlay(ctx: CanvasRenderingContext2D, t: number) {
  ctx.clearRect(0, 0, W, H)
  if (props.mode === 'none') return
  const [r, g, b] = ACCENT_RGB[props.accent]

  for (const o of objs) {
    const cx = o.x * W
    const cy = o.y * H
    const ow = o.w * W
    const oh = o.h * H
    const x = cx - ow / 2
    const y = cy - oh / 2

    if (o.cls === 'apriltag') {
      // green box + crosshair + axis
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.9)`
      ctx.lineWidth = 1.5
      // corner crosses
      const cs = 8
      ;[[x, y], [x + ow, y], [x, y + oh], [x + ow, y + oh]].forEach(([px, py]) => {
        ctx.beginPath()
        ctx.moveTo(px - cs, py); ctx.lineTo(px + cs, py)
        ctx.moveTo(px, py - cs); ctx.lineTo(px, py + cs)
        ctx.stroke()
      })
      // dashed bbox
      ctx.setLineDash([4, 4])
      ctx.strokeRect(x, y, ow, oh)
      ctx.setLineDash([])
      // axis (red x, green y, blue z) — fake 3D from centre
      const ax = cx
      const ay = cy
      const len = 28
      ctx.lineWidth = 2
      ctx.strokeStyle = '#ef4444'
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + len, ay); ctx.stroke()
      ctx.strokeStyle = '#22c55e'
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay - len); ctx.stroke()
      ctx.strokeStyle = '#60a5fa'
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax - len * 0.6, ay + len * 0.6); ctx.stroke()
      // tag id label
      const label = `tag36h11  id=${o.tag_id}`
      ctx.font = '600 11px "JetBrains Mono Variable", monospace'
      const tw = ctx.measureText(label).width + 10
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.92)`
      ctx.fillRect(x, y - 18, tw, 16)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 5, y - 6)
    } else {
      // YOLO bbox (corner brackets style)
      const conf = 0.82 + 0.12 * Math.sin(t / 300 + o.x * 10)
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.95)`
      ctx.lineWidth = 1.5
      const cs = 12
      // four corner L-brackets
      ctx.beginPath()
      ctx.moveTo(x, y + cs); ctx.lineTo(x, y); ctx.lineTo(x + cs, y)
      ctx.moveTo(x + ow - cs, y); ctx.lineTo(x + ow, y); ctx.lineTo(x + ow, y + cs)
      ctx.moveTo(x, y + oh - cs); ctx.lineTo(x, y + oh); ctx.lineTo(x + cs, y + oh)
      ctx.moveTo(x + ow - cs, y + oh); ctx.lineTo(x + ow, y + oh); ctx.lineTo(x + ow, y + oh - cs)
      ctx.stroke()
      // faint inner box
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.18)`
      ctx.lineWidth = 1
      ctx.strokeRect(x, y, ow, oh)
      // label chip
      const label = `${o.id}  ${conf.toFixed(2)}`
      ctx.font = '600 11px "JetBrains Mono Variable", monospace'
      const tw = ctx.measureText(label).width + 12
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.95)`
      ctx.fillRect(x, y - 18, tw, 16)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 6, y - 6)
    }
  }

  // top-left HUD
  ctx.font = '500 10px "JetBrains Mono Variable", monospace'
  ctx.fillStyle = 'rgba(255,255,255,0.55)'
  const hud = props.mode === 'apriltag' ? `tag36h11  ${objs.length} found`
    : props.mode === 'depth' ? `depth-map  ${objs.length} obs`
    : `yolo-world  ${objs.length} det`
  ctx.fillText(hud, 8, 14)
  // pulsing record dot
  const a = 0.4 + 0.4 * Math.abs(Math.sin(t / 250))
  ctx.fillStyle = `rgba(225, 29, 72, ${a})`
  ctx.beginPath(); ctx.arc(W - 14, 12, 3.5, 0, Math.PI * 2); ctx.fill()
  ctx.fillStyle = 'rgba(255,255,255,0.7)'
  ctx.fillText('REC', W - 38, 15)
}

// ---------- emit detection events ----------
let lastEmit = 0
function emitDetections(t: number) {
  if (!props.onDetect) return
  if (t - lastEmit < 1500) return
  lastEmit = t
  // pick a random one to emit so log feels live
  const o = objs[Math.floor(Math.random() * objs.length)]
  if (!o) return
  const conf = 0.78 + Math.random() * 0.20
  props.onDetect({
    id: `${o.id}-${(t | 0)}`,
    camera_id: props.camera.id,
    cls: o.cls,
    conf,
    bbox: [o.x - o.w / 2, o.y - o.h / 2, o.w, o.h],
    tag_id: o.tag_id,
    at_ms: Date.now(),
  })
  detCount.value += 1
}

// ---------- raf loop ----------
let raf = 0
let lastT = 0
function frame(t: number) {
  const dt = lastT === 0 ? 16 : t - lastT
  lastT = t
  step(dt)
  const c = canvasRef.value
  const o = overlayRef.value
  if (c) {
    const ctx = c.getContext('2d')
    if (ctx) drawScene(ctx, t)
  }
  if (o) {
    const ctx = o.getContext('2d')
    if (ctx) drawOverlay(ctx, t)
  }
  if (props.camera.online) emitDetections(t)
  raf = requestAnimationFrame(frame)
}

onMounted(() => {
  seedObjects()
  if (canvasRef.value) {
    canvasRef.value.width = W
    canvasRef.value.height = H
  }
  if (overlayRef.value) {
    overlayRef.value.width = W
    overlayRef.value.height = H
  }
  raf = requestAnimationFrame(frame)
})
onBeforeUnmount(() => cancelAnimationFrame(raf))
watch(() => props.mode, () => seedObjects())
</script>

<template>
  <div class="cam-tile card-elevated">
    <div class="cam-head">
      <div class="cam-head-left">
        <span class="cam-icon">◉</span>
        <div>
          <div class="cam-label">{{ camera.label }}</div>
          <div class="cam-meta mono">{{ res }} · {{ fps.toFixed(0) }} fps</div>
        </div>
      </div>
      <span class="chip" :class="camera.online ? 'chip-ok' : 'chip-err'">
        <span class="dot" :class="camera.online ? 'dot-ok' : 'dot-err'"></span>
        {{ camera.online ? 'live' : 'offline' }}
      </span>
    </div>

    <div class="cam-view">
      <canvas ref="canvasRef" class="cam-canvas"></canvas>
      <canvas ref="overlayRef" class="cam-overlay"></canvas>
      <div v-if="!camera.online" class="cam-offline">
        <div class="off-mark">⊘</div>
        <div class="off-text">stream offline</div>
      </div>
    </div>

    <div class="cam-foot">
      <span class="foot-label">{{ mode === 'apriltag' ? 'AprilTag pose' : mode === 'depth' ? 'depth obstacle' : 'YOLO-World det' }}</span>
      <span class="foot-val mono">{{ detCount }} cum</span>
    </div>
  </div>
</template>

<style scoped>
.cam-tile {
  padding: 12px;
  display: flex; flex-direction: column; gap: 10px;
  transition: transform 0.22s var(--ease-out-quint);
}
.cam-tile:hover { transform: translateY(-2px); }
.cam-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.cam-head-left { display: flex; gap: 10px; align-items: center; min-width: 0; }
.cam-icon { font-size: 1.0rem; color: var(--ink-tertiary); }
.cam-label { font-size: 0.82rem; font-weight: 600; color: var(--ink-primary); }
.cam-meta { font-size: 0.66rem; color: var(--ink-muted); margin-top: 2px; }

.cam-view {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0a0e16;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05), 0 6px 18px -10px rgba(0,0,0,0.4);
}
.cam-canvas, .cam-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.cam-overlay { pointer-events: none; }
.cam-offline {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 6px; color: rgba(255,255,255,0.6);
}
.off-mark { font-size: 2rem; opacity: 0.6; }
.off-text { font-size: 0.78rem; }

.cam-foot { display: flex; justify-content: space-between; align-items: center; }
.foot-label { font-size: 0.68rem; color: var(--ink-tertiary); text-transform: uppercase; letter-spacing: 0.05em; }
.foot-val { font-size: 0.74rem; color: var(--ink-secondary); }
</style>
