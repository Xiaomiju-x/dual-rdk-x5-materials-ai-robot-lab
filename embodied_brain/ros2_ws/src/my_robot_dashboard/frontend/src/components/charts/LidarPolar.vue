<script setup lang="ts">
// LidarPolar (G3) — 雷达极坐标面板.
// 真数据: telemetry.packet.bridge.scan (桥在线, base frame 降采样点) → teal 实点.
// 桥离线: 用 /api/map.json 栅格从当前位姿做光线投射, 模拟"该位置应看到的扫描",
// 明确标"演示 · 地图光线投射" — 不冒充真雷达。两种来源图上分色 + 标签区分。
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const cv = ref<HTMLCanvasElement | null>(null)
const srcLabel = ref<'real' | 'sim' | 'none'>('none')

interface MapData { w: number; h: number; res: number; ox: number; oy: number; grid: Uint8Array }
let map: MapData | null = null
let mapTimer: number | null = null

async function fetchMap() {
  try {
    const r = await fetch('/api/map.json').then((x) => x.json())
    if (!r.ok) { map = null; return }
    const grid = new Uint8Array(r.w * r.h)
    let pos = 0
    for (let i = 0; i < r.rle.length; i += 2) {
      grid.fill(r.rle[i], pos, pos + r.rle[i + 1])
      pos += r.rle[i + 1]
    }
    map = { w: r.w, h: r.h, res: r.res, ox: r.ox, oy: r.oy, grid }
  } catch { /* noop */ }
}

function occupied(wx: number, wy: number): boolean {
  if (!map) return false
  const c = Math.floor((wx - map.ox) / map.res)
  const r = Math.floor((wy - map.oy) / map.res)
  if (c < 0 || r < 0 || c >= map.w || r >= map.h) return false
  return map.grid[r * map.w + c] >= 65
}

/** 桥离线时: 从位姿向 72 个方向步进光线投射地图, 返回 base frame 点 */
function simScan(): Array<[number, number]> {
  const pkt = telemetry.packet
  if (!pkt || !map) return []
  const { x, y, yaw } = pkt.pose
  const pts: Array<[number, number]> = []
  const RMAX = 4.0, STEP = 0.04
  for (let i = 0; i < 72; i++) {
    const a = (i / 72) * Math.PI * 2
    const wa = yaw + a
    for (let r = 0.15; r <= RMAX; r += STEP) {
      if (occupied(x + r * Math.cos(wa), y + r * Math.sin(wa))) {
        pts.push([r * Math.cos(a), r * Math.sin(a)])
        break
      }
    }
  }
  return pts
}

function draw() {
  const c = cv.value
  if (!c) return
  const ctx = c.getContext('2d')!
  const W = c.width, H = c.height
  const cx = W / 2, cy = H / 2
  const RMAX = 4.0
  const scale = (Math.min(W, H) / 2 - 22) / RMAX
  ctx.clearRect(0, 0, W, H)
  // 距离环 1-4m
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.18)'
  ctx.fillStyle = 'rgba(100, 116, 139, 0.6)'
  ctx.font = '10px sans-serif'
  ctx.lineWidth = 1
  for (let r = 1; r <= RMAX; r++) {
    ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, Math.PI * 2); ctx.stroke()
    ctx.fillText(`${r}m`, cx + r * scale - 16, cy - 4)
  }
  // 十字
  ctx.beginPath(); ctx.moveTo(cx - RMAX * scale, cy); ctx.lineTo(cx + RMAX * scale, cy)
  ctx.moveTo(cx, cy - RMAX * scale); ctx.lineTo(cx, cy + RMAX * scale); ctx.stroke()

  const real = telemetry.packet?.bridge?.scan
  let pts: Array<[number, number]>
  if (real && real.length) { pts = real; srcLabel.value = 'real' }
  else { pts = simScan(); srcLabel.value = pts.length ? 'sim' : 'none' }

  ctx.fillStyle = srcLabel.value === 'real' ? 'rgba(8, 145, 178, 0.85)' : 'rgba(217, 119, 6, 0.75)'
  for (const [px, py] of pts) {
    // base frame: x 前 (画布上), y 左 (画布左)
    const dx = -py * scale, dy = -px * scale
    if (Math.hypot(px, py) > RMAX) continue
    ctx.beginPath(); ctx.arc(cx + dx, cy + dy, 2, 0, Math.PI * 2); ctx.fill()
  }
  // 车体朝向箭头 (始终向上 = base frame 前方)
  ctx.fillStyle = '#2563eb'
  ctx.beginPath()
  ctx.moveTo(cx, cy - 10); ctx.lineTo(cx - 6, cy + 7); ctx.lineTo(cx + 6, cy + 7)
  ctx.closePath(); ctx.fill()
}

let raf = 0
function tick() { draw(); raf = window.setTimeout(tick, 500) as unknown as number }

onMounted(() => {
  fetchMap()
  mapTimer = window.setInterval(fetchMap, 15_000)
  tick()
})
onBeforeUnmount(() => {
  if (mapTimer !== null) window.clearInterval(mapTimer)
  window.clearTimeout(raf)
})
watch(() => telemetry.packet?.bridge?.scan, draw)
</script>

<template>
  <div class="polar-wrap">
    <div class="polar-head">
      <span class="section-label">LD14 雷达 · 极坐标</span>
      <span
        class="chip" :class="srcLabel === 'real' ? 'chip-ok' : srcLabel === 'sim' ? 'chip-warn' : 'chip-info'"
      >
        {{ srcLabel === 'real' ? '● 真机点云' : srcLabel === 'sim' ? '⚠ 演示 · 地图光线投射' : '○ 无数据' }}
      </span>
    </div>
    <canvas ref="cv" width="340" height="300" class="polar-cv"></canvas>
  </div>
</template>

<style scoped>
.polar-wrap { display: flex; flex-direction: column; gap: 8px; }
.polar-head { display: flex; align-items: center; justify-content: space-between; }
.polar-cv { width: 100%; max-width: 380px; margin: 0 auto; }
</style>
