<script setup lang="ts">
// LiveMap · 实战地图 — SLAM 真地图 + 语义地标 + 虚拟围栏 + 安全层 (第 3 期 #2 + #5)
// 地图: GET /api/map.json (RLE) · 地标: /api/landmarks · 围栏/限速/急停: /api/safety*
import { computed, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useToastStore } from '@/stores/toast'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import Map3D from '@/components/three/Map3D.vue'

const telemetry = useTelemetryStore()
const toasts = useToastStore()

const bridgeAlive = computed(() => telemetry.packet?.bridge?.alive ?? false)
const estop = computed(() => telemetry.packet?.bridge?.estop ?? false)
const safety = computed(() => telemetry.packet?.bridge?.safety ?? null)
const poseIsReal = computed(() => telemetry.packet?.real?.pose === true)

// ---------------- 地图数据 ----------------
interface MapMeta { w: number; h: number; res: number; ox: number; oy: number; etag: number }
const mapMeta = ref<MapMeta | null>(null)
let mapBitmap: HTMLCanvasElement | null = null   // 离屏原生分辨率
let lastEtag = -1
// G3: 2D/3D 切换 — 3D 用同一张栅格挤出墙体
const viewMode = ref<'2d' | '3d'>('2d')
const mapGrid = shallowRef<Uint8Array | null>(null)
const mapMock = ref(false)
const mapUnavailable = ref(false)
const pose3d = computed(() => telemetry.packet?.pose ?? null)
const poseSourceLabel = computed(() => poseIsReal.value ? ' (live odom)' : bridgeAlive.value ? ' (unavailable)' : ' (fixture)')

async function fetchMap() {
  try {
    const r = await fetch('/api/map.json').then((x) => x.json())
    if (!r.ok) {
      mapMeta.value = null
      mapGrid.value = null
      mapBitmap = null
      mapMock.value = false
      mapUnavailable.value = !!r.unavailable
      draw()
      return
    }
    if (r.etag === lastEtag) return
    lastEtag = r.etag
    mapMock.value = !!r.mock
    mapUnavailable.value = false
    const { w, h, rle } = r as MapMeta & { rle: number[] }
    const grid = new Uint8Array(w * h)
    let pos = 0
    for (let i = 0; i < rle.length; i += 2) {
      const v = rle[i], run = rle[i + 1]
      grid.fill(v, pos, pos + run)
      pos += run
    }
    // 渲到离屏 canvas (行 0 在 oy → 画到底部, 翻转 y)
    const off = document.createElement('canvas')
    off.width = w; off.height = h
    const ictx = off.getContext('2d')!
    const img = ictx.createImageData(w, h)
    for (let row = 0; row < h; row++) {
      const cy = h - 1 - row
      for (let col = 0; col < w; col++) {
        const v = grid[row * w + col]
        const o = (cy * w + col) * 4
        if (v === 255) { img.data[o] = 241; img.data[o + 1] = 245; img.data[o + 2] = 249; img.data[o + 3] = 255 }      // unknown 浅灰
        else if (v >= 65) { img.data[o] = 30; img.data[o + 1] = 41; img.data[o + 2] = 59; img.data[o + 3] = 255 }       // occupied 深蓝黑
        else { img.data[o] = 255; img.data[o + 1] = 255; img.data[o + 2] = 255; img.data[o + 3] = 255 }                  // free 白
      }
    }
    ictx.putImageData(img, 0, 0)
    mapBitmap = off
    mapGrid.value = grid
    mapMeta.value = { w: r.w, h: r.h, res: r.res, ox: r.ox, oy: r.oy, etag: r.etag }
    draw()
  } catch { /* 桥离线 */ }
}

// etag 变了就拉新图; 兜底 10s 轮询
watch(() => telemetry.packet?.bridge?.map_etag, (e) => { if (e !== undefined && e !== lastEtag) fetchMap() })

// ---------------- 视图变换 ----------------
const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
// 无地图时默认视野 ±4m
const view = { scale: 60, cx: 0, cy: 0 }   // px/m, 世界中心

function fitView() {
  const cv = canvasRef.value
  if (!cv) return
  const m = mapMeta.value
  if (m) {
    const mw = m.w * m.res, mh = m.h * m.res
    view.scale = Math.min(cv.width / mw, cv.height / mh) * 0.92
    view.cx = m.ox + mw / 2
    view.cy = m.oy + mh / 2
  } else {
    view.scale = Math.min(cv.width, cv.height) / 8
    view.cx = telemetry.packet?.pose.x ?? 0
    view.cy = telemetry.packet?.pose.y ?? 0
  }
}
function w2c(x: number, y: number): [number, number] {
  const cv = canvasRef.value!
  return [cv.width / 2 + (x - view.cx) * view.scale, cv.height / 2 - (y - view.cy) * view.scale]
}
function c2w(px: number, py: number): [number, number] {
  const cv = canvasRef.value!
  return [view.cx + (px - cv.width / 2) / view.scale, view.cy - (py - cv.height / 2) / view.scale]
}

// ---------------- 地标 ----------------
interface Landmark { name: string; x: number; y: number; source: string; created_at: string }
const landmarks = ref<Landmark[]>([])
async function refreshLandmarks() {
  try {
    const r = await fetch('/api/landmarks').then((x) => x.json())
    landmarks.value = r.landmarks ?? []
  } catch { /* noop */ }
}
const newName = ref('')
async function addLandmarkHere() {
  if (!newName.value.trim()) { toasts.push({ tone: 'warn', title: '先输入地标名' }); return }
  const r = await fetch('/api/landmarks', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: newName.value.trim() }),
  }).then((x) => x.json())
  if (r.ok) { toasts.push({ tone: 'ok', title: `已记住当前位置 → ${newName.value}` }); newName.value = ''; refreshLandmarks() }
  else toasts.push({ tone: 'err', title: '失败', detail: r.error })
}
async function delLandmark(n: string) {
  await fetch(`/api/landmarks/${encodeURIComponent(n)}`, { method: 'DELETE' })
  refreshLandmarks()
}
const goingTo = ref('')
async function gotoLandmark(n: string) {
  goingTo.value = n
  toasts.push({ tone: 'info', title: `🧭 直线低速导航 → ${n}` })
  try {
    const r = await fetch(`/api/landmarks/${encodeURIComponent(n)}/goto`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'direct' }),
    }).then((x) => x.json())
    toasts.push(r.ok ? { tone: 'ok', title: `已到达 ${n}` } : { tone: 'err', title: '导航失败', detail: r.error })
  } finally { goingTo.value = '' }
}

// ---------------- 交互模式 ----------------
type Mode = 'view' | 'landmark' | 'fence'
const mode = ref<Mode>('view')
const fenceDraft = ref<Array<[number, number]>>([])

function onCanvasClick(ev: MouseEvent) {
  const cv = canvasRef.value
  if (!cv) return
  const rect = cv.getBoundingClientRect()
  const px = (ev.clientX - rect.left) * (cv.width / rect.width)
  const py = (ev.clientY - rect.top) * (cv.height / rect.height)
  const [wx, wy] = c2w(px, py)
  if (mode.value === 'fence') {
    fenceDraft.value.push([Number(wx.toFixed(3)), Number(wy.toFixed(3))])
    draw()
  } else if (mode.value === 'landmark') {
    const n = window.prompt(`在 (${wx.toFixed(2)}, ${wy.toFixed(2)}) 加地标, 名字:`)
    if (n?.trim()) {
      fetch('/api/landmarks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: n.trim(), x: wx, y: wy }),
      }).then((x) => x.json()).then((r) => {
        if (r.ok) refreshLandmarks()
        else toasts.push({ tone: 'err', title: '失败', detail: r.error })
      })
    }
  }
}

// ---------------- 安全层 ----------------
const speedCap = ref(0.2)
watch(safety, (s) => { if (s) speedCap.value = s.speed_cap }, { immediate: true })

async function postSafety(body: Record<string, unknown>, okMsg: string) {
  const r = await fetch('/api/safety', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((x) => x.json())
  toasts.push(r.ok ? { tone: 'ok', title: okMsg } : { tone: 'err', title: '失败', detail: r.error })
}
function saveFence() {
  if (fenceDraft.value.length < 3) { toasts.push({ tone: 'warn', title: '围栏至少 3 个点' }); return }
  postSafety({ fence: fenceDraft.value }, `围栏已生效 (${fenceDraft.value.length} 点)`)
  fenceDraft.value = []
  mode.value = 'view'
}
function clearFence() { postSafety({ fence: [] }, '围栏已清除') }
function undoFencePt() { fenceDraft.value.pop(); draw() }
async function doEstop() {
  const r = await fetch('/api/safety/estop', { method: 'POST' }).then((x) => x.json())
  toasts.push(r.ok ? { tone: 'err', title: '🟥 急停已置位' } : { tone: 'err', title: '急停失败', detail: r.error })
}
async function clearEstop() {
  const r = await fetch('/api/safety/clear_estop', { method: 'POST' }).then((x) => x.json())
  toasts.push(r.ok ? { tone: 'ok', title: '急停已解除' } : { tone: 'err', title: '失败', detail: r.error })
}

// ---------------- 绘制 ----------------
const trail: Array<[number, number]> = []
function draw() {
  const cv = canvasRef.value
  if (!cv) return
  const ctx = cv.getContext('2d')!
  ctx.clearRect(0, 0, cv.width, cv.height)

  // 地图位图
  const m = mapMeta.value
  if (m && mapBitmap) {
    const [x0, y0] = w2c(m.ox, m.oy + m.h * m.res)     // 左上角 (世界左上 = ox, oy+mh)
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(mapBitmap, x0, y0, m.w * m.res * view.scale, m.h * m.res * view.scale)
  }

  // 1m 网格
  ctx.strokeStyle = 'rgba(37, 99, 235, 0.07)'
  ctx.lineWidth = 1
  const [wl, wt] = c2w(0, 0), [wr, wb] = c2w(cv.width, cv.height)
  for (let gx = Math.ceil(wl); gx <= Math.floor(wr); gx++) {
    const [px] = w2c(gx, 0)
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, cv.height); ctx.stroke()
  }
  for (let gy = Math.ceil(wb); gy <= Math.floor(wt); gy++) {
    const [, py] = w2c(0, gy)
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(cv.width, py); ctx.stroke()
  }

  // 生效中的围栏 (紫)
  const fence = safety.value?.fence
  if (fence && fence.length >= 3) {
    ctx.beginPath()
    fence.forEach((p, i) => {
      const [px, py] = w2c(p[0], p[1])
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    ctx.closePath()
    ctx.fillStyle = safety.value?.fence_enabled ? 'rgba(124, 58, 237, 0.08)' : 'rgba(148, 163, 184, 0.08)'
    ctx.fill()
    ctx.strokeStyle = safety.value?.fence_enabled ? 'rgba(124, 58, 237, 0.7)' : 'rgba(148, 163, 184, 0.6)'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([])
  }

  // 围栏草稿 (橙)
  if (fenceDraft.value.length) {
    ctx.beginPath()
    fenceDraft.value.forEach((p, i) => {
      const [px, py] = w2c(p[0], p[1])
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    ctx.strokeStyle = 'rgba(217, 119, 6, 0.9)'
    ctx.lineWidth = 2
    ctx.stroke()
    for (const p of fenceDraft.value) {
      const [px, py] = w2c(p[0], p[1])
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fillStyle = '#d97706'; ctx.fill()
    }
  }

  // 雷达点 (真数据, teal)
  const pkt = telemetry.packet
  const scan = pkt?.bridge?.scan
  if (scan && pkt) {
    const { x, y, yaw } = pkt.pose
    ctx.fillStyle = 'rgba(8, 145, 178, 0.55)'
    for (const [sx, sy] of scan) {
      const wx2 = x + sx * Math.cos(yaw) - sy * Math.sin(yaw)
      const wy2 = y + sx * Math.sin(yaw) + sy * Math.cos(yaw)
      const [px, py] = w2c(wx2, wy2)
      ctx.fillRect(px - 1.2, py - 1.2, 2.4, 2.4)
    }
  }

  // 地标
  ctx.font = '12px sans-serif'
  for (const lm of landmarks.value) {
    const [px, py] = w2c(lm.x, lm.y)
    ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(5, 150, 105, 0.18)'; ctx.fill()
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2)
    ctx.fillStyle = '#059669'; ctx.fill()
    ctx.fillStyle = '#047857'
    ctx.fillText(lm.name, px + 9, py + 4)
  }

  // 轨迹
  if (trail.length > 1) {
    ctx.beginPath()
    trail.forEach((p, i) => {
      const [px, py] = w2c(p[0], p[1])
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    ctx.strokeStyle = 'rgba(37, 99, 235, 0.35)'
    ctx.lineWidth = 1.5
    ctx.stroke()
  }

  // 车体 (三角 + 朝向)
  if (pkt) {
    const { x, y, yaw } = pkt.pose
    const [px, py] = w2c(x, y)
    ctx.save()
    ctx.translate(px, py)
    ctx.rotate(-yaw)   // 画布 y 翻转 → 角度取反
    ctx.beginPath()
    ctx.moveTo(11, 0); ctx.lineTo(-7, 6.5); ctx.lineTo(-7, -6.5)
    ctx.closePath()
    ctx.fillStyle = estop.value ? '#e11d48' : poseIsReal.value ? '#2563eb' : '#94a3b8'
    ctx.shadowColor = ctx.fillStyle as string
    ctx.shadowBlur = 10
    ctx.fill()
    ctx.restore()
  }
}

// 位姿更新 → 轨迹 + 重绘
watch(() => telemetry.packet?.pose, (p) => {
  if (!p) return
  const last = trail[trail.length - 1]
  if (!last || Math.hypot(p.x - last[0], p.y - last[1]) > 0.02) {
    trail.push([p.x, p.y])
    if (trail.length > 1500) trail.splice(0, trail.length - 1500)
  }
  draw()
})
watch(landmarks, draw)

let mapTimer: number | null = null
let resizeObs: ResizeObserver | null = null
onMounted(() => {
  const cv = canvasRef.value!
  const fit = () => {
    const r = wrapRef.value!.getBoundingClientRect()
    cv.width = Math.floor(r.width)
    cv.height = Math.max(420, Math.floor(window.innerHeight - r.top - 120))
    fitView()
    draw()
  }
  resizeObs = new ResizeObserver(fit)
  resizeObs.observe(wrapRef.value!)
  fit()
  fetchMap()
  refreshLandmarks()
  mapTimer = window.setInterval(fetchMap, 10_000)
})
onUnmounted(() => {
  if (mapTimer !== null) window.clearInterval(mapTimer)
  resizeObs?.disconnect()
})
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title"><KineticTitle text="LiveMap · 实战地图" gradient="teal-emerald" /></h1>
        <p class="page-subtitle">
          SLAM 真地图 + 语义地标按名导航 + 虚拟围栏 + 速度上限
          <span class="chip" :class="bridgeAlive ? 'chip-ok' : 'chip-warn'" style="margin-left: 8px;">
            {{ bridgeAlive ? '● ROS 桥在线' : '○ 桥离线 (fixture mode)' }}
          </span>
          <span v-if="estop" class="chip chip-err" style="margin-left: 6px;">🟥 急停置位中</span>
        </p>
      </div>
      <div class="header-actions">
        <div class="seg">
          <button class="seg-btn" :class="{ active: viewMode === '2d' }" @click="viewMode = '2d'">🗺 2D</button>
          <button class="seg-btn" :class="{ active: viewMode === '3d' }" @click="viewMode = '3d'">⛰ 3D 实景</button>
        </div>
        <div v-if="viewMode === '2d'" class="seg">
          <button class="seg-btn" :class="{ active: mode === 'view' }" @click="mode = 'view'">👁 查看</button>
          <button class="seg-btn" :class="{ active: mode === 'landmark' }" @click="mode = 'landmark'">📍 点图加地标</button>
          <button class="seg-btn" :class="{ active: mode === 'fence' }" @click="mode = 'fence'">🚧 画围栏</button>
        </div>
        <button v-if="!estop" class="btn btn-danger" @click="doEstop">🟥 急停</button>
        <button v-else class="btn btn-ok" @click="clearEstop">✓ 解除急停</button>
      </div>
    </header>

    <div class="map-grid">
      <div ref="wrapRef" class="card-elevated map-wrap">
        <canvas v-show="viewMode === '2d'" ref="canvasRef" class="map-canvas" :class="`cursor-${mode}`" @click="onCanvasClick"></canvas>
        <Map3D
          v-if="viewMode === '3d'"
          :grid="mapGrid" :meta="mapMeta" :pose="pose3d" :landmarks="landmarks"
        />
        <span v-if="mapMock" class="chip chip-warn mock-chip">⚠ 演示地图 (桥离线, 程序生成) — SLAM 上线即换真图</span>
        <span v-else-if="mapUnavailable" class="chip chip-warn mock-chip">/map unavailable · no fixture backfill</span>
        <div v-if="mode === 'fence' && viewMode === '2d'" class="fence-bar">
          <span class="mono">围栏草稿 {{ fenceDraft.length }} 点 (点图加点, ≥3 可保存)</span>
          <button class="mini-btn" :disabled="!fenceDraft.length" @click="undoFencePt">↶ 撤点</button>
          <button class="mini-btn" :disabled="fenceDraft.length < 3" @click="saveFence">💾 生效</button>
          <button class="mini-btn" @click="fenceDraft = []; draw()">✕ 放弃</button>
        </div>
        <div class="legend mono">
          <span><i class="lg lg-robot"></i>车体{{ poseSourceLabel }}</span>
          <span><i class="lg lg-scan"></i>雷达</span>
          <span><i class="lg lg-lm"></i>地标</span>
          <span><i class="lg lg-fence"></i>围栏</span>
        </div>
      </div>

      <div class="side">
        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">语义地标</span><span class="chip chip-info">{{ landmarks.length }}</span></div>
          <div class="add-row">
            <input v-model="newName" class="name-input" placeholder="记住当前位置为…" @keyup.enter="addLandmarkHere" />
            <button class="btn" :disabled="!bridgeAlive" @click="addLandmarkHere">📌 记住这里</button>
          </div>
          <div v-if="!landmarks.length" class="empty mono">还没有地标 — 开车到位后点"记住这里", 或切"点图加地标"模式</div>
          <div v-for="lm in landmarks" :key="lm.name" class="lm-row">
            <span class="lm-name">{{ lm.name }}</span>
            <span class="lm-xy mono">({{ lm.x.toFixed(2) }}, {{ lm.y.toFixed(2) }})</span>
            <button class="mini-btn" :disabled="!bridgeAlive || !!goingTo" @click="gotoLandmark(lm.name)">
              {{ goingTo === lm.name ? '🧭…' : '🧭 去' }}
            </button>
            <button class="mini-btn mini-del" @click="delLandmark(lm.name)">🗑</button>
          </div>
        </div>

        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">安全层</span></div>
          <div class="safe-row">
            <span class="safe-label">速度上限</span>
            <input v-model.number="speedCap" type="range" min="0.05" max="0.5" step="0.01" class="slider" />
            <span class="mono safe-val">{{ speedCap.toFixed(2) }} m/s</span>
            <button class="mini-btn" :disabled="!bridgeAlive" @click="postSafety({ speed_cap: speedCap }, `限速 ${speedCap.toFixed(2)} m/s 已生效`)">设</button>
          </div>
          <div class="safe-row">
            <span class="safe-label">虚拟围栏</span>
            <span class="mono safe-val">{{ safety?.fence ? `${safety.fence.length} 点 · ${safety.fence_enabled ? '启用' : '停用'}` : '未设置' }}</span>
            <button
v-if="safety?.fence" class="mini-btn" :disabled="!bridgeAlive"
                    @click="postSafety({ fence_enabled: !safety!.fence_enabled }, safety!.fence_enabled ? '围栏已停用' : '围栏已启用')">
              {{ safety!.fence_enabled ? '停用' : '启用' }}
            </button>
            <button v-if="safety?.fence" class="mini-btn mini-del" :disabled="!bridgeAlive" @click="clearFence">清</button>
          </div>
          <p class="safe-note">围栏与限速在<b>车端桥内强制执行</b> (independent of UI) — goto/forward/twist 越界即拒, 限速钳全部运动命令。</p>
        </div>

        <div v-if="telemetry.packet?.bridge?.detections?.length" class="card-elevated panel">
          <div class="panel-head"><span class="section-label">BPU 实时检测</span></div>
          <div class="det-chips">
            <span v-for="(d, i) in telemetry.packet!.bridge!.detections.slice(0, 10)" :key="i" class="chip chip-info">
              {{ d.label }} {{ (d.conf * 100).toFixed(0) }}%
            </span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.map-grid { display: grid; grid-template-columns: 1fr 330px; gap: 16px; align-items: start; }
@media (max-width: 1100px) { .map-grid { grid-template-columns: 1fr; } }

.map-wrap { position: relative; padding: 10px; overflow: hidden; }
.mock-chip { position: absolute; top: 16px; left: 16px; z-index: 3; }
.map-canvas { display: block; width: 100%; border-radius: 10px; background: linear-gradient(160deg, #fdfefe, #f4f7fb); }
.cursor-landmark { cursor: crosshair; }
.cursor-fence { cursor: crosshair; }

.fence-bar {
  position: absolute; top: 18px; left: 18px;
  display: flex; align-items: center; gap: 8px;
  background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(8px);
  border: 1px solid var(--line-border); border-radius: 9px;
  padding: 7px 12px; font-size: 0.72rem;
  box-shadow: 0 4px 14px -4px rgba(15, 23, 42, 0.18);
}
.legend {
  position: absolute; bottom: 18px; left: 18px;
  display: flex; gap: 14px; font-size: 0.66rem; color: var(--ink-tertiary);
  background: rgba(255, 255, 255, 0.85); border-radius: 8px; padding: 5px 10px;
}
.lg { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 4px; vertical-align: -1px; }
.lg-robot { background: #2563eb; }
.lg-scan  { background: rgba(8, 145, 178, 0.7); }
.lg-lm    { background: #059669; border-radius: 50%; }
.lg-fence { background: rgba(124, 58, 237, 0.6); }

.side { display: flex; flex-direction: column; gap: 14px; }
.panel { padding: 14px; }
.panel-head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }

.add-row { display: flex; gap: 8px; margin-bottom: 10px; }
.name-input {
  flex: 1; min-width: 0; border: 1px solid var(--line-border); border-radius: 7px;
  padding: 6px 10px; font-size: 0.78rem; font-family: inherit; background: rgba(255,255,255,0.85);
}
.lm-row { display: flex; align-items: center; gap: 8px; padding: 6px 2px; border-bottom: 1px dashed var(--line-divider); }
.lm-row:last-child { border-bottom: 0; }
.lm-name { font-size: 0.8rem; font-weight: 600; color: var(--ink-primary); }
.lm-xy { font-size: 0.66rem; color: var(--ink-muted); margin-left: auto; }

.safe-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; }
.safe-label { font-size: 0.76rem; color: var(--ink-secondary); width: 64px; flex-shrink: 0; }
.safe-val { font-size: 0.7rem; color: var(--ink-tertiary); }
.slider { flex: 1; accent-color: var(--accent-blue); }
.safe-note { font-size: 0.66rem; color: var(--ink-muted); margin: 8px 0 0; line-height: 1.5; }

.det-chips { display: flex; flex-wrap: wrap; gap: 6px; }

.btn {
  border: 1px solid var(--line-border); background: var(--bg-elevated); border-radius: 8px;
  padding: 7px 13px; cursor: pointer; font-size: 0.78rem; font-weight: 600;
  font-family: inherit; color: var(--ink-secondary);
  transition: all 0.15s var(--ease-out-quint);
}
.btn:hover:not(:disabled) { transform: translateY(-1px); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-danger { border-color: rgba(225, 29, 72, 0.4); color: #e11d48; background: rgba(225, 29, 72, 0.05); }
.btn-ok { border-color: rgba(5, 150, 105, 0.4); color: #059669; background: rgba(5, 150, 105, 0.05); }

.mini-btn {
  border: 1px solid var(--line-border); background: white; border-radius: 6px;
  padding: 3px 8px; cursor: pointer; font-size: 0.7rem; font-family: inherit;
}
.mini-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.mini-btn:hover:not(:disabled) { border-color: rgba(37, 99, 235, 0.4); }
.mini-del:hover { border-color: rgba(225, 29, 72, 0.5); }

.empty { color: var(--ink-muted); font-size: 0.72rem; padding: 6px 2px; }

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 999px; padding: 2px 9px; font-size: 0.66rem; font-weight: 700;
}
.chip-ok   { background: rgba(5, 150, 105, 0.10); color: #059669; }
.chip-warn { background: rgba(217, 119, 6, 0.10); color: #d97706; }
.chip-err  { background: rgba(225, 29, 72, 0.10); color: #e11d48; animation: pulseSoft 1.4s infinite; }
.chip-info { background: rgba(37, 99, 235, 0.10); color: #2563eb; }
</style>
