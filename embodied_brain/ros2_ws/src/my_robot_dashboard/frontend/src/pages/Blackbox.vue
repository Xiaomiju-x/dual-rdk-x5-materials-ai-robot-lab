<script setup lang="ts">
// Blackbox · 黑匣子 — 行车记录回放 (第 3 期 #4)
// 数据源: cockpit_bridge ~/blackbox/bb-YYYYMMDD.jsonl (1Hz 遥测 + 事件), 7 天滚动.
// API: /api/blackbox/days + /api/blackbox/window
import { computed, onMounted, ref, watch } from 'vue'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import TimeSeries from '@/components/charts/TimeSeries.vue'
import type { HistorySample } from '@/stores/telemetry'

interface TelRec {
  t: number
  pose: { x: number; y: number; yaw: number } | null
  vel: { linear: number; angular: number } | null
  estop: boolean
  cpu: number | null
  dets: number
}
interface EvRec { t: number; k: string; [key: string]: unknown }

const days = ref<Array<{ day: string; size_kb: number }>>([])
const day = ref('')
const loading = ref(false)
const loadErr = ref('')
const tel = ref<TelRec[]>([])
const events = ref<EvRec[]>([])

async function refreshDays() {
  try {
    const r = await fetch('/api/blackbox/days').then((x) => x.json())
    days.value = r.days ?? []
    if (!day.value && days.value.length) day.value = days.value[days.value.length - 1].day
  } catch { /* noop */ }
}

async function loadDay() {
  if (!day.value) return
  loading.value = true
  loadErr.value = ''
  try {
    const r = await fetch(`/api/blackbox/window?day=${day.value}&max_points=2000`).then((x) => x.json())
    if (!r.ok) { loadErr.value = r.error; tel.value = []; events.value = []; return }
    tel.value = (r.telemetry ?? []) as TelRec[]
    events.value = (r.events ?? []) as EvRec[]
    if (tel.value.length) scrub.value = tel.value[tel.value.length - 1].t
    drawTrail()
  } catch (e) {
    loadErr.value = (e as Error).message
  } finally {
    loading.value = false
  }
}
watch(day, loadDay)

// ---------------- 时间范围 + 滑块 ----------------
const t0 = computed(() => (tel.value.length ? tel.value[0].t : 0))
const t1 = computed(() => (tel.value.length ? tel.value[tel.value.length - 1].t : 1))
const scrub = ref(0)

function fmtT(t: number): string {
  return new Date(t * 1000).toLocaleTimeString('zh-CN', { hour12: false })
}

// 滑块时刻对应的遥测帧 (二分)
const frameAt = computed<TelRec | null>(() => {
  const arr = tel.value
  if (!arr.length) return null
  let lo = 0, hi = arr.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (arr[mid].t <= scrub.value) lo = mid
    else hi = mid - 1
  }
  return arr[lo]
})

// ---------------- 图表序列 ----------------
const speedSeries = computed(() => [{
  name: '线速度 m/s',
  samples: tel.value.filter((r) => r.vel).map((r) => ({ t: r.t * 1000, v: Math.abs(r.vel!.linear) })) as HistorySample[],
  accent: 'teal' as const,
}])
const cpuSeries = computed(() => [{
  name: 'CPU %',
  samples: tel.value.filter((r) => r.cpu != null).map((r) => ({ t: r.t * 1000, v: r.cpu! })) as HistorySample[],
  accent: 'violet' as const,
  target: 80,
}])

// ---------------- 轨迹画布 ----------------
const trailRef = ref<HTMLCanvasElement | null>(null)
function drawTrail() {
  const cv = trailRef.value
  if (!cv) return
  const ctx = cv.getContext('2d')!
  ctx.clearRect(0, 0, cv.width, cv.height)
  const pts = tel.value.filter((r) => r.pose).map((r) => ({ t: r.t, x: r.pose!.x, y: r.pose!.y, yaw: r.pose!.yaw }))
  if (!pts.length) {
    ctx.fillStyle = '#94a3b8'
    ctx.font = '12px sans-serif'
    ctx.fillText('该日无位姿记录', 16, 28)
    return
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  const spanX = Math.max(0.5, maxX - minX), spanY = Math.max(0.5, maxY - minY)
  const scale = Math.min((cv.width - 40) / spanX, (cv.height - 40) / spanY)
  const ox = (cv.width - spanX * scale) / 2 - minX * scale
  const oy = (cv.height + spanY * scale) / 2 + minY * scale
  const px = (x: number) => ox + x * scale
  const py = (y: number) => oy - y * scale

  // 全程淡轨迹
  ctx.beginPath()
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(px(p.x), py(p.y)) : ctx.lineTo(px(p.x), py(p.y))))
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)'
  ctx.lineWidth = 1.5
  ctx.stroke()

  // 已回放部分 (到 scrub)
  const upto = pts.filter((p) => p.t <= scrub.value)
  if (upto.length > 1) {
    ctx.beginPath()
    upto.forEach((p, i) => (i === 0 ? ctx.moveTo(px(p.x), py(p.y)) : ctx.lineTo(px(p.x), py(p.y))))
    ctx.strokeStyle = 'rgba(37, 99, 235, 0.8)'
    ctx.lineWidth = 2.5
    ctx.stroke()
  }

  // 事件钉 (alarm 红 / command 橙 / mission 紫)
  for (const e of events.value) {
    const near = pts.reduce((a, b) => (Math.abs(b.t - e.t) < Math.abs(a.t - e.t) ? b : a), pts[0])
    if (Math.abs(near.t - e.t) > 10) continue
    ctx.beginPath()
    ctx.arc(px(near.x), py(near.y), 3.4, 0, Math.PI * 2)
    ctx.fillStyle = e.k === 'alarm' ? '#e11d48' : e.k === 'mission' ? '#7c3aed' : '#d97706'
    ctx.fill()
  }

  // 当前帧车标
  const f = frameAt.value
  if (f?.pose) {
    ctx.save()
    ctx.translate(px(f.pose.x), py(f.pose.y))
    ctx.rotate(-f.pose.yaw)
    ctx.beginPath()
    ctx.moveTo(10, 0); ctx.lineTo(-6, 5.5); ctx.lineTo(-6, -5.5)
    ctx.closePath()
    ctx.fillStyle = f.estop ? '#e11d48' : '#2563eb'
    ctx.shadowColor = ctx.fillStyle as string
    ctx.shadowBlur = 9
    ctx.fill()
    ctx.restore()
  }
}
watch(scrub, drawTrail)

// ---------------- 播放 (G3: 可选倍速) ----------------
const playing = ref(false)
const RATES = [1, 5, 10, 30] as const
const rate = ref<number>(10)
let playTimer: number | null = null
function togglePlay() {
  if (playing.value) {
    playing.value = false
    if (playTimer !== null) { window.clearInterval(playTimer); playTimer = null }
    return
  }
  if (scrub.value >= t1.value) scrub.value = t0.value
  playing.value = true
  playTimer = window.setInterval(() => {
    scrub.value = Math.min(t1.value, scrub.value + rate.value)   // rate× 速 (0.1s tick)
    if (scrub.value >= t1.value) togglePlay()
  }, 100)
}
function setRate(r: number) {
  rate.value = r
  if (playing.value) { togglePlay(); togglePlay() }   // 重启 interval 取新速
}

// ---------------- 事件列表 ----------------
type Kind = 'all' | 'alarm' | 'command' | 'result' | 'mission'
const kindFilter = ref<Kind>('all')
const filteredEvents = computed(() =>
  (kindFilter.value === 'all' ? events.value : events.value.filter((e) => e.k === kindFilter.value)).slice(-300).reverse(),
)
function evSummary(e: EvRec): string {
  if (e.k === 'alarm') return `[${e.severity}] ${e.title}: ${String(e.description ?? '').slice(0, 70)}`
  if (e.k === 'command') return `${e.cmd} ${JSON.stringify(e.args ?? {}).slice(0, 70)}`
  if (e.k === 'result') return `${e.ok ? '✓' : '✗'} ${String(e.error ?? '').slice(0, 70)}`
  if (e.k === 'mission') return `${e.event} ${String(e.name ?? e.msg ?? '').slice(0, 70)}`
  return JSON.stringify(e).slice(0, 80)
}
function jumpTo(e: EvRec) { scrub.value = e.t }

const KIND_META: Record<string, { label: string; cls: string }> = {
  alarm: { label: '报警', cls: 'ek-alarm' },
  command: { label: '命令', cls: 'ek-cmd' },
  result: { label: '回执', cls: 'ek-res' },
  mission: { label: '任务', cls: 'ek-mis' },
  tel: { label: '遥测', cls: 'ek-tel' },
}

onMounted(() => {
  refreshDays()
  const cv = trailRef.value
  if (cv) { cv.width = cv.clientWidth || 560; cv.height = 320 }
})
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title"><KineticTitle text="Blackbox · 黑匣子" gradient="amber-rose" /></h1>
        <p class="page-subtitle">
          行车记录回放 · 1Hz 遥测 + 报警/命令/任务事件 · 7 天滚动 · 轨迹时间轴拖动
        </p>
      </div>
      <div class="header-actions">
        <div class="seg">
          <button v-for="d in days" :key="d.day" class="seg-btn" :class="{ active: day === d.day }" @click="day = d.day">
            {{ d.day.slice(4, 6) }}-{{ d.day.slice(6) }} <span class="day-size mono">{{ d.size_kb }}K</span>
          </button>
        </div>
        <button class="btn" @click="refreshDays(); loadDay()">⟳</button>
      </div>
    </header>

    <div v-if="!days.length" class="card-elevated empty-state">
      <p>📼 还没有黑匣子记录 — cockpit_bridge 跑起来后自动以 1Hz 写入 <span class="mono">~/blackbox/bb-*.jsonl</span></p>
    </div>
    <div v-else-if="loadErr" class="card-elevated empty-state"><p>⚠ {{ loadErr }}</p></div>

    <template v-else>
      <!-- 回放控制条 -->
      <div class="card-elevated play-bar">
        <button class="play-btn" @click="togglePlay">{{ playing ? '⏸' : '▶' }}</button>
        <span class="rate-seg">
          <button
            v-for="r in RATES" :key="r" class="rate-btn" :class="{ active: rate === r }"
            @click="setRate(r)"
          >{{ r }}×</button>
        </span>
        <span class="mono play-t">{{ fmtT(scrub) }}</span>
        <input
          v-model.number="scrub" type="range"
          :min="t0" :max="t1" step="1" class="scrubber"
        />
        <span class="mono play-range">{{ fmtT(t0) }} → {{ fmtT(t1) }}</span>
        <span class="chip chip-info mono">{{ tel.length }} 帧 · {{ events.length }} 事件</span>
        <template v-if="frameAt">
          <span class="chip mono" :class="frameAt.estop ? 'chip-err' : 'chip-ok'">
            {{ frameAt.estop ? 'ESTOP' : 'OK' }}
          </span>
          <span v-if="frameAt.vel" class="chip chip-info mono">v={{ frameAt.vel.linear.toFixed(2) }}m/s</span>
          <span v-if="frameAt.cpu != null" class="chip chip-info mono">CPU {{ frameAt.cpu.toFixed(0) }}%</span>
          <span class="chip chip-info mono">{{ frameAt.dets }} 检测</span>
        </template>
      </div>

      <div class="bb-grid">
        <!-- 左: 轨迹回放 -->
        <div class="card-elevated panel">
          <div class="panel-head">
            <span class="section-label">轨迹回放</span>
            <span class="legend-mini mono">
              <i class="dot-mini" style="background:#e11d48"></i>报警
              <i class="dot-mini" style="background:#d97706"></i>命令
              <i class="dot-mini" style="background:#7c3aed"></i>任务
            </span>
          </div>
          <canvas ref="trailRef" class="trail-canvas"></canvas>
        </div>

        <!-- 右: 事件时间线 -->
        <div class="card-elevated panel">
          <div class="panel-head">
            <span class="section-label">事件时间线</span>
            <div class="seg seg-sm">
              <button v-for="k in (['all', 'alarm', 'command', 'result', 'mission'] as const)" :key="k"
                      class="seg-btn" :class="{ active: kindFilter === k }" @click="kindFilter = k">
                {{ k === 'all' ? '全部' : KIND_META[k]?.label }}
              </button>
            </div>
          </div>
          <div class="ev-list">
            <div v-if="!filteredEvents.length" class="empty mono">无事件</div>
            <button v-for="(e, i) in filteredEvents" :key="i" class="ev-row" @click="jumpTo(e)">
              <span class="ev-t mono">{{ fmtT(e.t) }}</span>
              <span class="ev-kind mono" :class="KIND_META[e.k]?.cls">{{ KIND_META[e.k]?.label ?? e.k }}</span>
              <span class="ev-sum">{{ evSummary(e) }}</span>
            </button>
          </div>
        </div>
      </div>

      <!-- 底: 双图表 -->
      <div class="bb-grid">
        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">速度曲线</span></div>
          <TimeSeries :series="speedSeries" y-label="m/s" height="180px" />
        </div>
        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">CPU 负载</span></div>
          <TimeSeries :series="cpuSeries" y-label="%" height="180px" />
        </div>
      </div>
    </template>

    <div v-if="loading" class="loading mono">读取 {{ day }} …</div>
  </section>
</template>

<style scoped>
.empty-state { padding: 36px; text-align: center; color: var(--ink-tertiary); font-size: 0.84rem; }

.play-bar {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; margin-bottom: 16px; flex-wrap: wrap;
}
.play-btn {
  width: 38px; height: 38px; border-radius: 50%; border: none; cursor: pointer;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white; font-size: 0.95rem;
  box-shadow: 0 3px 12px -3px rgba(37, 99, 235, 0.45);
  transition: transform 0.15s var(--ease-out-quint);
}
.play-btn:hover { transform: scale(1.07); }
.rate-seg { display: inline-flex; gap: 4px; }
.rate-btn {
  border: 1px solid var(--line-soft, rgba(15,23,42,.12)); background: transparent; cursor: pointer;
  border-radius: 999px; padding: 3px 10px; font-size: 0.68rem; font-weight: 800;
  color: var(--ink-secondary, #64748b); transition: all 0.15s;
}
.rate-btn.active {
  color: #fff; border-color: transparent;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
}
.play-t { font-size: 0.9rem; font-weight: 700; color: var(--ink-primary); min-width: 76px; }
.scrubber { flex: 1; min-width: 200px; accent-color: var(--accent-blue); }
.play-range { font-size: 0.68rem; color: var(--ink-muted); }

.bb-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; align-items: start; }
@media (max-width: 1100px) { .bb-grid { grid-template-columns: 1fr; } }

.panel { padding: 14px; }
.panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; justify-content: space-between; }

.trail-canvas {
  display: block; width: 100%; height: 320px; border-radius: 10px;
  background: linear-gradient(160deg, #fdfefe, #f4f7fb);
  border: 1px solid var(--line-divider);
}
.legend-mini { font-size: 0.64rem; color: var(--ink-muted); display: flex; align-items: center; gap: 6px; }
.dot-mini { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 2px; }

.ev-list { max-height: 320px; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
.ev-row {
  display: flex; align-items: center; gap: 9px; text-align: left;
  border: none; background: transparent; cursor: pointer;
  padding: 5px 7px; border-radius: 7px; font-family: inherit;
  transition: background 0.12s;
}
.ev-row:hover { background: rgba(37, 99, 235, 0.05); }
.ev-t { font-size: 0.66rem; color: var(--ink-muted); flex-shrink: 0; }
.ev-kind {
  font-size: 0.62rem; font-weight: 700; border-radius: 5px; padding: 1px 6px; flex-shrink: 0;
}
.ek-alarm { background: rgba(225, 29, 72, 0.10); color: #e11d48; }
.ek-cmd   { background: rgba(217, 119, 6, 0.10); color: #d97706; }
.ek-res   { background: rgba(8, 145, 178, 0.10); color: #0891b2; }
.ek-mis   { background: rgba(124, 58, 237, 0.10); color: #7c3aed; }
.ek-tel   { background: rgba(148, 163, 184, 0.14); color: #64748b; }
.ev-sum { font-size: 0.72rem; color: var(--ink-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.day-size { font-size: 0.6rem; opacity: 0.65; margin-left: 3px; }
.seg-sm .seg-btn { padding: 3px 8px; font-size: 0.66rem; }

.btn {
  border: 1px solid var(--line-border); background: var(--bg-elevated); border-radius: 8px;
  padding: 7px 12px; cursor: pointer; font-size: 0.78rem; font-family: inherit;
}
.empty { color: var(--ink-muted); font-size: 0.72rem; padding: 8px 4px; }
.loading { color: var(--ink-muted); font-size: 0.74rem; padding: 10px 2px; }

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 999px; padding: 2px 9px; font-size: 0.64rem; font-weight: 700;
}
.chip-ok   { background: rgba(5, 150, 105, 0.10); color: #059669; }
.chip-err  { background: rgba(225, 29, 72, 0.10); color: #e11d48; }
.chip-info { background: rgba(37, 99, 235, 0.10); color: #2563eb; }
</style>
