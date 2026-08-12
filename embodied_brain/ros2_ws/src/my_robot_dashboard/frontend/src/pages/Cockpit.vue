<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useUiStore } from '@/stores/ui'
import SlamScene from '@/components/three/SlamScene.vue'
import SensorCard from '@/components/SensorCard.vue'
import Sparkline from '@/components/charts/Sparkline.vue'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import Odometer from '@/components/premium/Odometer.vue'
import GlowCard from '@/components/premium/GlowCard.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'
import MagneticBtn from '@/components/premium/MagneticBtn.vue'
import RingChart from '@/components/premium/RingChart.vue'

const SENSOR_ACCENT: Record<string, 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'> = {
  ld14: 'blue',
  astra_depth: 'teal',
  astra_rgb: 'teal',
  lift_cam: 'emerald',
  imu: 'violet',
  odom: 'amber',
  mic_array: 'rose',
  magnet: 'amber',
}

const telemetry = useTelemetryStore()
const ui = useUiStore()

const kpis = computed(() => telemetry.packet?.kpis ?? [])
const tasks = computed(() => telemetry.packet?.tasks ?? [])
const alarms = computed(() => telemetry.packet?.alarms ?? [])
const sensors = computed(() => telemetry.packet?.sensors ?? [])
const cameras = computed(() => telemetry.packet?.cameras ?? [])
const bpuSlots = computed(() => telemetry.packet?.bpu_slots ?? [])
const furnaces = computed(() => telemetry.packet?.furnaces ?? [])
const provenance = computed(() => telemetry.packet?.provenance)
const sourceLabel = computed(() => provenance.value?.mode === 'live_partial' ? 'LIVE PARTIAL' : 'FIXTURE ONLY')
const batteryAvailable = computed(() => !provenance.value?.unavailable_fields.includes('battery'))

/** map kpi.value (string) → 0..100 percent for the gradient bar */
function kpiPercent(key: string, raw: string): number {
  const n = parseFloat(raw)
  if (Number.isNaN(n)) return 0
  if (key === 'battery' || key === 'cpu' || key === 'bpu') return Math.max(0, Math.min(100, n))
  if (key === 'ram') return Math.max(0, Math.min(100, (n / 8.0) * 100))
  if (key === 'vel') return Math.max(0, Math.min(100, (n / 1.0) * 100))
  if (key === 'alarms') return n === 0 ? 100 : 100
  return 50
}

const sensorOnline = computed(() => sensors.value.filter((s) => s.health === 'ok').length)
const cameraOnline = computed(() => cameras.value.filter((c) => c.online).length)

/** numeric value for Odometer slot (KPI.value is a string like "73.4") */
function kpiNumeric(raw: string): number {
  const n = parseFloat(raw)
  return Number.isFinite(n) ? n : 0
}
function kpiDecimals(raw: string): number {
  const dot = raw.indexOf('.')
  if (dot < 0) return 0
  return Math.min(2, raw.length - dot - 1)
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Cockpit · 主驾驶舱" gradient="aurora" />
        </h1>
        <p class="page-subtitle">
          实时全景 ·
          <span class="mono">{{ telemetry.isConnected ? `${telemetry.observedHz.toFixed(1)} Hz` : 'offline' }}</span>
          · seq <span class="mono">{{ telemetry.packet?.heartbeat.sequence ?? 0 }}</span>
          · uptime <span class="mono">{{ telemetry.packet?.heartbeat.uptime_s.toFixed(0) ?? 0 }}s</span>
        </p>
        <div class="source-line">
          <span class="source-chip" :class="provenance?.mode === 'live_partial' ? 'source-live' : 'source-fixture'">{{ sourceLabel }}</span>
          <span class="source-note">{{ provenance?.note ?? 'Waiting for source provenance.' }}</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn" @click="ui.togglePaused">
          {{ ui.streamPaused ? '▶ Resume' : '⏸ Pause' }} Stream
        </button>
        <MagneticBtn :strength="0.32" :radius="120">
          <button class="btn btn-primary" @click="ui.openDispatch">▶ Dispatch Task</button>
        </MagneticBtn>
      </div>
    </header>

    <!-- KPI strip — driven by store -->
    <div class="kpi-grid">
      <GlowCard
        v-for="k in kpis"
        :key="k.key"
        :color="`rgba(${k.accent === 'blue' ? '37, 99, 235' : k.accent === 'teal' ? '8, 145, 178' : k.accent === 'violet' ? '124, 58, 237' : k.accent === 'amber' ? '217, 119, 6' : k.accent === 'emerald' ? '5, 150, 105' : '225, 29, 72'}, 0.18)`"
      >
        <div v-tilt="{ max: 4, scale: 1.0 }" class="kpi-card card-elevated kpi-clickable" :title="`Deep dive ${k.label}`" @click="ui.openKpiDrill(k.key)">
          <div class="kpi-top">
            <span class="section-label">{{ k.label }}</span>
            <span class="chip" :class="`chip-${k.tone}`">{{ k.trend }}</span>
          </div>
          <div class="kpi-num-row">
            <span class="kpi-num">
              <Odometer v-if="k.value !== 'N/A'" :value="kpiNumeric(k.value)" :decimals="kpiDecimals(k.value)" />
              <span v-else class="kpi-na">N/A</span>
            </span>
            <span class="kpi-unit">{{ k.unit }}</span>
          </div>
          <div class="kpi-bar">
            <div class="kpi-bar-fill" :class="`bar-${k.accent}`" :style="{ width: `${kpiPercent(k.key, k.value)}%` }"></div>
          </div>
        </div>
      </GlowCard>
      <template v-if="!kpis.length">
        <div v-for="n in 6" :key="`skel-${n}`" class="kpi-card card-elevated">
          <div class="skeleton" style="height: 12px; width: 60%;"></div>
          <div class="skeleton" style="height: 28px; width: 80%; margin-top: 10px;"></div>
          <div class="skeleton" style="height: 4px; width: 100%; margin-top: 10px;"></div>
        </div>
      </template>
    </div>

    <!-- Hero — Three.js SLAM scene (embed mode = auto-orbit, no controls) -->
    <div class="hero card-floating">
      <BorderBeam :duration="11" :size="220" :radius="22" />
      <SlamScene :embed="true" mode="orbit-auto" :cinematic="true" />
      <div class="hero-rings">
        <RingChart v-if="telemetry.packet && batteryAvailable" :pct="telemetry.packet.battery.pct" label="Battery" accent="blue" :size="84" :stroke="7" />
        <RingChart v-if="telemetry.packet" :pct="telemetry.packet.host.bpu_pct" label="BPU" accent="violet" :size="84" :stroke="7" />
        <RingChart v-if="telemetry.packet" :pct="telemetry.packet.host.cpu_pct" label="CPU" accent="teal" :size="84" :stroke="7" />
      </div>
    </div>

    <!-- Sensor strip — 8 cards full width -->
    <div class="sensor-strip">
      <div class="sensor-strip-head">
        <span class="section-label">Sensors</span>
        <span class="chip chip-ok">{{ sensorOnline }} / {{ sensors.length || 8 }} online · 60s history</span>
      </div>
      <div class="sensor-grid">
        <SensorCard
          v-for="s in sensors"
          :key="s.id"
          :sensor="s"
          :accent="SENSOR_ACCENT[s.id] ?? 'blue'"
        />
        <div v-if="telemetry.packet && !sensors.length" class="empty-state">No verified sensor sources</div>
        <template v-else-if="!sensors.length">
          <div v-for="n in 8" :key="`sk-${n}`" class="sensor-skel card-elevated">
            <div class="skeleton" style="height: 14px; width: 60%;"></div>
            <div class="skeleton" style="height: 26px; width: 50%; margin-top: 8px;"></div>
            <div class="skeleton" style="height: 56px; width: 100%; margin-top: 12px;"></div>
          </div>
        </template>
      </div>
    </div>

    <!-- Two-column row: cameras+bpu+host / tasks+furnaces -->
    <div class="dual-grid">
      <div class="card-elevated panel">
        <div class="panel-head">
          <span class="section-label">Cameras · BPU · Host</span>
          <span class="chip chip-info">{{ cameraOnline }} cams · {{ bpuSlots.length }} slots</span>
        </div>
        <ul class="row-list">
          <li v-for="c in cameras" :key="c.id" class="row">
            <span class="dot" :class="c.online ? 'dot-ok' : 'dot-idle'"></span>
            <span class="row-label">{{ c.label }}</span>
            <span class="row-detail mono">{{ c.online ? (c.width > 0 && c.height > 0 ? `${c.width}×${c.height}@${c.fps.toFixed(0)}` : `${c.fps.toFixed(1)} Hz · ${c.provenance ?? 'live'}`) : 'offline' }}</span>
          </li>
          <li v-if="telemetry.packet && !cameras.length" class="empty-row">No verified camera or depth stream</li>
          <li class="row-divider"></li>
          <li v-for="b in bpuSlots" :key="b.id" class="row">
            <span class="dot" style="background: var(--accent-violet); box-shadow: 0 0 0 3px rgba(124,58,237,0.18);"></span>
            <span class="row-label">{{ b.label }}</span>
            <span class="row-detail mono">{{ b.last_ms.toFixed(1) }} ms · {{ b.util_available === false ? `${b.state ?? 'unknown'} / ${b.runtime ?? 'unknown'}` : `${b.util_pct.toFixed(0)}%` }}</span>
          </li>
          <li v-if="telemetry.packet && !bpuSlots.length" class="empty-row">No verified BPU runtime</li>
          <li class="row-divider"></li>
          <li class="host-row">
            <div class="host-meta">
              <span class="dot" style="background: var(--accent-blue); box-shadow: 0 0 0 3px rgba(37,99,235,0.18);"></span>
              <span>CPU</span>
              <span class="mono host-val">{{ telemetry.packet?.host.cpu_pct.toFixed(0) ?? '—' }}%</span>
            </div>
            <div class="host-spark"><Sparkline :samples="telemetry.hostHistory.cpu" :y-range="[0, 100]" accent="blue" /></div>
          </li>
          <li class="host-row">
            <div class="host-meta">
              <span class="dot" style="background: var(--accent-violet); box-shadow: 0 0 0 3px rgba(124,58,237,0.18);"></span>
              <span>BPU</span>
              <span class="mono host-val">{{ telemetry.packet?.host.bpu_pct.toFixed(0) ?? '—' }}%</span>
            </div>
            <div class="host-spark"><Sparkline :samples="telemetry.hostHistory.bpu" :y-range="[0, 100]" accent="violet" /></div>
          </li>
        </ul>
      </div>

      <!-- tasks + alarms + furnaces -->
      <div class="card-elevated panel">
        <div class="panel-head">
          <span class="section-label">Tasks + Furnaces</span>
          <span class="chip" :class="alarms.length ? 'chip-err' : 'chip-ok'">
            {{ alarms.length ? `${alarms.length} alarm` : 'all clear' }}
          </span>
        </div>
        <ul class="row-list">
          <li v-for="t in tasks" :key="t.id" class="row task-row">
            <div class="task-head">
              <span class="row-label">{{ t.name }}</span>
              <span class="chip" :class="t.status === 'running' ? 'chip-info' : t.status === 'queued' ? 'chip-idle' : 'chip-ok'">{{ t.status }}</span>
            </div>
            <div v-if="t.status === 'running'" class="task-bar">
              <div class="task-bar-fill" :style="{ width: `${t.progress_pct}%` }"></div>
            </div>
            <div class="task-meta mono">
              <span>{{ t.progress_pct.toFixed(0) }}%</span>
              <span v-if="t.eta_s !== null">ETA {{ t.eta_s.toFixed(0) }}s</span>
            </div>
          </li>
          <li v-if="telemetry.packet && !tasks.length" class="empty-row">No active bridge task</li>
          <li class="row-divider"></li>
          <li v-for="f in furnaces" :key="f.id" class="row furnace-row">
            <span class="dot" :class="f.state === 'heating' ? 'dot-warn' : f.state === 'fault' ? 'dot-err' : 'dot-idle'"></span>
            <span class="row-label">{{ f.id }}</span>
            <span class="row-detail mono">PV {{ f.pv.toFixed(0) }}°C / SV {{ f.sv.toFixed(0) }}°C</span>
          </li>
          <li v-if="telemetry.packet && !furnaces.length" class="empty-row">Furnace telemetry unavailable</li>
        </ul>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  max-width: 1680px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 22px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
}

.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.header-actions { display: flex; gap: 10px; }
.source-line { display: flex; align-items: center; gap: 8px; margin-top: 7px; }
.source-chip {
  display: inline-flex; align-items: center; height: 22px; padding: 0 7px;
  border-radius: 5px; font-family: 'JetBrains Mono Variable', monospace;
  font-size: 0.62rem; font-weight: 700; white-space: nowrap;
}
.source-live { color: #047857; background: rgba(5, 150, 105, 0.09); border: 1px solid rgba(5, 150, 105, 0.22); }
.source-fixture { color: #92400e; background: rgba(217, 119, 6, 0.09); border: 1px solid rgba(217, 119, 6, 0.24); }
.source-note { color: var(--ink-muted); font-size: 0.7rem; }

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 14px;
}
@media (max-width: 1280px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 720px)  {
  .kpi-grid { grid-template-columns: 1fr; }
  .kpi-grid > * { min-width: 0; }
  .page-header { flex-direction: column; align-items: stretch; gap: 12px; }
  .header-actions { justify-content: flex-start; flex-wrap: wrap; }
  .source-line { align-items: flex-start; flex-wrap: wrap; }
  .source-note { flex: 1 1 100%; line-height: 1.45; }
  .sensor-strip-head { align-items: flex-start; flex-direction: column; gap: 7px; }
  .hero { height: 320px; }
}

.kpi-card {
  padding: 16px 18px;
  display: flex; flex-direction: column; gap: 10px;
  transition: transform 0.22s var(--ease-out-quint), box-shadow 0.22s var(--ease-out-quint);
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-elevated); }
.kpi-clickable { cursor: pointer; }

.kpi-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kpi-num-row { display: flex; align-items: baseline; gap: 6px; }
.kpi-num { font-size: 2.25rem; line-height: 1.05; font-variant-numeric: tabular-nums; font-weight: 700; letter-spacing: 0; }
.kpi-na { font-size: 1.55rem; color: var(--ink-muted); letter-spacing: 0; }
.kpi-unit { font-size: 0.78rem; color: var(--ink-tertiary); font-weight: 500; }

.kpi-bar { height: 4px; border-radius: 999px; background: var(--line-hairline); overflow: hidden; }
.kpi-bar-fill { height: 100%; border-radius: 999px; transition: width 0.45s var(--ease-out-quint); }
.bar-blue    { background: linear-gradient(90deg, var(--accent-blue), #60a5fa); }
.bar-teal    { background: linear-gradient(90deg, var(--accent-teal), #22d3ee); }
.bar-emerald { background: linear-gradient(90deg, var(--accent-emerald), #34d399); }
.bar-violet  { background: linear-gradient(90deg, var(--accent-violet), #a78bfa); }
.bar-amber   { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
.bar-rose    { background: linear-gradient(90deg, var(--accent-rose), #fb7185); }

.hero {
  position: relative;
  height: 400px;
  overflow: hidden;
  border-radius: 22px;
  padding: 0;
}
.hero-rings {
  position: absolute;
  right: 18px;
  bottom: 18px;
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--line-divider);
  border-radius: 18px;
  z-index: 3;
}
[data-theme='dark'] .hero-rings { background: rgba(20, 25, 38, 0.55); }
.kpi-num { display: inline-flex; align-items: baseline; }

.sensor-strip { display: flex; flex-direction: column; gap: 10px; }
.sensor-strip-head { display: flex; align-items: center; justify-content: space-between; padding: 0 4px; }
.sensor-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
@media (max-width: 1280px) { .sensor-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 720px)  { .sensor-grid { grid-template-columns: 1fr; } }
.sensor-skel { padding: 14px 16px; }
.empty-state {
  grid-column: 1 / -1; min-height: 96px; display: grid; place-items: center;
  color: var(--ink-muted); border: 1px dashed var(--line-border); border-radius: 7px;
  font-size: 0.76rem;
}

.dual-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
@media (max-width: 1024px) { .dual-grid { grid-template-columns: 1fr; } }

.panel { padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; min-height: 280px; }
.host-row { display: flex; align-items: center; gap: 12px; padding: 6px 8px; }
.host-meta { display: flex; align-items: center; gap: 6px; min-width: 110px; font-size: 0.78rem; color: var(--ink-secondary); }
.host-val { margin-left: auto; }
.host-spark { flex: 1; height: 32px; min-width: 0; }
.panel-head { display: flex; align-items: center; justify-content: space-between; }

.row-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 8px;
  border-radius: 8px;
  font-size: 0.78rem;
  transition: background 0.18s var(--ease-out-quint);
}
.row:hover { background: rgba(241, 245, 249, 0.7); }
.row-label { flex: 1; color: var(--ink-secondary); }
.row-detail { color: var(--ink-tertiary); font-size: 0.72rem; }
.row-divider { height: 1px; background: var(--line-divider); margin: 6px 0; }
.empty-row { padding: 10px 8px; color: var(--ink-muted); font-size: 0.74rem; }

.task-row { flex-direction: column; align-items: stretch; gap: 4px; padding: 8px; background: var(--bg-elevated); }
.task-head { display: flex; align-items: center; justify-content: space-between; }
.task-bar { height: 4px; border-radius: 999px; background: var(--line-hairline); overflow: hidden; }
.task-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal)); transition: width 0.32s var(--ease-out-quint); }
.task-meta { display: flex; justify-content: space-between; color: var(--ink-muted); font-size: 0.7rem; }

.furnace-row { font-size: 0.78rem; }
</style>
