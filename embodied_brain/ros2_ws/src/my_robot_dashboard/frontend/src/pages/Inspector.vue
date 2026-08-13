<script setup lang="ts">
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import TimeSeries from '@/components/charts/TimeSeries.vue'
import Sparkline from '@/components/charts/Sparkline.vue'
import LidarPolar from '@/components/charts/LidarPolar.vue'
import type { HistorySample } from '@/stores/telemetry'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import Odometer from '@/components/premium/Odometer.vue'

const telemetry = useTelemetryStore()
const sensors = computed(() => telemetry.packet?.sensors ?? [])

const RANGES = [
  { id: '60s',  label: '60s',  ms:   60_000 },
  { id: '5m',   label: '5m',   ms:  300_000 },
  { id: 'all',  label: 'live', ms: null    },
] as const
type RangeId = (typeof RANGES)[number]['id']
const range = ref<RangeId>('60s')

function trim(buf: HistorySample[], rangeMs: number | null): HistorySample[] {
  if (rangeMs == null) return buf
  if (buf.length === 0) return buf
  const cutoff = Date.now() - rangeMs
  const idx = buf.findIndex((s) => s.t >= cutoff)
  return idx <= 0 ? buf : buf.slice(idx)
}

function bufFor(id: string): HistorySample[] {
  const rmsObj = RANGES.find((r) => r.id === range.value)
  return trim(telemetry.sensorBuffer(id), rmsObj?.ms ?? null)
}

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

const TARGETS: Record<string, number> = {
  ld14: 10, astra_depth: 30, astra_rgb: 30, lift_cam: 30, imu: 200, odom: 50, mic_array: 16, magnet: 0,
}

/** stats from current trim window */
function statsOf(buf: HistorySample[]) {
  if (!buf.length) return null
  let min = Infinity, max = -Infinity, sum = 0
  for (const s of buf) {
    if (s.v < min) min = s.v
    if (s.v > max) max = s.v
    sum += s.v
  }
  const avg = sum / buf.length
  let varSum = 0
  for (const s of buf) varSum += (s.v - avg) ** 2
  const std = Math.sqrt(varSum / buf.length)
  return { min, max, avg, std, n: buf.length }
}

function exportCsv() {
  const lines = ['sensor_id,epoch_ms,hz']
  for (const s of sensors.value) {
    const buf = telemetry.sensorBuffer(s.id)
    for (const sample of buf) lines.push(`${s.id},${sample.t},${sample.v}`)
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `navcockpit_sensors_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// host series for top combined view
const hostSeries = computed(() => [
  { name: 'CPU %',  samples: trim(telemetry.hostHistory.cpu, RANGES.find((r) => r.id === range.value)?.ms ?? null), accent: 'blue' as const,    target: 70 },
  { name: 'BPU %',  samples: trim(telemetry.hostHistory.bpu, RANGES.find((r) => r.id === range.value)?.ms ?? null), accent: 'violet' as const,  target: 80 },
])
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Inspector · 传感器诊断" gradient="amber-rose" />
        </h1>
        <p class="page-subtitle">
          8 路传感器实时频率 + 滚动历史 · 偏差监测 · CSV 导出 ·
          <span class="mono">{{ telemetry.observedHz.toFixed(1) }} Hz</span>
        </p>
      </div>
      <div class="header-actions">
        <div class="seg">
          <button
            v-for="r in RANGES"
            :key="r.id"
            class="seg-btn"
            :class="{ active: range === r.id }"
            @click="range = r.id"
          >{{ r.label }}</button>
        </div>
        <button class="btn" @click="exportCsv">⤓ CSV</button>
      </div>
    </header>

    <!-- Host top: combined CPU + BPU + 雷达极坐标 (G3) -->
    <div class="host-row">
      <div class="card-elevated host-panel">
        <div class="panel-head">
          <span class="section-label">Host Load · CPU + BPU</span>
          <span class="chip chip-info">{{ telemetry.hostHistory.cpu.length }} samples</span>
        </div>
        <TimeSeries :series="hostSeries" y-label="%" height="200px" />
      </div>
      <div class="card-elevated host-panel">
        <LidarPolar />
      </div>
    </div>

    <!-- Sensor cards -->
    <div class="sensor-detail-grid">
      <div v-for="s in sensors" :key="s.id" v-tilt="{ max: 3 }" class="card-elevated detail-card">
        <div class="detail-head">
          <div>
            <div class="detail-label">{{ s.label }}</div>
            <div class="detail-sub mono">{{ s.kind }} · {{ s.detail }}</div>
          </div>
          <span class="chip" :class="`chip-${s.health}`">{{ s.health }}</span>
        </div>

        <div class="detail-num">
          <span class="num">
            <Odometer v-if="s.hz > 0" :value="s.hz" :decimals="2" />
            <template v-else>—</template>
          </span>
          <span class="unit">Hz</span>
          <span v-if="TARGETS[s.id] && TARGETS[s.id] > 0" class="target mono">
            target {{ TARGETS[s.id] }} · {{ ((s.hz - TARGETS[s.id]!) / TARGETS[s.id]! * 100).toFixed(1) }}%
          </span>
        </div>

        <div class="detail-chart">
          <TimeSeries
            :series="[{ name: 'Hz', samples: bufFor(s.id), accent: SENSOR_ACCENT[s.id] ?? 'blue', target: TARGETS[s.id] }]"
            y-label="Hz"
            height="180px"
          />
        </div>

        <div v-if="statsOf(bufFor(s.id))" class="detail-stats">
          <div v-for="st in [statsOf(bufFor(s.id))!]" :key="`stats-${s.id}-${st.n}`" class="stat">
            <div><span class="stat-label">min</span><span class="stat-val mono">{{ st.min.toFixed(2) }}</span></div>
            <div><span class="stat-label">avg</span><span class="stat-val mono">{{ st.avg.toFixed(2) }}</span></div>
            <div><span class="stat-label">max</span><span class="stat-val mono">{{ st.max.toFixed(2) }}</span></div>
            <div><span class="stat-label">σ</span><span class="stat-val mono">{{ st.std.toFixed(3) }}</span></div>
            <div><span class="stat-label">n</span><span class="stat-val mono">{{ st.n }}</span></div>
          </div>
        </div>

        <div v-else class="detail-stats-empty">
          <Sparkline :samples="[]" />
        </div>
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
  gap: 18px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.header-actions { display: flex; gap: 10px; align-items: center; }

.seg {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 8px;
  overflow: hidden;
}
.seg-btn {
  background: transparent;
  border: none;
  padding: 6px 14px;
  font-size: 0.78rem;
  color: var(--ink-tertiary);
  cursor: pointer;
  font-weight: 500;
  transition: background 0.15s var(--ease-out-quint), color 0.15s var(--ease-out-quint);
}
.seg-btn:hover { color: var(--ink-primary); background: rgba(241, 245, 249, 0.6); }
.seg-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white;
}

.host-row { display: grid; grid-template-columns: 1fr 400px; gap: 14px; }
@media (max-width: 980px) { .host-row { grid-template-columns: 1fr; } }
.host-panel { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; }

.sensor-detail-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
}
@media (max-width: 1280px) { .sensor-detail-grid { grid-template-columns: 1fr; } }

.detail-card { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
.detail-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.detail-label { font-size: 0.92rem; font-weight: 600; color: var(--ink-primary); }
.detail-sub { font-size: 0.68rem; color: var(--ink-muted); margin-top: 2px; }

.detail-num { display: flex; align-items: baseline; gap: 6px; }
.num { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em; font-variant-numeric: tabular-nums; color: var(--ink-primary); }
.unit { font-size: 0.78rem; color: var(--ink-tertiary); font-weight: 500; }
.target { font-size: 0.7rem; color: var(--ink-muted); margin-left: auto; }

.detail-chart { width: 100%; }
.detail-stats { padding: 6px 0 0; }
.stat { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.72rem; color: var(--ink-tertiary); }
.stat-label { color: var(--ink-muted); margin-right: 6px; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.66rem; }
.stat-val { color: var(--ink-primary); font-weight: 600; }
.detail-stats-empty { height: 16px; }
</style>
