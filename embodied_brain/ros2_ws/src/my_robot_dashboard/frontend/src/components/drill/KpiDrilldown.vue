<script setup lang="ts">
/**
 * KpiDrilldown — full-screen modal for deep-diving a single KPI.
 *
 *  - 60s scrubbable history (uses TimeSeries chart)
 *  - now vs 1h ago / threshold lines
 *  - related KPIs side cards (e.g. clicking battery shows voltage/current/temp_c)
 *  - close on backdrop, Esc, ×
 */
import { computed, onBeforeUnmount, onMounted, watch } from 'vue'
import { useTelemetryStore, type HistorySample } from '@/stores/telemetry'
import { useUiStore } from '@/stores/ui'
import TimeSeries from '@/components/charts/TimeSeries.vue'
import Sparkline from '@/components/charts/Sparkline.vue'
import Odometer from '@/components/premium/Odometer.vue'
import RingChart from '@/components/premium/RingChart.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'

const telemetry = useTelemetryStore()
const ui = useUiStore()

const KPI_DEFS: Record<string, {
  label: string
  unit: string
  accent: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  pctOf: (pkt: NonNullable<typeof telemetry.packet>) => number   // 0..100 for ring + threshold
  value: (pkt: NonNullable<typeof telemetry.packet>) => number
  series: () => HistorySample[]
  thresholds: { warn: number; crit: number }  // pct values
  related: { label: string; value: (pkt: NonNullable<typeof telemetry.packet>) => string }[]
  description: string
}> = {
  battery: {
    label: 'Battery',
    unit: '%',
    accent: 'blue',
    pctOf: (p) => p.battery.pct,
    value:  (p) => p.battery.pct,
    series: () => telemetry.hostHistory.cpu.map((s) => ({ t: s.t, v: telemetry.packet?.battery.pct ?? s.v })),
    thresholds: { warn: 25, crit: 10 },
    related: [
      { label: 'Voltage', value: (p) => `${p.battery.voltage.toFixed(2)} V` },
      { label: 'Current', value: (p) => `${p.battery.current.toFixed(2)} A` },
      { label: 'Temp',    value: (p) => `${p.battery.temp_c.toFixed(1)} °C` },
    ],
    description: '14.8V 4S LiPo, charge protocol from STM32 BMS line. 临界 10% 时自动巡航回 dock.',
  },
  cpu: {
    label: 'CPU',
    unit: '%',
    accent: 'teal',
    pctOf: (p) => p.host.cpu_pct,
    value:  (p) => p.host.cpu_pct,
    series: () => telemetry.hostHistory.cpu,
    thresholds: { warn: 70, crit: 90 },
    related: [
      { label: 'RAM',    value: (p) => `${p.host.ram_gb.toFixed(2)} / ${p.host.ram_total_gb.toFixed(1)} GB` },
      { label: 'BPU',    value: (p) => `${p.host.bpu_pct.toFixed(0)} %` },
      { label: 'CMA',    value: (p) => `${p.host.cma_mb.toFixed(1)} / ${p.host.cma_total_mb.toFixed(0)} MB` },
    ],
    description: 'RDK X5 4 ARM Cortex-A55 cores. Nav2 + slam_toolbox + 8 BPU drivers + ROS2 DDS.',
  },
  bpu: {
    label: 'BPU',
    unit: '%',
    accent: 'violet',
    pctOf: (p) => p.host.bpu_pct,
    value:  (p) => p.host.bpu_pct,
    series: () => telemetry.hostHistory.bpu,
    thresholds: { warn: 75, crit: 92 },
    related: [
      { label: 'CMA',    value: (p) => `${p.host.cma_mb.toFixed(1)} MB` },
      { label: 'Slots',  value: (p) => `${p.bpu_slots.length} active` },
      { label: 'CPU',    value: (p) => `${p.host.cpu_pct.toFixed(0)} %` },
    ],
    description: 'Bayes-e 10 TOPS INT8. YOLO-World + PP-OCRv4 + XFeat + MPPI cost MLP swap-loaded.',
  },
  ram: {
    label: 'RAM',
    unit: 'GB',
    accent: 'emerald',
    pctOf: (p) => (p.host.ram_gb / p.host.ram_total_gb) * 100,
    value:  (p) => p.host.ram_gb,
    series: () => telemetry.hostHistory.ram,
    thresholds: { warn: 75, crit: 90 },
    related: [
      { label: 'Total', value: (p) => `${p.host.ram_total_gb.toFixed(1)} GB` },
      { label: 'CMA',   value: (p) => `${p.host.cma_mb.toFixed(1)} MB` },
      { label: 'CPU',   value: (p) => `${p.host.cpu_pct.toFixed(0)} %` },
    ],
    description: 'X5 8GB DDR4 LPDDR. budget target ≤6GB so 2GB free for buffer.',
  },
  vel: {
    label: 'Velocity',
    unit: 'm/s',
    accent: 'amber',
    pctOf: (p) => (p.velocity.linear / 1.0) * 100,
    value:  (p) => p.velocity.linear,
    series: () => telemetry.hostHistory.cpu.map((s) => ({ t: s.t, v: telemetry.packet?.velocity.linear ?? 0 })),
    thresholds: { warn: 80, crit: 95 },
    related: [
      { label: 'Angular', value: (p) => `${p.velocity.angular.toFixed(2)} rad/s` },
      { label: 'Pose x',  value: (p) => `${p.pose.x.toFixed(2)} m` },
      { label: 'Pose y',  value: (p) => `${p.pose.y.toFixed(2)} m` },
    ],
    description: 'Cruise spec 0.42 m/s; turn-in-place 0.6 rad/s. STM32 reports back via 0xAA55 @ 50Hz.',
  },
  alarms: {
    label: 'Alarms',
    unit: '',
    accent: 'rose',
    pctOf: (p) => Math.min(100, (p.alarms.length / 5) * 100),
    value:  (p) => p.alarms.length,
    series: () => telemetry.hostHistory.cpu.map((s) => ({ t: s.t, v: telemetry.packet?.alarms.length ?? 0 })),
    thresholds: { warn: 40, crit: 80 },
    related: [
      { label: 'History', value: () => `${telemetry.alarmHistory.length} total` },
      { label: 'Source',  value: (p) => p.alarms[0]?.source ?? '—' },
      { label: 'Latest',  value: (p) => p.alarms[0]?.title ?? '—' },
    ],
    description: 'I1 ~ I6 4 类报警: 温度超界 / 偏差大 / 电源指示熄 / 烟雾或火焰.',
  },
}

const def = computed(() => ui.kpiDrillKey ? KPI_DEFS[ui.kpiDrillKey] : null)
const pkt = computed(() => telemetry.packet)

function close() { ui.closeKpiDrill() }

const open = computed(() => ui.kpiDrillKey != null && def.value != null)

function onKey(e: KeyboardEvent) { if (e.key === 'Escape' && open.value) close() }
onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
watch(open, (v) => { document.body.style.overflow = v ? 'hidden' : '' })

function backdropClick(e: MouseEvent) {
  if ((e.target as HTMLElement).classList.contains('kd-back')) close()
}

const currentSeries = computed(() => def.value?.series() ?? [])
const currentValue  = computed(() => (pkt.value && def.value) ? def.value.value(pkt.value) : 0)
const currentPct    = computed(() => (pkt.value && def.value) ? def.value.pctOf(pkt.value) : 0)

const stats = computed(() => {
  const arr = currentSeries.value
  if (!arr.length) return null
  let min = Infinity, max = -Infinity, sum = 0
  for (const s of arr) { if (s.v < min) min = s.v; if (s.v > max) max = s.v; sum += s.v }
  return { min, max, avg: sum / arr.length, n: arr.length }
})

const tone = computed(() => {
  if (!def.value) return 'idle'
  if (currentPct.value >= def.value.thresholds.crit) return 'err'
  if (currentPct.value >= def.value.thresholds.warn) return 'warn'
  return 'ok'
})
</script>

<template>
  <Teleport to="body">
    <Transition name="kd">
      <div v-if="open && def" class="kd-back" role="dialog" aria-label="KPI drill" @click="backdropClick">
        <div class="kd-card card-floating">
          <BorderBeam :duration="11" :size="220" :radius="22" :color-from="`rgba(37, 99, 235, 0.85)`" :color-to="`rgba(124, 58, 237, 0.85)`" />

          <header class="kd-head">
            <div class="kd-title">
              <span class="section-label">KPI · Deep Dive</span>
              <h2 class="kd-name">{{ def.label }}</h2>
              <p class="kd-desc">{{ def.description }}</p>
            </div>
            <button class="kd-x" aria-label="close" @click="close">×</button>
          </header>

          <div class="kd-grid">
            <!-- left: live value + ring -->
            <div class="kd-now">
              <RingChart :pct="currentPct" :accent="def.accent" :size="180" :stroke="14" suffix="" />
              <div class="kd-val-row">
                <span class="kd-val mono"><Odometer :value="currentValue" :decimals="2" /></span>
                <span class="kd-unit">{{ def.unit }}</span>
              </div>
              <span class="chip" :class="`chip-${tone}`">
                {{ tone === 'ok' ? 'nominal' : tone === 'warn' ? 'warning' : 'critical' }}
              </span>
              <div class="kd-thresh mono">
                warn ≥ {{ def.thresholds.warn }}% · crit ≥ {{ def.thresholds.crit }}%
              </div>
            </div>

            <!-- right: history series -->
            <div class="kd-history">
              <div class="kd-history-head">
                <span class="section-label">Last 60 s</span>
                <span v-if="stats" class="kd-stat-row mono">
                  min {{ stats.min.toFixed(2) }} · avg {{ stats.avg.toFixed(2) }} · max {{ stats.max.toFixed(2) }} · n={{ stats.n }}
                </span>
              </div>
              <TimeSeries
                :series="[{ name: def.label, samples: currentSeries, accent: def.accent, target: def.thresholds.warn }]"
                :y-label="def.unit"
                height="220px"
              />
              <div class="kd-spark-grid">
                <div class="kd-spark">
                  <span class="section-label">CPU</span>
                  <Sparkline :samples="telemetry.hostHistory.cpu" :y-range="[0, 100]" accent="blue" />
                </div>
                <div class="kd-spark">
                  <span class="section-label">BPU</span>
                  <Sparkline :samples="telemetry.hostHistory.bpu" :y-range="[0, 100]" accent="violet" />
                </div>
                <div class="kd-spark">
                  <span class="section-label">RAM</span>
                  <Sparkline :samples="telemetry.hostHistory.ram" :y-range="[0, telemetry.packet?.host.ram_total_gb ?? 8]" accent="emerald" />
                </div>
              </div>
            </div>
          </div>

          <!-- related KPIs -->
          <footer v-if="def && pkt" class="kd-related">
            <span class="section-label">Related</span>
            <div class="kd-related-grid">
              <div v-for="r in def.related" :key="r.label" class="kd-related-cell">
                <span class="kd-related-label">{{ r.label }}</span>
                <span class="kd-related-val mono">{{ r.value(pkt) }}</span>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.kd-back {
  position: fixed; inset: 0; z-index: 90;
  background: rgba(11, 18, 32, 0.55);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  display: flex; align-items: flex-start; justify-content: center;
  padding-top: 8vh;
}
.kd-card {
  position: relative;
  width: 920px;
  max-width: calc(100vw - 32px);
  max-height: 84vh;
  overflow-y: auto;
  padding: 22px 26px;
  display: flex; flex-direction: column;
  gap: 18px;
  border-radius: 22px;
}

.kd-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.kd-title { display: flex; flex-direction: column; gap: 4px; }
.kd-name { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.015em; color: var(--ink-primary); margin: 0; }
.kd-desc { font-size: 0.78rem; color: var(--ink-tertiary); margin: 0; max-width: 640px; }
.kd-x {
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  color: var(--ink-tertiary);
  font-size: 1.4rem; cursor: pointer; border-radius: 8px;
  flex-shrink: 0;
}
.kd-x:hover { background: var(--bg-card); color: var(--ink-primary); }

.kd-grid {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 22px;
}
@media (max-width: 760px) { .kd-grid { grid-template-columns: 1fr; } }

.kd-now {
  display: flex; flex-direction: column; align-items: center; gap: 12px;
  padding: 18px;
  border-radius: 16px;
  background: var(--bg-elevated);
}
.kd-val-row { display: flex; align-items: baseline; gap: 6px; }
.kd-val { font-size: 1.6rem; font-weight: 700; color: var(--ink-primary); letter-spacing: -0.015em; }
.kd-unit { font-size: 0.86rem; color: var(--ink-tertiary); }
.kd-thresh { font-size: 0.66rem; color: var(--ink-muted); }

.kd-history { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.kd-history-head { display: flex; align-items: baseline; justify-content: space-between; }
.kd-stat-row { font-size: 0.72rem; color: var(--ink-tertiary); }

.kd-spark-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.kd-spark {
  height: 56px;
  padding: 8px 10px;
  background: var(--bg-elevated);
  border-radius: 8px;
  display: flex; flex-direction: column; gap: 4px;
}

.kd-related { display: flex; flex-direction: column; gap: 8px; padding-top: 12px; border-top: 1px solid var(--line-divider); }
.kd-related-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
@media (max-width: 760px) { .kd-related-grid { grid-template-columns: 1fr 1fr; } }
.kd-related-cell {
  padding: 10px 12px;
  background: var(--bg-elevated);
  border-radius: 8px;
  display: flex; flex-direction: column; gap: 2px;
}
.kd-related-label { font-size: 0.66rem; color: var(--ink-muted); letter-spacing: 0.06em; text-transform: uppercase; }
.kd-related-val { font-size: 0.86rem; color: var(--ink-primary); font-weight: 600; }

.kd-enter-from { opacity: 0; }
.kd-enter-from .kd-card { transform: translateY(-20px) scale(0.96); }
.kd-leave-to { opacity: 0; }
.kd-leave-to .kd-card { transform: translateY(-12px) scale(0.98); }
.kd-enter-active, .kd-leave-active { transition: opacity 0.28s var(--ease-out-quint); }
.kd-enter-active .kd-card, .kd-leave-active .kd-card { transition: transform 0.28s var(--ease-out-quint); }
</style>
