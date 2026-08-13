<script setup lang="ts">
import { computed } from 'vue'
import Sparkline from './charts/Sparkline.vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { Sensor } from '@/types/telemetry'

export interface Props {
  sensor: Sensor
  /** which colour the spark uses */
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  /** expected hz so we can draw a reference line */
  target?: number
}
const props = withDefaults(defineProps<Props>(), { accent: 'blue', target: undefined })

const telemetry = useTelemetryStore()
const samples = computed(() => telemetry.sensorBuffer(props.sensor.id))

const targetHz = computed(() => {
  if (props.target != null) return props.target
  const t = TARGET_HZ[props.sensor.id]
  return t == null ? undefined : t
})

const iconFor = computed(() => ICON_FOR[props.sensor.kind] ?? '◇')
const driftPct = computed(() => {
  if (!targetHz.value || targetHz.value === 0) return null
  return ((props.sensor.hz - targetHz.value) / targetHz.value) * 100
})
</script>

<script lang="ts">
const TARGET_HZ: Record<string, number> = {
  ld14: 10,
  astra_depth: 30,
  astra_rgb: 30,
  lift_cam: 30,
  imu: 200,
  odom: 50,
  mic_array: 16,
  magnet: 0,
}
const ICON_FOR: Record<string, string> = {
  lidar: '⟿',
  depth_cam: '⊡',
  rgb_cam: '◉',
  imu: '⚙',
  odom: '⤤',
  mic_array: '◈',
  temp: '🌡',
  magnet: '✪',
}
</script>

<template>
  <div class="sensor-card card-elevated">
    <div class="head">
      <div class="head-left">
        <span class="kind-icon">{{ iconFor }}</span>
        <div class="head-text">
          <div class="label">{{ sensor.label }}</div>
          <div class="detail mono">{{ sensor.detail || sensor.kind }}</div>
        </div>
      </div>
      <span class="chip" :class="`chip-${sensor.health}`">
        <span class="dot" :class="`dot-${sensor.health}`"></span>
        {{ sensor.health }}
      </span>
    </div>

    <div class="num-row">
      <span class="hz-num">{{ sensor.hz > 0 ? sensor.hz.toFixed(1) : '—' }}</span>
      <span class="hz-unit">Hz</span>
      <span v-if="targetHz != null && targetHz > 0" class="target mono">/ {{ targetHz }}</span>
      <span v-if="driftPct != null" class="drift mono" :class="Math.abs(driftPct) > 5 ? 'drift-warn' : 'drift-ok'">
        {{ driftPct >= 0 ? '+' : '' }}{{ driftPct.toFixed(1) }}%
      </span>
    </div>

    <div class="spark-wrap">
      <Sparkline :samples="samples" :target="targetHz" :accent="accent" />
    </div>
  </div>
</template>

<style scoped>
.sensor-card {
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: transform 0.22s var(--ease-out-quint), box-shadow 0.22s var(--ease-out-quint);
}
.sensor-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-elevated); }

.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.head-left { display: flex; align-items: flex-start; gap: 10px; min-width: 0; }
.kind-icon {
  font-size: 1.15rem; line-height: 1;
  color: var(--ink-tertiary);
  width: 22px; text-align: center;
  margin-top: 2px;
}
.head-text { min-width: 0; }
.label { font-size: 0.82rem; font-weight: 600; color: var(--ink-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.detail { font-size: 0.68rem; color: var(--ink-muted); margin-top: 2px; }

.num-row { display: flex; align-items: baseline; gap: 6px; }
.hz-num { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--ink-primary); }
.hz-unit { font-size: 0.7rem; color: var(--ink-tertiary); font-weight: 500; }
.target { font-size: 0.7rem; color: var(--ink-muted); margin-left: 4px; }
.drift { font-size: 0.66rem; margin-left: auto; padding: 2px 6px; border-radius: 999px; }
.drift-ok { color: #047857; background: rgba(16, 185, 129, 0.10); }
.drift-warn { color: #b45309; background: rgba(245, 158, 11, 0.12); }

.spark-wrap { height: 56px; }
</style>
