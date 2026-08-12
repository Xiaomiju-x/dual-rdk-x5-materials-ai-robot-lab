<script setup lang="ts">
import Sparkline from '@/components/charts/Sparkline.vue'
import type { HistorySample } from '@/stores/telemetry'
import type { Accent, Tone } from '@/types/telemetry'

interface Props {
  label: string
  value: string
  unit?: string
  icon?: string
  accent?: Accent
  tone?: Tone
  samples?: HistorySample[]
  trend?: string
}
withDefaults(defineProps<Props>(), { unit: '', icon: '', accent: 'blue', tone: 'idle', samples: () => [], trend: '' })
</script>

<template>
  <div class="stat-card card" :class="`accent-${accent}`">
    <div class="sc-head">
      <span class="sc-icon">{{ icon }}</span>
      <span class="sc-label section-label">{{ label }}</span>
      <span v-if="tone !== 'idle'" class="dot" :class="`dot-${tone}`" style="margin-left:auto"></span>
    </div>
    <div class="sc-value-row">
      <span class="kpi-num sc-value">{{ value }}</span>
      <span v-if="unit" class="sc-unit">{{ unit }}</span>
      <span v-if="trend" class="sc-trend mono">{{ trend }}</span>
    </div>
    <div v-if="samples.length" class="sc-spark">
      <Sparkline :samples="samples" :accent="accent" height="38px" />
    </div>
  </div>
</template>

<style scoped>
.stat-card { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; position: relative; overflow: hidden; }
.stat-card::after {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; border-radius: 3px 0 0 3px; opacity: 0.85;
}
.accent-blue::after { background: var(--accent-blue); }
.accent-teal::after { background: var(--accent-teal); }
.accent-emerald::after { background: var(--accent-emerald); }
.accent-violet::after { background: var(--accent-violet); }
.accent-amber::after { background: var(--accent-amber); }
.accent-rose::after { background: var(--accent-rose); }
.sc-head { display: flex; align-items: center; gap: 7px; }
.sc-icon { font-size: 0.95rem; }
.sc-label { color: var(--ink-muted); }
.sc-value-row { display: flex; align-items: baseline; gap: 6px; }
.sc-value { font-size: 1.75rem; line-height: 1; }
.sc-unit { font-size: 0.8rem; color: var(--ink-tertiary); font-weight: 600; }
.sc-trend { margin-left: auto; font-size: 0.7rem; color: var(--ink-muted); }
.sc-spark { margin: -2px -4px -4px; }
</style>
