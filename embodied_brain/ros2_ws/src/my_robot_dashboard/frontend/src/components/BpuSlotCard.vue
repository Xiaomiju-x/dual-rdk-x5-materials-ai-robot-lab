<script setup lang="ts">
import { computed } from 'vue'
import Sparkline from './charts/Sparkline.vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { BpuSlot } from '@/types/telemetry'

interface Props {
  slot: BpuSlot
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
}
const props = withDefaults(defineProps<Props>(), { accent: 'violet' })

const telemetry = useTelemetryStore()
const samples = computed(() => telemetry.bpuSlotBuffer(props.slot.id))

const heatLevel = computed(() => {
  const u = props.slot.util_pct
  if (u >= 70) return 'hot'
  if (u >= 35) return 'warm'
  if (u >= 5)  return 'idle'
  return 'cold'
})

const heatColor = computed(() => ({
  hot:  '#e11d48',
  warm: '#d97706',
  idle: '#0891b2',
  cold: '#64748b',
}[heatLevel.value]))

const isActive = computed(() => props.slot.util_pct >= 5)
</script>

<template>
  <div class="bpu-card card-elevated" :class="{ 'bpu-active': isActive }">
    <div class="bpu-head">
      <div class="bpu-head-left">
        <span class="bpu-glyph" :style="{ background: heatColor }">
          <span class="glyph-inner" :class="{ pulse: isActive }"></span>
        </span>
        <div>
          <div class="bpu-label">{{ slot.label }}</div>
          <div class="bpu-model mono">{{ slot.model }}</div>
        </div>
      </div>
      <span class="bpu-size mono">{{ slot.size_mb.toFixed(2) }} MB</span>
    </div>

    <div class="bpu-util-row">
      <span class="util-num">{{ slot.util_pct.toFixed(0) }}</span>
      <span class="util-unit">%</span>
      <span class="util-tone mono" :class="`tone-${heatLevel}`">{{ heatLevel }}</span>
      <span class="latency mono">{{ slot.last_ms.toFixed(2) }} ms</span>
    </div>

    <div class="heat-bar">
      <div class="heat-fill" :style="{ width: `${slot.util_pct}%`, background: `linear-gradient(90deg, ${heatColor}aa, ${heatColor})` }"></div>
      <div class="heat-glow" :style="{ width: `${slot.util_pct}%`, background: heatColor, opacity: isActive ? 0.5 : 0.0 }"></div>
    </div>

    <div class="bpu-spark">
      <Sparkline :samples="samples" :y-range="[0, 100]" :accent="accent" />
    </div>
  </div>
</template>

<style scoped>
.bpu-card {
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 8px;
  position: relative;
  overflow: hidden;
}
.bpu-card.bpu-active::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 0% 0%, rgba(124, 58, 237, 0.08), transparent 60%);
  pointer-events: none;
}

.bpu-head { display: flex; align-items: center; justify-content: space-between; }
.bpu-head-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
.bpu-glyph {
  width: 22px; height: 22px; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 8px -2px currentColor;
  flex-shrink: 0;
}
.glyph-inner {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.6);
}
.glyph-inner.pulse { animation: pulse-dot 1.6s var(--ease-out-quint) infinite; }
@keyframes pulse-dot {
  0%   { box-shadow: 0 0 0 0 rgba(255,255,255,0.6); }
  70%  { box-shadow: 0 0 0 6px rgba(255,255,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}

.bpu-label { font-size: 0.82rem; font-weight: 600; color: var(--ink-primary); }
.bpu-model { font-size: 0.66rem; color: var(--ink-muted); margin-top: 1px; }
.bpu-size { font-size: 0.7rem; color: var(--ink-tertiary); }

.bpu-util-row { display: flex; align-items: baseline; gap: 6px; }
.util-num { font-size: 1.55rem; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; color: var(--ink-primary); }
.util-unit { font-size: 0.7rem; color: var(--ink-tertiary); }
.util-tone {
  font-size: 0.62rem; padding: 2px 7px; border-radius: 999px;
  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
}
.tone-hot  { color: #b91c1c; background: rgba(225, 29, 72, 0.10); }
.tone-warm { color: #b45309; background: rgba(217, 119, 6, 0.10); }
.tone-idle { color: #0e7490; background: rgba(8, 145, 178, 0.10); }
.tone-cold { color: #475569; background: rgba(100, 116, 139, 0.08); }
.latency { font-size: 0.7rem; color: var(--ink-muted); margin-left: auto; }

.heat-bar {
  position: relative;
  height: 6px;
  border-radius: 3px;
  background: rgba(15, 23, 42, 0.06);
  overflow: hidden;
}
.heat-fill {
  position: absolute; inset: 0 auto 0 0;
  border-radius: 3px;
  transition: width 0.4s var(--ease-out-quint);
}
.heat-glow {
  position: absolute; inset: 0 auto 0 0;
  border-radius: 3px;
  filter: blur(6px);
  transition: width 0.4s var(--ease-out-quint), opacity 0.4s ease;
}

.bpu-spark { height: 36px; }
</style>
