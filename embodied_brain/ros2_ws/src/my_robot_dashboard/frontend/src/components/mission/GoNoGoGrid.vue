<script setup lang="ts">
/**
 * GoNoGoGrid — 8 subsystem cells, NASA flight-director GO/NO-GO grid.
 *
 * Two layouts: 'compact' (8 inline pill chips, default for topbar ribbon)
 * and 'panel' (full grid w/ labels + detail, used in mission HUD popovers).
 */
import { useMissionStore } from '@/stores/mission'

interface Props {
  layout?: 'compact' | 'panel'
}
const props = withDefaults(defineProps<Props>(), { layout: 'compact' })

const mission = useMissionStore()
</script>

<template>
  <div class="gng" :class="`gng-${props.layout}`">
    <span v-if="props.layout === 'compact'" class="gng-summary">
      <span class="gng-go">{{ mission.subsystemsGo }}</span>
      <span class="gng-sep">/</span>
      <span class="gng-all">{{ mission.subsystemsAll }}</span>
      <span class="gng-label">GO</span>
    </span>
    <div class="gng-cells">
      <div
        v-for="s in mission.subsystems"
        :key="s.id"
        class="gng-cell"
        :class="`gng-${s.status.toLowerCase().replace('-', '')}`"
        :title="`${s.label}: ${s.status} — ${s.detail}`"
      >
        <span class="gng-name">{{ s.label }}</span>
        <span v-if="props.layout === 'panel'" class="gng-detail mono">{{ s.detail }}</span>
        <span v-else class="gng-status">{{ s.status === 'NO-GO' ? '✕' : s.status === 'WARN' ? '!' : '✓' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.gng { display: inline-flex; align-items: center; gap: 8px; }

.gng-summary {
  display: inline-flex; align-items: baseline; gap: 2px;
  font-family: 'JetBrains Mono Variable', monospace;
}
.gng-go { font-size: 0.92rem; font-weight: 700; color: var(--status-ok); text-shadow: 0 0 12px rgba(16, 185, 129, 0.35); }
.gng-sep { color: var(--ink-muted); }
.gng-all { color: var(--ink-tertiary); font-size: 0.78rem; }
.gng-label { font-size: 0.58rem; letter-spacing: 0.18em; color: var(--ink-muted); font-weight: 700; padding-left: 4px; }

.gng-cells { display: inline-flex; gap: 3px; }
.gng-compact .gng-cells .gng-cell {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  border: 1px solid var(--line-divider);
  background: var(--bg-elevated);
  font-family: 'JetBrains Mono Variable', monospace;
  transition: background 0.18s var(--ease-out-quint);
}
.gng-compact .gng-cell .gng-status { font-size: 0.7rem; }
.gng-compact .gng-go .gng-status   { color: var(--status-ok); }
.gng-compact .gng-warn .gng-status { color: var(--status-warn); }
.gng-compact .gng-nogo .gng-status { color: var(--status-err); }

.gng-cell.gng-go { background: rgba(16, 185, 129, 0.08); color: #047857; border-color: rgba(16, 185, 129, 0.25); }
.gng-cell.gng-warn { background: rgba(245, 158, 11, 0.08); color: #b45309; border-color: rgba(245, 158, 11, 0.30); animation: pulseSoft 2.2s ease-in-out infinite; }
.gng-cell.gng-nogo { background: rgba(239, 68, 68, 0.10); color: #b91c1c; border-color: rgba(239, 68, 68, 0.35); animation: pulseSoft 1.6s ease-in-out infinite; }

[data-theme='dark'] .gng-cell.gng-go   { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; }
[data-theme='dark'] .gng-cell.gng-warn { background: rgba(245, 158, 11, 0.18); color: #fcd34d; }
[data-theme='dark'] .gng-cell.gng-nogo { background: rgba(239, 68, 68, 0.22); color: #fca5a5; }

/* panel layout */
.gng-panel { display: block; }
.gng-panel .gng-cells {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
}
.gng-panel .gng-cell {
  display: flex; flex-direction: column; gap: 2px;
  padding: 8px 10px;
  border-radius: 8px;
}
.gng-panel .gng-name { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
.gng-panel .gng-detail { font-size: 0.66rem; color: inherit; opacity: 0.75; }

@media (prefers-reduced-motion: reduce) {
  .gng-cell.gng-warn, .gng-cell.gng-nogo { animation: none; }
}
[data-reduce-motion='true'] .gng-cell.gng-warn,
[data-reduce-motion='true'] .gng-cell.gng-nogo { animation: none; }
</style>
